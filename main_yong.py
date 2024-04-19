import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from mv_generate import mv_generate
from mv_rand_generate import mv_rand_generate
from split_datasets import split_and_convert_to_tensor
# from data import load_data
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from get_args import get_args

class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        batch_size=1,
        dim_model=8,
        num_head=8,
        num_layers=5,
        dim_ff=8,
        dropout=0.1,
        num_classes=3,
    ):
        super(TabularTransformer, self).__init__()
        self.num_features = num_features
        self.dim_model = dim_model
        self.batch_size = batch_size
        self.embedding = nn.Linear(1, dim_model)

        self.pos_encoder = PositionalEncoding(dim_model, dropout)

        encoder_layers = TransformerEncoderLayer(
            dim_model, num_head, dim_ff, dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.decoder = nn.Linear(dim_model, num_classes)
        self.imputation_decoder = nn.Linear(dim_model, 1)

    def forward(self, src):
        batch_size = src.size(0)
        srcs = []
        for i in range(self.num_features):
            srcs.append(self.embedding(torch.tensor([src[0,i]])))
        src = torch.stack(srcs, dim=1)
        src = torch.reshape(src, (-1, self.num_features, self.dim_model))
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)

        src = torch.cat((cls_tokens, src), dim=1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        cls_output = output[:, 0, :]
        tokens = torch.tensor([])
        for i in range(self.num_features):
            token = output[:, i+1, :]
            tokens = torch.cat((tokens, self.imputation_decoder(token)))
        cls_output = self.decoder(cls_output)
        return cls_output, tokens


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)
        )
        pe = torch.zeros(max_len, 1, dim_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
    

def train_one_epoch(model, loss_fn, optimizer, x, y, z):
    model.train()
    mse = nn.MSELoss()

    for data, target, complete in tqdm(zip(x, y, z)): 
        data = data.unsqueeze(0) 
        target = target.unsqueeze(0)
        complete = complete.unsqueeze(0)
        optimizer.zero_grad()
        y_pred, output = model(data)
        output = torch.reshape(output, (1, -1))
        loss = loss_fn(y_pred, target) + mse(output, complete)*10
        loss.backward()
    
    optimizer.step()
        
    return loss

def metric(y_pred, y_true, threshold=0.5):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred > threshold, 1, 0)

    for i in range(len(y_pred)):
        if (y_pred[i].item() == 1 and y_true[i] == 1):
            tp += 1
        elif (y_pred[i].item() == 0 and y_true[i] == 0):
            tn += 1
        elif (y_pred[i].item() == 1 and y_true[i] == 0):
            fp += 1
        elif (y_pred[i].item() == 0 and y_true[i] == 1):
            fn += 1
    
    accuracy = (tp+tn)/(tp+tn+fp+fn+1e-10)
    Sensitivity = tp/(tp+fn+1e-10)
    Specificity = tn/(tn+fp+1e-10)
    Precision = tp/(tp+fp+1e-10)
    f1 = (2*tp)/(2*tp+fp+fn+1e-10)

    return accuracy, Sensitivity, Specificity, Precision, f1

def validate(model, loss_fn, x, y, z, args):
    total_loss = 0
    total_mse_loss = 0
    mse = nn.MSELoss()
    pred = []

    for data, target, complete in zip(x, y, z):
        temp_grad = []
        data = data.unsqueeze(0)
        target = target.unsqueeze(0)
        complete = complete.unsqueeze(0)
        y_pred, output = model(data)
        output = torch.reshape(output, (1, -1))
        pred.append(torch.argmax(y_pred).item())
        loss = loss_fn(y_pred, target)
        mse_loss = mse(output, complete) * 10
        
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()

    accuracy, Sensitivity, Specificity, Precision, f1 = metric(pred, y, threshold=args.threshold/100)
    # auc = roc_auc_score(y, pred)
    total_loss /= len(x)
    total_mse_loss /= len(x)

    return total_loss, accuracy, Sensitivity, Specificity, Precision, f1, total_mse_loss #, auc

def train(args, model, loss_fn, optimizer, x_train, y_train, x_val, y_val,z_train, z_val):
    patience = 50
    best_loss = 1e9
    best_epoch = 0
    counter = 0    

    for epoch in tqdm(range(args.epochs)):

        # Train one epoch
        model.train()
        train_loss = train_one_epoch(model, loss_fn, optimizer, x_train, y_train,z_train)

        model.eval()
        val_loss, val_accuracy, Sensitivity, Specificity, Precision, f1, total_mse = validate(model, loss_fn, x_val, y_val, z_val, args)

        if (val_loss + total_mse*10) < best_loss:
            best_loss = val_loss + total_mse*10
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_{args.model_name}.pth')
            counter = 0
        else:
            counter += 1

        if counter > patience:
            break

        # if epoch % 10 == 0:
        if True:
            print(f'Epoch: {epoch}, Train loss: {train_loss}, val loss: {val_loss}, Valid acc: {val_accuracy}, valid mse: {total_mse}')

    print(f'Best epoch: {best_epoch}, Best loss: {best_loss}')

def test(args, model, loss_fn, x_test, y_test, z_test):
    model.load_state_dict(torch.load(f'best_{args.model_name}.pth'))
    model.eval()
    for threshold in range(10, 100, 10):
        args.threshold = threshold
        test_loss, test_accuracy, Sensitivity, Specificity, Precision, f1, total_mse = validate(model, loss_fn, x_test, y_test, z_test, args)
        print('#############################################')
        print(f'model: {args.model_name}')
        print(f'Threshold: {threshold}')
        print(f'Test accuracy: {test_accuracy}')
        print(f'Sensitivity(R find): {Sensitivity}')
        print(f'Specificity(X find): {Specificity}')
        print(f'Precision: {Precision}')
        print(f'f1: {f1}')
        print(f'Test loss: {test_loss}')
        print(f'Test mse: {total_mse}')
    print('#############################################')
    # print(f'auc: {auc}')
    # return auc


def main():

    args = get_args()
    
    if args.data == 'wine':
        # sparse_df, complete_df = mv_generate()
            # Generate the data
        x_df, z_df, y_df = mv_rand_generate(
            max_remove_count=args.max_remove_count,
            new_num_per_origin=args.new_num_per_origin,
        )

        # Split the data and transform it into tensors
        x_train, x_val, x_test, y_train, y_val, y_test, z_train, z_val, z_test = (
            split_and_convert_to_tensor(
                x_df, z_df, y_df, args.val_size, args.test_size, args.random_state
            )
        )
        model = TabularTransformer(num_features=13)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    auc = 0
    best_auc = 0
    counter = 0
    pat = 10
    # while auc < 0.7:

    # # load model
    # model.load_state_dict(torch.load(f'best_{args.model_name}.pth'))
    # model.eval()

    # test(args, model, loss_fn, x_test, y_test, z_test)
    # print()


    if True:
        train(args, model, loss_fn, optimizer, x_train, y_train, x_val, y_val, z_train, z_val)
        test(args, model, loss_fn, x_test, y_test, z_test)
        # auc = test(args, model, loss_fn, x_test, y_test)
        # if auc > best_auc:
        #     best_auc = auc
        #     if auc > 0.6:
        #         torch.save(model.state_dict(), f'best_{args.model_name}_{auc:03f}.pth')
        #     counter = 0
        # else:
        #     counter += 1

        # if counter > pat:             
        #     model = TabularTransformer(num_features=29)
        #     best_auc = 0
        #     counter = 0


if __name__ == "__main__":
    main()
