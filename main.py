from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from get_args import get_args, print_args
from get_dataloaders import get_dataloaders
from tabular_transformer import TabularTransformer

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=10000)

def train_one_epoch(args, model, train_loader, optimizer):

    cross_entropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    train_loss = 0

    model.train()
    for batch_idx, (missing_data, complete_data, y) in enumerate(train_loader):
        
        # Move to the CUDA device.
        missing_data = missing_data.to(args.device)
        complete_data = complete_data.to(args.device)
        y = y.to(args.device)

        optimizer.zero_grad()
        y_pred, recon_data = model(missing_data)

        loss = cross_entropy_loss(y_pred, y) + mse_loss(recon_data, complete_data) * 10
        loss.backward()
        train_loss += loss.item()
    
        optimizer.step()
    
    return train_loss / len(train_loader)


@torch.no_grad()
def valid_model(args, model, val_loader):
    cross_entropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    val_loss = 0

    model.eval()
    for batch_idx, (missing_data, complete_data, y) in enumerate(val_loader):
        
        # Move to the CUDA device.
        missing_data = missing_data.to(args.device)
        complete_data = complete_data.to(args.device)
        y = y.to(args.device)

        y_pred, recon_data = model(missing_data)

        loss = cross_entropy_loss(y_pred, y) + mse_loss(recon_data, complete_data) * args.num_features

        val_loss += loss.item()
        if batch_idx==3:
            print(f"\nM: {missing_data[:1, :5]}")
            print(f"R: {recon_data[:1, :5]}")
            print(f"C: {complete_data[:1, :5]}")
    
    return val_loss / len(val_loader)


def train_and_validate(args, model, train_loader, val_loader, optimizer):
    
    patience = 100
    best_loss = 1e9
    best_epoch = 0
    counter = 0    

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = train_one_epoch(args, model, train_loader, optimizer)

        model.eval()
        val_loss = valid_model(args, model, val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_{args.model_name}.pth')
            counter = 0
        else:
            counter += 1

        if counter > patience:
            break

        print(f'Epoch: {epoch}, Train loss: {train_loss}, val loss: {val_loss}')

    print(f'Best epoch: {best_epoch}, Best Val loss: {best_loss}')


@torch.no_grad()
def test(args, model, test_loader):
    
    model.load_state_dict(torch.load(f'best_{args.model_name}.pth'))

    model.eval()
    for threshold in range(10, 100, 10):
        args.threshold = threshold
        test_loss = valid_model(args, model, test_loader)
        print('===========================================')
        print(f'model: {args.model_name}')
        print(f'Threshold: {threshold}')
        print(f'Test loss: {test_loss}')
    print('===========================================')


def main(args):

    # 1. Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(args)
    
    # 2. Define the model and optimizer
    model = TabularTransformer(args).to(args.device)
    args.num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    print_args(args)
    # 3. Train and validate the model
    train_and_validate(args, model, train_loader, val_loader, optimizer)

    # 4. Test the model
    test(args, model, test_loader)


if __name__ == "__main__":
    args = get_args()
    main(args)