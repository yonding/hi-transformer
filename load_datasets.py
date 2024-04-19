from sklearn.datasets import load_wine # 178 rows | 13 features (float) | 3 classes
from sklearn.datasets import load_boston # 506 rows | 13 features (float) | regression
from sklearn.datasets import fetch_covtype # 581012 rows | 54 features (int) | 7 classes
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from generate_missing_data import generate_missing_data
    
def load_datasets(args):
    # 178 rows | 13 features (float) | 3 classes
    if args.dataset_name == 'wine':
        wine = load_wine()
        X_df = pd.DataFrame(wine.data, columns=wine.feature_names)
        y_df = pd.DataFrame(wine.target, columns=['target'])
    # 506 rows | 13 features (float) | regression
    elif args.dataset_name == 'boston':
        boston = load_boston()
        X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
        y_df = pd.DataFrame(boston.target, columns=['target'])
    # 581012 rows | 54 features (int) | 7 classes
    elif args.dataset_name == 'covtype':
        covtype = fetch_covtype()
        X_df = pd.DataFrame(covtype.data)
        y_df = pd.DataFrame(covtype.target, columns=["target"])
    elif args.dataset_name == 'ortho':
        df = pd.read_excel('./datasets/ortho_datasets.xlsx', sheet_name="all")
        columns_to_remove = [21, 22, 25, 26, 27, 28, 29, 30]
        df = df.drop(df.columns[columns_to_remove], axis=1).dropna()

        df.rename(columns={'class': 'target'}, inplace=True)
        df = df.dropna(axis=0, subset=["target", "name", "age", "sex"], how="any")
        df["target"] = df["target"].replace({"R": 1, "F": 0, "C": 0, "I": 0, "U": 0, "X": 0}) # R or not
        X_df = df.drop(["target", "name"], axis=1)
        y_df = df["target"]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df, 
        y_df,
        test_size = args.val_rate + args.test_rate,
        random_state=328,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=args.test_rate / (args.val_rate + args.test_rate),
        random_state=328,
    )

    X_miss_train, Z_miss_train, y_miss_train= generate_missing_data(args, X_train, y_train)
    X_miss_val, Z_miss_val, y_miss_val= generate_missing_data(args, X_val, y_val)
    X_miss_test, Z_miss_test, y_miss_test= generate_missing_data(args, X_test, y_test)

    X_miss_train = torch.tensor(X_miss_train.values, dtype=torch.float32)
    Z_miss_train = torch.tensor(Z_miss_train.values, dtype=torch.float32)

    X_miss_val = torch.tensor(X_miss_val.values, dtype=torch.float32)
    Z_miss_val = torch.tensor(Z_miss_val.values, dtype=torch.float32)

    X_miss_test = torch.tensor(X_miss_test.values, dtype=torch.float32)
    Z_miss_test = torch.tensor(Z_miss_test.values, dtype=torch.float32)

    args.num_features = X_miss_train.shape[1]
    args.num_classes = y_miss_train.nunique()
    
    return X_miss_train, Z_miss_train, y_miss_train, X_miss_val, Z_miss_val, y_miss_val, X_miss_test, Z_miss_test, y_miss_test