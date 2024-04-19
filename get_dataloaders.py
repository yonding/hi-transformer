from load_datasets import load_datasets
from torch.utils.data import Dataset, DataLoader

def get_dataloaders(args):
    X_miss_train, Z_miss_train, y_miss_train, X_miss_val, Z_miss_val, y_miss_val, X_miss_test, Z_miss_test, y_miss_test = load_datasets(args)
    
    train_dataset = MissingDataset(X_miss_train, Z_miss_train, y_miss_train)
    val_dataset = MissingDataset(X_miss_val, Z_miss_val, y_miss_val)
    test_dataset = MissingDataset(X_miss_test, Z_miss_test, y_miss_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print("\n----------------- DATA SHAPE -----------------")
    print(f"TRAIN  : ({len(train_loader.dataset)//args.batch_size * args.batch_size}, {next(iter(train_loader))[0].shape[1]})")
    print(f"VALID  : ({len(val_loader.dataset)//args.batch_size * args.batch_size}, {next(iter(val_loader))[0].shape[1]})")
    print(f"TEST   : ({len(test_loader.dataset)//args.batch_size * args.batch_size}, {next(iter(test_loader))[0].shape[1]})")
    print("----------------------------------------------\n")

    return train_loader, val_loader, test_loader


class MissingDataset(Dataset):
    def __init__(self, missing_data, complete_data, y):
        self.missing_data = missing_data
        self.complete_data = complete_data
        self.y = y

    def __len__(self):
        return len(self.missing_data)
    
    def __getitem__(self, idx):
        return self.missing_data[idx], self.complete_data[idx], self.y[idx]