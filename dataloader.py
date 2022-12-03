import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split


class TwitterDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.attrs = torch.load('inputs.pt')
        self.labels = torch.load('labels.pt')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.attrs[idx, :]
        y = self.labels[idx]
        return X, y

    def get_dataloaders(self, ratio=[0.8, 0.1, 0.1]):
        # train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
        # train, val = random_split(train_data, [55000, 5000]) # training size 55000, val size 5000

        # train_loader = DataLoader(train, batch_size=32)
        # val_loader = DataLoader(val, batch_size=32)

        train_set, val_set, test_set = random_split(self, ratio)    

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True) #, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False) #, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False) #, num_workers=4)

        return train_loader, val_loader, test_loader