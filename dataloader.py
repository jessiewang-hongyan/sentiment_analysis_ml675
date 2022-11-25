from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders():
    # TODO: change the dataset to our custom one
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_data, [55000, 5000]) # training size 55000, val size 5000

    train_loader = DataLoader(train, batch_size=32)
    val_loader = DataLoader(val, batch_size=32)

    return train_loader, val_loader