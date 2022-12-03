import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from model import BertClassifier
from dataloader import TwitterDataset


class ModelTraining(pl.LightningModule):
    def __init__(self, dataset: TwitterDataset):
        super().__init__()
        self.model = BertClassifier(num_label=3)
        
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy()

        self.train_dl, self.val_dl, self.test_dl = dataset.get_dataloaders([0.8, 0.1, 0.1])

        # self.logger = TensorBoardLogger("tb_logs", name="my_model")

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        # b = x.size(0)
        # x = x.view(b, -1)

        logits = self.model(x)
        J = self.loss(logits, y)
        acc = self.acc(logits, y)

        pbar = {'acc': {'train': acc}, 'losses': {'train': J}, 'loss': J}
        return pbar

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["losses"]['train'] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"]['train'] for x in outputs]).mean()
        self.logger.experiment.add_scalars('avg_loss',{"train": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars('avg_acc',{"train": avg_acc}, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # b = x.size(0)
        # x = x.view(b, -1)

        logits = self.model(x)
        J = self.loss(logits, y)
        acc = self.acc(logits, y)

        pbar = {'acc': {'val': acc}, 'loss': {'val': J}}

        return pbar

    def validation_epoch_end(self, outputs):
        avg_va_loss = torch.stack([x["loss"]['val'] for x in outputs]).mean()
        avg_va_acc = torch.stack([x["acc"]['val'] for x in outputs]).mean()

        self.logger.experiment.add_scalars('avg_loss',{'valid': avg_va_loss}, self.current_epoch)
        self.logger.experiment.add_scalars('avg_acc',{'valid': avg_va_acc}, self.current_epoch)
    
    def train_dataloader(self):
        # train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
        # train_loader = DataLoader(train_data, batch_size=32)
        return self.train_dl

    def val_dataloader(self):
        # val_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
        # val_loader = DataLoader(val_data, batch_size=32)
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


class TwitterClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertClassifier(num_label=3)


# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, random_split
# from pytorch_lightning.loggers import TensorBoardLogger
# def train_model(model, optimizer, loss, train_loader, val_loader, num_loops, use_cuda=False):
#     if use_cuda:
#         model = model.cuda()

#     for iter in range(num_loops):
#         # training
#         losses = list()
#         accuracies = list()

#         for batch in train_loader:
#             x, y = batch
#             b = x.size(0)
#             x = x.view(b, -1)

#             if use_cuda:
#                 x = x.cuda()
#                 y = y.cuda()

#             # forward
#             logits = model(x)
#             # compute loss
#             J = loss(logits, y)
#             # clean gradient, you can also use optimizer to clear
#             model.zero_grad()
#             # accumulating the partial derivatives
#             J.backward()
#             # step opposite the gradient
#             optimizer.step() 

#             losses.append(J.item().cpu())
#             accuracies.append(y.eq(logits.detach().argmax(dim=1).cpu()).float().mean())

#         print(f'Epoch {iter+1} train loss: {torch.tensor(losses).mean():.4f}, accuracy: {torch.tensor(accuracies).mean():.4f}.')

#         # validation
#         model.eval()
#         losses = list()
#         accuracies = list()

#         for batch in val_loader:
#             x, y = batch
#             b = x.size(0)
#             x = x.view(b, -1)

            
#             if use_cuda:
#                 x = x.cuda()
#                 y = y.cuda()

#             # zero grad
#             with torch.no_grad():
#                 logits = model(x)
#             # compute loss
#             J = loss(logits, y)

#             losses.append(J.item().cpu())
#             accuracies.append(y.eq(logits.detach().argmax(dim=1).cpu()).float().mean())

#         print(f'Epoch {iter+1} val loss: {torch.Tensor(losses).mean():.4f}, accuracy: {torch.tensor(accuracies).mean():.4f}.')