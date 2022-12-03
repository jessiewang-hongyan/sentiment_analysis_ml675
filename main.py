import pytorch_lightning as pl
from trainer import ModelTraining
from dataloader import TwitterDataset
import torch


if __name__ == '__main__':
    X = torch.load('inputs.pt')
    y = torch.load('labels.pt')
    print(X.shape)
    print(y.shape)

    dataset = TwitterDataset()
    model = ModelTraining(dataset)

    trainer = pl.Trainer(
        max_epochs=5,
        # gpus=1
    )
    trainer.fit(model)

    # # training loop
    # num_loops = 10
    # train_model(model, optimizer, loss, train_loader, val_loader, num_loops, use_cuda)