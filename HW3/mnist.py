import argparse
import os

from pytorch_lightning import callbacks
from LeNet5 import LeNet5

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from typing import Optional

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.dataset
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        self.batch_size = args.bz

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--dataset', metavar='PATH', type=str, default='datasets', help='dataset path')
    parser.add_argument('--log', metavar='PATH', type=str, default='out/lenet5/logs', help='log path')
    parser.add_argument('--modeldir', metavar='PATH', type=str, default='out/lenet5/models', help='model path')
    parser.add_argument('--out', metavar='PATH', type=str, default='out/lenet5/results', help='out path')

    parser.add_argument('--gpus', metavar='gpus', type=int, default=None, help='GPU id')
    parser.add_argument('--min_epochs', metavar='MIN_EPOCHS', type=int, default=50, help='min epochs')
    parser.add_argument('--max_epochs', metavar='MAX_EPOCHS', type=int, default=100, help='max epochs')
    parser.add_argument('--bz', metavar='batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', metavar='lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', metavar='weight_decay', type=float, default=1e-4, help='weight decay')

    args = parser.parse_args()

    mnist = MNISTDataModule(args)

    model = LeNet5(args)
    tb_logger = pl_loggers.TensorBoardLogger(args.log)
    es = EarlyStopping(monitor='val/loss', patience=10)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.modeldir,
        save_top_k=3,
        monitor="val/loss",
        save_last=True
    )
    trainer = pl.Trainer(
        gpus=args.gpus,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=tb_logger,
        callbacks=[es, checkpoint_callback],
    )
    trainer.fit(model, datamodule=mnist)
