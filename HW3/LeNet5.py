import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR
from utils import *


class LeNet5(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, 10),
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.batch_size = args.bz

    def forward(self, x):
        logits = self.net(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = cal_acc_from_logits(logits, y)
        self.log("train/loss", loss)
        self.log('train/accuracy', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = cal_acc_from_logits(logits, y)
        self.log("val/loss", loss, sync_dist=True, batch_size=self.batch_size)
        self.log('val/accuracy', acc, sync_dist=True, batch_size=self.batch_size)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = cal_acc_from_logits(logits, y)
        self.log("test/loss", loss, sync_dist=True, batch_size=self.batch_size)
        self.log('test/accuracy', acc, sync_dist=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1
            },
        }
