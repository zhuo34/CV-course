import argparse
import os

from pytorch_lightning import callbacks
from resnet18 import myresnet18
import torchvision

import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from typing import Optional

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--dataset', metavar='PATH', type=str, default='datasets', help='dataset path')
    parser.add_argument('--log', metavar='PATH', type=str, default='out/cifar10/logs', help='log path')
    parser.add_argument('--modeldir', metavar='PATH', type=str, default='out/cifar10/models', help='model path')
    parser.add_argument('--out', metavar='PATH', type=str, default='out/cifar10/results', help='out path')

    parser.add_argument('--gpus', metavar='gpus', type=int, default=None, help='GPU id')
    parser.add_argument('--min_epochs', metavar='MIN_EPOCHS', type=int, default=50, help='min epochs')
    parser.add_argument('--max_epochs', metavar='MAX_EPOCHS', type=int, default=100, help='max epochs')
    parser.add_argument('--bz', metavar='batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', metavar='lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', metavar='weight_decay', type=float, default=1e-4, help='weight decay')

    args = parser.parse_args()

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=args.dataset,
        batch_size=args.bz,
        # num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    model = myresnet18(args)
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
    trainer.fit(model, datamodule=cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)