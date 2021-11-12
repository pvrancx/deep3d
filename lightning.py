import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms

from dataset import StereoVisionDataset
from deep3d import Deep3dNet


def shift(tensor, i: int):
    shifted = torch.roll(tensor, (i,), -1)
    shifted[:, :, :, :i] = tensor[:, :, :, :1]
    return shifted


class Deep3dModule(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        """
        Create Deep3dModule

        :param learning_rate: learning rate for optimizer
        """
        super().__init__()
        self.model = Deep3dNet((384, 160))
        self.save_hyperparameters()

    def forward(self, x):
        pred = self.model(x)
        probs = nn.functional.softmax(pred, dim=1)
        shifted = torch.stack([shift(x, i) for i in range(33)], -1)
        out = torch.einsum('bchwd,bdhw->bchw', shifted, probs)
        return out

    def training_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.l1_loss(y, y_hat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.l1_loss(y, y_hat)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class Deep3dDataModule(pl.LightningDataModule):
    def __init__(self, root: str, batch_size: int = 64, num_workers: int = 16):
        """
        Create datamodule for deep3d stereoscopic training

        :param root: dataset root directory
        :param batch_size:  batch size to use
        :param num_workers: number of workers for dataloaders
        """
        super().__init__()
        self.dataset_path = root
        self.train_set, self.val_set, self.test_set = None, None, None
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(size=(432, 180)),
                transforms.RandomCrop(size=(384, 160)),
                transforms.ToTensor()
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(size=(432, 180)),
                transforms.CenterCrop(size=(384, 160)),
                transforms.ToTensor()
            ]
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        train_path = os.path.join(self.dataset_path, 'train')
        test_path = os.path.join(self.dataset_path, 'test')

        full_set = StereoVisionDataset(train_path, transforms=self.train_transform)
        n_imgs = len(full_set)
        n_train = int(0.95 * n_imgs)
        self.train_set, self.val_set = random_split(full_set, [n_train, n_imgs - n_train])

        self.test_set = StereoVisionDataset(test_path, transforms=self.test_transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == '__main__':
    def _main():
        model = Deep3dModule()
        print(model(torch.randn(5, 3, 384, 160)).shape)
        model.training_step((torch.randn(5, 3, 384, 160), torch.randn(5, 3, 384, 160)), 1)
    _main()

