import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from dataset import StereoVisionDataset
from deep3d import Deep3dNet


def shift(tensor, i: int):
    shifted = torch.roll(tensor, (i,), -1)
    shifted[:, :, :, :i] = tensor[:, :, :, :1]
    return shifted


class Deep3dModule(pl.LightningModule):
    def __init__(self, learning_rate: float = 2e-3, momentum=0.9, scheduler_epochs=250):
        """
        Create Deep3dModule

        :param learning_rate: learning rate for optimizer
        :param scheduler_epochs: number of epochs for learning rate scheduling

        """
        super().__init__()
        self.model = Deep3dNet((160, 384))
        self.save_hyperparameters()

    def forward(self, x):
        pred = self.model(x)
        probs = nn.functional.softmax(pred, dim=1)
        shifted = torch.stack([shift(x, i) for i in range(-16, 17)], -1)
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
        if idx == 0:
            indx = torch.randint(x.shape[0],size=(3,), device=self.device)
            inp = x.index_select(0, indx)
            gt = y.index_select(0, indx)
            samples = y_hat.index_select(0, indx)
            imgs = torch.stack([inp, gt, samples]).view(-1, *samples.shape[-3:])
            grid = make_grid(imgs, nrow=3)
            depth = nn.functional.softmax(self.model(inp[:1,]), dim=1).detach().view(-1, 1, 160, 384)
            depth_grid = make_grid(depth, nrow=6)
            self.logger.experiment.add_image("output image", grid, global_step=self.trainer.global_step)
            self.logger.experiment.add_image("depth", depth_grid, global_step=self.trainer.global_step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)
        scheduler = StepLR(optimizer, step_size=int(self.hparams.scheduler_epochs * 0.1), gamma=0.1)
        return [optimizer], [scheduler]


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
                transforms.Resize(size=(180, 432)),
                transforms.CenterCrop(size=(160, 384)),
                transforms.ToTensor()
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(size=(180, 432)),
                transforms.CenterCrop(size=(160, 384)),
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
        print(model(torch.randn(5, 3, 160, 384)).shape)
        datamodule = Deep3dDataModule(root='./data', batch_size=2, num_workers=0)
        datamodule.setup()
        dl = datamodule.train_dataloader()
        x, y = next(iter(dl))
        print(x)

        model.training_step((x, y), 1)

        transforms.ToPILImage()(x[0, ]).show()
        transforms.ToPILImage()(y[0, ]).show()

    _main()

