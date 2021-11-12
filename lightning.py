import pytorch_lightning as pl
import torch
import torch.nn as nn

from deep3d import Deep3dNet


def shift(tensor, i: int):
    shifted = torch.roll(tensor, (i,), -1)
    shifted[:, :, :, :i] = tensor[:, :, :, :1]
    return shifted


class Deep3dModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Deep3dNet((384, 160))

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    model = Deep3dModule()
    print(model(torch.randn(5, 3, 384, 160)).shape)
    model.training_step((torch.randn(5, 3, 384, 160),torch.randn(5, 3, 384, 160)), 1)
