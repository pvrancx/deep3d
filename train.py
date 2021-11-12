from pytorch_lightning.utilities.cli import LightningCLI

from lightning import Deep3dModule, Deep3dDataModule


def train():
    LightningCLI(Deep3dModule, Deep3dDataModule)


if __name__ == '__main__':
    train()

