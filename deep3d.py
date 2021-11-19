import torch
import torch.nn as nn
from typing import Tuple
from torchvision.models import vgg11


def create_fc_group(n_in: int, n_hidden: int):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.Dropout(p=0.5)
    )


def bilinear_weights(size: int, stride: int):
    c = (2 * stride - 1 - (stride % 2)) / size
    rows = torch.ones((size, size)) * torch.arange(size).view(-1, 1)
    cols = torch.ones((size, size)) * torch.arange(size).view(1, -1)
    # note: paper uses abs(i / (S - C)), code uses abs(i / S - C)
    weight = (1 - torch.abs(rows / (stride - c))) * (1 - torch.abs(cols / (stride - c)))
    return weight


class DeconvBlock(nn.Module):
    def __init__(self, n_features, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_features)
        self.conv = nn.Conv2d(n_features, 33, kernel_size=(3, 3), padding=(1, 1,))
        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose2d(33, 33, kernel_size=(kernel_size, kernel_size),
                                         stride=(stride, stride), padding=(padding, padding))

    def forward(self, x):
        out1 = self.bn(x)
        out2 = self.relu(self.conv(out1))
        return self.deconv(out2)


class Deep3dNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, int]):
        super().__init__()

        vgg11_net = vgg11(pretrained=True)
        feat_size = vgg11_net.features(torch.randn(1, 3, *input_shape)).shape[-2:]

        # extract trained vgg feature layers
        # Note: paper mentions vgg16 but repo uses vgg11 architecture
        layers = list(vgg11_net.features.children())
        self.feat1 = nn.Sequential(*layers[:3])
        self.feat2 = nn.Sequential(*layers[3:6])
        self.feat3 = nn.Sequential(*layers[6:11])
        self.feat4 = nn.Sequential(*layers[11:16])
        self.feat5 = nn.Sequential(*layers[16:])

        self.fc1 = create_fc_group(512 * feat_size[0] * feat_size[1], 512)
        self.fc2 = create_fc_group(512, 512)
        self.fc3 = nn.Linear(512, 33 * 12 * 5)

        # add deconv upsample blocks
        # Note: these values from paper repo are hardcoded for 384 x 160 inputs
        self.deconv1 = DeconvBlock(64, 1, 1, 0)
        self.deconv2 = DeconvBlock(128, 4, 2, 1)
        self.deconv3 = DeconvBlock(256, 8, 4, 2)
        self.deconv4 = DeconvBlock(512, 16, 8, 4)

        self.deconv5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(33, 33, kernel_size=(32, 32), stride=(16, 16), padding=(8, 8))
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(33, 33, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(33, 33, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'feat' in n:
                    # these layers are initialised from trained vgg
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                with torch.no_grad():
                    m.weight[:, :, ] = bilinear_weights(m.kernel_size[0], m.stride[0])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.feat1(x)
        out2 = self.feat2(out1)
        out3 = self.feat3(out2)
        out4 = self.feat4(out3)
        out5 = self.feat5(out4)

        out6 = self.fc1(torch.flatten(out5, 1, -1))
        out7 = self.fc2(out6)
        out8 = self.fc3(out7)

        pred5 = self.deconv5(out8.view(-1, 33, 5, 12))
        pred4 = self.deconv4(out4)
        pred3 = self.deconv3(out3)
        pred2 = self.deconv2(out2)
        pred1 = self.deconv1(out1)

        feat = pred1 + pred2 + pred3 + pred4 + pred5

        return self.up(feat)


if __name__ == '__main__':
    model = Deep3dNet((384, 160))
    print(model(torch.randn(1, 3, 384, 160)).shape)
    print(model.fc1[0].weight.shape)
