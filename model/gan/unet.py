import torch
import torch.nn as nn
import datetime
import torchvision.transforms.functional as TF
from torchsummary import summary

from export import export


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        kernel_size = 2
        stride = 2
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=kernel_size, stride=stride
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        step = 2
        for idx in range(0, len(self.ups), step):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//step]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

    def test():
        x = torch.randn((3, 1, 161, 161))
        model = UNET(1, 1)
        preds = model(x)
        print(preds.shape)
        assert preds.shape == x.shape
        print("Success!")

    def export(self):
        export(self, torch.randn((3, 1, 161, 161)),
               f"./blah-unet-generator.onnx")


if __name__ == "__main__":
    UNET.test()
    model = UNET(1, 1, [64, 128])
    model.export()
