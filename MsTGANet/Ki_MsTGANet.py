import torch
import torch.nn as nn
from MsGCS import MsGCS as MsM
from Ki_MsTNL import MsTNL


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class MsTGANet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, feature_scale=2):
        super(MsTGANet, self).__init__()
        print("================ MsTGANet ================")

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]

        # Encoder
        # Utilized existing up-sample block (up_conv) and convolution block (conv_block)
        self.Up_conv1 = conv_block(ch_in=in_channels, ch_out=filters[0])

        self.Up_conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Up1 = up_conv(ch_in=filters[0], ch_out=filters[1])

        self.Up_conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[2])

        self.Up_conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[3])

        self.Up_conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])
        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[4])

        # MsTNL Block
        # Same command as U-MsTGANet, changes made in Ki-MsTNL
        self.trans = MsTNL(train_dim=512, filters=filters)

        # Decoder
        # Used same maxpool as U-MsTGANet, same convolution block (conv_block)
        # MsGCS remains the same in Ki_MsTGANet and U-MsTGANet (only change is order of size passed in)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])
        self.Att5 = MsM(F_g=filters[3], F_l=filters[3], F_int=filters[2], size=(512, 512))

        self.Conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])
        self.Att4 = MsM(F_g=filters[2], F_l=filters[2], F_int=filters[1], size=(256, 256))

        self.Conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])
        self.Att3 = MsM(F_g=filters[1], F_l=filters[1], F_int=filters[0], size=(128, 128))

        self.Conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        self.Att2 = MsM(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2, size=(64, 64))

        # Final Layer
        # Same final layer as U-MsTGANet
        self.Conv_1x1 = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # Up-Sampling Encoder
        x1 = self.Up_conv1(x)

        x2 = self.Up_conv2(x1)
        x2 = self.Up1(x2)

        x3 = self.Up_conv3(x2)
        x3 = self.Up2(x3)

        x4 = self.Up_conv4(x3)
        x4 = self.Up3(x4)

        x5 = self.Up_conv5(x4)
        x5 = self.Up4(x5)

        # MsTNL
        x5 = self.trans(x5, [x1, x2, x3, x4])

        # Maxpool Decoder
        d5 = self.Maxpool(x5)
        d5 = self.Conv5(d5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)

        d4 = self.Maxpool(d5)
        d4 = self.Conv4(d4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)

        d3 = self.Maxpool(d4)
        d3 = self.Conv3(d3)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)

        d2 = self.Maxpool(d3)
        d2 = self.Conv2(d2)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)

        d1 = self.Conv_1x1(d2)

        return torch.sigmoid(d1)
