import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gating_channels, skip_channels, intermediate_channels):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, intermediate_channels, 1),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, 1),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gating, skip):
        attention = self.relu(
            self.W_g(gating) + self.W_x(skip)
        )
        attention = self.psi(attention)

        return skip * attention


# Building U-Net
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# Building Attention U-Net with deep supervision
class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.att1 = AttentionGate(32, 32, 16)
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(32, 1, 1)

        self.aux3 = nn.Conv2d(128, 1, 1)
        self.aux2 = nn.Conv2d(64, 1, 1)
        self.aux1 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        s4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self.up3(d4)
        s3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        main_output = self.output(d1)

        aux_output3 = F.interpolate(
            self.aux3(d3), size=input_size,
            mode="bilinear", align_corners=False
        )

        aux_output2 = F.interpolate(
            self.aux2(d2), size=input_size,
            mode="bilinear", align_corners=False
        )

        aux_output1 = F.interpolate(
            self.aux1(d1), size=input_size,
            mode="bilinear", align_corners=False
        )

        return main_output, aux_output1, aux_output2, aux_output3


# Road U-Net
class RoadUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)


# Road Attention U-Net with deep supervision
class RoadAttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.att1 = AttentionGate(32, 32, 16)
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(32, 1, 1)

        self.aux3 = nn.Conv2d(128, 1, 1)
        self.aux2 = nn.Conv2d(64, 1, 1)
        self.aux1 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        u4 = self.up4(b)
        s4 = self.att4(u4, e4)
        d4 = self.dec4(torch.cat([u4, s4], dim=1))

        u3 = self.up3(d4)
        s3 = self.att3(u3, e3)
        d3 = self.dec3(torch.cat([u3, s3], dim=1))

        u2 = self.up2(d3)
        s2 = self.att2(u2, e2)
        d2 = self.dec2(torch.cat([u2, s2], dim=1))

        u1 = self.up1(d2)
        s1 = self.att1(u1, e1)
        d1 = self.dec1(torch.cat([u1, s1], dim=1))

        main = self.output(d1)

        aux3 = F.interpolate(
            self.aux3(d3), size=input_size,
            mode="bilinear", align_corners=False
        )

        aux2 = F.interpolate(
            self.aux2(d2), size=input_size,
            mode="bilinear", align_corners=False
        )

        aux1 = F.interpolate(
            self.aux1(d1), size=input_size,
            mode="bilinear", align_corners=False
        )

        return main, aux1, aux2, aux3


def build_model(name="unet", task="roads"):
    if task == "buildings":
        if name == "unet":
            return UNet()
        if name == "attention_unet":
            return AttentionUNet()

    elif task == "roads":
        if name == "unet":
            return RoadUNet()
        if name == "attention_unet":
            return RoadAttentionUNet()

    raise ValueError(f"Unknown model/task combination: {name}/{task}")