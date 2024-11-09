import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # 编码器（下采样）
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 中间层
        self.middle = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(512, 1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )
        
        # 解码器（上采样）
        self.decoder3 = self.upconv_block(1024, 256)
        self.decoder2 = self.upconv_block(512, 128)
        self.decoder1 = self.upconv_block(256, 64)

        # 输出层
        self.final = nn.Sequential(
            self.conv_block(128, 64),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def conv_block(self, in_channels, out_channels):
        """卷积块：包含卷积层、ReLU激活和批归一化"""
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        """上采样块：包括转置卷积和卷积块"""
        block = nn.Sequential(
            self.conv_block(in_channels, in_channels // 2),
            nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=2, stride=2),
        )
        return block

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)                         # torch.Size([2, 64, 512, 512])
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))     # torch.Size([2, 128, 256, 256])
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))     # torch.Size([2, 256, 128, 128])
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))     # torch.Size([2, 512, 64, 64])

        # 中间层
        dec4 = self.middle(enc4)                        # torch.Size([2, 512, 64, 64])

        # 解码器
        dec4 = torch.cat([dec4, enc4], dim=1)           # torch.Size([2, 1024, 64, 64])
        dec3 = self.decoder3(dec4)                      # torch.Size([2, 256, 128, 128])
        dec3 = torch.cat([dec3, enc3], dim=1)           # torch.Size([2, 512, 128, 128])
        dec2 = self.decoder2(dec3)                      # torch.Size([2, 128, 256, 256])
        dec2 = torch.cat([dec2, enc2], dim=1)           # torch.Size([2, 256, 256, 256])
        dec1 = self.decoder1(dec2)                      # torch.Size([2, 64, 512, 512])
        dec1 = torch.cat([dec1, enc1], dim=1)           # torch.Size([2, 128, 512, 512])
        
        # 输出
        return self.final(dec1)


if __name__ == '__main__':
    unet = UNet()
    x = torch.zeros(size=(2, 1, 512, 512))
    print(x.shape, unet(x).shape)   # torch.Size([2, 1, 512, 512]) torch.Size([2, 1, 512, 512])
    