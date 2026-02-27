import torch
from torch import nn
from torch.nn import functional as F



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        print('----------------------------------------already load UNet but not use---------------------------------------')
        # 编码器部分
        self.enc_conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        self.enc_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU())
        self.enc_conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU())
        self.enc_conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU())
        self.enc_conv5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.ReLU())

        # 解码器部分
        self.up_conv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv5 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU())
        self.up_conv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU())
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU())
        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())

        # 最终的1x1卷积层，输出与输入同通道数
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        enc1 = self.enc_conv1(x)
        
        enc2 = self.enc_conv2(enc1)
        enc3 = self.enc_conv3(enc2)
        enc4 = self.enc_conv4(enc3)
        enc5 = self.enc_conv5(enc4)
        
        # 解码器部分
        x = self.up_conv5(enc5)
        x = torch.cat((x, enc4), dim=1)
        x = self.dec_conv5(x)
        x = self.up_conv4(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.dec_conv4(x)
        x = self.up_conv3(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.dec_conv3(x)
        x = self.up_conv2(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.dec_conv2(x)
        print('----------------------------------------------------------unet已经加载---------------------------------')
        # 最后的输出层
        x = self.final_conv(x)
        return x
