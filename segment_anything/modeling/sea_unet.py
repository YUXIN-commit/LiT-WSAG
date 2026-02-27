import torch
from torch import nn
import os
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F

Dropout=0.5
class Deconv(nn.Module):
    def __init__(self, in_channel, out_channel,up_set='conv'):
        super(Deconv, self).__init__()
        if up_set == 'conv':
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                #nn.ReLU()
            )
        elif up_set == 'bilinear':
            self.layer = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layer(x)

class Att(nn.Module):
    def __init__(self,channel, up_channel, Dropout=Dropout):
        super(Att, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(up_channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
    def forward(self, x,g):
        x1 = self.conv1(x)
        g1 = self.conv2(g)
        add = self.conv3(x1+g1)
        return x*add
        #消融 去掉Att
        #return x


class Conv_block(nn.Module):
    def __init__(self,in_channel, out_channel, Dropout=Dropout):
        super(Conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(Dropout),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)


class SE_Res(nn.Module):
    def __init__(self, channel, ratio):
        super(SE_Res, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, padding_mode ='reflect', bias=False),
            nn.BatchNorm2d(channel)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // ratio, channel, 1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.conv(x)
        # (batch,channel,height,width) (2,512,8,8)
        b, c, _, _ = y.size()
        # 全局平均池化 (2,512,8,8) -> (2,512,1,1) -> (2,512)
        a = self.avg_pool(x).view(b, c)
        # (2,512) -> (2,512//reducation) -> (2,512) -> (2,512,1,1)
        a = self.fc(a).view(b, c, 1, 1)
        # (2,512,8,8)* (2,512,1,1) -> (2,512,8,8)
        pro = a * y
        #return self.relu(x + pro)
        #消融
        return x

class SEA_Unet_5(nn.Module):
    def __init__(self, input_channels=1, num_classes=1,  **kwargs):
        print('-----------------------------laod sea_unet_5-----------------------------------------')
        super(SEA_Unet_5, self).__init__()
        #         0    1    2    3    4    5    6    7
        filter = [64, 128, 256, 512, 512, 512, 512, 512]
        #           0     1     2     3     4    5    6    7
        Deconvs = [512, 1024, 1024, 1024, 1024, 512, 256, 128] 
        self.cov1 = Conv_block(input_channels, 64)
        self.cov2 = Conv_block(64,128)
        self.cov3 = Conv_block(128, 256)
        self.cov4 = Conv_block(256, 512)
        self.cov5 = Conv_block(512, 512)
        self.cov6 = Conv_block(512, 512)
        

        self.se1 = SE_Res(512, 8)
        self.se2 = SE_Res(256, 8)
        self.se3 = SE_Res(128, 8)
        self.se4 = SE_Res(64, 8)
        
        
        

        self.dev1 = Deconv(512, 512)
        self.dev2 = Deconv(1024, 256)
        self.dev3 = Deconv(512, 128)
        self.dev4 = Deconv(256, 64)
        
        
        
        self.att1 = Att(512, 512)
        self.att2 = Att(256, 256)
        self.att3 = Att(128, 128)
        self.att4 = Att(64, 64)
        
       
        

        self.out = nn.ConvTranspose2d(Deconvs[7], num_classes, kernel_size=4, stride=2, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
            # print("x=",x.shape)
            r1 = self.cov1(x)#1/2
            #print("r1=",r1)
            r2 = self.cov2(r1)#1/4
            r3 = self.cov3(r2)#1/8
            r4 = self.cov4(r3)#1/16
            r5 = self.cov5(r4)#1/32
            
           
         

   

           
            r9 = torch.cat([self.att1(self.dev1(r5), r4), self.se1(r4)], dim=1)
            r10 = torch.cat([self.att2(self.dev2(r9), r3), self.se2(r3)], dim=1)
            r11 = torch.cat([self.att3(self.dev3(r10), r2), self.se3(r2)], dim=1)
            r12 = torch.cat([self.att4(self.dev4(r11), r1), self.se4(r1)], dim=1)
            
            
            

            
            out = self.out(r12)
            out_simgmoid=self.Th(out)

            #return out
            return out_simgmoid
