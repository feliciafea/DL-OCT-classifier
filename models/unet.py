import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetClassifier(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # downsampling 
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # bridge
        self.bridge = DoubleConv(128, 256)
        
        # upsampling 
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(64, 32)
        
        # drooout
        self.dropout = nn.Dropout(0.2)

        # output classifier
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.down1(x)
        x2 = self.maxpool(x1)
        
        x3 = self.down2(x2)
        x4 = self.maxpool(x3)
        
        x5 = self.down3(x4)
        x6 = self.maxpool(x5)
        
        # bridge
        x7 = self.bridge(x6)
        
        # decoder
        x = self.up1(x7)
        x = torch.cat([x5, x], dim=1) # skip connect
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1) # skip 
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1) # skip
        x = self.up_conv3(x)
        x = self.dropout(x)
        
        return self.out(x)