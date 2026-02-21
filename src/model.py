import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # Output Head
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)
        
        e4 = self.enc4(p3)
        p4 = F.max_pool2d(e4, 2)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        u4 = self.up4(b)
        # Resize/Crop handling? Assuming 256x256 input which is power of 2, so shapes match.
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        out = self.final_conv(d1)
        return out

if __name__ == "__main__":
    # Test
    # History=3, S1(3)+S2(8)=11ch. Total=33. Terrain=5. Total In=38.
    # Forecast=5. Out=5.
    model = SimpleUNet(in_channels=38, out_channels=5)
    x = torch.randn(1, 38, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
