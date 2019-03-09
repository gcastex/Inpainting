
# UNet
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = down(1, 4)
        self.down2 = down(4, 16)
        self.down3 = down(16, 32)
        self.down4 = down(32, 64)
        self.up1 = up(64, 32)
        self.up2 = up(64, 16)
        self.up3 = up(32, 4)
        self.up4 = up(8, 4)
        self.outc = outconv(4, 1)
    def forward(self, x):
        #x1 = self.inc(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x,x4],dim=1)
        x = self.up2(x)
        x = torch.cat([x,x3],dim=1)
        x = self.up3(x)
        x = torch.cat([x,x2],dim=1)
        x = self.up4(x)
        #x = torch.cat([x,x1],dim=0)
        x = self.outc(x)
        return x

