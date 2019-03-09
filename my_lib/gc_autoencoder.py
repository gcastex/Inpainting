import numpy as np
import h5py
import torch
import torch.utils.data
from torch import nn
import matplotlib.pylab as plt
from random import randint
from my_lib.gc_nn_lib import *

# Autoencoder - conv_activation_BN
class autoencoder_cab(nn.Module):
    def __init__(self):
        super(autoencoder_cab, self).__init__()
        self.down1 = down(1, 4, bn ='ab')
        self.down2 = down(4, 16, bn ='ab')
        self.down3 = down(16, 32, bn ='ab')
        self.down4 = down(32, 64, bn ='ab')
        self.up1 = up(64, 32, bn ='ab')
        self.up2 = up(32, 16, bn ='ab')
        self.up3 = up(16, 4, bn ='ab')
        self.up4 = up(4, 4, bn ='ab')
        self.outc = outconv(4, 1)
    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        return x

# Autoencoder - conv_BN_activation
class autoencoder_cba(nn.Module):
    def __init__(self):
        super(autoencoder_cba, self).__init__()
        self.down1 = down(1, 4, bn ='ba', lbias = False)
        self.down2 = down(4, 16, bn ='ba', lbias = False)
        self.down3 = down(16, 32, bn ='ba', lbias = False)
        self.down4 = down(32, 64, bn ='ba', lbias = False)
        self.up1 = up(64, 32, bn ='ba', lbias = False)
        self.up2 = up(32, 16, bn ='ba', lbias = False)
        self.up3 = up(16, 4, bn ='ba', lbias = False)
        self.up4 = up(4, 4, bn ='ba', lbias = False)
        self.outc = outconv(4, 1)
    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        return x


class autoencoder_cba2(nn.Module):
    def __init__(self):
        super(autoencoder_cba2, self).__init__()
        self.down1 = down(1, 4, bn ='ba')
        self.down2 = down(4, 16, bn ='ba')
        self.down3 = down(16, 32, bn ='ba')
        self.down4 = down(32, 64, bn ='ba')
        self.up1 = up(64, 32, bn ='ba')
        self.up2 = up(32, 16, bn ='ba')
        self.up3 = up(16, 4, bn ='ba')
        self.up4 = up(4, 4, bn ='ba')
        self.outc = outconv(4, 1)
    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        return x





