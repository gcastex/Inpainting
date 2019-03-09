
import torch
#import torch.utils.data
import torch.nn as nn
#import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x



# Down layer. 
#'bn' param: Batch Normalization
#	'bn' == 'ba': BN before Activation
#       'bn' == 'ab': BN after Activation, Affine = False
#       'bn' == 'aba': BN after Activation, Affine = True
#	'bn' == 'none': Default. No BN.
class down(nn.Module):
    def __init__(self, in_ch, out_ch, bn='none', lbias = True):
        super(down, self).__init__()
        if bn == 'ba':
                # BN if before Activation, no bias for Conv layer
                self.convmp = nn.Sequential(
                    torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias = lbias),
                    nn.BatchNorm2d(out_ch)
                )
        else:
                # Conv layer with bias
                self.convmp = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # Activation
        self.convmp = nn.Sequential(
            self.convmp,
            nn.ReLU(True)
        )	
        # BN after Activation
        if bn == 'ab':
                self.convmp = nn.Sequential(
                    self.convmp,
                    nn.BatchNorm2d(out_ch, affine = False)
                )
	# BN with learnable linear parameters (shouldn't be necessary before linear layer)
        if bn == 'aba':
                self.convmp = nn.Sequential(
                    self.convmp,
                    nn.BatchNorm2d(out_ch, affine = True)
                )
        # Max pooling
        self.convmp = nn.Sequential(
            self.convmp,
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
    def forward(self, x):
        x = self.convmp(x)
        return x



class up(nn.Module):
    def __init__(self, in_ch, out_ch, bn = 'none', lbias = True):
        super(up, self).__init__()
        # Upscale and Convolutional layer
        self.upscale = nn.Sequential(
            Interpolate(scale_factor = 2, mode='bilinear'),
            nn.ReflectionPad2d(1)
        )
        # BN if before Activation
        if bn == 'ba':
                self.upscale = nn.Sequential(
                    self.upscale,
                    nn.Conv2d(in_ch, out_ch,kernel_size=3, stride=1, padding=0, bias = lbias),
                    nn.BatchNorm2d(out_ch)
                )
        else:
                self.upscale = nn.Sequential(
                    self.upscale,
                    nn.Conv2d(in_ch, out_ch,kernel_size=3, stride=1, padding=0)
                )
        # Activation
        self.upscale = nn.Sequential(
            self.upscale,
            nn.ReLU(True)
        )
        # BN after Activation
        if bn == 'ab':
                self.upscale = nn.Sequential(
                    self.upscale,
                    nn.BatchNorm2d(out_ch, affine = False)
                )
        # BN with learnable linear parameters (shouldn't be necessary before linear layer)
        if bn == 'aba':
                self.convmp = nn.Sequential(
                    self.convmp,
                    nn.BatchNorm2d(out_ch, affine = True)
                )
    def forward(self, x):
        x = self.upscale(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.out = nn.Sequential(
            #Interpolate(scale_factor = 2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch,kernel_size=3, stride=1, padding=0)
        )
    def forward(self, x):
        x = self.out(x)
        return x









