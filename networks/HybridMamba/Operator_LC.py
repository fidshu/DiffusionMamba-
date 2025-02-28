import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            dilation=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation = dilation,
#            groups = in_channels,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)
class IPATR(nn.Module):
    def __init__(self,channel):
      super(IPATR, self).__init__()
      self.channel = channel
      self.conv1 = nn.Conv3d(self.channel,self.channel,kernel_size=1,padding=0)
      

      self.conv2 = nn.Conv3d(self.channel,self.channel,kernel_size=3,padding=1,stride=1)
      

      self.conv3 = nn.Conv3d(self.channel,self.channel,kernel_size=3,padding=2,stride=1,dilation=2)
      self.sg = torch.nn.Sigmoid()
    
    def max_min(self,x):
        min_val = x.min(dim=3, keepdim=True)[0]
        max_val = x.max(dim=3, keepdim=True)[0]
        min_val = min_val.min(dim=4, keepdim=True)[0]
        max_val = max_val.max(dim=4, keepdim=True)[0]
        normalized_tensor = (x - min_val + 1e-8) / (max_val - min_val + 1e-8)
        return normalized_tensor

    
    def zero_scoreHW(self,x):
        mean = x.mean(dim=[3, 4], keepdim=True)
        std = x.std(dim=[3, 4], keepdim=True)
        normalized_tensor = (x - mean) / (std + 1e-8)
        return normalized_tensor
    def ipatr(self,x,choice):
      if(choice ==1):
        conv = self.conv1
      elif(choice == 2):
        conv = self.conv2
      else:
        conv = self.conv3
      mu = conv(x)
    
      segma = self.sg(conv(x-mu))
      mscn = torch.div(x-mu+0.001,segma+0.001)
      # print(segma)
      return mscn
    def forward(self,x):
      x1 = self.ipatr(x,choice = 1)
      x2 = self.ipatr(x,choice = 2)
      x3 = self.ipatr(x,choice = 3)
      mpcm =torch.div(x1+x2+x3,3)
      mpcm = self.max_min(mpcm)
#       mpcm = x+mpcm*x

      return x