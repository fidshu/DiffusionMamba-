import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.HybridMamba.Operator_LC import IPATR

class Conv3dReLU(nn.Sequential):
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

# Semantic difference module
class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(SDC, self).__init__() 
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv3dbn(guidance_channels, in_channels, kernel_size=3, padding=1)                         
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize for skipped features
        # x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        # x_initial[:, :, np.int(kernel_size / 2), np.int(kernel_size / 2), np.int(kernel_size / 2)] = -1
        # self.x_kernel_diff = nn.Parameter(x_initial)
        # self.x_kernel_diff[:, :, np.int(kernel_size / 2), np.int(kernel_size / 2), np.int(kernel_size / 2)].detach()
        
        self.x_kernel_diff = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)

#        self.x_kernel_diff = IPATR(in_channels)
        # initialize for guidance features
        # guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        # guidance_initial[:, :, np.int(kernel_size / 2), np.int(kernel_size / 2), np.int(kernel_size / 2)] = -1
        # self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        # self.guidance_kernel_diff[:, :, np.int(kernel_size / 2), np.int(kernel_size / 2), np.int(kernel_size / 2)].detach()
        
#        self.guidance_kernel_diff = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.guidance_kernel_diff = IPATR(in_channels)

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size
        
        guidance = self.conv1(guidance)

        # x_diff = F.conv3d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1, groups=in_channels)
        x_diff = self.x_kernel_diff(x)
        
        
        # guidance_diff = F.conv3d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1, groups=in_channels)
        guidance_diff = self.guidance_kernel_diff(guidance)
        # print(self.guidance_kernel_diff[0, 0], self.x_kernel_diff[0, 0])
        out = self.conv(x_diff * guidance_diff)
        return out, x_diff, guidance, guidance_diff
    

    
    
class SDN(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(SDN, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced, x_diff, guidance, guidance_diff = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature          
        return boundary_enhanced
