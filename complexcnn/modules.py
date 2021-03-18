# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

class ComplexLinear(nn.Module):
    def __init__(self, inputsize, outputsize, bias=True):
        super(ComplexLinear,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## Model components
        self.re=nn.Linear(inputsize, outputsize)
        self.im=nn.Linear(inputsize, outputsize)
        
    def forward(self, x): 
        real=self.re(x[:,0])-self.im(x[:,1])
        imag=self.re(x[:,1])+self.im(x[:,0])
        output = torch.stack((real,imag),dim=1)
        return output

class ComplexMultiLinear(nn.Module):
    def __init__(self, channel, inputsize, outputsize, bias=True):
        super(ComplexMultiLinear,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## Model components
        self.re=nn.ModuleList()
        self.im=nn.ModuleList()
        for i in range(channel):
            self.re.append(nn.Linear(inputsize, outputsize))
            self.im.append(nn.Linear(inputsize, outputsize))
        self.channel=channel
        
    def forward(self, x): 
        # x: B, 2, C, ..., inputsize
        # out: B, 2, C, ..., outputsize
        output=[]
        for i in range(self.channel):
            real=self.re[i](x[:,0, i])-self.im[i](x[:,1, i])
            imag=self.re[i](x[:,1, i])+self.im[i](x[:,0, i])
            output.append(torch.stack((real,imag),dim=1))
        
        return torch.stack(output, dim=2)
    
class ComplexSTFTWrapper(nn.Module):
    def __init__(self, win_length, hop_length):
        super(ComplexSTFTWrapper,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.win_length=win_length
        self.hop_length=hop_length
        
    def transform(self, input_data):
        channel=input_data.shape[-2]
        result=[]
        for i in range(channel):
            r=torch.stft(input_data[:, i, :], n_fft=self.win_length, hop_length=self.hop_length)
            result.append(r.permute(0, 3, 1, 2))
        return torch.stack(result, dim=2) # B, 2, C, F, L
                              
    def reverse(self, input_data):
        channel=input_data.shape[-3]
        result=[]
        for i in range(channel):
            d=input_data[:, :, i, :, :].permute(0,2,3,1)
            r=torch.istft(d, n_fft=self.win_length, hop_length=self.hop_length) # B, L
            result.append(r)
        return torch.stack(result, dim=1) # B, C, L
   
    def forward(self, x):
        return self.reverse(self.transform(x))
            
        
    
class ComplexConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv1d,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real,imaginary),dim=1)
        return output

class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConvTranspose1d,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_re = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real,imaginary),dim=1)
        return output

class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real,imaginary),dim=1)
        return output
        
#%%
if __name__ == "__main__":
    ## Random Tensor for Input
    ## shape : [batchsize,2,channel,axis1_size,axis2_size]
    ## Below dimensions are totally random
    x = torch.randn((10,2,3,100,100))
    
    # 1. Make ComplexConv Object
    ## (in_channel, out_channel, kernel_size) parameter is required
    complexConv = ComplexConv(3,10,(5,5))
    
    # 2. compute
    y = complexConv(x)

