# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

def cExp(tensor):
    # tensor: B, 2, ...
    e=torch.exp(tensor[:,0])
    real=e*torch.cos(tensor[:,1])
    imag=e*torch.sin(tensor[:,1])
    return torch.stack([real, imag], dim=1)

def cLog(tensor):
    real=torch.log(torch.sqrt(tensor[:,0]**2+tensor[:,1]**2))
    imag=torch.arctan(tensor[:,1]/tensor[:,0])
    return torch.stack([real, imag], dim=1)

def cMul(t1,t2):
    real=t1[:,0]*t2[:,0]-t1[:,1]*t2[:,1]
    imag=t1[:,1]*t2[:,0]+t1[:,0]*t2[:,1]
    return torch.stack((real,imag),dim=1)


def toComplex(tensor):
    return torch.stack((tensor, tensor-tensor), dim=1)

def toReal(tensor):
    return tensor[:,0,...]

class ComplexLinear(nn.Module):
    def __init__(self, inputsize, outputsize, bias=True):
        super(ComplexLinear,self).__init__()
        
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
        self.win_length=win_length
        self.hop_length=hop_length
        
    def transform(self, input_data):
        channel=input_data.shape[-2]
        result=[]
        for i in range(channel):
            r=torch.stft(input_data[:, i, :], n_fft=self.win_length, hop_length=self.hop_length, return_complex=False)
            result.append(r.permute(0, 3, 1, 2))
        return torch.stack(result, dim=2) # B, 2, C, F, L
                              
    def reverse(self, input_data):
        channel=input_data.shape[-3]
        result=[]
        for i in range(channel):
            d=input_data[:, :, i, :, :].permute(0,2,3,1)
            r=torch.istft(d, n_fft=self.win_length, hop_length=self.hop_length, return_complex=False) # B, L
            result.append(r)
        return torch.stack(result, dim=1) # B, C, L
   
    def forward(self, x):
        return self.reverse(self.transform(x))
            
        
    
class ComplexConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv1d,self).__init__()
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
        self.padding = padding

        ## Model components
        self.conv_re = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real,imaginary),dim=1)
        return output

class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d,self).__init__()
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real,imaginary),dim=1)
        return output

class CausalComplexConv1d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, dilation=1, fullconv=False, activation=nn.LeakyReLU(0.1)):
        super().__init__()
        # input is [batch, channel, length]
        # depthwise + 1x1
        self.kernel_size=kernel_size
        self.stride=stride
        self.pad=nn.ConstantPad1d((kernel_size*dilation-dilation+1-stride,0), 0.0)
        self.fullconv=fullconv
        self.activation=activation
        if not fullconv:
            self.l1=ComplexConv1d(channel_in, channel_out, 1)
            self.l2=ComplexConv1d(channel_out, channel_out, kernel_size, stride=stride, dilation=dilation, groups=channel_out, padding=0)
        else:
            self.l1=ComplexConv1d(channel_in, channel_out, kernel_size, stride=stride, dilation=dilation, padding=0)
            self.l2=nn.Identity()
    def forward(self, x):
        if not self.fullconv:
            return self.l2(self.pad(self.activation(self.l1(x))))
        else:
            return self.l1(self.pad(x))

class CausalComplexConvTrans1d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, fullconv=False):
        super().__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        self.fullconv=fullconv
        self.activation=nn.LeakyReLU(0.1)
        if not fullconv:
            self.l1=ComplexConv1d(channel_in, channel_out, 1)
            self.l2=ComplexConvTranspose1d(channel_out, channel_out, kernel_size, stride=stride, groups=channel_out)
        else:
            self.l=ComplexConvTranspose1d(channel_in, channel_out, kernel_size, stride=stride)
            
    def forward(self, x):
        if not self.fullconv:
            return self.l2(self.activation(self.l1(x)))[..., self.kernel_size-self.stride:]
        else:
            return self.l(x)[..., self.kernel_size-self.stride:]

class CausalComplexConv2d(nn.Module):
    def __init__(self, in_channels, freq_channels, out_channels, freq_kernel_size, time_kernel_size):
        super().__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.freq_kernel_size=freq_kernel_size
        self.time_kernel_size=time_kernel_size
        self.freq_channels=freq_channels
        self.conv_f1=ComplexConv1d(in_channels, freq_channels, 1)
        self.conv_f2=CausalComplexConv1d(freq_channels, freq_channels, freq_kernel_size, fullconv=True)
        self.conv_t=CausalComplexConv1d(freq_channels, out_channels, time_kernel_size)
        self.act=nn.LeakyReLU(0.1)
        
    def forward(self, x):
        B,_,C,F,T=x.shape
        # input: B, 2, C, F, T
        y=x.permute(0,4,1,2,3).view(B*T, 2, C, F)
        y=self.conv_f1(y)
        y=self.act(y)
        y=self.conv_f2(y)
        y=self.act(y)
        y=y.view(B,T,2,self.freq_channels,-1).permute(0,4,2,3,1).view(-1,2,self.freq_channels,T)
        y=self.conv_t(y)
        y=self.act(y)
        y=y.view(B,-1,2,self.out_channels,T).permute(0,2,3,1,4)
        return y
