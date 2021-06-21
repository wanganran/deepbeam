import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import numpy as np

import asteroid
from torch.utils.checkpoint import checkpoint

class ModTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, eps=1e-7):
        x_re, x_im = x[0], x[1]
        norm = torch.sqrt(x_re ** 2 + x_im ** 2 + eps)
        phase_re, phase_im = x_re / norm, x_im / norm
        activated_norm = torch.tanh(norm)
        return torch.stack(
            [activated_norm * phase_re, activated_norm * phase_im], 0
        )
    
class TReLU(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.transform = torch.nn.Parameter(torch.randn((2,2)+input_shape))
        self.bias = torch.nn.Parameter(torch.randn((2,)+input_shape))
        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, tensor):
        real=tensor[0]
        imag=tensor[1]
        
        newreal=real*self.transform[0,0]+imag*self.transform[0,1]+self.bias[0]
        newimag=real*self.transform[1,0]+imag*self.transform[1,1]+self.bias[1]
        
        return torch.stack([
            self.relu(newreal),
            self.relu(newimag)], dim=0)

def cMul(t1,t2):
    real=t1[0]*t2[0]-t1[1]*t2[1]
    imag=t1[1]*t2[0]+t1[0]*t2[1]
    return torch.stack((real,imag),dim=0)

class ComplexSTFTWrapper(nn.Module):
    def __init__(self, win_length, hop_length, center=True):
        super(ComplexSTFTWrapper,self).__init__()
        self.win_length=win_length
        self.hop_length=hop_length
        self.center=center
        
    def transform(self, input_data):
        B,C,L=input_data.shape
        input_data=input_data.view(B*C, L)
        r=torch.stft(input_data, n_fft=self.win_length, hop_length=self.hop_length, center=self.center, return_complex=False)
        _,F,T,_=r.shape
        return r.view(B,C,F,T,2).permute(4,0,1,2,3)    
                              
    def reverse(self, input_data):
        _,B,C,F,T=input_data.shape
        input_data=input_data.permute(1,2,3,4,0).reshape(B*C,F,T,2)
        r=torch.istft(input_data, n_fft=self.win_length, hop_length=self.hop_length, center=self.center, return_complex=False) # B, L
        return r.view(B,C,-1)
        
    def forward(self, x):
        return self.reverse(self.transform(x))

class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=(0,0), dilation=1, groups=1, bias=True, complex_mul=True, causal=True):
        super(ComplexConv2d,self).__init__()
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.complex_mul=complex_mul
        self.causal=causal
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        if self.complex_mul:
            real = self.conv_re(x[0]) - self.conv_im(x[1])
            imag = self.conv_re(x[1]) + self.conv_im(x[0])
        else:
            real = self.conv_re(x[0])
            imag = self.conv_im(x[1])
        
        if self.causal and self.padding[1]>0:
            real=real[..., :-self.padding[1]]
            imag=imag[..., :-self.padding[1]]

        output = torch.stack((real,imag),dim=0)
        return output
    
class ComplexConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, causal=True):
        super(ComplexConv1d,self).__init__()
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.causal=causal
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[0]) - self.conv_im(x[1])
        imag = self.conv_re(x[1]) + self.conv_im(x[0])
        if self.causal and self.padding>0:
            real=real[..., :-self.padding]
            imag=imag[..., :-self.padding]
        output = torch.stack((real,imag),dim=0)
        return output
    
def depthwise_conv2d(ch_in, ch_out, kernel, dilation):
        return nn.Sequential(ComplexConv2d(ch_in, ch_out, (1,1)),
                       TReLU((ch_out, 1,1)),
                       ComplexConv2d(ch_out, ch_out, (1, kernel), dilation=(1, dilation), padding=(0, dilation*(kernel-1)), groups=ch_out))

def depthwise_conv1d(ch_in, ch_out, kernel, dilation):
        return nn.Sequential(ComplexConv1d(ch_in, ch_out, 1),
                       TReLU((ch_out, 1)),
                       ComplexConv1d(ch_out, ch_out, kernel, dilation=dilation, padding=dilation*(kernel-1), groups=ch_out))

class ComplexTCN(nn.Module):
    def __init__(self, ch_mid, ch_hid, kernel, dilation):
        super().__init__()
        self.conv1=depthwise_conv1d(ch_mid, ch_hid, kernel, dilation)
        self.act=TReLU((ch_hid, 1))
        self.norm=nn.BatchNorm1d(ch_mid, momentum=0.02, affine=True)
        self.conv2=ComplexConv1d(ch_hid, ch_mid, 1)
        self.conv3=ComplexConv1d(ch_hid, ch_mid, 1)
        
    def forward(self, t):
        B=t.shape[1]
        tori=t
        t=self.conv1(t)
        t=self.act(t)
        tres=self.conv2(t)
        t=self.conv3(t)
        _, B, C, T=t.shape
        t=self.norm(torch.flatten(t, 0, 1)).view(2,B,C,T)
        return t+tori, tres

class ComplexTCN2d(nn.Module):
    def __init__(self, ch_mid, ch_hid, freq, kernel, dilation):
        super().__init__()
        self.conv1=depthwise_conv2d(ch_mid, ch_hid, kernel, dilation)
        self.act=TReLU((ch_hid, freq, 1))
        self.norm=nn.BatchNorm2d(ch_mid, momentum=0.02, affine=True)
        self.conv2=ComplexConv2d(ch_hid, ch_mid, (1,1))
        self.conv3=ComplexConv2d(ch_hid, ch_mid, (1,1))
        
    def forward(self, t):
        B=t.shape[1]
        tori=t
        t=self.conv1(t)
        t=self.act(t)
        tres=self.conv2(t)
        t=self.conv3(t)
        _, B, C, F, T=t.shape
        t=self.norm(torch.flatten(t, 0, 1)).view(2,B,C,F,T)
        return t+tori, tres 
    
class BeamformerModel(nn.Module):
    def __init__(self, ch_in, ch_bf, ch_hid, ch_mid, synth_mid, synth_hid, block_size, freq, kernel, bf_layer, bf_rep, synth_layer, synth_rep):
        super().__init__()
        self.stage=0
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        self.ch_in=ch_in
        self.ch_bf=ch_bf
        self.ch_hid=ch_hid
        self.ch_mid=ch_mid
        self.freq=freq
        self.kernel=kernel
        
        self.bf_conv=nn.ModuleList()
        
        for r in range(bf_rep):
            dilation=1
            for i in range(bf_layer):
                self.bf_conv.append(ComplexTCN2d(ch_mid, ch_hid, freq, kernel, dilation))
                dilation*=kernel

        self.bf_final=nn.Sequential(
                ComplexConv2d(ch_mid, ch_hid, (1, 1)),
                TReLU((ch_hid, freq, 1)),
                ComplexConv2d(ch_hid, 1, (1, 1))
            )            
        
        self.synth_first=ComplexConv1d((block_size//2+1)*(1+ch_bf), synth_mid, 1)
        self.synth_final=nn.Sequential(
            ComplexConv1d(synth_mid, synth_hid, 1),
            TReLU((synth_hid, 1)),
            ComplexConv1d(synth_hid, synth_mid, 1))
        
        self.synth_last=ComplexConv1d(synth_mid, block_size//2, 2, causal=False)
        self.synth_act=ModTanh()
        self.synth_conv=nn.ModuleList()
        
        for r in range(synth_rep):
            dilation=1
            for i in range(synth_layer):
                self.synth_conv.append(ComplexTCN(synth_mid, synth_hid, kernel, dilation))
                dilation*=kernel
        
        self.freq_shuffle=ComplexConv1d(block_size//2+1, freq, 1)
        self.freq_rec=ComplexConv1d(freq, block_size//2+1, 1)
        self.ch_shuffle=ComplexConv2d(ch_in, ch_mid, (1,1))
        
        
    def bf_forward(self, spec, convs, final, freq_shuffle, ch_shuffle, freq_rec):
        # spec: 2, B, C, F, T
        _, B, C, F, T=spec.shape
        tl=freq_shuffle(spec.reshape(2,B*C,-1,T)).view(2,B,C,-1,T)
        tl=ch_shuffle(tl)
        
        res=None
        for i in range(len(convs)):
            tl, shortcut=convs[i](tl)
            if res is None:
                res=shortcut
            else:
                res+=shortcut
        tl=final(res)
        tl=freq_rec(tl.squeeze(2)).unsqueeze(2)
        return tl
        
    def forward(self, spec): # spec: B, 2, C, F, T
        spec=spec.permute(1,0,2,3,4)
        
        if self.stage==0:
            # learn bf
            spec=self.bf_forward(spec[:,:,:self.ch_in], self.bf_conv, self.bf_final, self.freq_shuffle, self.ch_shuffle, self.freq_rec)
            return self.stft.reverse(spec)
        else:
            # learn synth
            if self.stage==2:
                with torch.no_grad():
                    left=self.bf_forward(spec[:,:,:self.ch_in], self.bf_conv, self.bf_final, self.freq_shuffle, self.ch_shuffle, self.freq_rec)
            else:
                left=self.bf_forward(spec[:,:,:self.ch_in], self.bf_conv, self.bf_final, self.freq_shuffle, self.ch_shuffle, self.freq_rec)
                
            bf=spec[:,:,self.ch_in:]
            spec=torch.cat([left, bf], dim=2)
            spec=torch.flatten(spec, 2,3) # 2, B, freq*(1+ch_bf), T
            spec=self.synth_first(spec)
            t=spec
            
            res=None
            for i in range(len(self.synth_conv)):
                spec, shortcut=self.synth_conv[i](spec)
                if res is None:
                    res=shortcut
                else:
                    res+=shortcut
            spec=self.synth_final(res)
            spec=cMul(self.synth_act(spec), t)
            spec=self.synth_last(spec) # 2,B,S,L
            return torch.flatten(spec[0], 1,2)
        