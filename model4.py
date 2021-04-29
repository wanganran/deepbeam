import torch
import torch.nn as nn
import importlib
import complexcnn.modules
import numpy as np
importlib.reload(complexcnn.modules)
from complexcnn.modules import cExp, cLog, ModReLU, ModTanh, cMul, toComplex, toReal, ComplexSTFTWrapper, ComplexConv1d, CausalComplexConv1d, CausalComplexConvTrans1d, ComplexConv2d,ModMaxPool2d, modLog
import asteroid
from torch.utils.checkpoint import checkpoint

from nnAudio.Spectrogram import STFT
class TrainableSTFTWrapper(nn.Module): # assume same win_length and hop_length
    def __init__(self, win_length):
        super(TrainableSTFTWrapper,self).__init__()
        self.win_length=win_length
        self.F=win_length//2+1
        self.stft=STFT(win_length, hop_length=win_length, fmin=0, fmax=12000, output_format='Complex', sr=24000, trainable=True, center=False)
        
    def transform(self, input_data):
        #input: B,C,T
        B, C, T=input_data.shape
        r=self.stft(input_data.view(-1, T)) #-1, F, T
        r=r.view(B, C, self.F, -1, 2).permute(0,4,1,2,3)
        return r
    
    def forward(self, x):
        return self.transform(x)
        

class TReLU(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.transform = torch.nn.Parameter(torch.randn((2,2)+input_shape))
        self.bias = torch.nn.Parameter(torch.randn((2,)+input_shape))
        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, tensor):
        real=tensor[:,0]
        imag=tensor[:,1]
        
        newreal=real*self.transform[0,0]+imag*self.transform[0,1]+self.bias[0]
        newimag=real*self.transform[1,0]+imag*self.transform[1,1]+self.bias[1]
        
        return torch.stack([
            self.relu(newreal),
            self.relu(newimag)], dim=1)

def cov_complex(t, conjugate=True):
    # B,2,C,F,T
    t=t.permute(0,1,3,4,2)
    # *, 2, *, C
    shape=t.shape
    B=t.shape[0]
    C=t.shape[-1]
    target=shape[:-1]+torch.Size([C*C])
    real=t[:,0].reshape(-1,C)
    imag=t[:,1].reshape(-1,C)
    if not conjugate:
        newreal=torch.bmm(real.unsqueeze(-1), real.unsqueeze(-2))-torch.bmm(imag.unsqueeze(-1), imag.unsqueeze(-2))
        newimag=torch.bmm(imag.unsqueeze(-1), real.unsqueeze(-2))+torch.bmm(real.unsqueeze(-1), imag.unsqueeze(-2))
    else:
        newreal=torch.bmm(real.unsqueeze(-1), real.unsqueeze(-2))+torch.bmm(imag.unsqueeze(-1), imag.unsqueeze(-2))
        newimag=torch.bmm(imag.unsqueeze(-1), real.unsqueeze(-2))-torch.bmm(real.unsqueeze(-1), imag.unsqueeze(-2))

    real=newreal.view(B,-1,C*C)
    imag=newimag.view(B,-1,C*C)

    r=torch.stack([real,imag], dim=1).view(target)
    return r.permute(0,1,4,2,3)
            
def feature_norm(tensor, ch, bias=False, EPS=1e-4):
    if bias:
        avg=tensor.mean(ch, keepdim=True)
        tensor=tensor-avg
    
    avg_len=torch.mean(tensor[:,0]**2+tensor[:,1]**2, dim=ch, keepdim=True)
    tensor=tensor/(torch.sqrt(avg_len+EPS))
    
    return tensor
        
    
    
class ParallelConv1d(nn.Module):
    def __init__(self, num, ch_in, ch_out, kernel_size, dilation=1, complex_mul=True):
        super().__init__()
        self.padding=kernel_size*dilation-dilation
        self.ch_out=ch_out
        self.conv=ComplexConv2d(in_channel=num, 
                        out_channel=num*ch_out, 
                        kernel_size=(ch_in, kernel_size), 
                        stride=1, 
                        padding=(0, self.padding), 
                        dilation=(1, dilation), 
                        groups=num, complex_mul=complex_mul)
    def forward(self, tensor):
        # tensor: B,2,C,N,T
        B,_,Cin,N,T=tensor.shape
        t=self.conv(tensor.permute(0,1,3,2,4))  #B,2,N*COUT,1,T+padding-1
        t=t[..., self.padding:]
        t=t.view(B,2,N,self.ch_out,T)
        return t.permute(0,1,3,2,4)
    
    
class MiniBeamformer(nn.Module):
    def __init__(self, ch, F, freq_kernel, time_kernel, dilation=1, ch_in=None):
        super().__init__()
        
        self.ch=ch
        self.padding=time_kernel*dilation-dilation
        
        self.conv1=ParallelConv1d(F, ch if ch_in is None else ch_in, ch, 1)
        self.act1=ModReLU((F,1))
        self.conv2=ComplexConv2d(ch*ch, ch, (freq_kernel, 1), padding=(freq_kernel//2, 0))
        self.act2=TReLU((F,1))
        self.conv3=ComplexConv1d(ch*F,ch*F, time_kernel, padding=self.padding, dilation=dilation, groups=ch*F)
        #self.pool=ModMaxPool2d(time_kernel, 1)
        self.act3=TReLU((F,1))
        
        
    def forward(self, tensor):
        B,_, Cin, F, T=tensor.shape
        t=self.conv1(tensor)
        tori=t
        #print(torch.cuda.memory_allocated())
        
        #print("  ", torch.isnan(t).any(), t[0,0])
        t=self.act1(t)
        t=cov_complex(t) # B,2,ch*ch,F,T
        #print("  ", torch.isnan(t).any(), t[0,0])
        t=modLog(t)
        #t=feature_norm(t, 2)
        t=self.act2(t)
        #print("  ", torch.isnan(t).any(), t[0,0])
        t=self.conv2(t) # B,2,ch,F,T
        
        #print("  ", torch.isnan(t).any(), t[0,0])
        t=self.act3(t)
        #print("  ", torch.isnan(t).any(), t[0,0])
        t=t.view(B,2,-1,T)
        t=self.conv3(t)[..., self.padding:]
        t=t.view(B,2,-1,F,T)
        
        #print("  ", torch.isnan(t).any(), t[0,0])
        #t=self.pool(t) # B,2,ch,F,T
        
        return t, cMul(t, tori) 
        
        
class MiniBeamformerModel(nn.Module):
    def __init__(self, ch_in, ch, layers, block_size, time_kernel=4):
        super().__init__()
        self.F=block_size//2+1
        
        self.beamformers=nn.ModuleList()
        dilation=1
        for i in range(layers):
            if i==0:
                self.beamformers.append(MiniBeamformer(ch, self.F, 3, time_kernel, dilation, ch_in))
            else:
                self.beamformers.append(MiniBeamformer(ch, self.F, 3, time_kernel, dilation))
            dilation*=time_kernel
        
        self.last=nn.Sequential(ModTanh(), ParallelConv1d(self.F, ch, 1, 1))
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        #self.loss_fn=nn.L1Loss()
        
    def forward(self, tensor):
        #print("begin")
        t=self.stft.transform(tensor)
        
        #print(torch.cuda.memory_allocated())
        for l in self.beamformers:
            if self.training and t.requires_grad:
                _, t=checkpoint(l,t)
            else:
                _, t=l(t)
            #print(torch.isnan(t).any())
            #_, t=l(t)
        
        t=self.last(t)
        return self.stft.reverse(t)
    
     
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))             
                 
class NaiveModel(nn.Module):
    def __init__(self, ch_in, ch_hidden, block_size):
        super().__init__()
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        self.freq=block_size//2+1
        self.l1=ParallelConv1d(self.freq, ch_in, ch_hidden, 1)
        self.act=ModReLU((self.freq, 1))
        self.l2=ParallelConv1d(self.freq, ch_hidden, 1, 1)
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
    def forward(self, mix):
        t=self.stft.transform(mix) # B, 2, C, F, T
        t=self.l1(t)
        t=self.act(t)
        t=self.l2(t)
        return self.stft.reverse(t)
    
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))
     
        
class CausalTCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dilation, kernel, activation=nn.LeakyReLU(0.1)):
        super().__init__()
        
        self.layers=nn.ModuleList()
        self.layers.append(CausalComplexConv1d(in_channels, hidden_channels, kernel, dilation=dilation, activation=activation, fullconv=True))
        self.layers.append(activation)
        self.layers.append(ComplexConv1d(hidden_channels, out_channels, 1))
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y=x
        for l in self.layers:
            y=l(y)
        return x+y
    
class HybridModel(nn.Module):
    def __init__(self, ch_in, ch, ch_wav, ch_hid, layers, block_size, wav_kernel, time_kernel=4, reception=64):
        super().__init__()
        self.F=block_size//2+1
        self.block_size=block_size
        self.layers=layers
        
        self.beamformers=nn.ModuleList()
        #self.convs=nn.ModuleList()
        #self.mappings=nn.ModuleList()
        
        self.wav_first=nn.Sequential(
            CausalComplexConv1d(ch_in, ch_wav, reception, fullconv=True), #B,2,C,L
            ModReLU([ch_wav, 1]),
            CausalComplexConv1d(ch_wav, ch_hid, wav_kernel, activation=ModReLU([ch_hid, 1])), #B,2,C,L
            ModReLU([ch_hid, 1]),
            ComplexConv1d(ch_hid, ch_wav, 1)
            )
        
        dilation=1
        for i in range(layers):
            if i==0:
                self.beamformers.append(MiniBeamformer(ch, self.F, 3, time_kernel, dilation, ch_in))
            else:
                self.beamformers.append(MiniBeamformer(ch, self.F, 3, time_kernel, dilation))
            dilation*=time_kernel
            
        
        #for i in range(layers):
        #    self.convs.append(CausalTCN(ch_wav, ch_wav, ch_hid, 1, wav_kernel, ModReLU((ch_hid, 1))))
        #    self.mappings.append(ComplexConv1d(self.F*ch, ch_wav, 1))
            
        self.wav_last=nn.Sequential(
            ModTanh(),
            CausalComplexConvTrans1d(ch_wav*2, 1, reception, fullconv=True)
        )
        self.wav_mapping=ComplexConv1d(self.F*ch, ch_wav, 1)
        self.weight_mapping=ComplexConv1d(self.F*ch, ch_wav, 1)
        
        self.stft=ComplexSTFTWrapper(block_size, hop_length=block_size, center=False)
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
    
    def __interpolate(self, spec_in):
        # spec_in: B,2,*,T
        B,_,C,T=spec_in.shape
        padding=torch.repeat_interleave(spec_in[:,:,:,0:1], self.block_size-1, dim=3)
        spec_in=nn.functional.interpolate(spec_in, (C, (T-1)*self.block_size+1), mode='bilinear', align_corners=True)
        spec_in=torch.cat([padding, spec_in], dim=3)
        return spec_in
    
    def forward(self, tensor):
        spec_next=self.stft.transform(tensor)
        
        mix=toComplex(tensor)
        wav_t=self.wav_first(mix) # B,2,C,L
        
        #print(torch.cuda.memory_allocated())
        
        for i in range(self.layers):
            #if self.training and wav_t.requires_grad:
            #    wav_t=checkpoint(self.convs[i], wav_t)
            #else:
            #wav_t=self.convs[i](wav_t)
            spec_w, spec_next=self.beamformers[i](spec_next)
            #spec_t=spec_next.view(spec_next.shape[0], 2, -1, spec_next.shape[-1])
            #spec_t=self.mappings[i](spec_t) #B,2,Cwav, T
            #spec_t=self.__interpolate(spec_t) # B,2,Cwav,T
            #wav_t=spec_t+wav_t
        
        B,_,C,F,T=spec_w.shape
        spec_w=spec_w.view(B, 2, -1, T)
        spec_w=self.weight_mapping(spec_w)
        spec_w=self.__interpolate(spec_w)
        t=cMul(wav_t, spec_w)
        
        spec_t=spec_next.view(B,2,-1,T)
        spec_t=self.wav_mapping(spec_t)
        spec_t=self.__interpolate(spec_t)
        
        t=torch.cat([t, spec_t], dim=2)
        t=self.wav_last(t)
        return toReal(t)
     
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))             
                 