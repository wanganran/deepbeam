import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import complexcnn.modules
import numpy as np
importlib.reload(complexcnn.modules)
from complexcnn.modules import ModReLU, ModTanh, cMul, toComplex, toReal, ComplexSTFTWrapper, ComplexConv1d, ComplexConvTranspose1d, CausalComplexConv1d, CausalComplexConvTrans1d, ComplexConv2d,ModMaxPool2d, modLog
from model4 import cExp, cLog, cLN, TGate, TReLU, ParallelConv1d
import asteroid
from torch.utils.checkpoint import checkpoint


class MiniBeamformer(nn.Module):
    def __init__(self, n_freq, ch_in, ch_hid, ch_out, kernel, dilation, freq_shuffle=True):
        super().__init__()
        if(freq_shuffle):
            self.freq_shuffle=ComplexConv2d(n_freq, n_freq, (1,1))
        self.padding=kernel*dilation-dilation
        self.kernel=kernel
        self.conv1r=ComplexConv2d(ch_in, ch_hid, (1,1))
        self.conv1i=ComplexConv2d(ch_hid, ch_hid, (1,kernel), padding=(0,self.padding), dilation=(1, dilation), groups=ch_hid)
        self.conv2r=ComplexConv2d(ch_hid, ch_out, (1,1))
        self.act1=TReLU((ch_hid, n_freq, 1))
        
    def forward(self, tensor): # tensor: B,2,C,F,T
        tl=self.freq_shuffle(tensor.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        tl=self.conv1r(tl)
        tl=F.leaky_relu(tl)
        tl=self.conv1i(tl)[..., :-self.padding]
        tl=self.act1(tl)
        tl=self.conv2r(tl)
        return tl
    
class BeamformerModel(nn.Module):
    def __init__(self, ch_in, ch_hidden, ch_mid, block_size, freqs, kernel, layers=2):
        super().__init__()
        self.fftr=nn.Conv2d(1, freqs, (1, block_size), stride=(1, block_size//2))
        self.ffti=nn.Conv2d(1, freqs, (1, block_size), stride=(1, block_size//2))
        
        dilation=1
        self.layers=layers
        self.bfl=nn.ModuleList()
        self.bfr=nn.ModuleList()
        
        for l in range(layers):
            ch=ch_in if l==0 else ch_mid
            dilation*=kernel
            self.bfl.append(MiniBeamformer(freqs, ch, ch_hidden, ch_mid, kernel, dilation))
            self.bfr.append(MiniBeamformer(freqs, ch, ch_hidden, ch_mid, kernel, dilation))
        
        self.finall=ComplexConv2d(ch_mid, 1, (1,1))
        self.finalr=ComplexConv2d(ch_mid, ch_in, (1,1))
        
        self.freq_recl=ComplexConv2d(freqs, freqs, (1,1))
        self.freq_recr=ComplexConv2d(freqs, freqs, (1,1))
        
        
        self.convw=MiniBeamformer(freqs, ch_in+2, ch_hidden, 1, kernel, 1)
        self.actw=TGate((1, freqs, 1))
        
        self.ifft=ComplexConvTranspose1d(freqs, 1, block_size, stride=block_size//2)
        
    def forward(self, tensor): # B, C, L
        tensor=tensor.unsqueeze(1) # B, 1, C, L
        ts=torch.stack([self.fftr(tensor), self.ffti(tensor)], dim=1).permute(0,1,3,2,4) # B, 2, C, F, T
        tsl=ts
        tsr=ts
        
        for l in range(self.layers):
            tsl=self.bfl[l](tsl)
            tsr=self.bfr[l](tsr)
        
        tsl=self.finall(tsl)
        tsr=self.finalr(tsr)
        
        tsl=self.freq_recl(tsl.permute(0,1,3,2,4)).permute(0,1,3,2,4) # B,2,1,F,T
        tsr=torch.sum(ts-self.freq_recr(tsr.permute(0,1,3,2,4)).permute(0,1,3,2,4), dim=2, keepdim=True)/ts.shape[2] # B,2,1,F,T
        
        tw=self.actw(self.convw(torch.cat([ts, tsl, tsr], dim=2))).unsqueeze(1)
        ts=tsl*tw+tsr*(1-tw) # B,2,1,F,T
        
        return self.ifft(ts.squeeze(2))[:, 0] # B, 1, L
        

class FuseModel(nn.Module):
    def __init__(self, ch_in, ch_bf, ch_mid, ch_hid, freqs, k_reception, k_mid, bfmodel, stride):
        super().__init__()
        
        self.ch_in=ch_in
        self.ch_bf=ch_bf
        
        self.bfmodel=bfmodel
        self.fft=ComplexConv2d(1, freqs, (1, k_reception), stride=(1, stride))
        
        self.tcn1=MiniBeamformer(freqs, ch_bf+2, ch_hid, ch_mid, k_mid, 1)
        self.tcn2=MiniBeamformer(freqs, ch_mid, ch_hid, ch_mid, k_mid, k_mid)
        self.tcn3=MiniBeamformer(freqs, ch_mid, ch_hid, ch_mid, k_mid, k_mid*k_mid)
        self.tcn4=MiniBeamformer(freqs, ch_mid, ch_hid, ch_mid, k_mid, k_mid*k_mid*k_mid)
        
        self.act=TReLU((ch_mid, freqs, 1))
        self.final=ComplexConv2d(ch_mid, ch_bf+2, (1,1))
        self.gate=ModTanh()
        self.ifft=ComplexConvTranspose1d(freqs, 1, k_reception, stride=stride)
        
    def forward(self, tensor):
        #with torch.no_grad():
        naive_out=self.bfmodel(tensor[:, :self.ch_in].contiguous()) # B, C, L
        
        t=torch.cat([tensor[:, self.ch_in:], naive_out, tensor[:, 0:1]], dim=1) # B, 5, L
        t=toComplex(t).unsqueeze(2) # B,2,1,5, L
        t=self.fft(t).permute(0,1,3,2,4) # B,2,C,F,T
        tp=t
        
        t=self.tcn1(t)
        t=t+self.tcn2(t)
        t=t+self.tcn3(t)
        t=self.tcn4(t)
        
        t=self.act(t)
        t=self.final(t)
        t=self.gate(t)
        
        t=cMul(t,tp) # B,2,ch_bf+2, F,T
        t=self.ifft(torch.sum(t,dim=2)) # B, 2, 1, L
        return toReal(t) # B,1,L
    
        
