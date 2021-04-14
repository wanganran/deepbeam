import torch
import torch.nn as nn
import importlib
import complexcnn.modules
import numpy as np
importlib.reload(complexcnn.modules)
from complexcnn.modules import cExp, cLog, ModReLU, ModTanh, cMul, toComplex, toReal, ComplexSTFTWrapper, ComplexConv1d, CausalComplexConv1d, CausalComplexConvTrans1d, ComplexConv2d
import asteroid

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
    def __init__(self, freqs, freq_kernel, ch_in, ch_inner, time_kernel, dilation, beamform_last=True, ch_out=None):
        super().__init__()
        
        assert(freq_kernel%2==1)
        if ch_out is None: ch_out=ch_in
        
        self.freqs=freqs
        self.freq_kernel=freq_kernel
        self.beamform_last=beamform_last
        
        self.channel_shuffle=ParallelConv1d(freqs, ch_in, ch_inner, 1, complex_mul=False)
        self.freq_conv=ComplexConv2d(ch_inner, ch_inner, (freq_kernel,1), padding=(freq_kernel//2,0), groups=ch_inner, complex_mul=False)
        self.time_conv=ParallelConv1d(freqs, ch_inner, ch_in, time_kernel, dilation, complex_mul=False)
        self.last_shuffle=ParallelConv1d(freqs, ch_in, ch_out, 1)
        if beamform_last:
            self.last= ParallelConv1d(freqs, ch_out, ch_out, 1)
            
    def cAct(self, tensor):
        weight=(torch.cos(tensor[:,1])+1)/2
        return torch.stack([weight*tensor[:,0], torch.remainder(tensor[:,1], 2*np.pi)], dim=1)
    
    def forward(self, tensor):
        # input: B,2,C,F,T
        T=tensor.shape[-1]
        t=cLog(tensor)
        t=self.channel_shuffle(t)
        t=self.cAct(t)
        
        t=self.freq_conv(t) # B,2,C,F,T
        t=self.cAct(t)
        
        t=self.time_conv(t)
        t=cExp(t)
        
        t=self.last_shuffle(t)
        
        if self.beamform_last:
            t=cMul(t,tensor)
            t=self.last(t)
        
        return t
    
class MiniBeamformerModel(nn.Module):
    def __init__(self, in_channels, spec_channels, hidden_channels, out_channels, block_size, layers, freq_kernel_size, time_kernel_size, return_last=True):
        super().__init__()
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        self.block_size=block_size
        self.freqs=block_size//2
        self.first=ParallelConv1d(self.freqs, in_channels, spec_channels, 1)
        self.return_last=return_last
        
        self.bs=nn.ModuleList()
        dilation=1
        for i in range(layers-1):
            self.bs.append(ModReLU((self.freqs,1)))
            self.bs.append(MiniBeamformer(self.freqs, freq_kernel_size, spec_channels, hidden_channels, time_kernel_size, dilation))
            dilation*=time_kernel_size
        
        if return_last:
            self.bs.append(MiniBeamformer(self.freqs, freq_kernel_size, spec_channels, hidden_channels, time_kernel_size, dilation, False))
            #self.last=ParallelConv1d(self.freqs, out_channels, spec_channels, 1)
        
        else:
            self.bs.append(MiniBeamformer(self.freqs, freq_kernel_size, spec_channels, hidden_channels, time_kernel_size, dilation, False, out_channels))
        
        self.act=ModReLU((spec_channels, self.freqs, 1))
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        
    def forward(self,mix):
        t=self.stft.transform(mix) # B, 2, C, F, T
        trest=torch.sum(t[:,:,:,self.block_size//2:,:], dim=2, keepdim=True)
        t=t[:,:,:,:self.block_size//2,:]
        t=self.first(t)
        tori=t
        
        for l in self.bs:
            t=l(t)
        
        if self.return_last:
            #t=self.last(t)
            t=self.act(t)
            t=cMul(t, tori)
            t=torch.sum(t, dim=2, keepdim=True)
            t=torch.cat([t, trest], 3)
            t=self.stft.reverse(t)
            return t
        else:
            return t
        
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))   
    
class FuseModel(nn.Module):
    def __init__(self, in_channels, spec_channels, hidden_channels, out_channels, block_size, layers, freq_kernel_size, time_kernel_size, wav_channels, wav_kernel):
        super().__init__()
        self.block_size=block_size
        F=block_size//2
        self.beamformerModel=MiniBeamformerModel(in_channels, spec_channels, hidden_channels, out_channels, block_size, layers, freq_kernel_size, time_kernel_size, False)
        self.wav_first=CausalComplexConv1d(in_channels, wav_channels, wav_kernel, fullconv=True) #B,2,C,L
        self.act1=ModReLU((wav_channels, 1))
        self.act2=ModReLU((wav_channels, 1))
        self.shuffle1=ComplexConv1d(out_channels*F, wav_channels, 1)
        self.shuffle2=ComplexConv1d(out_channels*F, wav_channels, 1)
        self.fuse1=ComplexConv1d(wav_channels, wav_channels, 1)
        self.fuse2=ComplexConv1d(wav_channels, wav_channels, 1)
        self.wav_last=CausalComplexConvTrans1d(wav_channels, 1, wav_kernel, fullconv=True)
    
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        
    def forward(self, mix):
        beamformer_out=self.beamformerModel(mix) # B,2,Cout, F, T
        beamformer_out=beamformer_out[..., :-1]
        B, _, Cout, F, T=beamformer_out.shape
        step=self.block_size//2
        spec_in=beamformer_out.permute(0,1,4,2,3).reshape(B,2,T, -1) # B,2,T,C*F
        padding=torch.repeat_interleave(spec_in[:,:,0:1,:], step-1, dim=2)
        spec_in=nn.functional.interpolate(spec_in, ((T-1)*step+1, Cout*F), mode='bilinear', align_corners=True)
        spec_in=torch.cat([padding, spec_in], dim=2)
        spec_in=spec_in.permute(0,1,3,2) # B,2,C*F,T
        
        mix=toComplex(mix)
        t=self.wav_first(mix)
        
        b1=self.shuffle1(spec_in)
        b1=self.act1(b1)
        t=cMul(t, b1)
        t=self.fuse1(t)
        
        b2=self.shuffle2(spec_in)
        b2=self.act2(b2)
        t=cMul(t, b2)
        t=self.fuse2(t)
        
        t=self.wav_last(t)
        return toReal(t)
    
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:])) 