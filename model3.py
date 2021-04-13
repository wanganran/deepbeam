import torch
import torch.nn as nn
from complexcnn.modules import cExp, cLog, ModReLU, ModTanh, cMul, toComplex, toReal, ComplexSTFTWrapper, ComplexConv1d, CausalComplexConv1d, ComplexConv2d
import asteroid

class ParallelConv1d(nn.Module):
    def __init__(self, num, ch_in, ch_out, kernel_size, dilation=1):
        super().__init__()
        self.padding=kernel_size*dilation-dilation
        self.ch_out=ch_out
        self.conv=ComplexConv2d(in_channel=num, 
                        out_channel=num*ch_out, 
                        kernel_size=(ch_in, kernel_size), 
                        stride=1, 
                        padding=(0, self.padding), 
                        dilation=(1, dilation), 
                        groups=num)
    def forward(self, tensor):
        # tensor: B,2,C,N,T
        B,_,Cin,N,T=tensor.shape
        t=self.conv(tensor.permute(0,1,3,2,4))  #B,2,N*COUT,1,T+padding-1
        t=t[..., self.padding:]
        t=t.view(B,2,N,self.ch_out,T)
        return t.permute(0,1,3,2,4)
    
class MiniBeamformer(nn.Module):
    def __init__(self, freqs, freq_kernel, ch_in, ch_inner, time_kernel, dilation, beamform_last=True):
        super().__init__()
        
        assert(freq_kernel%2==1)
        
        self.freqs=freqs
        self.freq_kernel=freq_kernel
        self.beamform_last=beamform_last
        
        self.channel_shuffle=ParallelConv1d(freqs, ch_in, ch_inner, 1)
        self.freq_conv=ComplexConv2d(ch_inner, ch_inner, (freq_kernel,1), padding=(freq_kernel//2,0), groups=ch_inner)
        self.time_conv=ParallelConv1d(freqs, ch_inner, ch_in, time_kernel, dilation)
        
        if beamform_last:
            self.last= ParallelConv1d(freqs, ch_in, ch_in, 1)
            
    def cAct(self, tensor):
        return torch.stack([torch.tanh(tensor[:,0]), torch.relu(tensor[:,1])], dim=1)
    
    def forward(self, tensor):
        # input: B,2,C,F,T
        T=tensor.shape[-1]
        t=cLog(tensor)
        t=self.channel_shuffle(t)
        t=self.cAct(t)
        
        t=self.freq_conv(t) # B,2,C,F,T
        t=self.cAct(t)
        
        t=self.time_conv(t)
        t=self.cAct(t)
        
        t=cExp(t)
        
        if self.beamform_last:
            t=self.last(t)
        
        return t
    
class MiniBeamformerModel(nn.Module):
    def __init__(self, in_channels, spec_channels, hidden_channels, block_size, layers, freq_kernel_size, time_kernel_size):
        super().__init__()
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        self.block_size=block_size
        self.freqs=block_size//2
        self.first=ParallelConv1d(self.freqs, in_channels, spec_channels, 1)
        
        self.bs=nn.ModuleList()
        dilation=1
        for i in range(layers-1):
            self.bs.append(ModReLU((self.freqs,1)))
            self.bs.append(MiniBeamformer(self.freqs, freq_kernel_size, spec_channels, hidden_channels, time_kernel_size, dilation))
            dilation*=time_kernel_size
        
        self.bs.append(MiniBeamformer(self.freqs, freq_kernel_size, spec_channels, hidden_channels, time_kernel_size, dilation, False))
        
        self.last=ParallelConv1d(self.freqs, spec_channels, in_channels, 1)
        
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        
    def forward(self,mix):
        t=self.stft.transform(mix) # B, 2, C, F, T
        trest=t[:,:,:,self.block_size//2:,:]
        t=t[:,:,:,:self.block_size//2,:]
        tori=t
        t=self.first(t)
        
        for l in self.bs:
            t=l(t)
        
        t=self.last(t)
        
        t=cMul(t, tori)
        
        t=torch.cat([t, trest], 3)
        t=torch.sum(t, dim=2, keepdim=True)
        t=self.stft.reverse(t)
        return t
        
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))   