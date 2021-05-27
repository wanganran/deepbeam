import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import complexcnn.modules
import numpy as np
importlib.reload(complexcnn.modules)
from complexcnn.modules import ModReLU, ModTanh, cMul, toComplex, toReal, ComplexSTFTWrapper, ComplexConv1d, CausalComplexConv1d, CausalComplexConvTrans1d, ComplexConv2d,ModMaxPool2d, modLog
import asteroid
from torch.utils.checkpoint import checkpoint

EPS=1e-4

from nnAudio.Spectrogram import STFT

def modExp(tensor, exp_max):
    return torch.where(tensor>exp_max, (tensor-exp_max)*np.exp(exp_max)+np.exp(exp_max), torch.exp(tensor))

def cExp(tensor):
    # tensor: B, 2, ...
    e=modExp(tensor[:,0], 5)-1
    real=e*torch.cos(tensor[:,1]-1)
    imag=e*torch.sin(tensor[:,1]-1)
    
    return torch.stack([real, imag], dim=1)

def cLog(tensor):
    eps_max=torch.ones(1, device=tensor.device)*EPS
    eps_min=torch.ones(1, device=tensor.device)*(-EPS)

    t=tensor[:,1]    
    
    t=torch.where((t>=0) & (t<EPS), eps_max, t)
    t=torch.where((t<0) & (t>-EPS), eps_min, t)
    
    t2=tensor[:,0]    
    
    t2=torch.where((t2>=0) & (t2<EPS), eps_max, t2)
    t2=torch.where((t2<0) & (t2>-EPS), eps_min, t2)
    
    real=torch.log(torch.sqrt(t**2+t2**2+EPS**2)+1)
    imag=torch.arctan(t/t2)
    
    return torch.stack([real, imag], dim=1)



class cLN(nn.Module):
    def __init__(self, eps=1e-8, withbias=False):
        super(cLN, self).__init__()
        
        self.eps = eps
        self.withbias=withbias
        
    def forward(self, input):
        # input size: (Batch, 2, *, Time)
        # cumulative mean for each time step
        batch_size = input.size(0)
        channel = input.size(2)
        time_step = input.size(3)
        
        step_sum = input.sum(2)  # B, 2, T
        cum_sum = torch.cumsum(step_sum, dim=2)  # B, 2, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1,1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, 2, T
        
        if self.withbias:
            step_pow_sum = (input-cum_mean).pow(2).sum((1,2))  # B, T
        else:
            step_pow_sum = input.pow(2).sum((1,2))  # B, T
        
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        
        cum_var = cum_pow_sum / entry_cnt[:,0]  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(2) # B,2,1,T
        cum_std = cum_std.unsqueeze(1).unsqueeze(1) # B,1, 1,T
        
        if self.withbias:
            x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        else:
            x = input / cum_std.expand_as(input)
        return x

class cLN_2d(nn.Module):
    def __init__(self, eps=1e-8, withbias=False):
        super().__init__()
        self.cln=cLN(eps, withbias)
        
    def forward(self, input):
        input=input.permute(0,3,1,2,4) # B,F, 2, C, T
        B,F,_,C,T=input.shape
        output=self.cln(input.flatten(0,1)).view(B,F,2,C,T)
        return output.permute(0,2,3,1,4)

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
        
class TGate(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.transform = torch.nn.Parameter(torch.randn((2,2)+input_shape))
        self.bias = torch.nn.Parameter(torch.randn((2,)+input_shape))
        self.act = torch.nn.Sigmoid()
    
    def forward(self, tensor):
        real=tensor[:,0]
        imag=tensor[:,1]
        
        newreal=real*self.transform[0,0]+imag*self.transform[0,1]+self.bias[0]
        newimag=real*self.transform[1,0]+imag*self.transform[1,1]+self.bias[1]
        
        return self.act(newreal)*self.act(newimag)+(1-self.act(newreal))*(1-self.act(newimag))

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
    
    avg_len=torch.mean(tensor[:,0:1]**2+tensor[:,1:2]**2, dim=ch, keepdim=True)
    tensor=tensor/(torch.sqrt(avg_len+EPS**2))
    
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
        if self.padding!=0:
            t=t[..., :-self.padding]
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
        self.pool=ModMaxPool2d(time_kernel, 1)
        self.act3=ModReLU((F,1))
        
        
    def forward(self, tensor):
        B,_, Cin, F, T=tensor.shape
        t=self.conv1(tensor)
        tori=t
        #print(torch.cuda.memory_allocated())
        
        t=self.act1(t)
        t=cov_complex(t) # B,2,ch*ch,F,T
        t=modLog(t)
        t=feature_norm(t, 2)
        t=self.act2(t)
        t=self.conv2(t) # B,2,ch,F,T
        
        t=self.act3(t)
        t=t.view(B,2,-1,T)
        t=self.conv3(t)[..., self.padding:]
        t=t.view(B,2,-1,F,T)
        
        t=self.pool(t) # B,2,ch,F,T
        
        return t, cMul(t, tori) 
        
class MiniBeamformerV2(nn.Module):
    def __init__(self, ch, ch_cov, F, freq_kernel, time_kernel, dilation=1, ch_hid=24, return_gate=False):
        super().__init__()
        
        self.ch=ch
        self.time_kernel=time_kernel
        self.padding=time_kernel*dilation-dilation
        self.return_gate=return_gate
        
        self.conv1=nn.Sequential(
            ParallelConv1d(F, ch_cov, ch_hid, 1),
            ComplexConv2d(ch_hid, ch_hid, (1, time_kernel), padding=(0, self.padding), dilation=(1, dilation), groups=ch_hid)
        )
        self.act1=TReLU((ch_hid, F, 1))
        self.conv2=nn.Conv2d(ch_hid, ch*ch, (1, 1))
        self.w=torch.nn.Parameter(torch.randn(1, 2, ch*ch*F))
        self.act2=ModReLU((ch, F, 1))
        self.conv3=nn.Sequential(
            ComplexConv2d(ch*2, ch*2, (freq_kernel, 1), padding=(freq_kernel//2, 0)),
            ParallelConv1d(F, ch*2, ch, time_kernel, dilation=dilation)
        )
        #nn.Sequential(
            #ComplexConv2d(ch*2, ch_hid, (freq_kernel, time_kernel), padding=(freq_kernel//2, self.padding), dilation=(1, dilation)),
            #ModReLU((ch_hid, F, 1)),
            #ComplexConv2d(ch_hid, ch, (1,1))
        #)
        
    def __norm(self, w, EPS=1e-4):
        r=w[:, 0]
        i=w[:, 1]
        d=torch.sqrt(r*r+i*i+EPS*EPS)
        return torch.stack([r/d, i/d], dim=1)
    
    def __complex_bmm(self, a, b):
        newreal=torch.bmm(a[:,0], b[:,0])-torch.bmm(a[:,1], b[:,1])
        newimag=torch.bmm(a[:,1], b[:,0])+torch.bmm(a[:,0], b[:,1])
        return torch.stack([newreal, newimag], dim=1)
    
    def forward(self, tensor, cov):
        B, _, C, F, T=tensor.shape
        
        cov=self.conv1(cov)[...,:-self.padding]
        tw=self.act1(cov) # B, ch_hid, F, T
        tw=torch.sqrt(tw[:,0]**2+tw[:,1]**2+EPS**2)
        tw_result=tw
        
        tw=self.conv2(tw) # B,ch*ch, F, T
        
        tw=tw.permute(0,3,1,2).flatten(2,3).flatten(0,1).unsqueeze(1)# B*T, 1, C*C*F
        tw=tw*self.__norm(self.w)
        tw=tw.view(B,T,2,C,C,F).permute(0,5,1,2,3,4) # B,F,T,2,C,C
        
        #normalizing tw
        tw_sum=torch.sum(torch.abs(tw), dim=(-3, -2), keepdim=True)
        tw=tw/(tw_sum+EPS)
        
        t2=tensor.permute(0,3,4,1,2) # B,F,T,2,C
        
        tf=self.__complex_bmm(t2.flatten(0,2).unsqueeze(-2), tw.flatten(0,2)).reshape(t2.shape).permute(0,3,4,1,2)
        tf=self.act2(tf)
        tf=self.conv3(torch.cat([tf, tensor], dim=2))
        
        if self.return_gate:
            return tf, tw_result
        else:
            return tf
        
class MiniBeamformerModelV2(nn.Module):
    def __init__(self, ch_in, ch, layers, block_size, time_kernel=4, ch_hid=128):
        super().__init__()
        self.F=block_size//2+1
        
        self.beamformers=nn.ModuleList()
        dilation=1
        
        self.freq_shuffle=ComplexConv2d(self.F,self.F,(1,1))
        self.freq_shuffleback=ComplexConv2d(self.F,self.F,(1,1))
        self.first=ParallelConv1d(self.F, ch, ch, 1)
        self.cln=cLN()
        
        for i in range(layers):
            self.beamformers.append(MiniBeamformerV2(ch, ch*ch, self.F, 5, time_kernel, dilation, ch_hid))
            dilation*=2
        
        self.last=nn.Sequential(ParallelConv1d(self.F, ch, 1, 1))
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        
    def forward(self, tensor):
        t=self.stft.transform(tensor) # B,2,C,F,T
        tf=self.freq_shuffle(t.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        tc=cov_complex(tf) #B,2,ch*ch, F, T
        tc=self.freq_shuffle(tc.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        tc=self.cln(tc.flatten(2,3)).view(tc.shape)
                
        t=self.first(t)
        for l in self.beamformers:
            t=l(t, tc)
                
        t=self.last(t)
        return self.stft.reverse(t)
    
     
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))             
        
                 
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

class SISDRLoss(nn.Module):
    def __init__(self, offset, l=1e-3):
        super().__init__()
        
        self.l=l
        self.offset=offset
        
    def forward(self, signal, gt):
        return torch.sum(asteroid.losses.pairwise_neg_sisdr(signal[..., self.offset:], gt[..., self.offset:]))+self.l*torch.sum(signal**2)/signal.shape[-1]          
    
class L1Loss(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset=offset
        self.loss=nn.L1Loss()
        
    def forward(self, signal, gt):
        return self.loss(signal[..., self.offset:], gt[..., self.offset:])
    
    
class FuseLoss(nn.Module):
    def __init__(self, offset, r=50):
        super().__init__()
        self.offset=offset
        self.l1loss=nn.L1Loss()
        self.sisdrloss=asteroid.losses.pairwise_neg_sisdr
        self.r=r
        
    def forward(self, signal, gt):
        return self.l1loss(signal[..., self.offset:], gt[..., self.offset:])*self.r+torch.mean(self.sisdrloss(signal[..., self.offset:], gt[..., self.offset:]))
    

class NaiveModel2(nn.Module):
    def __init__(self, ch_in, ch_hidden, block_size):
        super().__init__()
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        self.freq=block_size//2+1
        self.freq_shuffle=ComplexConv2d(self.freq, self.freq, (1,1))
        self.freq_rec=ComplexConv2d(self.freq, self.freq, (1,1))
        
        self.conv1r=ComplexConv2d(ch_in, ch_hidden, (1,1))
        self.conv1i=ComplexConv2d(ch_hidden, ch_hidden, (1,1))
        self.conv2r=ComplexConv2d(ch_hidden, ch_hidden, (1,8), padding=(0,7))
        self.conv2i=ComplexConv2d(ch_hidden, ch_hidden, (1,8), padding=(0,7))
        self.act1=TReLU((ch_hidden, self.freq, 1))
        self.act2=TReLU((ch_hidden, self.freq, 1))
        
        self.conv3r=ComplexConv2d(ch_in, ch_hidden, (1,1))
        self.conv3i=ComplexConv2d(ch_hidden, ch_hidden, (1,1))
        self.conv4r=ComplexConv2d(ch_hidden, ch_hidden, (1,8), padding=(0,7))
        self.conv4i=ComplexConv2d(ch_hidden, ch_hidden, (1,8), padding=(0,7))
        self.act3=TReLU((ch_hidden, self.freq, 1))
        self.act4=TReLU((ch_hidden, self.freq, 1))
        
        self.final1=ComplexConv2d(ch_hidden, ch_in, (1,1))
        self.final2=ComplexConv2d(ch_hidden, 1, (1,1))
        
        self.convw1=ComplexConv2d(ch_in, ch_hidden, (1,1))
        self.convw2=ComplexConv2d(ch_hidden, ch_hidden, (1,1))
        self.convw3=ComplexConv2d(ch_hidden, ch_hidden, (1,8), padding=(0,7))
        self.convw4=ComplexConv2d(ch_hidden, 1, (1,1))
        self.actw=TGate((1, self.freq, 1))
        
        self.loss_fn=nn.L1Loss()
        
    def forward(self, mix):
        ts=self.stft.transform(mix) # B, 2, C, F, T
        ts=self.freq_shuffle(ts.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        
        tl=ts
        tl=self.conv1r(tl)
        tl=F.leaky_relu(tl)
        tl=self.conv1i(tl)
        tl=self.act1(tl)
        tl=self.conv2r(tl)[..., :-7]
        tl=F.leaky_relu(tl)
        tl=self.conv2i(tl)[..., :-7]
        tl=self.act2(tl)
        tl=self.final1(tl)        
        td1=torch.sum(ts-tl, dim=2, keepdim=True)/ts.shape[2]
        
        tl=ts
        tl=self.conv3r(tl)
        tl=F.leaky_relu(tl)
        tl=self.conv3i(tl)
        tl=self.act3(tl)
        tl=self.conv4r(tl)[..., :-7]
        tl=F.leaky_relu(tl)
        tl=self.conv4i(tl)[..., :-7]
        tl=self.act4(tl)
        td2=self.final2(tl)
        
        tl=self.convw1(ts)
        tl=F.leaky_relu(tl)
        tl=self.convw2(tl)
        tl=F.leaky_relu(tl)
        tl=self.convw3(tl)[..., :-7]
        tl=F.leaky_relu(tl)
        tl=self.convw4(tl)
        tw=self.actw(tl).unsqueeze(1)
        
        td=td1*tw+td2*(1-tw)
        
        t=self.freq_rec(td.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        
        return self.stft.reverse(t)
    
    
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))
    
    
class NaiveModel3(nn.Module):
    def __init__(self, ch_in, ch_hidden, ch_mid, block_size):
        super().__init__()
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        self.freq=block_size//2+1
        self.freq_shuffle1=ComplexConv2d(self.freq, self.freq, (1,1))
        self.freq_rec1=ComplexConv2d(self.freq, self.freq, (1,1))
        self.freq_shuffle2=ComplexConv2d(self.freq, self.freq, (1,1))
        self.freq_rec2=ComplexConv2d(self.freq, self.freq, (1,1))
        
        self.conv1r=ComplexConv2d(ch_in, ch_hidden, (1,1))
        self.conv1i=ComplexConv2d(ch_hidden, ch_mid, (1,1))
        self.conv2r=ComplexConv2d(ch_mid, ch_hidden, (1,4), padding=(0,3))
        self.conv2i=ComplexConv2d(ch_hidden, ch_mid, (1,4), padding=(0,3))
        self.conv3r=ComplexConv2d(ch_mid, ch_hidden, (1,4), dilation=(1,4), padding=(0,12))
        self.conv3i=ComplexConv2d(ch_hidden, ch_mid, (1,4), dilation=(1,4), padding=(0,12))
        self.act1=TReLU((ch_mid, self.freq, 1))
        self.act2=TReLU((ch_mid, self.freq, 1))
        self.act3=TReLU((ch_mid, self.freq, 1))
        
        self.conv4r=ComplexConv2d(ch_in, ch_hidden, (1,1))
        self.conv4i=ComplexConv2d(ch_hidden, ch_mid, (1,1))
        self.conv5r=ComplexConv2d(ch_mid, ch_hidden, (1,4), padding=(0,3))
        self.conv5i=ComplexConv2d(ch_hidden, ch_mid, (1,4), padding=(0,3))
        self.conv6r=ComplexConv2d(ch_mid, ch_hidden, (1,4), dilation=(1,4), padding=(0,12))
        self.conv6i=ComplexConv2d(ch_hidden, ch_mid, (1,4), dilation=(1,4), padding=(0,12))
        self.act4=TReLU((ch_mid, self.freq, 1))
        self.act5=TReLU((ch_mid, self.freq, 1))
        self.act6=TReLU((ch_mid, self.freq, 1))
        
        self.final1=ComplexConv2d(ch_mid, 1, (1,1))
        self.final2=ComplexConv2d(ch_mid, ch_in, (1,1))        
        
        self.convw=nn.Sequential(
            ComplexConv2d(ch_in+2, ch_hidden, (1,1)),
            nn.LeakyReLU(),
            ComplexConv2d(ch_hidden, ch_mid, (1,4), padding=(0,3)),
            TReLU((ch_mid, self.freq, 1)),
            ComplexConv2d(ch_mid, 1, (1,1))
        )
        self.actw=TGate((1, self.freq, 1))
        
        self.loss_fn=nn.L1Loss()
            
    def forward(self, mix):
        ts=self.stft.transform(mix) # B, 2, C, F, T
        
        tl=self.freq_shuffle1(ts.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        
        tl=self.conv1r(tl)
        tl=F.leaky_relu(tl)
        tl=self.conv1i(tl)
        tl=self.act1(tl)
        tl=self.conv2r(tl)[..., :-3]
        tl=F.leaky_relu(tl)
        tl=self.conv2i(tl)[..., :-3]
        tl=self.act2(tl)
        tl=self.conv3r(tl)[..., :-12]
        tl=F.leaky_relu(tl)
        tl=self.conv3i(tl)[..., :-12]
        tl=self.act3(tl)
        tl=self.final1(tl)
        td1=self.freq_rec1(tl.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        
        tl=self.freq_shuffle2(ts.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        
        tl=self.conv4r(tl)
        tl=F.leaky_relu(tl)
        tl=self.conv4i(tl)
        tl=self.act4(tl)
        tl=self.conv5r(tl)[..., :-3]
        tl=F.leaky_relu(tl)
        tl=self.conv5i(tl)[..., :-3]
        tl=self.act5(tl)
        tl=self.conv6r(tl)[..., :-12]
        tl=F.leaky_relu(tl)
        tl=self.conv6i(tl)[..., :-12]
        tl=self.act6(tl)
        tl=self.final2(tl)
        tl=torch.sum(ts-tl, dim=2, keepdim=True)/ts.shape[2]
        td2=self.freq_rec2(tl.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        
        tw=self.actw(self.convw(torch.cat([ts, td1, td2], dim=2))[..., :-3]).unsqueeze(1)
        td=td1*tw+td2*(1-tw)
            
        return self.stft.reverse(td)
    
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))
    
class FuseModel(nn.Module):
    def __init__(self, ch_in, ch_bf, ch_mid, ch_hid, k_reception, k_mid, naivemodel):
        super().__init__()
        
        self.ch_in=ch_in
        self.ch_bf=ch_bf
        
        self.naivemodel=naivemodel
        self.conv1=CausalComplexConv1d(ch_bf+2, ch_mid, k_reception, fullconv=True)
        self.act1=nn.LeakyReLU()
        self.tcn1=CausalTCN(ch_mid, ch_mid, ch_hid, 1, k_mid)
        self.tcn2=CausalTCN(ch_mid, ch_mid, ch_hid, k_mid, k_mid)
        self.tcn3=CausalTCN(ch_mid, ch_mid, ch_hid, k_mid*k_mid, k_mid)
        self.tcn4=CausalTCN(ch_mid, ch_mid, ch_hid, k_mid*k_mid*k_mid, k_mid)
        self.act2=TReLU((ch_mid, 1))
        self.final=ComplexConv1d(ch_mid, ch_mid, 1)
        #self.gate=TGate((ch_mid, 1))
        self.gate=ModTanh()
        self.conv2=CausalComplexConvTrans1d(ch_mid, 1, k_reception, fullconv=True)
        
    def forward(self, tensor):
        #with torch.no_grad():
        naive_out=self.naivemodel(tensor[:, :self.ch_in].contiguous()) # B, C, L
        
        t=torch.cat([tensor[:, self.ch_in:], naive_out, tensor[:, 0:1]], dim=1) # B, 5, L
        t=toComplex(t) # B,2, C+1, L
        
        t=self.conv1(t)
        tp=t
        t=self.act1(t)
        t=self.tcn1(t)
        t=self.tcn2(t)
        t=self.tcn3(t)
        t=self.tcn4(t)
        t=self.act2(t)
        t=self.final(t)
        t=self.gate(t)
        
        t=cMul(t,tp)
        t=self.conv2(t) # B, 2, 1, L
        return toReal(t)
    
class NaiveModel(nn.Module):
    def __init__(self, ch_in, ch_hidden, block_size):
        super().__init__()
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        self.freq=block_size//2+1
        self.freq_shuffle=ComplexConv2d(self.freq, self.freq, (1,1))
        self.freq_rec=ComplexConv2d(self.freq, self.freq, (1,1))
        
        self.conv1r=nn.Conv2d(ch_in, ch_hidden, (1,1))
        self.conv1i=nn.Conv2d(ch_hidden, ch_hidden, (1,1))
        self.conv1=ComplexConv2d(ch_hidden, ch_hidden, (1,1))
        self.conv2r=nn.Conv2d(ch_hidden, ch_hidden, (1,8), padding=(0,7))
        self.conv2i=nn.Conv2d(ch_hidden, ch_hidden, (1,8), padding=(0,7))
        self.conv2=ComplexConv2d(ch_in, ch_hidden, (1,8), padding=(0,7))
        self.act1=TReLU((ch_hidden, self.freq, 1))
        self.act2=TReLU((ch_hidden, self.freq, 1))
        
        self.conv3r=nn.Conv2d(ch_in, ch_hidden, (1,1))
        self.conv3i=nn.Conv2d(ch_hidden, ch_hidden, (1,1))
        self.conv3=ComplexConv2d(ch_hidden, ch_hidden, (1,1))
        self.conv4r=nn.Conv2d(ch_hidden, ch_hidden, (1,8), padding=(0,7))
        self.conv4i=nn.Conv2d(ch_hidden, ch_hidden, (1,8), padding=(0,7))
        self.conv4=ComplexConv2d(ch_in, ch_hidden, (1,8), padding=(0,7))
        self.act3=TReLU((ch_hidden, self.freq, 1))
        self.act4=TReLU((ch_hidden, self.freq, 1))
        
        self.final1=ComplexConv2d(ch_hidden, ch_in, (1,1))
        self.final2=ComplexConv2d(ch_hidden, 1, (1,1))
        
        self.convw=ComplexConv2d(ch_hidden*2, 1, (1,1))
        
        self.fuse=TGate((1, self.freq, 1))
        
        self.loss_fn=nn.L1Loss() #asteroid.losses.pairwise_neg_sisdr 
    def forward(self, mix):
        ts=self.stft.transform(mix) # B, 2, C, F, T
        ts=self.freq_shuffle(ts.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        t=ts
        
        tl=feature_norm(t, 2)
        tl=cLog(t)
        tl=torch.stack([self.conv1r(tl[:,0]), self.conv1r(tl[:,1])], dim=1)
        tl=torch.stack([self.conv1i(F.leaky_relu(tl[:,0])), self.conv1i(tl[:,1])], dim=1)
        tl=cExp(tl)
        tl=self.act1(tl)
        t=self.conv1(tl)
        
        tl=feature_norm(t, 2)
        tl=cLog(t)
        tl=torch.stack([self.conv2r(tl[:,0]), self.conv2r(tl[:,1])], dim=1)[..., :-7]
        tl=torch.stack([self.conv2i(F.leaky_relu(tl[:,0])), self.conv2i(tl[:,1])], dim=1)[..., :-7]
        tl=cExp(tl)
        tl1=tl
        tc=self.conv2(ts)[..., :-7]
        t=self.act2(cMul(tl,tc))
        
        t=self.final1(t)
        td1=torch.sum(ts-t, dim=2, keepdim=True)/ts.shape[2]
       
        t=ts
        tl=feature_norm(t, 2)
        tl=cLog(t)
        tl=torch.stack([self.conv3r(tl[:,0]), self.conv3r(tl[:,1])], dim=1)
        tl=torch.stack([self.conv3i(F.leaky_relu(tl[:,0])), self.conv3i(tl[:,1])], dim=1)
        tl=cExp(tl)
        tl=self.act3(tl)
        t=self.conv3(tl)
        
        tl=feature_norm(t, 2)
        tl=cLog(t)
        tl=torch.stack([self.conv4r(tl[:,0]), self.conv4r(tl[:,1])], dim=1)[..., :-7]
        tl=torch.stack([self.conv4i(F.leaky_relu(tl[:,0])), self.conv4i(tl[:,1])], dim=1)[..., :-7]
        tl=cExp(tl)
        tc=self.conv4(ts)[..., :-7]
        t=self.act4(cMul(tl,tc))
        
        td2=self.final2(t)
        
        f=self.convw(torch.cat([tl1, tl], dim=2)) # B,2,1,F,T
        f=self.fuse(f).unsqueeze(1) # B, 1, F, T
        td=td1*f+td2*(1-f)
        
        t=self.freq_rec(td.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        
        return self.stft.reverse(t)
    
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))
     
        
class CausalTCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dilation, kernel, activation=nn.LeakyReLU(0.1)):
        super().__init__()
        
        self.layers=nn.ModuleList()
        self.layers.append(CausalComplexConv1d(in_channels, hidden_channels, kernel, dilation=dilation, activation=activation))
        self.layers.append(activation)
        self.layers.append(ComplexConv1d(hidden_channels, out_channels, 1))
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y=x
        for l in self.layers:
            y=l(y)
        return x+y if x.shape==y.shape else y
    
class HybridModel(nn.Module):
    def __init__(self, ch_in, ch_raw, ch, ch_wav, ch_hid, blayers, players, block_size, wav_kernel, time_kernel=4, reception=64):
        super().__init__()
        self.F=block_size//2+1
        self.block_size=block_size
        self.blayers=blayers
        self.players=players
        self.ch_raw=ch_raw
        self.ch=ch
        self.cov_hid=64
        
        self.beamformers=nn.ModuleList()
        self.postfilters=nn.ModuleList()
        self.mappings=nn.ModuleList()
        
        self.wav_first=CausalComplexConv1d(ch_in, ch_wav, reception, fullconv=True) #B,2,C,L
        self.cov_first=ParallelConv1d(self.F, ch_raw*ch_raw, ch*ch, 1)
        self.spec_first=ParallelConv1d(self.F, ch_raw, ch, 1)
        
        dilation=1
        for i in range(blayers):
            self.beamformers.append(MiniBeamformerV2(ch, ch*ch, self.F, 5, time_kernel, dilation, ch_hid=self.cov_hid, return_gate=True))
            dilation*=time_kernel
            
        dilation=1
        for i in range(players):
            self.postfilters.append(CausalTCN(ch_wav*2, ch_wav, ch_hid, dilation, wav_kernel, ModReLU((ch_hid, 1))))
            self.mappings.append(ComplexConv1d(self.F*ch, ch_wav, 1))
            dilation*=wav_kernel
            
        self.gate_mapping=nn.Conv1d(self.cov_hid*self.F*blayers, ch_wav, 1) 
        
        self.wav_last=CausalComplexConvTrans1d(ch_wav, 1, reception, fullconv=True)
        
        
        self.stft=ComplexSTFTWrapper(block_size, hop_length=block_size, center=False)
        self.freq_shuffle=ComplexConv2d(self.F, self.F, (1,1))
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
    
    def __interpolate(self, spec_in):
        # spec_in: B,2,*,T
        B,_,C,T=spec_in.shape
        padding=torch.repeat_interleave(spec_in[:,:,:,0:1], self.block_size-1, dim=3)
        spec_in=nn.functional.interpolate(spec_in, (C, (T-1)*self.block_size+1), mode='bilinear', align_corners=True)
        spec_in=torch.cat([padding, spec_in], dim=3)
        return spec_in
    
    def forward(self, tensor):
        # B, C, L
        mix=toComplex(tensor)
        wav_t=self.wav_first(mix) # B,2,C,L
        
        spec=self.stft.transform(tensor[:, :self.ch_raw, :].contiguous()) # B,2,C,F,T
        
        spec=self.freq_shuffle(spec.permute(0,1,3,2,4)).permute(0,1,3,2,4)
        tc=cov_complex(spec) #B,2,ch_raw*ch_raw, F, T
        tc=self.cov_first(tc)
        tc=feature_norm(tc, 2)
        
        t=self.spec_first(spec)
        gs=[]
        for i in range(self.blayers):
            t, g=self.beamformers[i](t, tc) #B,2,ch,F,T
            gs.append(g) #B,ch*ch, F, T
            
        B,_,C,F,T=t.shape
        t=t.view(B,2,C*F,T)
        
        wav=wav_t
        for i in range(self.players):
            t1=self.mappings[i](t)
            t1=self.__interpolate(t1)
            wav_t=torch.cat([t1, wav_t], dim=2)
            wav_t=self.postfilters[i](wav_t)
            
        gs=torch.cat(gs, dim=1).view(B, self.blayers*self.cov_hid*F, -1)
        gs=self.gate_mapping(gs).unsqueeze(1)
        gs=self.__interpolate(gs)
        
        wav_t=wav_t*gs
        wav_t=self.wav_last(wav_t)
        return toReal(wav_t)
     
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))             
                 