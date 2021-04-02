import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib
import complexcnn.modules
importlib.reload(complexcnn.modules)
from complexcnn.modules import ComplexConv1d, ComplexConvTranspose1d, ComplexLinear, ComplexMultiLinear, ComplexSTFTWrapper, ComplexConv2d, cMul, toComplex, toReal, CausalComplexConv2d, CausalComplexConvTrans1d, CausalComplexConv1d
import util
importlib.reload(util)
import asteroid
from util import cudaMem



def cSymRelu6(input, inplace=False):
    return F.relu6(input+3, inplace=inplace)-3

def sineAct(input, inplace=False):
    return torch.sin((F.relu6(input+3, inplace=inplace)/6*np.pi)-np.pi/2)

class SineAct(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace=inplace
    def forward(self, x):
        return sineAct(x, self.inplace)

class CSymRelu6(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace=inplace
    def forward(self, x):
        return cSymRelu6(x, self.inplace)
    
def center_trim(tensor, reference):
    """
    Trim a tensor to match with the dimension of `reference`.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., diff // 2:-(diff - diff // 2)]
    return tensor




    
class SpectrogramCRN(nn.Module):
    def __init__(self, in_channels, mid_channels, freq_channels, final_channels, block_size, freq_kernel_size, time_kernel_size, n_conv):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.in_channels=in_channels
        self.block_size=block_size
        
        self.act=nn.LeakyReLU(0.1)
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        
        self.first_kernel=5
        F=block_size//2-self.first_kernel+1
        self.first=ComplexConv2d(in_channels, mid_channels, (self.first_kernel, self.first_kernel), padding=(0, self.first_kernel-1))
        self.convs=nn.ModuleList()
        for i in range(n_conv):
            self.convs.append(CausalComplexConv2d(mid_channels, freq_channels, mid_channels, freq_kernel_size, time_kernel_size))
        
        self.last=ComplexConv2d(mid_channels, 1, (1,1)) # B, 2, 1, F, T
        
        F2=F-(freq_kernel_size-1)*n_conv
        
        self.rnn_r=nn.GRU(F2, F2, 2, batch_first=True)
        self.rnn_i=nn.GRU(F2, F2, 2, batch_first=True)    
        
        self.final=ComplexLinear(F2, final_channels)
        
    def forward(self, mix:torch.Tensor):
        l0=self.stft.transform(mix) # B, 2, C, F, T
        l1=l0[:,:,:,:self.block_size//2,:]
        
        l2=self.first(l1)[:,:,:,:,:-self.first_kernel+1]
        
        for conv in self.convs:
            l2=conv(l2)
        l3=self.last(l2).permute(0,1,4,3,2).squeeze(-1)
        l3_r=l3[:,0]
        l3_i=l3[:,1]
        
        r2r_out = self.rnn_r(l3_r)[0]
        r2i_out = self.rnn_i(l3_r)[0]
        i2r_out = self.rnn_r(l3_i)[0]
        i2i_out = self.rnn_i(l3_i)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out 
        l4=torch.stack((real_out, imag_out), dim=1) # B,2,T,F
        
        return self.final(l4) # B,2,T,final
        
    def getStep(self):
        return self.block_size//2
    
class CausalTCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dilation, kernel=5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.layers=nn.ModuleList()
        self.layers.append(CausalComplexConv1d(in_channels, hidden_channels, kernel, dilation=dilation))
        self.layers.append(nn.LeakyReLU(0.1))
        self.out=ComplexConv1d(hidden_channels, out_channels, 1)
        
        #self.out2=ComplexConv1d(hidden_channels, out_channels, 1)
    
    def forward(self, x):
        y=x
        for l in self.layers:
            y=l(y)
        return x+self.out(y)
        
class FuseModel(nn.Module):
    
    def __init__(self, in_channels, reception, channels, hidden_channels, mid_channels, freq_channels, block_size, kernel_size_freq, kernel_size_time, kernel_size_1d, dilation=2, layers_crn=4, layers_1d=6):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 
        self.crn=SpectrogramCRN(in_channels, mid_channels, freq_channels, channels, block_size, kernel_size_freq, kernel_size_time, layers_crn)
        self.first=CausalComplexConv1d(in_channels, channels, reception, fullconv=True)
        self.last=CausalComplexConvTrans1d(channels, 1, reception, fullconv=True)
        
        self.mixer=ComplexConv1d(channels*2, channels, 1)
        
        self.encoders=nn.ModuleList()
        
        d=1
        for i in range(layers_1d):
            self.encoders.append(CausalTCN(channels, channels, hidden_channels, d))
            d*=dilation
        
        self.block_size=block_size
        self.activation=nn.LeakyReLU(0.1)
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 

    def forward(self, mix:torch.Tensor):
        # mix: [batch, channel, length]
        # make sure tensor are the same size
        residual=mix.shape[2]%self.block_size
        if residual!=0:
            mix=mix[:,:,:-residual]
        
        #cudaMem()
        
        #spectrogram
        spec_out=self.crn(mix)[:,:,:-1,:] # B,2,T,C
        spec_out=torch.repeat_interleave(spec_out, repeats=self.block_size//2, dim=2)
        spec_out=spec_out.permute(0,1,3,2) # B,2,C,L
        
        #cudaMem()
        
        mix=toComplex(mix)
        x=self.first(mix)
        final=x
        x=self.activation(x) # B,2,C,L
        
        
        # mix
        x=torch.cat((x, spec_out), 2) # B,2,C*2, L
        x=self.mixer(x)
        x=self.activation(x)
        
        
        # encoders
        for encode in self.encoders:
            #cudaMem()
            x=encode(x)
            
        # mask
        final=cMul(F.tanh(x), final)
        
        # last layer
        final=self.last(final)
        
        return toReal(final)
    
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))   
    
class NSNet(nn.Module):
    def __init__(self, in_channels, block_size, ff_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.in_channels=in_channels
        self.ff_size=ff_size
        self.block_size=block_size
        
        self.act=nn.LeakyReLU(0.1)
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        self.shuffle=ComplexLinear(in_channels, in_channels)
        self.ff1=ComplexMultiLinear(in_channels, block_size//2, ff_size)
        self.shuffle2=ComplexLinear(in_channels, in_channels)
        self.ff2=ComplexMultiLinear(in_channels, ff_size, ff_size)
        self.bottleneck=ComplexLinear(in_channels, 1)
        self.rnn_r=nn.GRU(ff_size, ff_size, 2, batch_first=True)
        self.rnn_i=nn.GRU(ff_size, ff_size, 2, batch_first=True)
        self.expand=ComplexLinear(1, in_channels)
        self.ff3=ComplexMultiLinear(in_channels, ff_size, ff_size)
        self.ff4=ComplexMultiLinear(in_channels, ff_size, block_size//2)
        
        self.alayers=nn.ModuleList()
        self.alayers.append(ComplexLinear(1, in_channels*block_size//2))
        for i in range(3):
            self.alayers.append(ComplexLinear(2, in_channels*ff_size))
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        
    def forward(self, mix:torch.Tensor, angle:torch.Tensor):
        #angle=angle.view(-1,2,1)
        B=mix.shape[0]
        
        l0=self.stft.transform(mix) # B, 2, C, F, T
        l1=l0[:,:,:,:self.block_size//2,:]
        ori=l1
        a1=self.alayers[0](angle)
        a1=a1.view(-1,2,self.in_channels, self.block_size//2, 1)
        l1=cMul(l1,a1)
        l1=self.act(l1)
        
        l2=self.shuffle(l1.permute(0,1,3,4,2)) # -1,2,F,T,C
        l2=l2.permute(0,1,4,3,2) # -1,2,C,T,F
        l2=self.ff1(l2)
        a2=self.alayers[1](angle)
        a2=a2.view(-1,2,self.in_channels, 1, self.ff_size)
        l2=cMul(l2,a2)
        l2=self.act(l2)
        l2=l2.permute(0, 1, 2, 4, 3) # -1, 2, C, F, T
        
        l3=self.shuffle2(l2.permute(0,1,3,4,2)) # -1,2,F,T,C
        l3=l3.permute(0,1,4,3,2) # -1,2,C,T,F
        l3=self.ff2(l3)
        a3=self.alayers[2](angle)
        a3=a3.view(-1,2, self.in_channels, 1, self.ff_size)
        l3=cMul(l3,a3)
        l3=self.act(l3)
        l3=l3.permute(0, 1, 2, 4, 3) # -1, 2, C, F, T
        
        l4=l3.permute(0,1,3,4,2) # B,2,F,T,C
        l4=self.bottleneck(l4).squeeze(-1).permute(0,1,3,2) # B, 2, T, F
        
        l5_r, _=self.rnn_r(l4[:,0]) # B, T, F
        l5_i, _=self.rnn_i(l4[:,1])
        l5=torch.stack((l5_r, l5_i), dim=1) # B,2,T,F
        
        l6=self.expand(l5.unsqueeze(-1)).permute(0,1,4,2,3)
        l6=self.ff3(l6)
        a6=self.alayers[3](angle)
        a6=a6.view(-1, 2, self.in_channels, 1, self.ff_size)
        l6=cMul(l6,a6)
        l6=self.act(l6)
        
        l7=self.ff4(l6)
        #l7=self.act(l7) # B,2, C, T,F
        l7=l7.permute(0,1,2, 4,3) # B,2,C,F,T
        
        result=cMul(ori, l7)
        result=torch.sum(result, dim=2, keepdim=True) # B,2,1, F,T
        result=torch.cat([result, l0[:,:,0:1,self.block_size//2:,:]], 3)
        
        return self.stft.reverse(result) # B, L
       
    def loss(self, signal, gt, mix):
        
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))   
    
class SimpleNetwork(nn.Module):        
    def __init__(self, in_channels, channels, kernel_size, reception, lstm_layers=2, depth=2, stride=2, wnet_reception=4, wnet_layers=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 
        # simple model
        self.encoders=nn.ModuleList()
        self.decoders=nn.ModuleList()
        
        self.kernel_size=kernel_size
        self.reception=reception
        self.stride=stride
        self.activation=nn.LeakyReLU(0.1)
        
        # added wavenet-like convs
        self.wnet_reception=wnet_reception
        self.wnet_layers=wnet_layers
        
        self.wconvs=nn.ModuleList()
        self.alayers=nn.ModuleList()
        
        dilation=1
        for i in range(wnet_layers):
            wconv=nn.ModuleList()
            wconv.append(CausalComplexConv1d(channels, channels, wnet_reception, stride=1, dilation=dilation, fullconv=True))
            wconv.append(self.activation)
            self.wconvs.append(wconv)
            
            al=nn.ModuleList()
            al.append(ComplexLinear(2, channels))
            al.append(SineAct())
            self.alayers.append(al) # B, C
            
            dilation*=wnet_reception
        
        self.first=CausalComplexConv1d(in_channels, channels, reception, fullconv=True)
        self.last=CausalComplexConvTrans1d(channels*2, 1, reception, fullconv=True)
        for i in range(depth):
            encode=nn.ModuleDict()
            encode["conv"]=CausalComplexConv1d(channels, channels, kernel_size, stride=stride)
            encode["relu"]=self.activation
            
            self.encoders.append(encode)
            
            decode=nn.ModuleDict()
            if i>0:
                decode["conv"]=CausalComplexConvTrans1d(channels*2, channels, kernel_size, stride=stride) 
            else:
                decode["conv"]=CausalComplexConvTrans1d(channels, channels, kernel_size, stride=stride) 
            
            decode["relu"]=self.activation
            self.decoders.append(decode)
            
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 

    def forward(self, mix:torch.Tensor, angle:torch.Tensor):
        # mix: [batch, channel, length]
        # mix=torch.view_as_complex(torch.stack((mix, mix-mix), dim=-1))
        # angle: (B, 2)
        
        
        mix=toComplex(mix)
        x=self.first(mix)
        x=self.activation(x)
        
        # attention
        x2=x # B, channels, L
        for i in range(self.wnet_layers):
            xa=angle
            for layer in self.alayers[i]:
                xa=layer(xa)
            for layer in self.wconvs[i]:
                x2=layer(x2)
            x2=cMul(x2, xa.unsqueeze(-1))
        x=cMul(x,x2)
        
        saved=[]
        
        # encoders
        id=0
        for encode in self.encoders:
            saved.append(x)
            
            x=encode["conv"](x)
            x=encode["relu"](x)
            id+=1
        
        '''
        x=toReal(x)
        x = x.permute(2,0,1)
        x=self.lstm(x)[0]
        x = x.permute(1,2,0)
        x=toComplex(x)
        '''
        # decoders
        for decode in self.decoders:
            x=decode["conv"](x)
            x=decode["relu"](x)
            skip=saved.pop(-1)
            x=torch.cat((x, skip), -2)
            
        # last layer
        x=self.last(x)
        
        return toReal(x)
    
    def loss(self, signal, gt, mix):
        
        # less weight on first 0.5s
        '''
        total_len=signal.shape[-1]
        l1=F.l1_loss(signal[..., :12000], gt[..., :12000], reduction='sum')
        l2=F.l1_loss(signal[..., 12000:24000], gt[..., 12000:24000], reduction='sum')
        l3=F.l1_loss(signal[..., 24000:], gt[..., 24000:], reduction='sum')
        return (l1*0.1+l2*0.5+l3)/total_len
        '''
        #return si_sdr_loss(signal, gt)
        return F.l1_loss(signal[..., 24000:], gt[..., 24000:])
        #return util.wsdr_loss(mix, signal, gt)
        #return torch.sum(self.loss_fn(signal, gt))
    