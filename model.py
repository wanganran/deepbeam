import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcomplex.nn as cnn
import torchcomplex.nn.functional as cF
import numpy as np
import importlib
import complexcnn.modules
importlib.reload(complexcnn.modules)
from complexcnn.modules import ComplexConv1d, ComplexConvTranspose1d
import util
importlib.reload(util)
#from util import si_sdr_loss


def cSymRelu6(input, inplace=False):
    return F.relu6(input+3, inplace=inplace)-3

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

def toComplex(tensor):
    return torch.stack((tensor, tensor-tensor), dim=1)

def toReal(tensor):
    return tensor[:,0,...]

class CausalComplexConv1d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, dilation=1, fullconv=False):
        super().__init__()
        # input is [batch, channel, length]
        # depthwise + 1x1
        self.kernel_size=kernel_size
        self.stride=stride
        self.pad=nn.ConstantPad1d((kernel_size*dilation-dilation+1-stride,0), 0)
        self.fullconv=fullconv
        if not fullconv:
            self.l1=ComplexConv1d(channel_in, channel_in, kernel_size, stride=stride, dilation=dilation, groups=channel_in, padding=0)
            self.l2=ComplexConv1d(channel_in, channel_out, 1)
        else:
            self.l=ComplexConv1d(channel_in, channel_out, kernel_size, stride=stride, dilation=dilation, padding=0)
    def forward(self, x):
        if not self.fullconv:
            return self.l2(cSymRelu6(self.l1(self.pad(x))))
        else:
            return self.l(self.pad(x))

class CausalComplexConvTrans1d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, fullconv=False):
        super().__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        self.fullconv=fullconv
        if not fullconv:
            self.l1=ComplexConv1d(channel_in, channel_out, 1)
            self.l2=ComplexConvTranspose1d(channel_out, channel_out, kernel_size, stride=stride, groups=channel_out)
        else:
            self.l=ComplexConvTranspose1d(channel_in, channel_out, kernel_size, stride=stride)
            
    def forward(self, x):
        if not self.fullconv:
            return self.l2(cSymRelu6(self.l1(x)))[..., self.kernel_size-self.stride:]
        else:
            return self.l(x)[..., self.kernel_size-self.stride:]
    
class SimpleNetwork(nn.Module):        
    def __init__(self, in_channels, channels, kernel_size, reception, lstm_layers=2, depth=2, context=3, stride=2):
        super().__init__()
        
        # simple model
        self.encoders=nn.ModuleList()
        self.decoders=nn.ModuleList()
        
        self.kernel_size=kernel_size
        self.reception=reception
        self.context=context
        self.stride=stride
        self.lstm=nn.GRU(input_size=channels, hidden_size=channels, num_layers=lstm_layers)
        #self.lstm=nn.LSTM(input_size=channels, hidden_size=channels, num_layers=lstm_layers) 
        
        # added wavenet-like convs
        wnet_reception=4
        wnet_layers=4
        
        self.wconvs=nn.ModuleList()
        dilation=1
        for i in range(wnet_layers):
            self.wconvs.append(CausalComplexConv1d(channels, channels, wnet_reception, stride=1, dilation=dilation, fullconv=True))
            self.wconvs.append(CSymRelu6())
            dilation*=wnet_reception
        
        self.first=CausalComplexConv1d(in_channels, channels, reception)
        self.last=CausalComplexConvTrans1d(channels*2, 1, reception, fullconv=True)
        for i in range(depth):
            encode=nn.ModuleDict()
            encode["conv"]=CausalComplexConv1d(channels, channels, kernel_size, stride=stride, fullconv=True)
            encode["relu"]=CSymRelu6()
            
            self.encoders.append(encode)
            
            decode=nn.ModuleDict()
            decode["conv"]=CausalComplexConvTrans1d(channels*2, channels, kernel_size, stride=stride) 
            self.decoders.append(decode)

    def forward(self, mix:torch.Tensor):
        # mix: [batch, channel, length]
        # mix=torch.view_as_complex(torch.stack((mix, mix-mix), dim=-1))
        mix=toComplex(mix)
        x=self.first(mix)
        
        # attention
        x2=x # B, channels, L
        
        for layer in self.wconvs:
            x2=layer(x2)
        
        x=x*x2
        
        saved=[x]
        
        # encoders
        id=0
        for encode in self.encoders:
            x=encode["conv"](x)
            x=encode["relu"](x)
            saved.append(x)
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
            # skip=center_trim(saved.pop(-1), x)
            skip=saved.pop(-1)
            #x=x+skip
            x=torch.cat((x, skip), -2)
            x=decode["conv"](x)
        
        # last layer
        skip=saved.pop(-1)
        x=torch.cat((x, skip), -2)
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