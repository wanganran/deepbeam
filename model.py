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
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, dilation=1):
        super().__init__()
        # input is [batch, channel, length]
        # depthwise + 1x1
        self.kernel_size=kernel_size
        self.stride=stride
        self.pad=nn.ConstantPad1d((kernel_size*dilation-dilation+1-stride,0), 0)
        self.l1=ComplexConv1d(channel_in, channel_in, kernel_size, stride=stride, dilation=dilation, groups=channel_in, padding=0)
        #self.l1=ComplexConv1d(channel_in, channel_in, kernel_size, stride=stride, dilation=dilation, groups=channel_in, padding=kernel_size*dilation-dilation+1-stride)
        self.l2=ComplexConv1d(channel_in, channel_out, 1)
    
    def forward(self, x):
        #return self.l2(cSymRelu6(self.l1(x)[..., :(-self.kernel_size//self.stride+1)]))
        return self.l2(cSymRelu6(self.l1(self.pad(x))))
        

class CausalComplexConvTrans1d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1):
        super().__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        self.l1=ComplexConv1d(channel_in, channel_out, 1)
        self.l2=ComplexConvTranspose1d(channel_out, channel_out, kernel_size, stride=stride, groups=channel_out)
    
    def forward(self, x):
        return self.l2(cSymRelu6(self.l1(x)))[..., self.kernel_size-self.stride:]

    
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
            self.wconvs.append(CausalComplexConv1d(channels, channels, wnet_reception, stride=1, dilation=dilation))
            self.wconvs.append(CSymRelu6())
            dilation*=wnet_reception
        
        for i in range(depth):
            encode=nn.ModuleDict()
            encode["conv"]=CausalComplexConv1d(channels, channels, kernel_size, stride=stride) if i>0 else CausalComplexConv1d(in_channels, channels, reception)
            encode["relu"]=CSymRelu6()
            
            self.encoders.append(encode)
            
            decode=nn.ModuleDict()
            decode["conv"]=CausalComplexConvTrans1d(channels*2, channels, kernel_size, stride=stride) if i!=depth-1 else CausalComplexConvTrans1d(channels*2, 1, reception)
            if i<depth-1:
                decode["relu"]=CSymRelu6()
            self.decoders.append(decode)

    def forward(self, mix:torch.Tensor):
        # mix: [batch, channel, length]
        # mix=torch.view_as_complex(torch.stack((mix, mix-mix), dim=-1))
        mix=toComplex(mix)
        saved=[mix]
        x=mix
        id=0
        for encode in self.encoders:
            x=encode["conv"](x)
            x=encode["relu"](x)
            saved.append(x)
            #print(x.shape)
            id+=1
        
        x2=x
        for layer in self.wconvs:
            x=layer(x)
            #print(x.shape)
        
        x=x+x2
        
        '''
        x=toReal(x)
        x = x.permute(2,0,1)
        x=self.lstm(x)[0]
        x = x.permute(1,2,0)
        x=toComplex(x)
        '''
        
        for decode in self.decoders:
            # skip=center_trim(saved.pop(-1), x)
            skip=saved.pop(-1)
            #x=x+skip
            x=torch.cat((x, skip), -2)
            x=decode["conv"](x)
            if "relu" in decode:
                x=decode["relu"](x)
            # print(x.shape)
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
        #return F.l1_loss(signal[..., 24000:], gt[..., 24000:])
        return util.wsdr_loss(mix, signal, gt)