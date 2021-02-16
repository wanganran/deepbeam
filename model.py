import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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

def CausalConv1d(channel_in, channel_out, kernel_size, stride=1):
    # input is [batch, channel, length]
    return nn.Conv1d(channel_in, channel_out, kernel_size, stride=stride, padding=kernel_size-1)

def CausalConvTranspose1d(channel_in, channel_out, kernel_size, stride=1):
    return nn.ConvTranspose1d(channel_in, channel_out, kernel_size, stride=stride)
        
class SimpleNetwork(nn.Module):        
    def __init__(self, in_channels, channels, kernel_size, reception, depth=2, context=3, stride=2):
        super().__init__()
        
        # simple model
        self.encoders=nn.ModuleList()
        self.decoders=nn.ModuleList()
        
        self.kernel_size=kernel_size
        self.reception=reception
        self.context=context
        self.stride=stride
        
        for i in range(depth):
            encode=nn.ModuleDict()
            encode["conv1"]=CausalConv1d(channels, channels, kernel_size) if i>0 else CausalConv1d(in_channels, channels, reception)
            encode["relu"]=nn.ReLU()
            # 1x1 mapping
            encode["conv2"]=CausalConv1d(channels, 2*channels, 1, stride=stride)
            encode["activation"]=nn.GLU(dim=1)

            self.encoders.append(encode)
            
            decode=nn.ModuleDict()
            decode["conv1"]=CausalConv1d(channels, 2*channels, context)
            decode["activation"]=nn.GLU(dim=1)
            decode["conv2"]=CausalConvTranspose1d(channels, channels, kernel_size, stride=stride) if i!=depth-1 else CausalConvTranspose1d(channels, 1, kernel_size, stride=stride)
            if i!=depth-1:
                decode["relu"]=nn.ReLU()
            
            self.decoders.append(decode)

    def forward(self, mix:torch.Tensor):
        # mix: [batch, channel, length]
        saved=[mix]
        x=mix
        id=0
        for encode in self.encoders:
            x=encode["conv1"](x)
            x=x[:,:,:-self.kernel_size+1] if id>0 else x[:,:,:-self.reception+1] 
            x=encode["relu"](x)
            x=encode["conv2"](x)
            x=encode["activation"](x)
            saved.append(x)
            id+=1
        
        for decode in self.decoders:
            skip=center_trim(saved.pop(-1), x)
            x=x+skip
            x=decode["conv1"](x)[:,:,:-self.context+1]
            x=decode["activation"](x)
            x = decode["conv2"](x)[:,:,self.kernel_size-self.stride:]
            if "relu" in decode:
                x=decode["relu"](x)
            
        return x
    
    def loss(self, signal, gt):
        return F.l1_loss(signal, gt)
    
    
    