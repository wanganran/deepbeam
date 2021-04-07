import torch
import torch.nn as nn
import torch.functional as F
import importlib
import complexcnn.modules
importlib.reload(complexcnn.modules)
from complexcnn.modules import ComplexConv1d, ComplexConvTranspose1d, ComplexLinear, ComplexMultiLinear, ComplexSTFTWrapper, ComplexConv2d, cMul, cLog, cExp, toComplex, toReal, CausalComplexConv1d, CausalComplexConvTrans1d,CausalComplexConv2d,ModReLU

from sru import SRU

import torch.jit as jit

import convlstm
importlib.reload(convlstm)
from convlstm import ConvLSTM
import asteroid

class cTwoAvgPool(nn.Module):
    def __init__(self, channel=-1):
        super().__init__()
        
        self.channel=channel
    
    def forward(self, tensor):
        s=tensor.shape[self.channel]
        even=torch.arange(s//2, device=tensor.device)*2
        odd=torch.arange(s//2, device=tensor.device)*2+1
        return (torch.index_select(tensor, self.channel, even)+torch.index_select(tensor, self.channel, odd))/2

# transform time domain to freq domain, and a CNN to transform it
class SpectrogramNN(nn.Module):
    def __init__(self, in_channels, out_channels, block_size, freq_kernel_size, time_kernel_size):
        super().__init__()
        
        self.in_channels=in_channels
        self.block_size=block_size
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        
        self.F=block_size//2-freq_kernel_size+1
        assert(self.F%4==0)
        
        self.time_kernel_size=time_kernel_size
        self.first=nn.Sequential(
            ComplexConv2d(in_channels, out_channels, (freq_kernel_size, time_kernel_size), padding=(0, time_kernel_size-1)),
            ModReLU([out_channels, self.F, 1]),
            cTwoAvgPool(-2),
            ComplexConv2d(out_channels, out_channels, (1,1)),
            ModReLU([out_channels, self.F//2, 1]),
            cTwoAvgPool(-2)
        )
        
    def forward(self, mix):
        l0=self.stft.transform(mix) # B, 2, C, F, T
        l1=l0[:,:,:,:self.block_size//2,:]
        
        l2=self.first(l1)[:,:,:,:,:-self.time_kernel_size+1]
        return l2 # B,2,C,F,T
    
    def getStep(self):
        return self.block_size//2
    
    def getF(self):
        return self.F//4

class Fuser(nn.Module):
    def __init__(self, spec_channels, spec_F, wav_channels, spec_step, out_channels):
        super().__init__()
        self.spec_step=spec_step
        self.fuser=ComplexConv1d(spec_F*spec_channels+wav_channels, out_channels, 1)
        self.act=ModReLU([out_channels, 1])
        
    def forward(self, spec_in, wav_in):
        # spec_in: B,2,T,C,F (log)
        # wav_in: B,2,C,L (log)
        spec_in=spec_in.reshape(spec_in.shape[0], spec_in.shape[1], spec_in.shape[2], -1) # B,2,T,C*F
        padding=torch.repeat_interleave(spec_in[:,:,0:1,:], self.spec_step-1, dim=2)
        spec_in=nn.functional.interpolate(spec_in, ((spec_in.shape[2]-1)*self.spec_step+1, spec_in.shape[-1]), mode='bilinear', align_corners=True)
        spec_in=torch.cat([padding, spec_in], dim=2)
        #spec_in=torch.repeat_interleave(spec_in, repeats=self.spec_step, dim=2)
        
        spec_in=spec_in.permute(0,1,3,2) # B,2,C,L
        
        total=torch.cat([spec_in, wav_in], dim=2)
        return self.act(self.fuser(total)) # log
    
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
        
    
class DualModel(nn.Module):
    def __init__(self, in_channels, spec_channels, block_size, first_kernel_size, 
                 lstm_kernel_size, lstm_channels, lstm_layer, 
                 reception, wav_channels, wav_layers, hidden_channels, wav_kernel_size):
        super().__init__()
        
        self.spec_model=SpectrogramNN(in_channels, spec_channels, block_size, first_kernel_size[0], first_kernel_size[1])
        self.lstm_layer_real=ConvLSTM(spec_channels, lstm_channels, [1]+(lstm_layer-1)*[lstm_kernel_size], lstm_layer, True)
        self.lstm_layer_imag=ConvLSTM(spec_channels, lstm_channels, [1]+(lstm_layer-1)*[lstm_kernel_size], lstm_layer, True)
        
        #self.lstm_layers=CausalComplexConv2d(spec_channels, lstm_channels[-2], lstm_channels[-1], first_kernel_size[0], first_kernel_size[1])
        
        self.wav_first=nn.Sequential(
            CausalComplexConv1d(in_channels, wav_channels, reception, fullconv=True), #B,2,C,L
            ModReLU([wav_channels, 1]),
            CausalComplexConv1d(wav_channels, hidden_channels, wav_kernel_size, activation=ModReLU([hidden_channels, 1])), #B,2,C,L
            ModReLU([hidden_channels, 1]),
            ComplexConv1d(hidden_channels, wav_channels, 1)
            )
        
        self.wav_short=CausalTCN(wav_channels, wav_channels, hidden_channels, 1, wav_kernel_size, ModReLU([hidden_channels, 1]))
        
        self.fuser=Fuser(lstm_channels[-1], self.spec_model.getF(), wav_channels, self.spec_model.getStep(), wav_channels)
        
        self.wav_layers=nn.ModuleList()
        
        dilation=1
        for i in range(wav_layers):
            self.wav_layers.append(CausalTCN(wav_channels, wav_channels, hidden_channels, dilation, wav_kernel_size, ModReLU([hidden_channels, 1])))
            dilation*=wav_kernel_size
            
        self.last=CausalComplexConvTrans1d(wav_channels*2, 1, reception, fullconv=True)
        
        self.block_size=block_size
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        
        
    def forward(self, mix):
        residual=mix.shape[2]%self.block_size
        if residual!=0:
            mix=mix[:,:,:-residual]
          
        # spectrogram
        spec=self.spec_model(mix) #B,2,C,F,T
        spec_out=spec
        spec_out=cLog(spec)
        
        
        #print(torch.max(torch.abs(spec)), torch.min(torch.abs(spec)))
        #print(torch.max(torch.abs(spec_out)), torch.min(torch.abs(spec_out)))
        
        # conv LSTM
        spec_out=spec_out.permute(0,1,4,2,3) # B,2,T,C,F/2
        spec_out=spec_out[:,:,:-1]
        spec_out=torch.stack([
            self.lstm_layer_real(spec_out[:,0])[0], # B,2,T,C,F
            self.lstm_layer_imag(spec_out[:,1])[0]
        ], dim=1)
        #spec_out=self.lstm_layers(spec_out).permute(0,1,4,2,3)
        #spec_out=self.act(spec_out)
        
        
        #print(torch.max(torch.abs(spec_out)), torch.min(torch.abs(spec_out)))
        #print(torch.cuda.memory_allocated())
        
        # wav first half
        mix=toComplex(mix)
        wav=self.wav_first(mix) # B,2,C,L
        wav_out=wav
        wav_out=cLog(wav)
        wav_short=self.wav_short(wav)
        
        #print(torch.cuda.memory_allocated())
        #print(torch.max(torch.abs(wav_out)), torch.min(torch.abs(wav_out)))
        
        # fuse
        fused=self.fuser(spec_out, wav_out) # B,2,C,L
        
        #print(torch.cuda.memory_allocated())
        
        # wav second half
        for l in self.wav_layers:
            fused=l(fused)
        
        #print(torch.max(torch.abs(fused)), torch.min(torch.abs(fused)))
        #print(torch.cuda.memory_allocated())
        
        # apply to original
        fused=cExp(fused)
        
        
        final=cMul(fused, wav_short)
        final=torch.cat([final, wav_short], dim=2)
        final=self.last(final)
        
        #print(torch.cuda.memory_allocated())
        
        return toReal(final)
    
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))   
    
class BatchSRU(nn.Module):
    def __init__(self, batch, **kargs):
        super().__init__()
        self.srus=nn.ModuleList()
        for i in range(batch):
            self.srus.append(SRU(**kargs))
        
        
    def forward(self, tensor):
        #print("batchSRU", tensor.shape)
        # input: L, B, X
        return torch.stack([self.srus[i](tensor[..., i])[0] for i in range(len(self.srus))], dim=-1)
        
class DualSRUModel(nn.Module):
    def __init__(self, in_channels, spec_channels, block_size, first_kernel_size, 
                 lstm_num, lstm_channels, lstm_layer, 
                 reception, wav_channels, wav_layers, hidden_channels, wav_kernel_size):
        super().__init__()
        
        self.spec_model=SpectrogramNN(in_channels, spec_channels, block_size, first_kernel_size[0], first_kernel_size[1])
        
        F=self.spec_model.getF()
        #self.lstm_shuffle=ComplexMultiLinear(spec_channels, F, lstm_num)
        self.lstm_layer_real=nn.LSTM(
            input_size=F*spec_channels, hidden_size=lstm_num*lstm_channels[0],
            num_layers = lstm_layer,          # number of stacking RNN layers
            #dropout = 0.0,           # dropout applied between RNN layers
            #bidirectional = False,   # bidirectional RNN
            #layer_norm = False,      # apply layer normalization on the output of each layer
            #highway_bias = 0,        # initial bias of highway gate (<= 0)
            #rescale = True          # whether to use scaling correction
        )
        self.lstm_layer_imag=nn.LSTM( 
            input_size=F*spec_channels, hidden_size=lstm_num*lstm_channels[0],
            num_layers = lstm_layer,          # number of stacking RNN layers
            #dropout = 0.0,           # dropout applied between RNN layers
            #bidirectional = False,   # bidirectional RNN
            #layer_norm = False,      # apply layer normalization on the output of each layer
            #highway_bias = 0,        # initial bias of highway gate (<= 0)
            #rescale = True          # whether to use scaling correction
        )

        self.wav_first=nn.Sequential(
            CausalComplexConv1d(in_channels, wav_channels, reception, fullconv=True), #B,2,C,L
            ModReLU([wav_channels, 1]),
            CausalComplexConv1d(wav_channels, hidden_channels, wav_kernel_size, activation=ModReLU([hidden_channels, 1])), #B,2,C,L
            ModReLU([hidden_channels, 1]),
            ComplexConv1d(hidden_channels, wav_channels, 1)
            )
        
        self.wav_short=CausalTCN(wav_channels, wav_channels, hidden_channels, 1, wav_kernel_size, ModReLU([hidden_channels, 1]))
        
        self.fuser=Fuser(lstm_channels[-1], lstm_num, wav_channels, self.spec_model.getStep(), wav_channels)
        
        self.wav_layers=nn.ModuleList()
        
        dilation=1
        for i in range(wav_layers):
            self.wav_layers.append(CausalTCN(wav_channels, wav_channels, hidden_channels, dilation, wav_kernel_size, ModReLU([hidden_channels, 1])))
            dilation*=wav_kernel_size
            
        self.last=CausalComplexConvTrans1d(wav_channels*2, 1, reception, fullconv=True)
        
        self.block_size=block_size
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        
        self.init_weights()
       
    def init_weights(self):
        val_range = (3.0/64)**0.5
        params = list(self.lstm_layer_real.parameters()) + list(self.lstm_layer_imag.parameters())
        for p in params:
            if p.dim() > 1:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()
 
    def forward(self, mix):
        residual=mix.shape[2]%self.block_size
        if residual!=0:
            mix=mix[:,:,:-residual]
          
        # spectrogram
        spec=self.spec_model(mix) #B,2,C,F,T
        spec_out=spec
        spec_out=cLog(spec)
        
        
        # conv LSTM
        spec_out=spec_out.permute(0,1,4,2,3) # B,2,T,C,F/2
        spec_out=spec_out[:,:,:-1]
        T,B,C,F=spec_out.shape[2], spec_out.shape[0], spec_out.shape[-2], spec_out.shape[-1]
        spec_out=spec_out.permute(2,0,1,3,4)
        #print(spec_out.shape)
        
        spec_out=spec_out.view(T, B, 2, C*F) # T,B,2,C*F
        spec_out=torch.stack([
            self.lstm_layer_real(spec_out[:,:,0])[0],
            self.lstm_layer_imag(spec_out[:,:,1])[0]
        ], dim=2) # T,B,2,C*F
        #print(spec_out.shape)
        spec_out=spec_out.permute(1,2,0,3).unsqueeze(-1) # B,2,T,C,F
        
        #print(torch.max(torch.abs(spec_out)), torch.min(torch.abs(spec_out)))
        #print(torch.cuda.memory_allocated())
        
        
        
        # wav first half
        mix=toComplex(mix)
        wav=self.wav_first(mix) # B,2,C,L
        wav_out=wav
        wav_out=cLog(wav)
        wav_short=self.wav_short(wav)
        
        #print(torch.cuda.memory_allocated())
        #print(torch.max(torch.abs(wav_out)), torch.min(torch.abs(wav_out)))
        
        # fuse
        fused=self.fuser(spec_out, wav_out) # B,2,C,L
        
        #print(torch.cuda.memory_allocated())
        
        # wav second half
        for l in self.wav_layers:
            fused=l(fused)
        
        #print(torch.max(torch.abs(fused)), torch.min(torch.abs(fused)))
        #print(torch.cuda.memory_allocated())
        
        # apply to original
        fused=cExp(fused)
        
        
        final=cMul(fused, wav_short)
        final=torch.cat([final, wav_short], dim=2)
        final=self.last(final)
        
        #print(torch.cuda.memory_allocated())
        
        return toReal(final)
    
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))   
    

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=SRU(64, 128,
            num_layers = 2,          # number of stacking RNN layers
            dropout = 0.0,           # dropout applied between RNN layers
            bidirectional = False,   # bidirectional RNN
            layer_norm = False,      # apply layer normalization on the output of each layer
            highway_bias = 0,        # initial bias of highway gate (<= 0)
            rescale = True 
        )
        
        self.lstm2=SRU(64, 128,
            num_layers = 2,          # number of stacking RNN layers
            dropout = 0.0,           # dropout applied between RNN layers
            bidirectional = False,   # bidirectional RNN
            layer_norm = False,      # apply layer normalization on the output of each layer
            highway_bias = 0,        # initial bias of highway gate (<= 0)
            rescale = True 
        )
        self.last=nn.Conv1d(128,1,1)
        self.first=nn.Conv1d(9, 128,1)
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        
    def forward(self, tensor): # B, C, L
        print(tensor.shape)
        x=self.first(tensor)
        a=self.lstm2(x[:,:64].permute(2,0,1))[0].permute(1,2,0)
        b=self.lstm(x[:,64:].permute(2,0,1))[0].permute(1,2,0)
        return self.last(a+b)

    def loss(self, signal, gt, mix):
        
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))   
    
