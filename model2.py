import torch
import torch.nn as nn
import torch.functional as F
import importlib
import complexcnn.modules
importlib.reload(complexcnn.modules)
from complexcnn.modules import ComplexConv1d, ComplexConvTranspose1d, ComplexLinear, ComplexMultiLinear, ComplexSTFTWrapper, ComplexConv2d, cMul, cLog, cExp, toComplex, toReal, CausalComplexConv1d, CausalComplexConvTrans1d,CausalComplexConv2d

import torch.jit as jit

import convlstm
importlib.reload(convlstm)
from convlstm import ComplexConvLSTM
import asteroid




# transform time domain to freq domain, and a CNN to transform it
class SpectrogramNN(nn.Module):
    def __init__(self, in_channels, out_channels, block_size, freq_kernel_size, time_kernel_size):
        super().__init__()
        
        self.in_channels=in_channels
        self.block_size=block_size
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size)
        
        self.F=block_size//2-freq_kernel_size+1
        self.time_kernel_size=time_kernel_size
        self.first=ComplexConv2d(in_channels, out_channels, (freq_kernel_size, time_kernel_size), padding=(0, time_kernel_size-1))
        
    def forward(self, mix):
        l0=self.stft.transform(mix) # B, 2, C, F, T
        l1=l0[:,:,:,:self.block_size//2,:]
        
        l2=self.first(l1)[:,:,:,:,:-self.time_kernel_size+1]
        
        return l2 # B,2,C,F,T
    
    def getStep(self):
        return self.block_size//2
    
    def getF(self):
        return self.F

class Fuser(nn.Module):
    def __init__(self, spec_channels, spec_F, wav_channels, spec_step, out_channels):
        super().__init__()
        self.spec_step=spec_step
        self.fuser=ComplexConv1d(spec_F*spec_channels+wav_channels, out_channels, 1)
        
    def forward(self, spec_in, wav_in):
        # spec_in: B,2,T,C,F (log)
        # wav_in: B,2,C,L (log)
        spec_in=spec_in[:,:,:-1,:,:]
        spec_in=spec_in.reshape(spec_in.shape[0], spec_in.shape[1], spec_in.shape[2], -1)
        
        spec_in=torch.repeat_interleave(spec_in, repeats=self.spec_step, dim=2)
        
        spec_in=spec_in.permute(0,1,3,2) # B,2,C,L
        
        total=torch.cat([spec_in, wav_in], dim=2)
        return self.fuser(total) # log
    
class CausalTCN(jit.ScriptModule):
    def __init__(self, in_channels, out_channels, hidden_channels, dilation, kernel, activation=nn.LeakyReLU(0.1)):
        super().__init__()
        
        self.layers=nn.ModuleList()
        self.layers.append(CausalComplexConv1d(in_channels, hidden_channels, kernel, dilation=dilation, activation=activation, fullconv=True))
        self.layers.append(activation)
        self.layers.append(ComplexConv1d(hidden_channels, out_channels, 1))
        
    @jit.script_method
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
        
        self.act=nn.LeakyReLU(0.01)
        
        self.spec_model=SpectrogramNN(in_channels, spec_channels, block_size, first_kernel_size[0], first_kernel_size[1])
        #self.lstm_layers=ComplexConvLSTM(spec_channels, lstm_channels, [1]+(lstm_layer-1)*[lstm_kernel_size], lstm_layer, True)
        self.lstm_layers=CausalComplexConv2d(spec_channels, lstm_channels[-2], lstm_channels[-1], first_kernel_size[0], first_kernel_size[1])
        
        self.wav_first=nn.Sequential(
            CausalComplexConv1d(in_channels, wav_channels, reception, fullconv=True),
            self.act,
            ComplexConv1d(wav_channels, wav_channels, 1),
            self.act)
        
        self.fuser=Fuser(lstm_channels[-1], self.spec_model.getF()//2, wav_channels, self.spec_model.getStep(), wav_channels)
        #self.pooling=nn.AvgPool1d(2, stride=2)
        
        self.wav_layers=nn.ModuleList()
        
        for i in range(wav_layers):
            self.wav_layers.append(CausalTCN(wav_channels, wav_channels, hidden_channels, 1, wav_kernel_size, self.act))
        
        self.last=CausalComplexConvTrans1d(wav_channels, 1, reception, fullconv=True)
        
        self.block_size=block_size
        
        self.loss_fn=asteroid.losses.pairwise_neg_sisdr 
        
        
    def forward(self, mix):
        residual=mix.shape[2]%self.block_size
        if residual!=0:
            mix=mix[:,:,:-residual]
          
        # spectrogram
        spec=self.spec_model(mix) #B,2,C,F,T
        spec=spec[:,:,:,0::2,:]+spec[:,:,:,1::2,:] # reduce F
        spec_out=spec
        #spec_out=cLog(spec)
        
        #print(torch.cuda.memory_allocated())
        
        # conv LSTM
        #spec_out=spec_out.permute(0,1,4,2,3) # B,2,F/2,T,C
        #spec_out=self.lstm_layers(spec_out)[0] # B,2,T,C,F
        spec_out=self.lstm_layers(spec_out).permute(0,1,4,2,3)
        spec_out=self.act(spec_out)
        
        #print(torch.cuda.memory_allocated())
        
        # wav first half
        mix=toComplex(mix)
        wav=self.wav_first(mix) # B,2,C,L
        wav_out=wav
        #wav_out=cLog(wav)
        
        #print(torch.cuda.memory_allocated())
        
        # fuse
        fused=self.fuser(spec_out, wav_out) # B,2,C,L
        
        #print(torch.cuda.memory_allocated())
        
        # wav second half
        for l in self.wav_layers:
            fused=l(fused)
        
        #print(torch.cuda.memory_allocated())
        
        # apply to original
        fused=cExp(fused)
        final=cMul(fused, wav)
        final=self.last(final)
        
        #print(torch.cuda.memory_allocated())
        
        return toReal(final)
    
    def loss(self, signal, gt, mix):
        return torch.sum(self.loss_fn(signal[..., 24000:], gt[..., 24000:]))   
    
    
    
# simple logexp model
class LogExpModel(nn.Module):
    def __init__(self, in_channels, channel, block_size, ff_size):
        super().__init__()
        
        self.in_channels=in_channels
        self.ff_size=ff_size
        self.block_size=block_size
        
        self.act=nn.LeakyReLU(0.01)
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
        
    def forward(self, mix:torch.Tensor):
        B=mix.shape[0]
        
        l0=self.stft.transform(mix) # B, 2, C, F, T
        l1=l0[:,:,:,:self.block_size//2,:]
        ori=l1
        
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

    