import torch.nn as nn
import torch
from complexcnn.modules import ComplexConv1d, cMul
import torch.jit as jit
from typing import List, Tuple

# changed to use complex conv1d
class ComplexConvLSTMCell(jit.ScriptModule):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ComplexConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = ComplexConv1d(in_channel=self.input_dim + self.hidden_dim,
                              out_channel=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
    @jit.script_method
    def forward(self, input_tensor:torch.Tensor, cur_state:Tuple[torch.Tensor, torch.Tensor])->Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=2)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=2)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = cMul(f, c_cur) + cMul(i, g)
        h_next = cMul(o, torch.tanh(c_next))

        return h_next, c_next

class ComplexConvLSTM(jit.ScriptModule):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ComplexConvLSTM, self).__init__()

        #self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ComplexConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)
    
    @jit.script_method
    def forward(self, input_tensor:torch.Tensor)->List[torch.Tensor]:
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, 2, b, c, l) -> (b, 2, t, c, l)
            input_tensor = input_tensor.permute(2, 1, 0, 3, 4)

        b, _, _, _, l = input_tensor.size()

        # Since the init is done in forward. Can send image size here

        hidden_state=jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
        for i in range(self.num_layers):
            hid=(torch.zeros(b, 2, self.hidden_dim[i], l, device=input_tensor.device),
                torch.zeros(b, 2, self.hidden_dim[i], l, device=input_tensor.device))
            hidden_state.append(hid)


        layer_output_list = jit.annotate(List[torch.Tensor], [])
        last_state_list = jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])

        seq_len = input_tensor.size(2)
        cur_layer_input = input_tensor

        for layer_idx, cell in enumerate(self.cell_list):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in torch.unbind(cur_layer_input, dim=2):
                h, c = cell(input_tensor=t, cur_state=(h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=2)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param