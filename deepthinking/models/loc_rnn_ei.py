import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error

class LocRNNEIcell(nn.Module):
    """
    Implements recurrent inhibitory excitatory normalization w/ lateral connections
    params:
      input_dim: Number of channels in input
      hidden_dim: Number of hidden channels
      kernel_size: Size of kernel in convolutions
    """

    def __init__(self,
                 in_channels,
                 hidden_dim=None,
                 exc_fsize=7,
                 ):
        super(LocRNNEIcell, self).__init__()
        self.in_channels = in_channels
        if hidden_dim is None:
            self.hidden_dim = in_channels
        else:
            self.hidden_dim = hidden_dim
        # recurrent gates computation
        self.conv_recall_e = nn.Conv2d(self.hidden_dim + 3, self.hidden_dim, 3, 
                                    bias=False, padding=1, stride=1)
        
        self.g_exc = nn.Conv2d(self.in_channels + self.hidden_dim, self.hidden_dim, 1, bias=False)
        self.ln_out = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
        
        # horizontal connections (e->e, i->e, i->i, e->i)
        self.w_ee = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, exc_fsize, padding=(exc_fsize-1) // 2,
            bias=False)
        # disynaptic inhibition with pairs of E-I cells, E -> exciting surround I -> inhibiting surround E
        self.w_ie = nn.Conv2d(self.hidden_dim, self.hidden_dim, exc_fsize, padding=(exc_fsize-1) // 2,
                              bias=False)
        self.e_nl = nn.ReLU()
        
    def forward(self, input, exc, image=None): 
        # TODO(vveeraba): Think through architecture, this is a bit messy
        exc = self.conv_recall_e(torch.cat((exc, image), 1))
        g_e = torch.sigmoid(self.g_exc(torch.cat((exc, input), 1)))
        hor_e = torch.relu(self.w_ee(exc))
        hor_i = torch.relu(self.w_ie(exc))
        e_hat_t = exc + hor_e - hor_i
        exc = self.e_nl(self.ln_out(g_e * e_hat_t + (1 - g_e) * exc))
        return exc


class LocRNNEILayer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim=None,
                 exc_fsize=5,
                 timesteps=15,
                 device='cuda',
                 ):
        super(LocRNNEILayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.exc_fsize = exc_fsize
        self.timesteps = timesteps
        self.device = device
        self.rnn_cell = LocRNNEIcell(in_channels=self.in_channels,
                                    hidden_dim=self.hidden_dim,
                                    exc_fsize=self.exc_fsize,
                                    )

    def forward(self, input, iters_to_do, interim_thought=None, stepwise_predictions=False, image=None):
        outputs_e = []
        if interim_thought:
            state = (interim_thought[0], interim_thought[1])
        else:
            state = input
        for _ in range(iters_to_do):
            state = self.rnn_cell(input, state, image=image)
            outputs_e.append(state)
        # use this return in normal training
        if stepwise_predictions:
            return outputs_e
        return outputs_e[-1]