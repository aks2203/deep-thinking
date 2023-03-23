import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error


class LocRNNcell(nn.Module):
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
                 inh_fsize=5,
                 device='cuda',
                 recall=True,
                 x_to_h=False,
                 ):
        super(LocRNNcell, self).__init__()
        self.in_channels = in_channels
        if hidden_dim is None:
            self.hidden_dim = in_channels
        else:
            self.hidden_dim = hidden_dim
        # recurrent gates computation
        self.recall = False
        self.x_to_h = x_to_h
        if recall:
            self.conv_recall = nn.Conv2d(self.hidden_dim + 3, self.hidden_dim, 3, 
                                        bias=False, padding=1, stride=1)
            self.recall=True
        
            self.g_exc = nn.Conv2d(self.in_channels+self.hidden_dim, self.hidden_dim, 1, bias=False)
            self.g_inh = nn.Conv2d(self.in_channels+self.hidden_dim, self.hidden_dim, 1, bias=False)
            self.ln_out_e = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
            self.ln_out_i = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)

            if self.x_to_h:
                # feedforward stimulus drive
                self.w_exc_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
                self.w_inh_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
            
            # horizontal connections (e->e, i->e, i->i, e->i)
            self.w_exc_ei = nn.Conv2d(
                self.hidden_dim * 2, self.hidden_dim, exc_fsize, padding=(exc_fsize-1) // 2)
            # disynaptic inhibition with pairs of E-I cells, E -> exciting surround I -> inhibiting surround E
            self.w_inh_ei = nn.Conv2d(
                self.hidden_dim * 2, self.hidden_dim, inh_fsize, padding=(inh_fsize-1) // 2)
            self.e_nl = nn.ReLU()
            self.i_nl = nn.ReLU()
        else:
            # recurrent gates computation
            self.g_exc_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
            self.ln_e_x = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
            self.g_exc_e = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
            self.ln_e_e = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
            self.g_inh_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
            self.ln_i_x = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
            self.g_inh_i = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
            self.ln_i_i = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
            self.ln_out_e = nn.GroupNorm(
                num_groups=1, num_channels=self.hidden_dim)
            self.ln_out_i = nn.GroupNorm(
                num_groups=1, num_channels=self.hidden_dim)

            self.ln_out = nn.GroupNorm(
                num_groups=1, num_channels=self.hidden_dim)
            if self.x_to_h:
                # feedforward stimulus drive
                self.w_exc_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
                self.w_inh_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)

            # horizontal connections (e->e, i->e, i->i, e->i)
            self.w_exc_ei = nn.Conv2d(
                self.hidden_dim * 2, self.hidden_dim, exc_fsize, padding=(exc_fsize-1) // 2)
            # disynaptic inhibition with pairs of E-I cells, E -> exciting surround I -> inhibiting surround E
            # self.w_exc_i = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
            self.w_inh_ei = nn.Conv2d(
                self.hidden_dim * 2, self.hidden_dim, inh_fsize, padding=(inh_fsize-1) // 2)
            # self.w_inh_e = nn.Conv2d(self.hidden_dim, self.hidden_dim, inh_fsize, padding=(inh_fsize-1) // 2)
            # nonnegative_weights_init(self.div)

        
    def forward(self, input, hidden):
        exc, inh = hidden
        if self.recall:
            input = self.conv_recall(input)
        g_e = torch.sigmoid(self.g_exc(torch.cat([input, exc], 1)))
        g_i = torch.sigmoid(self.g_inh(torch.cat([input, inh], 1)))
        if self.x_to_h:
            e_hat_t = self.e_nl(
                self.w_exc_x(input) +
                self.w_exc_ei(torch.cat((exc, inh), 1)))
            
            i_hat_t = self.i_nl(
                self.w_inh_x(input) +
                self.w_inh_ei(torch.cat((exc, inh), 1)))
        else:
            e_hat_t = self.e_nl(self.w_exc_ei(torch.cat((exc, inh), 1)))
            i_hat_t = self.i_nl(self.w_inh_ei(torch.cat((exc, inh), 1)))
        exc = self.e_nl(self.ln_out_e(g_e * e_hat_t + (1 - g_e) * exc))
        inh = self.i_nl(self.ln_out_i(g_i * i_hat_t + (1 - g_i) * inh))
        return (exc, inh)


class LocRNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim=None,
                 exc_fsize=5,
                 inh_fsize=3,
                 timesteps=15,
                 device='cuda',
                 recall=True,
                 x_to_h=False,
                 ):
        super(LocRNNLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.exc_fsize = exc_fsize
        self.inh_fsize = inh_fsize
        self.timesteps = timesteps
        self.device = device
        self.recall = recall
        self.x_to_h = x_to_h
        self.rnn_cell = LocRNNcell(in_channels=self.in_channels,
                                    hidden_dim=self.hidden_dim,
                                    exc_fsize=self.exc_fsize,
                                    inh_fsize=self.inh_fsize,
                                    device=self.device,
                                    recall=recall,
                                    x_to_h=self.x_to_h)

    def forward(self, input, iters_to_do, interim_thought=None, stepwise_predictions=False, image=None):
        outputs_e = []
        outputs_i = []
        n, _, h, w = input.shape
        if interim_thought:
            state = (interim_thought[0], interim_thought[1])
        elif self.x_to_h:
            state = (torch.zeros(n, self.hidden_dim, h, w).to(self.device),
                     torch.zeros(n, self.hidden_dim, h, w).to(self.device))
        else:
            state = (input, input)
        for _ in range(iters_to_do):
            state = self.rnn_cell(input, state)
            outputs_e += [state[0]]
            outputs_i += [state[1]]
        # use this return in normal training
        if stepwise_predictions:
            return (outputs_e, outputs_i)
        return (outputs_e[-1], outputs_i[-1])