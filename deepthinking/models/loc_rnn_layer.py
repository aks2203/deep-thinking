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
                 ):
        super(LocRNNcell, self).__init__()
        self.in_channels = in_channels
        if hidden_dim is None:
            self.hidden_dim = in_channels
        else:
            self.hidden_dim = hidden_dim
        # recurrent gates computation
        self.recall = False
        if recall:
            self.conv_recall_e = nn.Conv2d(self.hidden_dim + 3, self.hidden_dim, 3, 
                                        bias=False, padding=1, stride=1)
            self.conv_recall_i = nn.Conv2d(self.hidden_dim + 3, self.hidden_dim, 3, 
                                        bias=False, padding=1, stride=1)
            self.recall=True
        
            self.g_exc = nn.Conv2d(self.in_channels+self.hidden_dim, self.hidden_dim, 1, bias=False)
            self.g_inh = nn.Conv2d(self.in_channels+self.hidden_dim, self.hidden_dim, 1, bias=False)
            self.ln_out_e = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
            self.ln_out_i = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)

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

        
    def forward(self, input, hidden, image=None):
        exc, inh = hidden
        if self.recall:
            exc = self.conv_recall_e(torch.cat((exc, image), 1))
            inh = self.conv_recall_i(torch.cat((inh, image), 1))
            g_e = torch.sigmoid(self.g_exc(torch.cat([input, exc], 1)))
            g_i = torch.sigmoid(self.g_inh(torch.cat([input, inh], 1)))
            e_hat_t = self.e_nl(
                self.w_exc_x(input) +
                self.w_exc_ei(torch.cat((exc, inh), 1)))
            
            i_hat_t = self.i_nl(
                self.w_inh_x(input) +
                self.w_inh_ei(torch.cat((exc, inh), 1)))

            exc = self.e_nl(self.ln_out_e(g_e * e_hat_t + (1 - g_e) * exc))
            inh = self.i_nl(self.ln_out_i(g_i * i_hat_t + (1 - g_i) * inh))
        else:
            g_exc = torch.sigmoid(self.ln_e_x(self.g_exc_x(
                input)) + self.ln_e_e(self.g_exc_e(exc)))
            g_inh = torch.sigmoid(self.ln_i_x(self.g_inh_x(
                input)) + self.ln_i_i(self.g_inh_i(inh)))

            e_hat_t = torch.relu(
                self.w_exc_x(input) +
                self.w_exc_ei(torch.cat((exc, inh), 1)))

            i_hat_t = torch.relu(
                self.w_inh_x(input) +
                self.w_inh_ei(torch.cat((exc, inh), 1)))

            exc = torch.relu(self.ln_out_e(g_exc * e_hat_t + (1 - g_exc) * exc))
            inh = torch.relu(self.ln_out_i(g_inh * i_hat_t + (1 - g_inh) * inh))
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
                 ):
        super(LocRNNLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.exc_fsize = exc_fsize
        self.inh_fsize = inh_fsize
        self.timesteps = timesteps
        self.device = device
        self.recall = recall
        self.rnn_cell = LocRNNcell(in_channels=self.in_channels,
                                    hidden_dim=self.hidden_dim,
                                    exc_fsize=self.exc_fsize,
                                    inh_fsize=self.inh_fsize,
                                    device=self.device,
                                    recall=recall)

    def forward(self, input, iters_to_do, interim_thought=None, stepwise_predictions=False, image=None):
        outputs_e = []
        outputs_i = []
        if interim_thought:
            state = (interim_thought[0], interim_thought[1])
        else:
            state = (torch.zeros_like(input), torch.zeros_like(input))
        for rnn_t in range(iters_to_do):
            if self.recall:
                state = self.rnn_cell(input, state, image=image)
            else:
                state = self.rnn_cell(input, state)
            outputs_e += [state[0]]
            outputs_i += [state[1]]
        # use this return in normal training
        if stepwise_predictions:
            return (outputs_e, outputs_i)
        return (outputs_e[-1], outputs_i[-1])