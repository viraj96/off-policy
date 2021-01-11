import torch
import torch.nn as nn
from offpolicy.utils.util import to_torch
from offpolicy.algorithms.utils.mlp import MLPBase
from offpolicy.algorithms.utils.rnn import RNNBase
from offpolicy.algorithms.utils.act import ACTLayer

class AgentQFunction(nn.Module):
    # GRU implementation of the Agent Q function

    def __init__(self, args, input_dim, act_dim, device):
        # input dim is agent obs dim + agent acf dim
        # output dim is act dim
        super(AgentQFunction, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self._use_rnn_layer = args.use_rnn_layer
        self._gain = args.gain
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self._use_rnn_layer:
            self.rnn = RNNBase(args, input_dim)
        else:
            self.mlp = MLPBase(args, input_dim)

        self.q = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, gain=self._gain)

        self.to(device)

    def forward(self, obs, rnn_states):
        # make sure input is a torch tensor
        obs = to_torch(obs).to(**self.tpdv)
        rnn_states = to_torch(rnn_states).to(**self.tpdv)

        no_sequence = False
        if len(obs.shape) == 2:
            # this means we're just getting one output (no sequence)
            no_sequence = True
            obs = obs[None]
            # obs is now of shape (seq_len, batch_size, obs_dim)
        if len(rnn_states.shape) == 2:
            # hiddens should be of shape (1, batch_size, dim)
            rnn_states = rnn_states[None]

        inp = obs

        if self._use_rnn_layer: 
            rnn_outs, h_final = self.rnn(inp, rnn_states) 
        else:
            rnn_outs = self.mlp(inp)
            h_final = rnn_states[0, :, :]

        # pass outputs through linear layer
        q_outs = self.q(rnn_outs, no_sequence)

        return q_outs, h_final
