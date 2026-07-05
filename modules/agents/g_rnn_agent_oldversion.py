import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from torch.cuda.amp import autocast

class OGRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GRNNAgent, self).__init__()
        self.args = args
        
        self.n_agent = args.env_args['n_agents']
        
        # raise RuntimeError("lie biao ")

        self.fc1 = nn.ModuleList([nn.Linear(input_shape, args.rnn_hidden_dim) for _ in range(self.n_agent)]) 
        self.rnn = nn.ModuleList([nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim) for _ in range(self.n_agent)]) 
        self.fc2 = nn.ModuleList([nn.Linear(args.rnn_hidden_dim, args.n_actions) for _ in range(self.n_agent)]) 

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = nn.ModuleList([LayerNorm(args.rnn_hidden_dim) for _ in range(self.n_agent)])  
        
        if getattr(args, "use_orthogonal", False):
            for idx in range(self.n_agent):
                orthogonal_init_(self.fc1[idx])
                orthogonal_init_(self.fc2[idx], gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # print(inputs.shape)
        b, a, e = inputs.size()
        
        outputs = []
        hhs = []

        for idx in range(self.n_agent):
            input_idx = inputs[:,idx,:].view(-1, e)
            x = F.relu(self.fc1[idx](input_idx), inplace=True)
            h_in = hidden_state[:, idx, :].reshape(-1, self.args.rnn_hidden_dim)
            hh = self.rnn[idx](x, h_in)

            if getattr(self.args, "use_layer_norm", False):
                q = self.fc2[idx](self.layer_norm[idx](hh))
            else:
                q = self.fc2[idx](hh)
            outputs.append(q)
            hhs.append(hh)
        
        q = th.stack(outputs, dim=1)
        hh = th.stack(hhs, dim=1)
        

        return q.view(b, a, -1), hh.view(b, a, -1)