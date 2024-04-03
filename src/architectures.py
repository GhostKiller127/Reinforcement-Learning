import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class DenseModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(config['input_dim'], config['hidden_dim']).to(device)
        self.linear2 = nn.Linear(config['hidden_dim'], config['hidden_dim']).to(device)
        self.value_head1 = nn.Linear(config['hidden_dim'], int(config['hidden_dim']/2)).to(device)
        self.value_head2 = nn.Linear(int(config['hidden_dim']/2), int(config['hidden_dim']/2)).to(device)
        self.value_head3 = nn.Linear(int(config['hidden_dim']/2), 1).to(device)
        self.advantage_head1 = nn.Linear(config['hidden_dim'], int(config['hidden_dim']/2)).to(device)
        self.advantage_head2 = nn.Linear(int(config['hidden_dim']/2), int(config['hidden_dim']/2)).to(device)
        self.advantage_head3 = nn.Linear(int(config['hidden_dim']/2), config['action_dim']).to(device)


    def forward(self, s):
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        s = F.relu(self.linear1(s))
        s = F.relu(self.linear2(s))
        v = F.relu(self.value_head1(s))
        v = F.relu(self.value_head2(v))
        v = self.value_head3(v)
        a = F.relu(self.advantage_head1(s))
        a = F.relu(self.advantage_head2(a))
        a = self.advantage_head3(a)
        return v, a

