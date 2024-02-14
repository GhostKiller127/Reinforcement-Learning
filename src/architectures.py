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



class TransformerModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.pos_enc = torch.unsqueeze(torch.tensor(np.linspace(-1, 1, num=config['max_sequence_length']), dtype=torch.float32, device=device), 1)
        self.src_mask = (torch.triu(torch.ones((config['max_sequence_length'], config['max_sequence_length']), device=device)) == 1).transpose(0, 1).float()
        self.src_mask = self.src_mask.masked_fill(self.src_mask == 0, float('-inf'))
        self.src_mask = self.src_mask.masked_fill(self.src_mask == 1, float(0.0))
        self.input_dim = config['input_dim'] + 1
        self.n_layers = config['n_layers']
        
        self.transformer_layers = nn.ModuleDict()
        self.transformer_layers['e'] = nn.Linear(self.input_dim, config['hidden_dim']).to(device)
        for i in range(self.n_layers):
            self.transformer_layers[f'l{i}'] = nn.TransformerEncoderLayer(d_model=config['hidden_dim'], nhead=config['n_heads'], dim_feedforward=config['feed_forward_dim'], batch_first=True, norm_first=True).to(device)

        self.value_head1 = nn.Linear(config['hidden_dim'], int(config['hidden_dim']/2)).to(device)
        self.value_head2 = nn.Linear(int(config['hidden_dim']/2), int(config['hidden_dim']/2)).to(device)
        self.value_head3 = nn.Linear(int(config['hidden_dim']/2), 1).to(device)
        self.advantage_head1 = nn.Linear(config['hidden_dim'], int(config['hidden_dim']/2)).to(device)
        self.advantage_head2 = nn.Linear(int(config['hidden_dim']/2), int(config['hidden_dim']/2)).to(device)
        self.advantage_head3 = nn.Linear(int(config['hidden_dim']/2), config['action_dim']).to(device)


    def forward(self, s):
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        pos_enc = self.pos_enc[:s.size()[s.dim()-2]]
        if s.dim() == 3:
            pos_enc = pos_enc.repeat(s.size()[0], 1, 1)
        s = torch.cat((s, pos_enc), s.dim()-1)
        s = self.transformer_layers['e'](s)
        for i in range(self.n_layers):
            s = self.transformer_layers[f'l{i}'](s, self.src_mask[:s.size()[s.dim()-2],:s.size()[s.dim()-2]])

        v = F.relu(self.value_head1(s))
        v = F.relu(self.value_head2(v))
        v = self.value_head3(v)
        a = F.relu(self.advantage_head1(s))
        a = F.relu(self.advantage_head2(a))
        a = self.advantage_head3(a)
        return v, a
