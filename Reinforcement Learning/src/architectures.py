import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseModel(nn.Module):
    def __init__(self, config, device, requires_grad=True):
        super().__init__()
        config = config['architecture_params']
        self.device = device
        self.linear_in = nn.Linear(config['input_dim'], config['hidden_dim']).to(device)
        self.value_head1 = nn.Linear(config['hidden_dim'], int(config['hidden_dim']/2)).to(device)
        self.value_head2 = nn.Linear(int(config['hidden_dim']/2), 1).to(device)
        self.advantage_head1 = nn.Linear(config['hidden_dim'], int(config['hidden_dim']/2)).to(device)
        self.advantage_head2 = nn.Linear(int(config['hidden_dim']/2), config['action_dim']).to(device)

    def forward(self, s):
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        s = F.relu(self.linear_in(s))
        v = F.relu(self.value_head1(s))
        a = F.relu(self.advantage_head1(s))
        v = self.value_head2(v)
        a = self.advantage_head2(a)
        return v, a


class TransformerModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.pos_enc = torch.unsqueeze(torch.tensor(np.linspace(-1, 1, num=config['MAX_SEQUENCE_LENGTH']), dtype=torch.float32, device=device), 1)
        self.src_mask = (torch.triu(torch.ones((config['MAX_SEQUENCE_LENGTH'], config['MAX_SEQUENCE_LENGTH']), device=device)) == 1).transpose(0, 1).float()
        self.src_mask = self.src_mask.masked_fill(self.src_mask == 0, float('-inf'))
        self.src_mask = self.src_mask.masked_fill(self.src_mask == 1, float(0.0))
        # self.input_length = data_processed_list[0].shape[1] + 1
        self.input_length = 167
        self.embed_dim = config['D_MODEL']
        self.LAYERS = config['LAYERS']
        
        self.layers = nn.ModuleDict()
        self.layers['e'] = nn.Linear(self.input_length, self.embed_dim).to(device)
        for i in range(self.LAYERS):
            self.layers[f'l{i}'] = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config['HEADS'], dim_feedforward=config['FC'], dropout=config['DROPOUT'], activation=config['ACTIVATION'], batch_first=True, norm_first=True).to(device)
        self.layers['p'] = nn.Linear(self.embed_dim, config['LABEL_LENGTH']).to(device)


    def forward(self, s):
        pos_enc = self.pos_enc[:s.size()[s.dim()-2]]
        if s.dim() == 3:
            pos_enc = pos_enc.repeat(s.size()[0], 1, 1)
        out = torch.cat((s, pos_enc), s.dim()-1)
        out = self.layers['e'](out)
        for i in range(self.LAYERS):
            out = self.layers[f'l{i}'](out, self.src_mask[:s.size()[s.dim()-2],:s.size()[s.dim()-2]])
        out = self.layers['p'](out)
        return out
