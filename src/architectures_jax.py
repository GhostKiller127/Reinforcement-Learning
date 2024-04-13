from flax import linen as nn


class DenseModelJax(nn.Module):
    config: dict
    
    def setup(self):
        self.dense1 = nn.Dense(self.config['hidden_dim'])
        self.dense2 = nn.Dense(self.config['hidden_dim'])
        self.value_head1 = nn.Dense(int(self.config['hidden_dim']/2))
        self.value_head2 = nn.Dense(int(self.config['hidden_dim']/2))
        self.value_head3 = nn.Dense(1)
        self.advantage_head1 = nn.Dense(int(self.config['hidden_dim']/2))
        self.advantage_head2 = nn.Dense(int(self.config['hidden_dim']/2))
        self.advantage_head3 = nn.Dense(self.config['action_dim'])
        self.dropout = nn.Dropout(rate=self.config['dropout'])
        self.layernorm = nn.LayerNorm()
    

    def __call__(self, x, training=False):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        # x = self.dropout(x, deterministic=not training)
        # x = self.layernorm(x)
        v = nn.relu(self.value_head1(x))
        v = nn.relu(self.value_head2(v))
        v = self.value_head3(v)
        a = nn.relu(self.advantage_head1(x))
        a = nn.relu(self.advantage_head2(a))
        a = self.advantage_head3(a)
        return v, a
