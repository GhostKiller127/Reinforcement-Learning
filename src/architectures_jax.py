import jax.numpy as jnp
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



class TransformerModelJax(nn.Module):
    config: dict
    
    def setup(self):
        self.input_linear = nn.Dense(self.config['hidden_dim'])
        self.encoder_layers = [TransformerEncoderLayer(self.config) for _ in range(self.config['num_layers'])]
        self.value_head1 = nn.Dense(int(self.config['hidden_dim']/2))
        self.value_head2 = nn.Dense(int(self.config['hidden_dim']/2))
        self.value_head3 = nn.Dense(1)
        self.advantage_head1 = nn.Dense(int(self.config['hidden_dim']/2))
        self.advantage_head2 = nn.Dense(int(self.config['hidden_dim']/2))
        self.advantage_head3 = nn.Dense(self.config['action_dim'])
        self.dropout = nn.Dropout(rate=self.config['dropout'])
    

    def positional_encoding(self, length, depth):
        depth = depth / 2
        positions = jnp.arange(length)[:, jnp.newaxis]
        depths = jnp.arange(depth)[jnp.newaxis, :] / depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        pos_encoding = jnp.concatenate([jnp.sin(angle_rads), jnp.cos(angle_rads)], axis=-1)
        return pos_encoding


    def __call__(self, x, training=False):
        x = self.input_linear(x)
        if x.ndim == 2:
            x = x[None, :, :]
        
        seq_len = x.shape[1]
        pos_encoding = self.positional_encoding(seq_len, self.config['hidden_dim'])
        x = x + pos_encoding[jnp.newaxis, :, :]

        x = self.dropout(x, deterministic=not training)
        for layer in self.encoder_layers:
            x = layer(x, deterministic=not training)
        v = nn.relu(self.value_head1(x))
        v = nn.relu(self.value_head2(v))
        v = self.value_head3(v)
        a = nn.relu(self.advantage_head1(x))
        a = nn.relu(self.advantage_head2(a))
        a = self.advantage_head3(a)
        return v, a



class TransformerEncoderLayer(nn.Module):
    config: dict

    def setup(self):
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.config['num_heads'],
            dropout_rate=self.config['dropout']
        )
        self.dense1 = nn.Dense(self.config['hidden_dim'])
        self.dense2 = nn.Dense(self.config['hidden_dim'])
        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.config['dropout'])


    def __call__(self, x, deterministic=None):
        norm_x = self.layer_norm1(x)
        attention_output = self.attention(norm_x, norm_x, deterministic=deterministic)
        x = x + self.dropout(attention_output, deterministic=deterministic)
        norm_x = self.layer_norm2(x)
        dense_output = self.dense2(nn.relu(self.dense1(norm_x)))
        x = x + self.dropout(dense_output, deterministic=deterministic)
        return x