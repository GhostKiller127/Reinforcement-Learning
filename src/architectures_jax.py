import jax.numpy as jnp
from flax import linen as nn

#region Dense

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
        v = v[:, -1]
        a = a[:, -1]
        return v, a

#endregion
#region Transformer

class TransformerModelJax(nn.Module):
    config: dict
    
    def setup(self):
        self.input_linear = nn.Dense(self.config['d_model'])
        self.encoder_layers = [TransformerEncoderLayer(self.config) for _ in range(self.config['n_layers'])]
        self.value_head1 = nn.Dense(int(self.config['decoder_dim']), kernel_init=nn.initializers.he_normal())
        self.value_head2 = nn.Dense(int(self.config['decoder_dim']), kernel_init=nn.initializers.he_normal())
        self.value_head3 = nn.Dense(1, kernel_init=nn.initializers.he_normal())
        self.advantage_head1 = nn.Dense(int(self.config['decoder_dim']), kernel_init=nn.initializers.he_normal())
        self.advantage_head2 = nn.Dense(int(self.config['decoder_dim']), kernel_init=nn.initializers.he_normal())
        self.advantage_head3 = nn.Dense(self.config['action_dim'], kernel_init=nn.initializers.he_normal())
        self.dropout = nn.Dropout(rate=self.config['dropout'])
    

    def positional_encoding(self, length, depth):
        depth = depth / 2
        positions = jnp.arange(length)[:, jnp.newaxis]
        depths = jnp.arange(depth)[jnp.newaxis, :] / depth
        angle_rates = 1 / (length**depths)
        angle_rads = positions * angle_rates
        pos_encoding = jnp.concatenate([jnp.sin(angle_rads), jnp.cos(angle_rads)], axis=-1)
        return pos_encoding


    def __call__(self, x, training=False):
        if x.ndim == 2:
            x = x[None, :, :]
        seq_len = x.shape[1]
        
        x = self.input_linear(x)
        x = x + self.positional_encoding(seq_len, self.config['d_model'])[jnp.newaxis, :, :]
        x = self.dropout(x, deterministic=not training)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        v = nn.gelu(self.value_head1(x))
        v = nn.gelu(self.value_head2(v))
        v = self.value_head3(v)
        a = nn.gelu(self.advantage_head1(x))
        a = nn.gelu(self.advantage_head2(a))
        a = self.advantage_head3(a)
        return v, a

#endregion
#region Transformer Layer


class TransformerEncoderLayer(nn.Module):
    config: dict

    def setup(self):
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.config['heads'],
            dropout_rate=self.config['dropout'],
            # kernel_init=nn.initializers.glorot_normal()
        )
        self.dense1 = nn.Dense(self.config['d_model'], kernel_init=nn.initializers.he_normal())
        self.dense2 = nn.Dense(self.config['d_model'], kernel_init=nn.initializers.he_normal())

        if self.config['batchnorm']:
            self.norm1 = nn.BatchNorm(momentum=self.config['bn_momentum'])
            self.norm2 = nn.BatchNorm(momentum=self.config['bn_momentum'])
        else:
            self.norm1 = nn.LayerNorm()
            self.norm2 = nn.LayerNorm()

        self.dropout = nn.Dropout(rate=self.config['dropout'])


    def __call__(self, x, training=False):
        if self.config['batchnorm']:
            norm_x = self.norm1(x, use_running_average=not training)
        else:
            norm_x = self.norm1(x)

        attention_output = self.attention(norm_x, norm_x, deterministic=not training)
        x = x + self.dropout(attention_output, deterministic=not training)

        if self.config['batchnorm']:
            norm_x = self.norm2(x, use_running_average=not training)
        else:
            norm_x = self.norm2(x)

        dense_output = self.dense2(nn.gelu(self.dense1(norm_x)))
        x = x + self.dropout(dense_output, deterministic=not training)

        return x
    
#endregion