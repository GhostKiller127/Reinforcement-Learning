import os
import sys
sys.path.append(os.path.abspath('../../'))

import jax.numpy as np
from flax import linen as nn
from jax.scipy.linalg import block_diag

from S5.s5.seq_model import GaussianRegressionModel
from S5.s5.ssm import init_S5SSM
from S5.s5.ssm_init import make_DPLR_HiPPO
from S5.s5.cru.util import CRN_CNN


class S5():
    def __init__(self, configs):
        self.configs = configs
        self.BatchGaussianRegressionModel = nn.vmap(
            GaussianRegressionModel,
            in_axes=(0, None),
            out_axes=0,
            variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
            split_rngs={"params": False, "dropout": True},
            axis_name='batch')
        self.ssm_init_fn = self.get_ssm_init_fn()
        self.s5 = self.BatchGaussianRegressionModel(self.ssm_init_fn,
                                                    d_output=self.configs['action_dim'],
                                                    d_model=self.configs['d_model'],
                                                    n_layers=self.configs['n_layers'],
                                                    padded=False,
                                                    activation=self.configs['activation'],
                                                    dropout=self.configs['dropout'],
                                                    prenorm=self.configs['prenorm'],
                                                    batchnorm=self.configs['batchnorm'],
                                                    decoder_dim=self.configs['decoder_dim'],
                                                    # encoder_fn=lambda d: CRN_CNN(d, input_shape=24)
                                                    )

    def get_ssm_init_fn(self):
        block_size = int(self.configs['ssm_size'] / self.configs['blocks'])
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(N=block_size)

        block_size = block_size // 2
        ssm_size = self.configs['ssm_size'] // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        Lambda = (Lambda * np.ones((self.configs['blocks'], block_size))).ravel()
        V = block_diag(*([V] * self.configs['blocks']))
        Vinv = block_diag(*([Vc] * self.configs['blocks']))

        return init_S5SSM(H=self.configs['d_model'],
                            P=ssm_size,
                            Lambda_re_init=Lambda.real,
                            Lambda_im_init=Lambda.imag,
                            V=V,
                            Vinv=Vinv,
                            C_init='trunc_standard_normal',
                            discretization="zoh",
                            dt_min=0.001,
                            dt_max=0.1,
                            variable_observation_interval=False,
                            conj_sym=True,
                            clip_eigs=False,
                            bidirectional=False)
    