import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as dist
from architectures import DenseModel
from architectures import TransformerModel


class Actor:
    def __init__(self, config, metric, device):
        self.device = device
        self.d_pull = config['d_pull']
        self.log_dir = f'models/{metric.run_name}'
        if config['architecture_params']['architecture'] == 'dense':
            self.actor1 = DenseModel(config, device)
            self.actor2 = DenseModel(config, device)
        if config['architecture_params']['architecture'] == 'transformer':
            self.actor1 = TransformerModel(config, device)
            self.actor2 = TransformerModel(config, device)
        
    def pull_weights(self, step):
        if step % self.d_pull == 0:
            self.actor1.load_state_dict(torch.load(f'{self.log_dir}/learner1.pth'))
            self.actor2.load_state_dict(torch.load(f'{self.log_dir}/learner1.pth'))

    def calculate_values(self, observations, indices):
        v1, a1 = self.actor1(observations)
        v2, a2 = self.actor2(observations)

        indices = torch.tensor(indices, dtype=torch.float32).to(self.device)
        tau1 = indices[:,0].reshape(-1, 1)
        tau2 = indices[:,1].reshape(-1, 1)
        epsilon = indices[:,2].reshape(-1, 1)

        softmax_1 = F.softmax(a1 * tau1, dim=1)
        softmax_2 = F.softmax(a2 * tau2, dim=1)
        
        policy = epsilon * softmax_1 + (1 - epsilon) * softmax_2
    
        return v1, v2, a1, a2, policy

    def get_action(self, policy, stochastic, random):
        if random:
            action = [self.env.single_action_space.sample() for _ in range(self.num_envs)]
            action_prob = np.ones(action)
        if stochastic:
            action_dist = dist.Categorical(policy)
            action = action_dist.sample()
            action_prob = policy.gather(1, action.unsqueeze(1))
        else:
            _, action = policy.max(1)
            action_prob = policy.gather(1, action.unsqueeze(1))
        
        return action, action_prob
    