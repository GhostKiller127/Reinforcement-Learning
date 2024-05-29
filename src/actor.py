import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as dist
from architectures import DenseModel


class Actor:
    def __init__(self, training_class):
        self.count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = training_class.config
        self.log_dir = f'{training_class.log_dir}/models'
        self.architecture_parameters = self.config['parameters'][self.config['architecture']]
        if self.config['architecture'] == 'dense':
            self.actor1 = DenseModel(self.architecture_parameters, self.device)
            self.actor2 = DenseModel(self.architecture_parameters, self.device)
            for param1, param2 in zip(self.actor1.parameters(), self.actor2.parameters()):
                param1.requires_grad_(False)
                param2.requires_grad_(False)
        
    def pull_weights(self, learner=None, training=True):
        if self.count % self.config['d_pull'] == 0:
            if training:
                self.actor1.load_state_dict(learner.learner1.state_dict())
                self.actor2.load_state_dict(learner.learner2.state_dict())
            else:
                self.actor1.load_state_dict(torch.load(f'{self.log_dir}/learner1.pth'))
                self.actor2.load_state_dict(torch.load(f'{self.log_dir}/learner2.pth'))
        self.count += 1


    def calculate_policy(self, observations, indices):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            v1, a1 = self.actor1(observations)
            v2, a2 = self.actor2(observations)
        v1, a1 = v1[:, -1], a1[:, -1, :]
        v2, a2 = v2[:, -1], a2[:, -1, :]

        indices = torch.tensor(indices, dtype=torch.float32).to(self.device)
        tau1 = indices[:,0].unsqueeze(1)
        tau2 = indices[:,1].unsqueeze(1)
        epsilon = indices[:,2].unsqueeze(1)

        softmax_1 = F.softmax(a1 * tau1, dim=1)
        softmax_2 = F.softmax(a2 * tau2, dim=1)
    
        policy = epsilon * softmax_1 + (1 - epsilon) * softmax_2
        return policy


    def get_actions(self, observations, indeces, stochastic=True, random=False, training=False):
        policy = self.calculate_policy(observations, indeces)
        if random:
            actions = [self.env.single_action_space.sample() for _ in range(self.config['train_envs'] + self.config['val_envs'])]
            action_probs = np.ones(actions) / (self.config['train_envs'] + self.config['val_envs'])
        elif stochastic:
            action_dist = dist.Categorical(policy)
            actions = action_dist.sample()
            action_probs = policy.gather(1, actions.unsqueeze(1))
        else:
            _, actions = policy.max(1)
            action_probs = policy.gather(1, actions.unsqueeze(1))
        if training:
            _, greedy_actions = policy.max(1)
            actions[-(self.config['val_envs'] // 2):] = greedy_actions[-(self.config['val_envs'] // 2):]
        actions = actions.cpu().numpy()
        action_probs = action_probs.cpu().numpy()
        return actions, action_probs
    