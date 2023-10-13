import numpy as np

class DataCollector:
    def __init__(self, config):
        self.num_envs = config['num_envs']
        self.batch_size = config['batch_size']
        self.sequence_length = config['sequence_length']
        self.bootstrap_length = config['bootstrap_length']
        self.env_return = np.zeros(self.num_envs)
        self.stepwise_data = []
        self.sequential_data = []
    
    def add_step_data(self, o, v1, v2, a1, a2, i, p, a, a_p, r, d, t):
        self.stepwise_data.append({'o': o, 'v1': v1, 'v2': v2, 'a1': a1, 'a2': a2, 'i': i, 'p': p, 'a': a, 'a_p': a_p, 'r': r, 'd': d, 't': t})
    
    def update_return(self):
        latest_data = self.stepwise_data[-1]
        self.env_return += latest_data['r']
        if np.all(np.logical_not(latest_data['d'])) and np.all(np.logical_not(latest_data['t'])):
            return None, self.env_return, []
        else:
            done_index = np.where(latest_data['d'])[0]
            truncated_index = np.where(latest_data['t'])[0]
            terminated_envs = np.union1d(done_index, truncated_index)
            returns = self.env_return[terminated_envs]
            indeces = latest_data['i'][terminated_envs]
            self.env_return[terminated_envs] = 0
            return indeces, returns, terminated_envs

    def check_save_sequence(self):
        if len(self.stepwise_data) == self.sequence_length + self.bootstrap_length:
            self.sequential_data.append(self.stepwise_data)
            self.stepwise_data = self.stepwise_data[self.sequence_length:]

    def load_sequence(self):
        num_batches = len(self.sequential_data) // self.batch_size
        if num_batches == 0:
            return None
        else:
            data_batches = self.sequential_data[:num_batches * self.batch_size]
            self.sequential_data = self.sequential_data[num_batches * self.batch_size:]
            return data_batches
    