import torch
import numpy as np

class DataCollector:
    def __init__(self, config):
        self.num_envs = config['num_envs']
        self.batch_size = config['batch_size']
        self.sequence_length = config['sequence_length']
        self.bootstrap_length = config['bootstrap_length']
        self.env_return = np.zeros(self.num_envs)
        self.stepwise_data = []
        self.stacked_sequential_data = {}
        self.batched_sequential_data = []
    
    def add_step_data(self, o, a, a_p, i, r, d, t):
        self.stepwise_data.append({'o': o, 'a': a, 'a_p': a_p, 'i': i, 'r': r, 'd': d, 't': t})

    def add_stacked_data(self, stacked_data):
        for key in stacked_data.keys():
            if key in self.stacked_sequential_data:
                if isinstance(stacked_data[key], np.ndarray):
                    self.stacked_sequential_data[key] = np.vstack((self.stacked_sequential_data[key], stacked_data[key]))
                elif isinstance(stacked_data[key], torch.Tensor):
                    self.stacked_sequential_data[key] = torch.vstack((self.stacked_sequential_data[key], stacked_data[key]))
            else:
                self.stacked_sequential_data[key] = stacked_data[key]

    def add_batched_sequential_data(self):
        while True:
            enough_entries = next(iter(self.stacked_sequential_data.values())).shape[0] >= self.batch_size
            if enough_entries:
                batch = {key: value[:self.batch_size] for key, value in self.stacked_sequential_data.items()}
                self.stacked_sequential_data = {key: value[self.batch_size:] for key, value in self.stacked_sequential_data.items()}
                self.batched_sequential_data.append(batch)
            else:
                break

    def check_save_sequence(self):
        if len(self.stepwise_data) == self.sequence_length + self.bootstrap_length + 1:
            stacked_data = {key: np.stack([d[key] for d in self.stepwise_data], axis=1) if isinstance(self.stepwise_data[0][key], np.ndarray)
                            else torch.stack([d[key] for d in self.stepwise_data], dim=1) for key in self.stepwise_data[0].keys()}
            self.add_stacked_data(stacked_data)
            self.add_batched_sequential_data()
            self.stepwise_data = self.stepwise_data[self.sequence_length:]
    
    def check_done_and_return(self):
        latest_data = self.stepwise_data[-1]
        self.env_return += latest_data['r']
        if np.all(np.logical_not(latest_data['d'])) and np.all(np.logical_not(latest_data['t'])):
            return None, None, []
        else:
            done_index = np.where(latest_data['d'])[0]
            truncated_index = np.where(latest_data['t'])[0]
            terminated_envs = np.union1d(done_index, truncated_index)
            returns = self.env_return[terminated_envs]
            indeces = latest_data['i'][terminated_envs]
            self.env_return[terminated_envs] = 0
            return indeces, returns, terminated_envs

    def load_batched_sequences(self):
        data = self.batched_sequential_data
        self.batched_sequential_data = []
        return data
    