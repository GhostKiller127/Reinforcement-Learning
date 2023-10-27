import torch
import numpy as np


class DataCollector:
    def __init__(self, training_class):
        self.config = training_class.config
        self.env_return = np.zeros(self.config['num_envs'])
        self.stepwise_data = []
        self.stacked_sequential_data = {}
        self.batched_sequential_data = []
        self.sequential_data = {}
        self.sequence_count = 0
        self.frame_count = 0
        self.max_sequences = int(self.config['per_buffer_size'] / self.config['sequence_length'])
        self.priorities = np.zeros(self.max_sequences)
    

    def add_step_data(self, o, a, a_p, i, r, d, t):
        self.stepwise_data.append({'o': o, 'a': a, 'a_p': a_p, 'i': i, 'r': r, 'd': d, 't': t})

    
    def add_sequential_data(self, stacked_data):
        for _ in range(self.config['num_envs']):
            sequence = {key: value[_] for key, value in stacked_data.items()}
            self.sequential_data[self.sequence_count % self.max_sequences] = sequence
            self.priorities[self.sequence_count % self.max_sequences] = 1e3
            self.frame_count += self.config['sequence_length']
            self.sequence_count += 1


    def add_stacked_data(self, stacked_data):
        for key in stacked_data:
            if key in self.stacked_sequential_data:
                self.stacked_sequential_data[key] = np.vstack((self.stacked_sequential_data[key], stacked_data[key]))
            else:
                self.stacked_sequential_data[key] = stacked_data[key]


    def add_batched_sequential_data(self):
        while True:
            enough_entries = next(iter(self.stacked_sequential_data.values())).shape[0] >= self.config['batch_size']
            if enough_entries:
                batch = {key: value[:self.config['batch_size']] for key, value in self.stacked_sequential_data.items()}
                self.stacked_sequential_data = {key: value[self.config['batch_size']:] for key, value in self.stacked_sequential_data.items()}
                self.batched_sequential_data.append(batch)
            else:
                break


    def check_save_sequence(self):
        if len(self.stepwise_data) == self.config['sequence_length'] + self.config['bootstrap_length'] + 1:
            stacked_data = {key: np.stack([d[key] for d in self.stepwise_data], axis=1) for key in self.stepwise_data[0]}
            if self.config['per_experience_replay']:
                self.add_sequential_data(stacked_data)
            else:
                self.add_stacked_data(stacked_data)
                self.add_batched_sequential_data()
            self.stepwise_data = self.stepwise_data[int(self.config['sequence_length'] / self.config['sample_reuse']):]
    

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
    

    def load_per_batched_sequences(self):
        probabilities = self.priorities**self.config['per_priority_exponent'] / np.sum(self.priorities**self.config['per_priority_exponent'])
        sequence_indeces = np.random.choice(len(self.priorities), size=self.config['batch_size'], p=probabilities, replace=False)
        sequence_batch = [self.sequential_data[index] for index in sequence_indeces]
        batched_sequence = {key: np.stack([d[key] for d in sequence_batch]) for key in sequence_batch[0]}
        return [batched_sequence], sequence_indeces
    

    def update_priorities(self, rtd1, rtd2, sequence_indeces):
        td_max1 = np.max(np.abs(rtd1), axis=1)
        td_max2 = np.max(np.abs(rtd2), axis=1)
        td_mean1 = np.mean(np.abs(rtd1), axis=1)
        td_mean2 = np.mean(np.abs(rtd2), axis=1)
        priority1 = self.config['per_priority_exponent'] * td_max1 + (1 - self.config['per_priority_exponent']) * td_mean1
        priority2 = self.config['per_priority_exponent'] * td_max2 + (1 - self.config['per_priority_exponent']) * td_mean2
        priorities = np.mean([priority1, priority2], axis=0)
        self.priorities[sequence_indeces] = priorities
