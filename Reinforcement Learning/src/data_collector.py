import numpy as np



class DataCollector:
    def __init__(self, training_class):
        self.config = training_class.config
        self.env_return = np.zeros(self.config['num_envs'])
        self.max_sequences = self.get_max_sequences()
        self.priorities = np.zeros(self.max_sequences)
        self.sequence_length = self.config['sequence_length'] + self.config['bootstrap_length'] + 1
        self.reused_length = self.sequence_length - int(self.config['sequence_length'] / self.config['sample_reuse'])
        self.step_count = 0
        self.sequence_count = 0
        self.frame_count = 0

        self.obsveration_sequence = np.zeros((self.config['num_envs'], self.sequence_length, self.config['dense_params']['input_dim']))
        self.action_sequence = np.zeros((self.config['num_envs'], self.sequence_length))
        self.action_probs_sequence = np.zeros((self.config['num_envs'], self.sequence_length, 1))
        self.index_sequence = np.zeros((self.config['num_envs'], self.sequence_length, 3))
        self.reward_sequence = np.zeros((self.config['num_envs'], self.sequence_length))
        self.done_sequence = np.zeros((self.config['num_envs'], self.sequence_length))
        self.truncated_sequence = np.zeros((self.config['num_envs'], self.sequence_length))
        self.sequence_data = {'o': self.obsveration_sequence, 'a': self.action_sequence, 'a_p': self.action_probs_sequence, 'i': self.index_sequence,
                              'r': self.reward_sequence, 'd': self.done_sequence, 't': self.truncated_sequence}

        self.obsveration_data = np.zeros((self.max_sequences, self.sequence_length, self.config['dense_params']['input_dim']))
        self.action_data = np.zeros((self.max_sequences, self.sequence_length))
        self.action_probs_data = np.zeros((self.max_sequences, self.sequence_length, 1))
        self.index_data = np.zeros((self.max_sequences, self.sequence_length, 3))
        self.reward_data = np.zeros((self.max_sequences, self.sequence_length))
        self.done_data = np.zeros((self.max_sequences, self.sequence_length))
        self.truncated_data = np.zeros((self.max_sequences, self.sequence_length))
        self.all_data = {'o': self.obsveration_data, 'a': self.action_data, 'a_p': self.action_probs_data, 'i': self.index_data,
                         'r': self.reward_data, 'd': self.done_data, 't': self.truncated_data}

    
    def get_max_sequences(self):
        max_sequences = int(self.config['per_buffer_size'] / self.config['sequence_length'])
        if max_sequences % self.config['num_envs'] == 0:
            return max_sequences
        else:
            r = max_sequences % self.config['num_envs']
            d = self.config['num_envs'] - r
            return max_sequences + d

    
    def add_sequential_data(self):
        if self.sequence_count == self.max_sequences:
            self.sequence_count = 0
        for key, value in self.sequence_data.items():
            self.all_data[key][self.sequence_count: self.sequence_count + self.config['num_envs']] = value
        self.priorities[self.sequence_count: self.sequence_count + self.config['num_envs']] = 1e3
        self.frame_count += self.config['num_envs'] * self.config['sequence_length']
        self.sequence_count += self.config['num_envs']


    def add_data(self, **kwargs):
        for key, value in kwargs.items():
            self.sequence_data[key][:, self.step_count] = value
        self.step_count += 1
        if self.step_count == self.sequence_length:
            self.add_sequential_data()
            for _, value in self.sequence_data.items():
                value[:, :self.reused_length] = value[:, -self.reused_length:]
            self.step_count = self.reused_length
    

    def check_done_and_return(self):
        latest_return = self.sequence_data['r'][:, self.step_count - 1]
        latest_done = self.sequence_data['d'][:, self.step_count - 1]
        latest_truncated = self.sequence_data['t'][:, self.step_count - 1]
        latest_index = self.sequence_data['i'][:, self.step_count - 1]
        self.env_return += latest_return
        if np.all(np.logical_not(latest_done)) and np.all(np.logical_not(latest_truncated)):
            return None, None, []
        else:
            done_index = np.where(latest_done)[0]
            truncated_index = np.where(latest_truncated)[0]
            terminated_envs = np.union1d(done_index, truncated_index)
            returns = self.env_return[terminated_envs]
            indeces = latest_index[terminated_envs]
            self.env_return[terminated_envs] = 0
            return indeces, returns, terminated_envs


    def load_batched_sequences(self):
        probabilities = self.priorities**self.config['per_priority_exponent'] / np.sum(self.priorities**self.config['per_priority_exponent'])
        sequence_indeces = np.random.choice(len(self.priorities), size=self.config['batch_size'], p=probabilities, replace=False)
        batched_sequence = {key: value[sequence_indeces] for key, value in self.all_data.items()}
        return batched_sequence, sequence_indeces
    

    def update_priorities(self, rtd1, rtd2, sequence_indeces):
        td_max1 = np.max(np.abs(rtd1), axis=1)
        td_max2 = np.max(np.abs(rtd2), axis=1)
        td_mean1 = np.mean(np.abs(rtd1), axis=1)
        td_mean2 = np.mean(np.abs(rtd2), axis=1)
        priority1 = self.config['per_priority_exponent'] * td_max1 + (1 - self.config['per_priority_exponent']) * td_mean1
        priority2 = self.config['per_priority_exponent'] * td_max2 + (1 - self.config['per_priority_exponent']) * td_mean2
        priorities = np.mean([priority1, priority2], axis=0)
        self.priorities[sequence_indeces] = priorities
