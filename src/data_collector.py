import numpy as np



class DataCollector:
    def __init__(self, training_class):
        self.config = training_class.config
        self.log_dir = f'{training_class.log_dir}/data_collector.npz'
        np.random.seed(self.config['jax_seed'])
        self.architecture_parameters = self.config['parameters'][self.config['architecture']]
        self.num_envs = self.config['train_envs'] + self.config['val_envs']
        self.max_sequences = self.get_max_sequences()
        self.sequence_length = self.config['sequence_length'] + self.config['bootstrap_length'] + 1
        self.reused_length = self.sequence_length - int(self.config['sequence_length'] / self.config['sample_reuse'])
        self.initialize_data_collector()
        if self.config['load_run'] is not None:
            self.load_data_collector()
        self.sequence_data = {'o': self.obsveration_sequence, 'a': self.action_sequence, 'a_p': self.action_probs_sequence, 'i': self.index_sequence,
                              'r': self.reward_sequence, 'd': self.done_sequence, 't': self.truncated_sequence}
        self.all_data = {'o': self.obsveration_data, 'a': self.action_data, 'a_p': self.action_probs_data, 'i': self.index_data,
                         'r': self.reward_data, 'd': self.done_data, 't': self.truncated_data}


    def initialize_data_collector(self):
        self.env_return = np.zeros(self.num_envs, dtype=np.float32)
        self.priorities = np.zeros(self.max_sequences, dtype=np.float32)
        self.step_count = 0
        self.sequence_count = 0
        self.frame_count = 0

        self.obsveration_sequence = np.zeros((self.num_envs, self.sequence_length, self.architecture_parameters['input_dim']), dtype=np.float32)
        self.action_sequence = np.zeros((self.num_envs, self.sequence_length), dtype=np.float32)
        self.action_probs_sequence = np.zeros((self.num_envs, self.sequence_length, 1), dtype=np.float32)
        self.index_sequence = np.zeros((self.num_envs, self.sequence_length, 3), dtype=np.float32)
        self.reward_sequence = np.zeros((self.num_envs, self.sequence_length), dtype=np.float32)
        self.done_sequence = np.zeros((self.num_envs, self.sequence_length), dtype=np.float32)
        self.truncated_sequence = np.zeros((self.num_envs, self.sequence_length), dtype=np.float32)

        self.obsveration_data = np.zeros((self.max_sequences, self.sequence_length, self.architecture_parameters['input_dim']), dtype=np.float32)
        self.action_data = np.zeros((self.max_sequences, self.sequence_length), dtype=np.float32)
        self.action_probs_data = np.zeros((self.max_sequences, self.sequence_length, 1), dtype=np.float32)
        self.index_data = np.zeros((self.max_sequences, self.sequence_length, 3), dtype=np.float32)
        self.reward_data = np.zeros((self.max_sequences, self.sequence_length), dtype=np.float32)
        self.done_data = np.zeros((self.max_sequences, self.sequence_length), dtype=np.float32)
        self.truncated_data = np.zeros((self.max_sequences, self.sequence_length), dtype=np.float32)
    

    def save_data_collector(self):
        np.savez(self.log_dir,
            priorities=self.priorities,
            sequence_count=self.sequence_count,
            frame_count=self.frame_count,
            obsveration_data=self.obsveration_data,
            action_data=self.action_data,
            action_probs_data=self.action_probs_data,
            index_data=self.index_data,
            reward_data=self.reward_data,
            done_data=self.done_data,
            truncated_data=self.truncated_data)


    def load_data_collector(self):
        data = np.load(self.log_dir)
        self.priorities = data['priorities']
        self.sequence_count = data['sequence_count']
        self.frame_count = data['frame_count']
        self.obsveration_data = data['obsveration_data']
        self.action_data = data['action_data']
        self.action_probs_data = data['action_probs_data']
        self.index_data = data['index_data']
        self.reward_data = data['reward_data']
        self.done_data = data['done_data']
        self.truncated_data = data['truncated_data']
    

    def get_max_sequences(self):
        max_sequences = int(self.config['per_buffer_size'] / self.config['sequence_length'])
        if max_sequences % self.config['train_envs'] == 0:
            return max_sequences
        else:
            r = max_sequences % self.config['train_envs']
            d = self.config['train_envs'] - r
            return max_sequences + d

    
    def add_sequential_data(self):
        if self.sequence_count == self.max_sequences:
            self.sequence_count = 0
        for key, value in self.sequence_data.items():
            self.all_data[key][self.sequence_count: self.sequence_count + self.config['train_envs']] = value[:self.config['train_envs']]
        self.priorities[self.sequence_count: self.sequence_count + self.config['train_envs']] = 1e3
        self.frame_count += self.config['train_envs'] * self.config['sequence_length']
        self.sequence_count += self.config['train_envs']


    def add_data(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'o':
                value = value[:, -1, :]
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
        
        done_envs = np.where(latest_done)[0]
        truncated_envs = np.where(latest_truncated)[0]
        terminated_envs = np.union1d(done_envs, truncated_envs)
        terminated_train_envs = [env for env in terminated_envs if env < self.config['train_envs']]
        terminated_val_envs = [env for env in terminated_envs if env >= self.config['train_envs']]
        train_returns = self.env_return[terminated_train_envs]
        val_returns = self.env_return[terminated_val_envs]
        train_indeces = latest_index[terminated_train_envs]
        val_indeces = latest_index[terminated_val_envs]
        self.env_return[terminated_envs] = 0
        return train_indeces, val_indeces, train_returns, val_returns, terminated_train_envs, terminated_val_envs


    def load_batched_sequences(self):
        probabilities = self.priorities**self.config['per_priority_exponent'] / np.sum(self.priorities**self.config['per_priority_exponent'])
        sequence_indeces = np.random.choice(len(self.priorities), size=self.config['batch_size'], p=probabilities, replace=False)
        batched_sequences = {key: value[sequence_indeces] for key, value in self.all_data.items()}
        return batched_sequences, sequence_indeces
    

    def update_priorities(self, rtd1, rtd2, sequence_indeces):
        td_max1 = np.max(np.abs(rtd1), axis=1)
        td_max2 = np.max(np.abs(rtd2), axis=1)
        td_mean1 = np.mean(np.abs(rtd1), axis=1)
        td_mean2 = np.mean(np.abs(rtd2), axis=1)
        priority1 = self.config['per_priority_exponent'] * td_max1 + (1 - self.config['per_priority_exponent']) * td_mean1
        priority2 = self.config['per_priority_exponent'] * td_max2 + (1 - self.config['per_priority_exponent']) * td_mean2
        priorities = np.mean([priority1, priority2], axis=0)
        self.priorities[sequence_indeces] = priorities
