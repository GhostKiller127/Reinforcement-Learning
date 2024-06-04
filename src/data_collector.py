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
        self.sequence_data = self.initialize_sequence_data()
        self.all_data = self.initialize_all_data()
        if self.config['load_run'] is not None:
            self.load_data_collector()


    def initialize_data_collector(self):
        self.env_return = np.zeros(self.num_envs, dtype=np.float32)
        self.priorities = np.zeros(self.max_sequences, dtype=np.float32)
        self.step_count = 0
        self.sequence_count = 0
        self.frame_count = 0


    def initialize_sequence_data(self):
        return {key: np.zeros((self.num_envs, self.sequence_length, *shape), dtype=np.float32)
                for key, shape in zip(['o', 'a', 'a_p', 'i', 'r', 'd', 't'],
                                      [(self.architecture_parameters['input_dim'],), (), (1,), (3,), (), (), ()])}


    def initialize_all_data(self):
        return {key: np.zeros((self.max_sequences, self.sequence_length, *shape), dtype=np.float32)
                for key, shape in zip(['o', 'a', 'a_p', 'i', 'r', 'd', 't'],
                                      [(self.architecture_parameters['input_dim'],), (), (1,), (3,), (), (), ()])}


    def save_data_collector(self):
        np.savez(self.log_dir, priorities=self.priorities, sequence_count=self.sequence_count,
                 frame_count=self.frame_count, **self.all_data)


    def load_data_collector(self):
        data = np.load(self.log_dir)
        self.priorities = data['priorities']
        self.sequence_count = data['sequence_count']
        self.frame_count = data['frame_count']
        for key in self.all_data.keys():
            self.all_data[key] = data[key]


    def get_max_sequences(self):
        max_sequences = int(self.config['per_buffer_size'] / self.config['sequence_length'])
        remainder = max_sequences % self.config['train_envs']
        return max_sequences if remainder == 0 else max_sequences + (self.config['train_envs'] - remainder)


    def add_sequential_data(self):
        if self.sequence_count == self.max_sequences:
            self.sequence_count = 0
        for key in self.sequence_data.keys():
            self.all_data[key][self.sequence_count:self.sequence_count + self.config['train_envs']] = self.sequence_data[key][:self.config['train_envs']]
        self.priorities[self.sequence_count:self.sequence_count + self.config['train_envs']] = 1e3
        self.frame_count += self.config['train_envs'] * self.config['sequence_length']
        self.sequence_count += self.config['train_envs']


    def add_data(self, **kwargs):
        for key, value in kwargs.items():
            self.sequence_data[key][:, self.step_count] = value[:, -1, :] if key == 'o' else value
        self.step_count += 1
        if self.step_count == self.sequence_length:
            self.add_sequential_data()
            for key in self.sequence_data.keys():
                self.sequence_data[key][:, :self.reused_length] = self.sequence_data[key][:, -self.reused_length:]
            self.step_count = self.reused_length


    def check_done_and_return(self):
        latest_return = self.sequence_data['r'][:, self.step_count - 1]
        latest_done = self.sequence_data['d'][:, self.step_count - 1]
        latest_truncated = self.sequence_data['t'][:, self.step_count - 1]
        latest_index = self.sequence_data['i'][:, self.step_count - 1]
        self.env_return += latest_return

        terminated_envs = np.union1d(np.where(latest_done)[0], np.where(latest_truncated)[0])
        terminated_train_envs = [env for env in terminated_envs if env < self.config['train_envs']]
        terminated_val_envs = [env for env in terminated_envs if env >= self.config['train_envs']]
        train_returns = self.env_return[terminated_train_envs]
        val_returns = self.env_return[terminated_val_envs]
        train_indices = latest_index[terminated_train_envs]
        val_indices = latest_index[terminated_val_envs]
        self.env_return[terminated_envs] = 0
        return train_indices, val_indices, train_returns, val_returns, terminated_train_envs, terminated_val_envs


    def load_batched_sequences(self):
        probabilities = self.priorities**self.config['per_priority_exponent']
        probabilities /= np.sum(probabilities)
        sequence_indices = np.random.choice(len(self.priorities), size=self.config['batch_size'], p=probabilities, replace=False)
        batched_sequences = {key: value[sequence_indices] for key, value in self.all_data.items()}
        return batched_sequences, sequence_indices


    def update_priorities(self, rtd1, rtd2, sequence_indices):
        td_max1 = np.max(np.abs(rtd1), axis=1)
        td_max2 = np.max(np.abs(rtd2), axis=1)
        td_mean1 = np.mean(np.abs(rtd1), axis=1)
        td_mean2 = np.mean(np.abs(rtd2), axis=1)
        priority1 = self.config['per_priority_exponent'] * td_max1 + (1 - self.config['per_priority_exponent']) * td_mean1
        priority2 = self.config['per_priority_exponent'] * td_max2 + (1 - self.config['per_priority_exponent']) * td_mean2
        self.priorities[sequence_indices] = np.mean([priority1, priority2], axis=0)
