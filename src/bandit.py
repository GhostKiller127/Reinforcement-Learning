import pickle
import random
import numpy as np
import itertools



class Bandit:
    def __init__(self, mode, l, r, acc, acc2, width, lr, d):
        self.mode = mode
        self.l = l
        self.r = r
        self.acc = acc
        self.acc2 = acc2
        self.width = width
        self.lr = lr
        self.d = d
        self.size = acc * acc2 + 1
        self.w = np.zeros(self.size)
        self.N = np.zeros(self.size)
        self.search_space = np.linspace(self.l, self.r, self.size)


    def evaluate(self):
        V = np.convolve(self.w, np.ones(2 * self.width + 1), mode='same') / (2 * self.width + 1)
        V[:self.width] *= [(2 * self.width + 1) / (self.width + 1 + i) for i in range(self.width)]
        V[-self.width:] *= [(2 * self.width + 1) / (self.width + 1 + i) for i in range(self.width)][::-1]
        return V


    def update(self, x, g):
        i = np.argmin(np.abs(self.search_space - x))
        slice_indices = slice(*np.clip([i - self.width, i + self.width + 1], 0, self.size))
        self.w[slice_indices] += self.lr * (g - self.evaluate()[i])
        self.N[i] += 1


    def sample(self):
        V = self.evaluate()
        epsilon = np.finfo(np.float32).eps
        scores = (V - np.mean(V)) / (np.std(V) + epsilon) + np.sqrt(np.log(1 + np.sum(self.N)) / (1 + self.N))

        candidates = None
        if self.mode == 'argmax':
            sorted_indices = np.argsort(scores)[::-1]
            candidates_indices = []
            for idx in sorted_indices:
                within_epsilon = np.isclose(scores[idx], scores, atol=float(epsilon))
                indices_within_epsilon = np.where(within_epsilon)[0]
                candidates_indices.extend(indices_within_epsilon)
                if len(candidates_indices) >= self.d:
                    break
            candidates = self.search_space[np.random.choice(candidates_indices, self.d, replace=False)]
        elif self.mode == 'random':
            probabilities = np.exp(scores) / np.sum(np.exp(scores))
            candidates_indices = np.random.choice(len(scores), self.d, replace=False, p=probabilities)
            candidates = self.search_space[candidates_indices]
        return candidates



class Bandits:
    def __init__(self, training_class):
        self.config = training_class.config
        self.bandit_params = self.config['bandit_params']
        self.log_dir = f'{training_class.log_dir}/bandits.pkl'
        if self.config['load_run'] is None:
            self.bandits = self.initialize_bandits()
        else:
            self.bandits = self.load_bandits()
        self.search_space = self.get_all_search_space()


    def save_bandits(self):
        with open(self.log_dir, 'wb') as file:
            pickle.dump(self.bandits, file)


    def load_bandits(self):
        with open(self.log_dir, 'rb') as file:
            self.bandits = pickle.load(file)
        return self.bandits


    def initialize_bandits(self):
        modes = self.bandit_params["mode"]
        acc2_values = self.bandit_params["acc2"]
        width_values = self.bandit_params["width"]
        lr_values = self.bandit_params["lr"]
        d = self.bandit_params["d"]
        acc = self.bandit_params["acc"]
        search_params = ["tau1", "tau2", "epsilon"]

        bandits = []
        for mode, acc2, width, lr in itertools.product(modes, acc2_values, width_values, lr_values):
            bandit_dict = {}
            for i, param in enumerate(search_params):
                l, r = self.bandit_params[param]
                bandit_dict[param] = Bandit(mode, l, r, acc[i], acc2, width, lr, d)
            bandits.append(bandit_dict)
        return bandits


    def update_bandits(self, tau1, tau2, epsilon, g):
        for bandit_dict in self.bandits:
            bandit_dict["tau1"].update(tau1, g)
            bandit_dict["tau2"].update(tau2, g)
            bandit_dict["epsilon"].update(epsilon, g)


    def get_candidates(self):
        candidates = {"tau1": [], "tau2": [], "epsilon": []}
        for bandit_dict in self.bandits:
            for param in ["tau1", "tau2", "epsilon"]:
                candidates[param].extend(bandit_dict[param].sample())
        return candidates


    def sample_candidate(self, candidates):
        tau1 = random.choice(candidates["tau1"])
        tau2 = random.choice(candidates["tau2"])
        epsilon = random.choice(candidates["epsilon"])
        return tau1, tau2, epsilon
    

    def get_all_search_space(self):
        tau1_search_space = np.array([])
        tau2_search_space = np.array([])
        epsilon_search_space = np.array([])
        
        for bandit_dict in self.bandits:
            tau1_search_space = np.union1d(tau1_search_space, bandit_dict["tau1"].search_space)
            tau2_search_space = np.union1d(tau2_search_space, bandit_dict["tau2"].search_space)
            epsilon_search_space = np.union1d(epsilon_search_space, bandit_dict["epsilon"].search_space)
        
        return tau1_search_space, tau2_search_space, epsilon_search_space


    def get_index_data(self, only_index=False):
        tau1_search_space, tau2_search_space, epsilon_search_space = self.search_space
        num_bandits = len(self.bandits)

        tau1_average_count = np.zeros(len(tau1_search_space))
        tau2_average_count = np.zeros(len(tau2_search_space))
        epsilon_average_count = np.zeros(len(epsilon_search_space))
        tau1_average_weight = np.zeros(len(tau1_search_space))
        tau2_average_weight = np.zeros(len(tau2_search_space))
        epsilon_average_weight = np.zeros(len(epsilon_search_space))

        for bandit_dict in self.bandits:
            tau1_bandit = bandit_dict["tau1"]
            tau2_bandit = bandit_dict["tau2"]
            epsilon_bandit = bandit_dict["epsilon"]

            tau1_indices = np.argmin(np.abs(tau1_bandit.search_space[:, None] - tau1_search_space), axis=0)
            tau2_indices = np.argmin(np.abs(tau2_bandit.search_space[:, None] - tau2_search_space), axis=0)
            epsilon_indices = np.argmin(np.abs(epsilon_bandit.search_space[:, None] - epsilon_search_space), axis=0)

            if not only_index:
                tau1_average_count += tau1_bandit.N[tau1_indices]
                tau2_average_count += tau2_bandit.N[tau2_indices]
                epsilon_average_count += epsilon_bandit.N[epsilon_indices]

            tau1_average_weight += tau1_bandit.w[tau1_indices]
            tau2_average_weight += tau2_bandit.w[tau2_indices]
            epsilon_average_weight += epsilon_bandit.w[epsilon_indices]

        tau1_average_count /= num_bandits
        tau2_average_count /= num_bandits
        epsilon_average_count /= num_bandits
        tau1_average_weight /= num_bandits
        tau2_average_weight /= num_bandits
        epsilon_average_weight /= num_bandits

        max_tau1 = tau1_search_space[np.argmax(tau1_average_weight)]
        max_tau2 = tau2_search_space[np.argmax(tau2_average_weight)]
        max_epsilon = epsilon_search_space[np.argmax(epsilon_average_weight)]

        if only_index:
            return max_tau1, max_tau2, max_epsilon
        return (
            max_tau1, max_tau2, max_epsilon,
            tau1_average_count, tau2_average_count, epsilon_average_count,
            tau1_average_weight, tau2_average_weight, epsilon_average_weight
        )


    def get_all_indeces(self, num_envs):
        indeces = []
        all_candidates = self.get_candidates()
        for _ in range(num_envs):
            indeces.append(self.sample_candidate(all_candidates))
        return np.array(indeces)


    def update_and_get_data(self, data_collector, train_indeces, train_returns, train_envs, val_envs):
        index_data = None
        if data_collector.frame_count >= self.config['per_min_frames']:
            for _ in range(len(train_returns)):
                tau1, tau2, epsilon = train_indeces[_]
                g = train_returns[_]
                self.update_bandits(tau1, tau2, epsilon, g)
            self.save_bandits()
            index_data = self.get_index_data()
            new_val_indeces = [index_data[:3] for _ in val_envs] if val_envs else None
        
        all_candidates = self.get_candidates()
        new_train_indeces = [self.sample_candidate(all_candidates) for _ in train_envs] if train_envs else None
        if not data_collector.frame_count >= self.config['per_min_frames']:
            new_val_indeces = [self.sample_candidate(all_candidates) for _ in val_envs] if val_envs else None

        return new_train_indeces, new_val_indeces, index_data
