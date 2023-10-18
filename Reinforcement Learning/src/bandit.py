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
        self.N[slice_indices] += 1


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
    def __init__(self, config, metric):
        self.bandit_params = config['bandit_params']
        self.log_dir = metric.log_dir
        self.bandits_file = f'{self.log_dir}/bandits.pkl'
        if config['load_run'] is None:
            self.bandits = self.initialize_bandits()
        else:
            self.bandits = self.load_bandits()


    def save_bandits(self):
        with open(self.bandits_file, 'wb') as file:
            pickle.dump(self.bandits, file)


    def load_bandits(self):
        with open(self.bandits_file, 'rb') as file:
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


    def get_all_indeces(self, num_envs):
        indeces = []
        all_candidates = self.get_candidates()
        for _ in range(num_envs):
            indeces.append(self.sample_candidate(all_candidates))
        return np.array(indeces)


    def update_and_get_new_indeces(self, terminated_indeces, returns):
        if terminated_indeces is None:
            return 0
        for _ in range(len(returns)):
            tau1, tau2, epsilon = terminated_indeces[_]
            g = returns[_]
            self.update_bandits(tau1, tau2, epsilon, g)
        self.save_bandits()
        new_indeces = []
        all_candidates = self.get_candidates()
        for _ in range(len(returns)):
            new_indeces.append(self.sample_candidate(all_candidates))
        return np.array(new_indeces)
