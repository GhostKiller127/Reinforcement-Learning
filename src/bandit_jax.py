import random
import numpy as np
import jax
import jax.numpy as jnp
import functools
import itertools



class Bandits:
    def __init__(self, training_class):
        self.config = training_class.config
        self.bandit_params = self.config['bandit_params']
        self.log_dir = f'{training_class.log_dir}/bandits.npz'
        self.main_rng = jax.random.PRNGKey(self.config['jax_seed'])
        if self.config['load_run'] is None:
            self.initialize_bandits()
        else:
            self.load_bandits()
        self.param_indeces_and_search_spaces = self.get_param_indeces_and_search_spaces()


    def initialize_bandits(self):
        self.modes = []
        self.lrs = []
        self.ws = []
        self.Ns = []
        self.search_spaces = []
        self.params = []
        self.d = self.bandit_params["d"]
        self.width = self.bandit_params["width_"]
        self.size = self.bandit_params["size"]

        for mode, lr in itertools.product(self.bandit_params["mode"], self.bandit_params["lr"]):
            for param in ["tau1", "tau2", "epsilon"]:
                l, r = self.bandit_params[param]
                if mode == 'argmax':
                    self.modes.append(True)
                elif mode == 'random':
                    self.modes.append(False)
                self.lrs.append(lr)
                self.ws.append(jnp.zeros(self.size))
                self.Ns.append(jnp.zeros(self.size))
                self.search_spaces.append(jnp.linspace(l, r, self.size))
                self.params.append(param)

        self.modes = jnp.array(self.modes)
        self.lrs = jnp.array(self.lrs)
        self.ws = jnp.array(self.ws)
        self.Ns = jnp.array(self.Ns)
        self.search_spaces = jnp.array(self.search_spaces)
    

    def save_bandits(self):
        np.savez(self.log_dir,
                 modes=self.modes,
                 lrs=self.lrs,
                 ws=self.ws,
                 Ns=self.Ns,
                 search_spaces=self.search_spaces)


    def load_bandits(self):
        self.params = []
        self.d = self.bandit_params["d"]
        self.width = self.bandit_params["width_"]
        self.size = self.bandit_params["size"]

        for mode, lr in itertools.product(self.bandit_params["mode"], self.bandit_params["lr"]):
            for param in ["tau1", "tau2", "epsilon"]:
                self.params.append(param)

        data = np.load(self.log_dir)
        self.modes = data['modes']
        self.lrs = data['lrs']
        self.ws = data['ws']
        self.Ns = data['Ns']
        self.search_spaces = data['search_spaces']
    

    def update_bandits(self, tau1, tau2, epsilon, g):
        x = {"tau1": tau1, "tau2": tau2, "epsilon": epsilon}
        xs = jnp.array([x[param] for param in self.params])
        gs = jnp.array([g for _ in xs])
        self.ws, self.Ns = update_vmap(xs, gs, self.ws, self.Ns, self.search_spaces, self.lrs, self.width)


    def get_candidates(self):
        candidates = {"tau1": [], "tau2": [], "epsilon": []}
        sample_rngs = jax.random.split(self.main_rng, num=len(self.params))
        self.main_rng, _ = jax.random.split(self.main_rng)
        all_candidates = sample_vmap(sample_rngs, self.ws, self.Ns, self.search_spaces, self.modes, self.d, self.width)
        all_candidates = np.array(all_candidates)
        for i, param in enumerate(self.params):
            candidates[param].extend(all_candidates[i])
        return candidates


    def sample_candidate(self, candidates):
        tau1 = random.choice(candidates["tau1"])
        tau2 = random.choice(candidates["tau2"])
        epsilon = random.choice(candidates["epsilon"])
        return tau1, tau2, epsilon
    

    def get_all_indeces(self, num_envs):
        indeces = []
        self.index_data = self.get_index_data()
        self.all_candidates = self.get_candidates()
        for _ in range(num_envs):
            indeces.append(self.sample_candidate(self.all_candidates))
        return np.array(indeces)


    def get_param_indeces_and_search_spaces(self):
        tau1_indeces = []
        tau2_indeces = []
        epsilon_indeces = []
        tau1_search_space = jnp.array([])
        tau2_search_space = jnp.array([])
        epsilon_search_space = jnp.array([])
        
        for i, param in enumerate(self.params):
            if param == 'tau1':
                tau1_indeces.append(i)
                tau1_search_space = jnp.union1d(tau1_search_space, self.search_spaces[i])
            elif param == 'tau2':
                tau2_indeces.append(i)
                tau2_search_space = jnp.union1d(tau2_search_space, self.search_spaces[i])
            elif param == 'epsilon':
                epsilon_indeces.append(i)
                epsilon_search_space = jnp.union1d(epsilon_search_space, self.search_spaces[i])
        
        return (jnp.array([tau1_indeces]), jnp.array([tau2_indeces]), jnp.array([epsilon_indeces]),
                tau1_search_space, tau2_search_space, epsilon_search_space)
    

    def get_index_data(self, only_index=False):
        tau1_indeces, tau2_indeces, epsilon_indeces, tau1_search_space, tau2_search_space, epsilon_search_space = self.param_indeces_and_search_spaces

        max_tau1, max_tau2, max_epsilon = get_max_indeces(tau1_search_space,
                                                          tau2_search_space,
                                                          epsilon_search_space,
                                                          self.ws[tau1_indeces][0],
                                                          self.ws[tau2_indeces][0],
                                                          self.ws[epsilon_indeces][0])
        if only_index:
            return max_tau1, max_tau2, max_epsilon
        return (max_tau1, max_tau2, max_epsilon, None, None, None, None, None, None)


    def update_and_get_data(self, data_collector, train_indeces, train_returns, train_envs, val_envs):
        new_train_indeces, new_val_indeces, index_data = None, None, None
        if data_collector.frame_count >= self.config['per_min_frames'] and self.config['bandits']:
            for _ in range(len(train_returns)):
                tau1, tau2, epsilon = train_indeces[_]
                g = train_returns[_]
                self.update_bandits(tau1, tau2, epsilon, g)
            if train_envs:
                self.index_data = self.get_index_data()
                index_data = (np.array(x) for x in self.index_data)
                all_candidates = self.get_candidates()
                new_train_indeces = [self.sample_candidate(all_candidates) for _ in train_envs]
            new_val_indeces = [self.index_data[:3] for _ in val_envs] if val_envs else None
        else:
            if train_envs or val_envs:
                all_candidates = self.get_candidates()
            new_train_indeces = [self.sample_candidate(all_candidates) for _ in train_envs] if train_envs else None
            new_val_indeces = [self.sample_candidate(all_candidates) for _ in val_envs] if val_envs else None
        
        return new_train_indeces, new_val_indeces, index_data


@functools.partial(jax.jit, static_argnums=())
def get_max_indeces(s1, s2, s3, w1, w2, w3):
    max1 = s1[jnp.argmax(jnp.sum(w1, axis=0))]
    max2 = s2[jnp.argmax(jnp.sum(w2, axis=0))]
    max3 = s3[jnp.argmax(jnp.sum(w3, axis=0))]
    return max1, max2, max3


@functools.partial(jax.jit, static_argnums=(1,))
def evaluate(w, width):
    V = jnp.convolve(w, jnp.ones(2 * width + 1), mode='same') / (2 * width + 1)
    V = V.at[:width].mul(jnp.array([(2 * width + 1) / (width + 1 + i) for i in range(width)]))
    V = V.at[-width:].mul(jnp.array([(2 * width + 1) / (width + 1 + i) for i in range(width)][::-1]))
    return V


@functools.partial(jax.jit, static_argnums=(6,))
def update_vmap(x, g, w, N, search_space, lr, width):
    return jax.vmap(update_, in_axes=[0, 0, 0, 0, 0, 0, None])(x, g, w, N, search_space, lr, width)


@functools.partial(jax.jit, static_argnums=(6,))
def update_(x, g, w, N, search_space, lr, width):
    i = jnp.argmin(jnp.abs(search_space - x))
    start_index = jnp.maximum(i - width, 0)
    w_slice = jax.lax.dynamic_slice(w, (start_index,), (2 * width + 1,))
    update_value = lr * (g - evaluate(w, width)[i])
    w = jax.lax.dynamic_update_slice(w, w_slice + update_value, (start_index,))
    N = N.at[i].add(1)
    return w, N


@functools.partial(jax.jit, static_argnums=(5, 6))
def sample_vmap(sample_rng, w, N, search_space, mode, d, width):
    return jax.vmap(sample_, in_axes=[0, 0, 0, 0, 0, None, None])(sample_rng, w, N, search_space, mode, d, width)


@functools.partial(jax.jit, static_argnums=(5, 6))
def sample_(sample_rng, w, N, search_space, mode, d, width):
    V = evaluate(w, width)
    epsilon = jnp.finfo(jnp.float32).eps
    scores = (V - jnp.mean(V)) / (jnp.std(V) + epsilon) + jnp.sqrt(jnp.log(1 + jnp.sum(N)) / (1 + N))

    def true_fun():
        candidates_indices = jnp.argsort(scores)[-d:]
        return search_space[candidates_indices]

    def false_fun():
        probabilities = jnp.exp(scores) / jnp.sum(jnp.exp(scores))
        candidates_indices = jax.random.choice(sample_rng, len(scores), shape=(d,), replace=False, p=probabilities)
        return search_space[candidates_indices]

    candidates = jax.lax.cond(mode, true_fun, false_fun)
    return candidates

