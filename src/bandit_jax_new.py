import numpy as np
import jax
import jax.numpy as jnp
import functools



class Bandits:
    def __init__(self, training_class):
        self.config = training_class.config
        self.bandit_params = self.config['bandit_params']
        self.log_dir = f'{training_class.log_dir}/bandits.npz'
        self.main_rng = jax.random.PRNGKey(self.config['jax_seed'])
        np.random.seed(self.config['jax_seed'])
        if self.config['load_run'] is None:
            self.initialize_bandits()
        else:
            self.load_bandits()


    def initialize_bandits(self):
        self.envs = []
        self.modes = []
        self.lrs = []
        self.ws = []
        self.Ns = []
        self.pad_lengths = []
        self.masks = []
        self.search_spaces = []
        self.params = []
        self.d = self.bandit_params['d']
        self.max_size = max(self.bandit_params['acc']) * max(self.bandit_params['acc2']) + 1

        for env in range(self.config['num_envs'] - self.config['val_envs']):
            for _ in range(self.bandit_params['num_bandits']):
                for i, param in enumerate(['tau1', 'tau2', 'epsilon']):
                    mode = np.random.binomial(1, 0.5)
                    acc2 = np.random.choice(self.bandit_params['acc2'])
                    lr = np.random.choice(self.bandit_params['lr'])

                    l, r = self.bandit_params[param]
                    if param != 'epsilon':
                        l, r = np.log(1 + l), np.log(1 + r)
                    size = self.bandit_params['acc'][i] * acc2 + 1
                    max_offset = (r - l) / (size - 1)
                    offset = np.random.uniform(1e-6, max_offset)
                    pad_length = self.max_size - size
                    w = np.full(self.max_size, -np.inf)
                    w[pad_length:] = self.bandit_params['weight_init']
                    mask = np.zeros(self.max_size, dtype=bool)
                    mask[pad_length:] = 1
                    search_space = np.linspace(l, r, size) + offset
                    search_space[-1] = r
                    search_space = np.pad(search_space, (pad_length + 1, 0), constant_values=l)

                    self.envs.append(env)
                    self.params.append(param)
                    self.modes.append(mode)
                    self.lrs.append(lr)
                    self.ws.append(w)
                    self.Ns.append(np.zeros(self.max_size))
                    self.pad_lengths.append(pad_length)
                    self.masks.append(mask)
                    self.search_spaces.append(search_space)

        self.envs = jnp.array(self.envs)
        self.params = np.array(self.params)
        self.modes = jnp.array(self.modes)
        self.lrs = jnp.array(self.lrs)
        self.ws = jnp.array(self.ws)
        self.Ns = jnp.array(self.Ns)
        self.pad_lengths = jnp.array(self.pad_lengths)
        self.masks = jnp.array(self.masks)
        self.search_spaces = jnp.array(self.search_spaces)


    def save_bandits(self):
        np.savez(self.log_dir,
                 envs=self.envs,
                 params=self.params,
                 modes=self.modes,
                 lrs=self.lrs,
                 ws=self.ws,
                 Ns=self.Ns,
                 pad_lengths=self.pad_lengths,
                 masks=self.masks,
                 search_spaces=self.search_spaces)


    def load_bandits(self):
        data = np.load(self.log_dir)
        self.envs = data['envs']
        self.modes = data['modes']
        self.params = data['params']
        self.lrs = data['lrs']
        self.ws = data['ws']
        self.Ns = data['Ns']
        self.pad_lengths = data['pad_lengths']
        self.masks = data['masks']
        self.search_spaces = data['search_spaces']
        self.d = self.bandit_params["d"]
        self.max_size = max(self.bandit_params['acc']) * max(self.bandit_params['acc2']) + 1


    def update_bandits(self, train_indeces, train_returns, train_envs):
        train_indeces = np.array(train_indeces)
        train_indeces[:, 0:2] = np.log(1 + train_indeces[:, 0:2])

        xs = np.zeros((self.config['num_envs'] - self.config['val_envs'], self.bandit_params['num_bandits'], 3))
        gs = np.zeros((self.config['num_envs'] - self.config['val_envs'], self.bandit_params['num_bandits'], 3))
        bools = np.zeros((self.config['num_envs'] - self.config['val_envs'], self.bandit_params['num_bandits'], 3))

        xs[np.array(train_envs), ...] = np.array(train_indeces)[:, None, :]
        gs[np.array(train_envs), ...] = np.array(train_returns)[:, None, None]
        bools[np.array(train_envs)] = 1

        xs = xs.ravel()
        gs = gs.ravel()
        bools = bools.ravel()

        self.ws, self.Ns = update_vmap(xs, gs, self.ws, self.Ns, self.search_spaces, self.lrs, bools)


    def get_all_candidates(self):
        sample_rngs = jax.random.split(self.main_rng, num=len(self.params))
        self.main_rng, _ = jax.random.split(self.main_rng)
        all_candidates = sample_vmap(sample_rngs, self.ws, self.Ns, self.search_spaces, self.modes, self.masks, self.d)
        return all_candidates
    

    def sample_all_candidates(self, train_envs=None):
        all_candidates = self.get_all_candidates()
        all_candidates = all_candidates.reshape((self.config['num_envs'] - self.config['val_envs'], self.bandit_params['num_bandits'], 3, self.d))
        sample_rngs = jax.random.split(self.main_rng, num=((self.config['num_envs'] - self.config['val_envs']), 3))
        self.main_rng, _ = jax.random.split(self.main_rng)
        all_candidates = sample_candidates_vmap(sample_rngs, all_candidates)
        if train_envs is not None:
            all_candidates = all_candidates[np.array(train_envs)]
        return all_candidates
    

    def get_all_indeces(self, num_envs=None):
        self.index_data = self.get_index_data(only_index=True)
        all_candidates = self.sample_all_candidates()
        val_indeces = np.array([self.index_data for _ in range(self.config['val_envs'])])
        all_indeces = np.vstack((all_candidates, val_indeces))
        return all_indeces


    def get_index_data(self, only_index=False):
        sample_rngs = jax.random.split(self.main_rng, num=3)
        self.main_rng, _ = jax.random.split(self.main_rng)
        max_tau1, max_tau2, max_epsilon = get_max_indeces_vmap(sample_rngs,
                                                               self.search_spaces,
                                                               self.ws,
                                                               self.config['num_envs'],
                                                               self.config['val_envs'],
                                                               self.bandit_params['num_bandits'])
        if only_index:
            return max_tau1, max_tau2, max_epsilon
        return (max_tau1, max_tau2, max_epsilon, None, None, None, None, None, None)


    def update_and_get_data(self, data_collector, train_indeces, train_returns, train_envs, val_envs):
        new_train_indeces, new_val_indeces, index_data = None, None, None
        if data_collector.frame_count >= self.config['per_min_frames'] and self.config['bandits'] and train_envs:
            self.update_bandits(train_indeces, train_returns, train_envs)
            self.index_data = self.get_index_data()
            index_data = (np.array(x) for x in self.index_data)
        if train_envs:
            new_train_indeces = np.array(self.sample_all_candidates(train_envs))
        new_val_indeces = np.array([self.index_data[:3] for _ in val_envs]) if val_envs else None
        return new_train_indeces, new_val_indeces, index_data


@jax.jit
def update_vmap(xs, gs, ws, Ns, search_spaces, lrs, bools):
    return jax.vmap(update_)(xs, gs, ws, Ns, search_spaces, lrs, bools)


@jax.jit
def update_(x, g, w, N, search_space, lr, bools):
    i = jnp.digitize(x, search_space)
    w = w.at[i-1].add(lr * (g - w[i-1]) * bools)
    N = N.at[i-1].add(bools)
    return w, N


@functools.partial(jax.jit, static_argnums=(6))
def sample_vmap(sample_rngs, ws, Ns, search_spaces, modes, masks, d):
    return jax.vmap(sample_, in_axes=[0, 0, 0, 0, 0, 0, None])(sample_rngs, ws, Ns, search_spaces, modes, masks, d)


@functools.partial(jax.jit, static_argnums=(6))
def sample_(sample_rng, w, N, search_space, mode, mask, d):
    epsilon = jnp.finfo(jnp.float32).eps
    w = jnp.where(mask, w, 0)
    N = jnp.where(mask, N, 0)
    mean_w = jnp.sum(w) / jnp.sum(mask)
    std_w = jnp.sqrt(jnp.sum(mask * (w - mean_w) ** 2) / jnp.sum(mask))

    scores = (w - mean_w) / (std_w + epsilon)
    scores += jnp.sqrt(jnp.log(1 + jnp.sum(N)) / (1 + N))
    scores = jnp.where(mask, scores, -jnp.inf)

    def true_fun():
        indices = jnp.argsort(scores)[-d:]
        intervals = jnp.stack((search_space[indices], search_space[indices + 1]), axis=-1)
        keys = jax.random.split(sample_rng, d)
        minvals = intervals[:, 0].reshape((d, 1))
        maxvals = intervals[:, 1].reshape((d, 1))

        canditates = jax.vmap(
            lambda k, minv, maxv: jax.random.uniform(k, shape=(1,), minval=minv, maxval=maxv),
            in_axes=(0, 0, 0)
        )(keys, minvals, maxvals).ravel()
        return canditates

    def false_fun():
        probabilities = jnp.exp(scores) / jnp.sum(jnp.exp(scores))
        indices = jax.random.choice(sample_rng, len(scores), shape=(d,), replace=False, p=probabilities)
        intervals = jnp.stack((search_space[indices], search_space[indices + 1]), axis=-1)
        keys = jax.random.split(sample_rng, d)
        minvals = intervals[:, 0].reshape((d, 1))
        maxvals = intervals[:, 1].reshape((d, 1))

        canditates = jax.vmap(
            lambda k, minv, maxv: jax.random.uniform(k, shape=(1,), minval=minv, maxval=maxv),
            in_axes=(0, 0, 0)
        )(keys, minvals, maxvals).ravel()
        return canditates

    candidates = jax.lax.cond(mode, true_fun, false_fun)
    return candidates


@jax.jit
def sample_candidates_vmap(sample_rngs, all_candidates):
    return jax.vmap(sample_candidates_)(sample_rngs, all_candidates)


@jax.jit
def sample_candidates_(sample_rng, candidates):
    candidates = candidates.at[:, 0:2, :].set(jnp.exp(candidates[:, 0:2, :]) - 1)
    candidates = candidates.transpose(1, 0, 2).reshape(3, -1)
    sampled_candidates = jax.vmap(lambda rng, row: jax.random.choice(rng, row, shape=(1,)))(sample_rng, candidates)
    return sampled_candidates.squeeze()


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def get_max_indeces_vmap(sample_rngs, search_spaces, ws, num_envs, val_envs, num_bandits):
    search_spaces = search_spaces.reshape((num_envs - val_envs, num_bandits, 3, -1))
    search_spaces = search_spaces.transpose(2, 0, 1, 3)
    search_spaces = search_spaces.reshape(3, (num_envs - val_envs) * num_bandits, -1)
    ws = ws.reshape((num_envs - val_envs, num_bandits, 3, -1))
    ws = ws.transpose(2, 0, 1, 3)
    ws = ws.reshape(3, (num_envs - val_envs) * num_bandits, -1)
    max_indeces = jax.vmap(get_max_indeces_)(sample_rngs, search_spaces, ws)
    max_indeces = max_indeces.at[0:2].set(jnp.exp(max_indeces[0:2]) - 1)
    return max_indeces


@jax.jit
def get_max_indeces_(sample_rng, search_spaces, ws):
    max_indeces = jnp.argmax(ws, axis=1)
    sample_rngs = jax.random.split(sample_rng, search_spaces.shape[0])
    samples = jax.vmap(get_elements)(search_spaces, max_indeces, sample_rngs)
    return jnp.mean(samples)


@jax.jit
def get_elements(search_space, max_index, sample_rng):
    interval = jax.lax.dynamic_slice(search_space, (max_index,), (2,))
    sample = jax.random.uniform(sample_rng, minval=interval[0], maxval=interval[1])
    return sample
