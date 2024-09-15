import asyncio
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import functools
import orbax.checkpoint as ocp
import optax
from architectures_jax import DenseModelJax, TransformerModelJax
from s5 import S5



class Actor:
    def __init__(self, training_class):
        self.count = 0
        self.config = training_class.config
        self.log_dir = f'{training_class.log_dir}/models'
        self.main_rng = jax.random.PRNGKey(self.config['jax_seed'])
        self.architecture_parameters = self.config['parameters'][self.config['architecture']]
        if self.config['architecture'] == 'dense_jax':
            self.architecture = DenseModelJax(self.architecture_parameters)
        elif self.config['architecture'] == 'transformer':
            self.architecture = TransformerModelJax(self.architecture_parameters)
        elif self.config['architecture'] == 'S5':
            self.architecture = S5(self.architecture_parameters).s5
        self.actor1_params = self.initialize_parameters()
        self.actor2_params = self.initialize_parameters()
        self.actor1 = train_state.TrainState.create(apply_fn=self.architecture.apply, params=self.actor1_params, tx=optax.adam(0))
        self.actor2 = train_state.TrainState.create(apply_fn=self.architecture.apply, params=self.actor2_params, tx=optax.adam(0))
        self.checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(write_tree_metadata=True))
    

    def initialize_parameters(self):
        self.main_rng, init_rng, drop_rng = jax.random.split(self.main_rng, num=3)
        if self.config['architecture'] == 'S5':
            return self.architecture.init({"params": init_rng, "dropout": drop_rng}, jnp.ones(self.architecture_parameters['input_shape']), None)["params"]
        else:
            return self.architecture.init({"params": init_rng, "dropout": drop_rng}, jnp.ones(self.architecture_parameters['input_shape']))['params']


    async def pull_weights(self, learner=None, training=True, target_eval=False):
        if self.count % self.config['d_pull'] == 0:
            if training and self.config['actor_target_network']:
                self.actor1 = self.actor1.replace(params=learner.target1.params)
                self.actor2 = self.actor2.replace(params=learner.target2.params)
            elif training and not self.config['actor_target_network']:
                self.actor1 = self.actor1.replace(params=learner.learner1.params)
                self.actor2 = self.actor2.replace(params=learner.learner2.params)
            elif target_eval:
                self.actor1 = await asyncio.get_event_loop().run_in_executor(None, self.checkpointer.restore, f'{self.log_dir}/target1', self.actor1)
                self.actor2 = await asyncio.get_event_loop().run_in_executor(None, self.checkpointer.restore, f'{self.log_dir}/target2', self.actor2)
            else:
                self.actor1 = await asyncio.get_event_loop().run_in_executor(None, self.checkpointer.restore, f'{self.log_dir}/learner1', self.actor1)
                self.actor2 = await asyncio.get_event_loop().run_in_executor(None, self.checkpointer.restore, f'{self.log_dir}/learner2', self.actor2)
        self.count += 1
        

    def get_actions(self, observations, indeces, stochastic=True, random_=False, training=False, return_values=False):
        self.main_rng, act_rng = jax.random.split(self.main_rng)
        actions, action_probs, q_action, v = calculate_actions(act_rng,
                                                  self.actor1,
                                                  self.actor2,
                                                  observations,
                                                  indeces,
                                                  self.architecture_parameters['action_dim'],
                                                  self.config['train_envs'],
                                                  self.config['val_envs'],
                                                  stochastic,
                                                  random_,
                                                  training,
                                                  return_values)
        actions = np.array(actions)
        action_probs = np.array(action_probs)
        if return_values:
            q_action = np.array(q_action)
            v = np.array(v)
            return actions, action_probs, q_action, v
        else:
            return actions, action_probs



@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11))
def calculate_actions(act_rng, actor1, actor2, observations, indices, action_dim, train_envs, val_envs, stochastic, random_, training, return_values):
    v1, a1 = actor1.apply_fn({'params': actor1.params}, observations, False)
    v2, a2 = actor2.apply_fn({'params': actor2.params}, observations, False)
    v1 = v1[:, -1]
    v2 = v2[:, -1]
    a1 = a1[:, -1]
    a2 = a2[:, -1]

    tau1 = indices[:, 0:1]
    tau2 = indices[:, 1:2]
    epsilon = indices[:, 2:3]
    softmax_1 = jax.nn.softmax(a1 * tau1, axis=1)
    softmax_2 = jax.nn.softmax(a2 * tau2, axis=1)
    policy = epsilon * softmax_1 + (1 - epsilon) * softmax_2

    if random_:
        actions = jax.random.randint(act_rng, (train_envs + val_envs,), 0, action_dim)
        action_probs = jnp.ones_like(actions) / action_dim
    elif stochastic:
        actions = jax.random.categorical(act_rng, jnp.log(policy), axis=-1)
        action_probs = jnp.take_along_axis(policy, actions[:, None], axis=1)
    else:
        actions = jnp.argmax(policy, axis=1)
        action_probs = jnp.take_along_axis(policy, actions[:, None], axis=1)
    if training:
        greedy_actions = jnp.argmax(policy, axis=1)
        actions = actions.at[-(val_envs // 2):].set(greedy_actions[-(val_envs // 2):])

    if return_values:
        q1 = a1 + v1
        q2 = a2 + v2
        v = epsilon * v1 + (1 - epsilon) * v2
        q = epsilon * q1 + (1 - epsilon) * q2
        q_action = jnp.take_along_axis(q, actions[:, None], axis=1)
        return actions, action_probs, q_action, v
    else:
        return actions, action_probs, None, None
