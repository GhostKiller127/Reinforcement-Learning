import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import serialization
from flax.training import train_state
import optax
import functools
from architectures_jax import DenseModelJax



class Learner:
    def __init__(self, training_class):
        self.update_count = 0
        self.step_count = 0
        self.config = training_class.config
        self.log_dir = f'{training_class.log_dir}/models'
        self.architecture_parameters = self.config['parameters'][self.config['architecture']]
        if self.config['architecture'] == 'dense':
            self.main_rng = random.PRNGKey(self.config['jax_seed'])
            self.architecture = DenseModelJax(self.architecture_parameters)
            self.learner1_params = self.initialize_parameters()
            self.learner2_params = self.initialize_parameters()
            self.target1_params = self.initialize_parameters()
            self.target2_params = self.initialize_parameters()
            self.learning_rate_fn = self.create_learning_rate_fn()
            self.learner1 = self.create_learner(self.learning_rate_fn, self.learner1_params)
            self.learner2 = self.create_learner(self.learning_rate_fn, self.learner2_params)
            self.target1 = self.create_learner(self.learning_rate_fn, self.target1_params)
            self.target2 = self.create_learner(self.learning_rate_fn, self.target2_params)
            if self.config['load_run'] is None:
                os.makedirs(self.log_dir, exist_ok=True)
                self.save_weights_and_data_collector()
            else:
                self.load_weights()
            self.update_target_weights()


    def create_learning_rate_fn(self):
        if self.config['load_run'] is not None:
            self.config['warmup_steps'] = 0
            initial_lr, end_lr = self.config["learning_rate"], self.config["learning_rate"]
        elif self.config["lr_finder"]:
            initial_lr, end_lr = 1e-8, 1e-2
            warmup_fn = optax.exponential_decay(init_value=initial_lr, decay_rate=end_lr/initial_lr, transition_steps=self.config['warmup_steps'])
            return warmup_fn
        else:
            self.config['warmup_steps'] = int(self.config['warmup_steps'] / 4)
            initial_lr, end_lr = self.config['learning_rate'] / 1e3, self.config['learning_rate']
        warmup_fn = optax.exponential_decay(init_value=initial_lr, decay_rate=end_lr/initial_lr, transition_steps=self.config['warmup_steps'])
        constant_fn = optax.constant_schedule(value=self.config['learning_rate'])
        schedule_fn = optax.join_schedules(schedules=[warmup_fn, constant_fn], boundaries=[self.config['warmup_steps']])
        # cosine_fn = optax.cosine_decay_schedule(init_value=self.config['learning_rate'], decay_steps=self.config['decay_steps'])
        # schedule_fn = optax.join_schedules(schedules=[warmup_fn, constant_fn, cosine_fn], boundaries=[self.config['warmup_steps'], self.configs['constant_steps']])
        return schedule_fn
    

    def save_weights_and_data_collector(self, data_collector=None):
        if self.update_count % self.config['d_push'] == 0:
            with open(f'{self.log_dir}/learner1.pkl', 'wb') as f:
                f.write(serialization.to_bytes(self.learner1))
            with open(f'{self.log_dir}/learner2.pkl', 'wb') as f:
                f.write(serialization.to_bytes(self.learner2))
            if data_collector is not None and not self.config['lr_finder']:
                data_collector.save_data_collector()


    def load_weights(self):
        with open(f'{self.log_dir}/learner1.pkl', 'rb') as f:
            loaded_state = serialization.from_bytes(train_state.TrainState, f.read())
            self.learner1 = self.learner1.replace(params=loaded_state['params'], opt_state=loaded_state['opt_state'])
        with open(f'{self.log_dir}/learner2.pkl', 'rb') as f:
            loaded_state = serialization.from_bytes(train_state.TrainState, f.read())
            self.learner2 = self.learner2.replace(params=loaded_state['params'], opt_state=loaded_state['opt_state'])


    def update_target_weights(self):
        if self.update_count % self.config['d_target'] == 0:
            self.target1 = self.target1.replace(params=self.learner1.params)
            self.target2 = self.target2.replace(params=self.learner2.params)
    

    def initialize_parameters(self):
        self.main_rng, init_rng, drop_rng = random.split(self.main_rng, num=3)
        if self.config['architecture'] == 'S5':
            return self.architecture.init({"params": init_rng, "dropout": drop_rng}, np.ones(self.architecture_parameters['input_shape']), None)["params"]
        else:
            return self.architecture.init({"params": init_rng, "dropout": drop_rng}, np.ones(self.architecture_parameters['input_shape']))['params']


    def create_learner(self, learning_rate_fn, parameters):
        tx = optax.adamw(learning_rate_fn, weight_decay=self.config['weight_decay'])
        return train_state.TrainState.create(apply_fn=self.architecture.apply, params=parameters, tx=tx)
    

    def train_batch(self, batch, data_collector):
        self.learner1, self.learner2, loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, grad_norm1, grad_norm2, lr, rt1, rt2, vt1, vt2, pt1, pt2, rtd1, rtd2 = train_batch_(self.learner1, self.learner2, self.target1, self.target2, batch, self.update_count,
                                                                                                self.config['reward_scaling_1'],
                                                                                                self.config['reward_scaling_2'],
                                                                                                self.config['c_clip'],
                                                                                                self.config['rho_clip'],
                                                                                                self.config['batch_size'],
                                                                                                self.config['bootstrap_length'],
                                                                                                self.config['sequence_length'],
                                                                                                self.config['discount'],
                                                                                                self.learning_rate_fn,
                                                                                                self.config['v_loss_scaling'],
                                                                                                self.config['q_loss_scaling'],
                                                                                                self.config['p_loss_scaling'])
        
        self.update_count += 1
        self.save_weights_and_data_collector(data_collector)
        self.update_target_weights()
        return loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, grad_norm1, grad_norm2, lr, rt1, rt2, vt1, vt2, pt1, pt2, rtd1, rtd2


    def check_and_update(self, data_collector):
        losses = None
        if self.step_count % self.config['update_frequency'] == 0 and data_collector.frame_count >= self.config['per_min_frames']:
            batched_sequences, sequence_indeces = data_collector.load_batched_sequences()
            loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, gradient_norm1, gradient_norm2, learning_rate, rt1, rt2, vt1, vt2, pt1, pt2, rtd1, rtd2 = self.train_batch(batched_sequences, data_collector)
            losses = loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, gradient_norm1, gradient_norm2, learning_rate
            losses = tuple(jax.device_get(loss) for loss in losses)
            targets = rt1, rt2, vt1, vt2, pt1, pt2
            targets = tuple(jax.device_get(target) for target in targets)
            data_collector.update_priorities(rtd1, rtd2, sequence_indeces)
        self.step_count += 1
        if losses is None:
            return None, None
        else:
            return losses, targets



@functools.partial(jax.jit, static_argnums=(1,))
def scale_value1(x, scale):
    x_log = jnp.log(jnp.abs(x) + 1) * scale
    x = jnp.where(jnp.sign(x) > 0, x_log * 2, x_log * -1)
    return x


@functools.partial(jax.jit, static_argnums=(1,))
def scale_value2(x, scale):
    x = (jnp.sign(x) * ((jnp.abs(x) + 1.)**(1/2) - 1.) + 0.001 * x) * scale
    return x


@functools.partial(jax.jit, static_argnums=(1,))
def invert_scale1(h, scale):
    h = h / scale
    h_ = jnp.where(jnp.sign(h) > 0, jnp.abs(h) / 2, jnp.abs(h))
    h_ = jnp.sign(h) * (jnp.exp(h_) - 1)
    return h_


@functools.partial(jax.jit, static_argnums=(1,))
def invert_scale2(h, scale):
    h = h / scale
    h = jnp.sign(h) * ((((1 + 4*0.001*(jnp.abs(h) + 1 + 0.001))**(1/2) - 1) / (2*0.001))**2 - 1)
    return h


@functools.partial(jax.jit, static_argnums=(1,))
def moving_window(matrix, window_shape):
    matrix_height = matrix.shape[0]
    matrix_width = matrix.shape[1]

    window_height = window_shape[0]
    window_width = window_shape[1]

    startsy = jnp.arange(matrix_height - window_height + 1)
    startsx = jnp.arange(matrix_width - window_width + 1)
    starts_xy = jnp.dstack(jnp.meshgrid(startsy, startsx)).reshape(-1, 2)

    return jax.vmap(lambda start: jax.lax.dynamic_slice(matrix, (start[0], start[1]), (window_height, window_width)))(starts_xy)


@jax.jit
def calculate_values_(params1,
                      params2,
                      learner1,
                      learner2,
                      target1,
                      target2,
                      batch):
    
    v1, a1 = learner1.apply_fn({'params': params1}, batch['o'])
    v2, a2 = learner2.apply_fn({'params': params2}, batch['o'])

    v1_, a1_ = target1.apply_fn({'params': target1.params}, batch['o'])
    v2_, a2_ = target2.apply_fn({'params': target2.params}, batch['o'])
    
    i = jnp.array(batch['i'], dtype=jnp.float32)
    a = jnp.expand_dims(jnp.array(batch['a'], dtype=jnp.int32), axis=2)
    tau1 = i[:, :, 0:1]
    tau2 = i[:, :, 1:2]
    epsilon = i[:, :, 2:3]
    softmax_1 = jax.nn.softmax(a1 * tau1, axis=2)
    softmax_2 = jax.nn.softmax(a2 * tau2, axis=2)
    softmax_1_ = jax.nn.softmax(a1_ * tau1, axis=2)
    softmax_2_ = jax.nn.softmax(a2_ * tau2, axis=2)
    policy1 = epsilon * softmax_1 + (1 - epsilon) * jax.lax.stop_gradient(softmax_2)
    policy2 = epsilon * jax.lax.stop_gradient(softmax_1) + (1 - epsilon) * softmax_2
    policy_ = epsilon * softmax_1_ + (1 - epsilon) * softmax_2_
    p1 = jnp.take_along_axis(policy1, a, axis=2)
    p2 = jnp.take_along_axis(policy2, a, axis=2)
    
    a1 = jnp.take_along_axis(a1, a, axis=2) + jnp.sum(jax.lax.stop_gradient(policy1) * a1, axis=2, keepdims=True)
    a2 = jnp.take_along_axis(a2, a, axis=2) + jnp.sum(jax.lax.stop_gradient(policy2) * a2, axis=2, keepdims=True)
    q1 = a1 + jax.lax.stop_gradient(v1)
    q2 = a2 + jax.lax.stop_gradient(v2)
    a1_ = a1_ + jnp.sum(policy_ * a1_, axis=2, keepdims=True)
    a2_ = a2_ + jnp.sum(policy_ * a2_, axis=2, keepdims=True)
    q1_ = a1_ + v1_
    q2_ = a2_ + v2_
    q1_m = jnp.sum(jax.lax.stop_gradient(policy1) * q1_, axis=2, keepdims=True)
    q2_m = jnp.sum(jax.lax.stop_gradient(policy2) * q2_, axis=2, keepdims=True)
    q1_ = jnp.take_along_axis(q1_, a, axis=2)
    q2_ = jnp.take_along_axis(q2_, a, axis=2)
    
    return v1, v2, q1, q2, p1, p2, v1_, v2_, q1_, q2_, q1_m, q2_m


@functools.partial(jax.jit, static_argnums=(8, 9, 10, 11, 12, 13, 14))
def calculate_retrace_targets_(q1,
                               q2,
                               q1_,
                               q2_,
                               q1_m,
                               q2_m,
                               p,
                               batch,
                               reward_scaling_1,
                               reward_scaling_2,
                               c_clip,
                               batch_size,
                               bootstrap_length,
                               sequence_length,
                               discount):
    
    q1 = jax.lax.stop_gradient(q1).squeeze()
    q2 = jax.lax.stop_gradient(q2).squeeze()
    p = jax.lax.stop_gradient(p)
    q1_ = invert_scale1(q1_.squeeze(), reward_scaling_1)
    q2_ = invert_scale2(q2_.squeeze(), reward_scaling_2)
    q1_m = invert_scale1(q1_m.squeeze(), reward_scaling_1)
    q2_m = invert_scale2(q2_m.squeeze(), reward_scaling_2)

    c = jnp.minimum(p / batch['a_p'], c_clip).squeeze()
    c = moving_window(c[:, :-2], (batch_size, bootstrap_length)).transpose((1, 0, 2))
    c = c.at[:, :, 0].set(1)
    c = jnp.cumprod(c, axis=2)
    gamma = discount * jnp.ones((batch_size, sequence_length, bootstrap_length))
    gamma = gamma.at[:, :, 0].set(1)
    gamma = jnp.cumprod(gamma, axis=2)

    mask = jnp.logical_or(batch['d'], batch['t'])[:, :-2]
    mask = moving_window(mask, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    mask = jnp.cumsum(mask, axis=2)
    mask_e = mask.astype(bool)
    mask_t = jnp.roll(mask_e, shift=1, axis=2)
    mask_t = mask_t.at[:, :, 0].set(False)
    mask_e = ~mask_e
    mask_t = ~mask_t

    td1_t = batch['r'][:, :-2] - q1_[:, :-2]
    td2_t = batch['r'][:, :-2] - q2_[:, :-2]
    td1_e = discount * q1_m[:, 1:-1]
    td2_e = discount * q2_m[:, 1:-1]
    td1_t = moving_window(td1_t, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    td2_t = moving_window(td2_t, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    td1_e = moving_window(td1_e, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    td2_e = moving_window(td2_e, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    td1 = td1_t * mask_t + td1_e * mask_e
    td2 = td2_t * mask_t + td2_e * mask_e
    rt1 = q1_[:, :sequence_length] + jnp.sum(gamma * c * td1, axis=2)
    rt2 = q2_[:, :sequence_length] + jnp.sum(gamma * c * td2, axis=2)

    rt1 = scale_value1(rt1, reward_scaling_1)
    rt2 = scale_value2(rt2, reward_scaling_2)
    rtd1 = rt1 - q1[:, :sequence_length]
    rtd2 = rt2 - q2[:, :sequence_length]
    rt1 = jnp.expand_dims(rt1, axis=2)
    rt2 = jnp.expand_dims(rt2, axis=2)
    
    return rt1, rt2, rtd1, rtd2


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def calculate_vtrace_targets_(v1_,
                              v2_,
                              p,
                              batch,
                              reward_scaling_1,
                              reward_scaling_2,
                              c_clip,
                              rho_clip,
                              batch_size,
                              bootstrap_length,
                              sequence_length,
                              discount):
    
    v1_ = invert_scale1(v1_.squeeze(), reward_scaling_1)
    v2_ = invert_scale2(v2_.squeeze(), reward_scaling_2)
    p = jax.lax.stop_gradient(p)
    c = jnp.minimum(p / batch['a_p'], c_clip).squeeze()
    rho_t = jnp.minimum(p / batch['a_p'], rho_clip).squeeze()

    rho = moving_window(rho_t[:, :-1], (batch_size, bootstrap_length)).transpose((1, 0, 2))
    c = moving_window(c[:, :-1], (batch_size, bootstrap_length)).transpose((1, 0, 2))
    c = jnp.roll(c, shift=1, axis=2)
    c = c.at[:, :, 0].set(1)
    c = jnp.cumprod(c, axis=2)

    gamma = discount * jnp.ones((batch_size, sequence_length + 1, bootstrap_length))
    gamma = gamma.at[:, :, 0].set(1)
    gamma = jnp.cumprod(gamma, axis=2)

    mask = jnp.logical_or(batch['d'], batch['t'])[:, :-1]
    mask = moving_window(mask, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    mask = jnp.cumsum(mask, axis=2)
    mask_e = mask.astype(bool)
    mask_t = jnp.roll(mask_e, shift=1, axis=2)
    mask_t = mask_t.at[:, :, 0].set(False)
    mask_e = ~mask_e
    mask_t = ~mask_t

    td1_t = batch['r'][:, :-1] - v1_[:, :-1]
    td2_t = batch['r'][:, :-1] - v2_[:, :-1]
    td1_e = discount * v1_[:, 1:]
    td2_e = discount * v2_[:, 1:]
    td1_t = moving_window(td1_t, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    td2_t = moving_window(td2_t, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    td1_e = moving_window(td1_e, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    td2_e = moving_window(td2_e, (batch_size, bootstrap_length)).transpose((1, 0, 2))
    td1 = td1_t * mask_t + td1_e * mask_e
    td2 = td2_t * mask_t + td2_e * mask_e
    vt1 = v1_[:, :sequence_length + 1] + jnp.sum(gamma * c * rho * td1, axis=2)
    vt2 = v2_[:, :sequence_length + 1] + jnp.sum(gamma * c * rho * td2, axis=2)

    pt1 = (rho_t[:, :sequence_length] *
           (batch['r'][:, :sequence_length] + discount * vt1[:, 1:] - v1_[:, :sequence_length]))
    pt2 = (rho_t[:, :sequence_length] *
           (batch['r'][:, :sequence_length] + discount * vt2[:, 1:] - v2_[:, :sequence_length]))

    vt1 = scale_value1(vt1, reward_scaling_1)
    vt2 = scale_value2(vt2, reward_scaling_2)
    pt1 = scale_value1(pt1, reward_scaling_1)
    pt2 = scale_value2(pt2, reward_scaling_2)

    vt1 = jnp.expand_dims(vt1[:,:-1], axis=2)
    vt2 = jnp.expand_dims(vt2[:,:-1], axis=2)
    pt1 = jnp.expand_dims(pt1, axis=2)
    pt2 = jnp.expand_dims(pt2, axis=2)

    return vt1, vt2, pt1, pt2


@functools.partial(jax.jit, static_argnums=(12,))
def calculate_losses_(v1, v2, q1, q2, p1, p2, rt1, rt2, vt1, vt2, pt1, pt2,
                                                                        sequence_length,
                                                                        v_loss_scaling,
                                                                        q_loss_scaling,
                                                                        p_loss_scaling):
    
    v_loss1 = jnp.mean((vt1 - v1[:,:sequence_length])**2)
    v_loss2 = jnp.mean((vt2 - v2[:,:sequence_length])**2)
    q_loss1 = jnp.mean((rt1 - q1[:,:sequence_length])**2)
    q_loss2 = jnp.mean((rt2 - q2[:,:sequence_length])**2)
    p_loss1 = jnp.mean(pt1 * -jnp.log(p1[:,:sequence_length] + 1e-6))
    p_loss2 = jnp.mean(pt2 * -jnp.log(p2[:,:sequence_length] + 1e-6))

    loss1 = ((v_loss_scaling * v_loss1 + q_loss_scaling * q_loss1 + p_loss_scaling * p_loss1) /
            (v_loss_scaling + q_loss_scaling + p_loss_scaling))
    loss2 = ((v_loss_scaling * v_loss2 + q_loss_scaling * q_loss2 + p_loss_scaling * p_loss2) /
            (v_loss_scaling + q_loss_scaling + p_loss_scaling))
    loss = loss1 + loss2

    return loss, loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2


@functools.partial(jax.jit, static_argnums=(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
def train_batch_(learner1, learner2, target1, target2, batch, step,
                                                            reward_scaling_1,
                                                            reward_scaling_2,
                                                            c_clip,
                                                            rho_clip,
                                                            batch_size,
                                                            bootstrap_length,
                                                            sequence_length,
                                                            discount,
                                                            learning_rate_fn,
                                                            v_loss_scaling,
                                                            q_loss_scaling,
                                                            p_loss_scaling):
    def loss_fn(params1, params2):
        
        v1, v2, q1, q2, p1, p2, v1_, v2_, q1_, q2_, q1_m, q2_m = calculate_values_(params1,
                                                                                    params2,
                                                                                    learner1,
                                                                                    learner2,
                                                                                    target1,
                                                                                    target2,
                                                                                    batch)
        
        rt1, rt2, rtd1, rtd2 = calculate_retrace_targets_(q1,
                                                        q2,
                                                        q1_,
                                                        q2_,
                                                        q1_m,
                                                        q2_m,
                                                        p1,
                                                        batch,
                                                        reward_scaling_1,
                                                        reward_scaling_2,
                                                        c_clip,
                                                        batch_size,
                                                        bootstrap_length,
                                                        sequence_length,
                                                        discount)
        
        vt1, vt2, pt1, pt2 = calculate_vtrace_targets_(v1_,
                                                        v2_,
                                                        p1,
                                                        batch,
                                                        reward_scaling_1,
                                                        reward_scaling_2,
                                                        c_clip,
                                                        rho_clip,
                                                        batch_size,
                                                        bootstrap_length,
                                                        sequence_length,
                                                        discount)
        
        loss, loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2 = calculate_losses_(v1, v2, q1, q2, p1, p2, rt1, rt2, vt1, vt2, pt1, pt2,
                                                                                                                                    sequence_length,
                                                                                                                                    v_loss_scaling,
                                                                                                                                    q_loss_scaling,
                                                                                                                                    p_loss_scaling)
        
        rest = loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, rt1, rt2, vt1, vt2, pt1, pt2, rtd1, rtd2
        return loss, rest
    
    (loss, rest), grads = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)(learner1.params, learner2.params)
    grads1, grads2 = grads
    loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, rt1, rt2, vt1, vt2, pt1, pt2, rtd1, rtd2 = rest

    learner1 = learner1.apply_gradients(grads=grads1)
    learner2 = learner2.apply_gradients(grads=grads2)
    lr = learning_rate_fn(step)
    grad_norm1, grad_norm2 = jnp.empty(0), jnp.empty(0)
    for grad1, grad2 in zip(jax.tree_leaves(grads1), jax.tree_leaves(grads2)):
        grad_norm1 = jnp.concatenate([grad_norm1, jnp.ravel(grad1)])
        grad_norm2 = jnp.concatenate([grad_norm2, jnp.ravel(grad2)])
    grad_norm1, grad_norm2 = jnp.linalg.norm(grad_norm1), jnp.linalg.norm(grad_norm2)

    return learner1, learner2, loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, grad_norm1, grad_norm2, lr, rt1, rt2, vt1, vt2, pt1, pt2, rtd1, rtd2