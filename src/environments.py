import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from laser_hockey_env import LaserHockeyEnv, BasicOpponent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Bybit.src.crypto_env import CryptoEnv

#region init

class Environments:
    def __init__(self, training_class, training=True, render_mode='rgb_array'):
        self.env_name = training_class.env_name
        self.config = training_class.config
        self.log_dir = training_class.log_dir
        self.training = training
        self.render_mode = render_mode
        np.random.seed(self.config['jax_seed'])
        self.rng = np.random.default_rng(self.config['jax_seed'])
        self.envs = self.get_environments()


    def get_environments(self):
        if self.training:
            if self.env_name == 'Crypto-v0':
                envs = CryptoEnv(train_configs=self.config, train_log_dir=self.log_dir)
            elif self.env_name == 'LaserHockey-v0':
                self.basic = BasicOpponent()
                self.action_space = self.get_action_space()
                train_envs = [lambda: LaserHockeyEnv(mode_='train', seed=self.rng.randint(1e9)) for _ in range(self.config['train_envs'])]
                val_envs = [lambda: LaserHockeyEnv(mode_='val', seed=self.rng.randint(1e9)) for _ in range(self.config['val_envs'])]
                envs = train_envs + val_envs
                envs = SyncVectorEnv(envs)
            else:
                envs = [lambda: gym.make(self.env_name) for _ in range(self.config['train_envs'] + self.config['val_envs'])]
                envs = SyncVectorEnv(envs)
        else:
            if self.env_name == 'Crypto-v0':
                envs = CryptoEnv(train_configs=self.config, train_log_dir=self.log_dir, mode='eval')
            elif self.env_name == 'LaserHockey-v0':
                self.basic = BasicOpponent()
                self.action_space = self.get_action_space()
                envs = LaserHockeyEnv(mode_='val')
            else:
                envs = gym.make(self.env_name, render_mode=self.render_mode)
        return envs

#endregion
#region main

    def reset(self, random=False):
        if self.env_name == 'Crypto-v0':
            observations, infos = self.envs.reset()
            return observations, infos
        else:
            if random:
                observations, infos = self.envs.reset()
            else:
                observations, infos = self.envs.reset(seed=self.config['jax_seed'])
            if observations.ndim == 1:
                observations = observations[np.newaxis, :]
            self.observations = np.tile(observations[:, np.newaxis, :], (1, self.config['observation_length'], 1))
            if self.env_name == 'LaserHockey-v0' and self.render_mode == 'human':
                self.envs.render()
            return self.observations, infos


    def step(self, actions, infos):
        if self.env_name == 'Crypto-v0':
            observations, rewards, terminated, truncated, new_infos = self.envs.step(actions)
            return observations, rewards, terminated, truncated, new_infos
        else:
            converted_actions = self.convert_actions(actions, infos)
            observations, rewards, terminated, truncated, new_infos = self.envs.step(converted_actions)
            if observations.ndim == 1:
                observations = observations[np.newaxis, :]
            
            reset_mask = np.atleast_1d(terminated) | np.atleast_1d(truncated)
            if reset_mask.any():
                reset_observations = observations[reset_mask]
                if reset_observations.ndim == 1:
                    reset_observations = reset_observations[np.newaxis, :]
                self.observations[reset_mask] = np.tile(reset_observations[:, np.newaxis, :], (1, self.config['observation_length'], 1))
            
            if self.config['observation_length'] > 1:
                self.observations[~reset_mask, :-1, :] = self.observations[~reset_mask, 1:, :]
            self.observations[~reset_mask, -1, :] = observations[~reset_mask]
            
            if self.env_name == 'LaserHockey-v0' and self.render_mode == 'human':
                self.envs.render()
            return self.observations, rewards, terminated, truncated, new_infos


    def render(self):
        return self.envs.render()


    def close(self):
        return self.envs.close()

#endregion
#region helper

    def save_environments(self):
        if self.env_name == 'Crypto-v0':
            self.envs.save_configs()
            self.envs.save_env_states()
    

    def preprocess_observations(self, observations):
        if self.env_name == 'LunarLander-v2':
            return self.scale_observations(observations)
        else:
            return observations
    
    
    def scale_observations(self, x):
        x = x * self.config['observations_scaling_x']
        x_log = np.log(np.abs(x) + 1)
        x = np.where(np.sign(x) > 0, x_log, -x_log)
        x = x * self.config['observations_scaling_y']
        return x


    def convert_actions(self, actions, infos):
        if self.env_name == 'LaserHockey-v0':
            if np.isscalar(infos['obs_agent_two'][0]):
                obs_agent2s = infos['obs_agent_two'][np.newaxis, :]
            else:
                obs_agent2s = np.vstack(infos['obs_agent_two'])
            basic_actions = self.basic.act(obs_agent2s)
            converted_actions = self.action_space[actions]
            total_actions = np.hstack([converted_actions, basic_actions])
            if self.render_mode == 'human':
                return total_actions[0]
            else:
                return total_actions
        else:
            return actions
        

    def get_action_space(self):
        values = [-1, 0, 1]
        x, y, z = np.meshgrid(values, values, values, indexing='ij')
        result = np.stack((x, y, z), axis=-1)
        reshaped_result = result.reshape(-1, result.shape[-1])
        return reshaped_result