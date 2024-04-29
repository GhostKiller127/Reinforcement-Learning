import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import laser_hockey_env as lh


class Environments:
    def __init__(self, training_class, training=True, render_mode='rgb_array'):
        self.env_name = training_class.env_name
        self.config = training_class.config
        self.training = training
        self.render_mode = render_mode
        self.envs = self.get_environments()
        self.basic = lh.BasicOpponent()
        self.action_space = self.get_action_space()


    def get_environments(self):
        if self.training:
            if self.env_name == 'LaserHockey-v0':
                train_envs = [lambda: lh.LaserHockeyEnv(mode_='train') for _ in range(self.config['num_envs'] - self.config['val_envs'])]
                val_envs = [lambda: lh.LaserHockeyEnv(mode_='val') for _ in range(self.config['val_envs'])]
                envs = train_envs + val_envs
                envs = AsyncVectorEnv(envs)
            else:
                envs = [lambda: gym.make(self.env_name) for _ in range(self.config['num_envs'])]
                envs = SyncVectorEnv(envs)
        else:
            if self.env_name == 'LaserHockey-v0':
                envs = lh.LaserHockeyEnv(mode_='val')
            else:
                envs = gym.make(self.env_name, render_mode=self.render_mode)
        return envs


    def reset(self):
        observations, infos = self.envs.reset()
        self.observations = np.tile(observations[:, np.newaxis, :], (1, self.config['observation_length'], 1))
        if self.env_name == 'LaserHockey-v0' and self.render_mode == 'human':
            self.envs.render()
        return self.observations, infos


    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.envs.step(actions)
        if self.config['observation_length'] > 1:
            self.observations[:, :-1, :] = self.observations[:, 1:, :]
        self.observations[:, -1, :] = observations
        if self.env_name == 'LaserHockey-v0' and self.render_mode == 'human':
            self.envs.render()
        return self.observations, rewards, terminated, truncated, infos


    def render(self):
        return self.envs.render()


    def close(self):
        return self.envs.close()


    def convert_actions(self, actions, infos, training=True):
        if self.env_name == 'LaserHockey-v0':
            obs_agent2s = infos['obs_agent_two']
            if not training:
                obs_agent2s = np.expand_dims(obs_agent2s, axis=0)
            basic_actions = [self.basic.act(obs_agent2) for obs_agent2 in obs_agent2s]
            converted_actions = self.action_space[actions]
            total_actions = np.hstack([converted_actions, basic_actions])
            return total_actions
        else:
            return actions
        

    def get_action_space(self):
        values = [-1, 0, 1]
        x, y, z = np.meshgrid(values, values, values, indexing='ij')
        result = np.stack((x, y, z), axis=-1)
        reshaped_result = result.reshape(-1, result.shape[-1])
        return reshaped_result