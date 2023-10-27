import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import laser_hockey_env as lh


class Environments:
    def __init__(self, training_class, training=True):
        self.env_name = training_class.env_name
        self.config = training_class.config
        self.envs = self.get_environments(training)
        self.basic = lh.BasicOpponent()


    def get_environments(self, training=True):
        if training:
            if self.env_name == 'LaserHockey-v0':
                envs = [lambda: lh.LaserHockeyEnv().reset() if _ % 2 == 0 else lh.LaserHockeyEnv() for _ in range(self.config['num_envs'])]
            else:
                envs = [lambda: gym.make(self.env_name) for _ in range(self.config['num_envs'])]
            envs = SyncVectorEnv(envs)
        else:
            if self.env_name == 'LaserHockey-v0':
                envs = lh.LaserHockeyEnv()
            else:
                # envs = gym.make(self.env_name, render_mode='rgb_array')
                envs = gym.make(self.env_name, render_mode='human')
        return envs


    def reset(self):
        return self.envs.reset()


    def step(self, actions):
        return self.envs.step(actions)


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
            converted_actions = self.convert_actions2(actions)
            total_actions = np.hstack([converted_actions, basic_actions])
            return total_actions
        else:
            return actions
        

    def convert_actions2(self, actions):
        if self.env_name == 'LaserHockey-v0':
            conversion_dict = {0: [-1,-1,-1], 1: [0,-1,-1], 2: [1,-1,-1],
                               3: [-1,0,-1], 4: [0,0,-1], 5: [1,0,-1],
                               6: [-1,1,-1], 7: [0,1,-1], 8: [1,1,-1],
                               9: [-1,-1,0], 10: [0,-1,0], 11: [1,-1,0],
                               12: [-1,0,0], 13: [0,0,0], 14: [1,0,0],
                               15: [-1,1,0], 16: [0,1,0], 17: [1,1,0],
                               18: [-1,-1,1], 19: [0,-1,1], 20: [1,-1,1],
                               21: [-1,0,1], 22: [0,0,1], 23: [1,0,1],
                               24: [-1,1,1], 25: [0,1,1], 26: [1,1,1]}
            converted_actions = [conversion_dict[action] for action in actions]
            return converted_actions
        