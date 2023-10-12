import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import laser_hockey_env as lh

class Training:
    def __init__(self, env_name, num_envs, num_steps):
        self.num_envs = num_envs
        self.num_steps = num_steps
        if env_name == "LaserHockey-v0":
            self.envs = [lambda: lh.LaserHockeyEnv() for _ in range(self.num_envs)]
        else:
            self.envs = [lambda: gym.make(env_name) for _ in range(self.num_envs)]
        self.env = SyncVectorEnv(self.envs)
        self.total_rewards = [0] * self.num_envs

    def run(self, actor):
        observations, info = self.env.reset()
        
        for step in range(self.num_steps):
            # actions = [self.env.single_action_space.sample() for _ in range(self.num_envs)]

            dummy_indeces = [[1,1,1]] * len(observations)
            v1, v2, a1, a2, policy = actor.calculate_values(observations, dummy_indeces)
            actions, action_probs = actor.get_action(policy, stochastic=True)

            observations, rewards, terminated, truncated, infos = self.env.step(actions)
            # print(terminated)

            self.total_rewards = [total_reward + reward for total_reward, reward in zip(self.total_rewards, rewards)]
            
            print(f"Step number: {step+1}", end='\r')

        self.env.close()