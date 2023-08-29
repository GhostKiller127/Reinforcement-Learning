import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

class TrainingLoop:
    def __init__(self, num_envs, num_steps):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.envs = [lambda: gym.make('CartPole-v1') for _ in range(self.num_envs)]
        self.env = SyncVectorEnv(self.envs)
        self.total_rewards = [0] * self.num_envs

    def run(self):
        observations = self.env.reset()

        for step in range(self.num_steps):
            actions = [self.env.single_action_space.sample() for _ in range(self.num_envs)]
            next_observations, rewards, dones, infos, _ = self.env.step(actions)

            self.total_rewards = [total_reward + reward for total_reward, reward in zip(self.total_rewards, rewards)]
            
            print(f"Step number: {step+1}", end='\r')

        self.env.close()