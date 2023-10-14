import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import laser_hockey_env as lh


class Training:
    def __init__(self, config, env_name):
        self.num_steps = config['num_steps']
        self.num_envs = config['num_envs']
        if env_name == "LaserHockey-v0":
            self.envs = [lambda: lh.LaserHockeyEnv() for _ in range(self.num_envs)]
        else:
            self.envs = [lambda: gym.make(env_name) for _ in range(self.num_envs)]
        self.env = SyncVectorEnv(self.envs)

    def run(self, actor, data_collector, bandits, metric):
        next_observations, info = self.env.reset()
        indeces = bandits.get_all_indeces(self.num_envs)
        
        for step in range(self.num_steps):
            observations = next_observations
            v1, v2, a1, a2, policy = actor.calculate_values(observations, indeces)
            actions, action_probs = actor.get_action(policy, stochastic=True)
            
            action = actions.cpu().numpy()
            # action = [self.env.single_action_space.sample() for _ in range(self.num_envs)]
            next_observations, rewards, terminated, truncated, infos = self.env.step(action)

            data_collector.add_step_data(o=observations, v1=v1, v2=v2, a1=a1, a2=a2, i=indeces, p=policy, a=actions, a_p=action_probs, r=rewards, d=terminated, t=truncated)
            data_collector.check_save_sequence()
            terminated_indeces, returns, terminated_envs = data_collector.update_return()
            # print(terminated_indeces)
            # print(returns)
            # print(terminated_envs)

            new_indeces = bandits.update_and_get_new_indeces(terminated_indeces, returns)
            indeces[terminated_envs] = new_indeces
            # print(indeces)
            # print(new_indeces)

            metric.add_return(returns, step)
            
            print(f"Step number: {step+1}", end='\r')

        self.env.close()