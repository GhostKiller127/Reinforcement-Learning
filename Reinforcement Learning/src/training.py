import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import laser_hockey_env as lh


class Training:
    def __init__(self, config, env_name):
        self.num_frames = config['num_frames']
        self.num_envs = config['num_envs']
        if env_name == "LaserHockey-v0":
            self.envs = [lambda: lh.LaserHockeyEnv() for _ in range(self.num_envs)]
        else:
            self.envs = [lambda: gym.make(env_name) for _ in range(self.num_envs)]
        self.env = SyncVectorEnv(self.envs)

    def run(self, actor, learner, bandits, data_collector, metric):
        next_observations, infos = self.env.reset()
        indeces = bandits.get_all_indeces(self.num_envs)
        
        num_frames = 0
        while num_frames < self.num_frames:
            observations = next_observations
            policy = actor.calculate_policy(observations, indeces)
            actions, action_probs = actor.get_action(policy, stochastic=True, random=False)
            
            next_observations, rewards, terminated, truncated, infos = self.env.step(actions.cpu().numpy())

            data_collector.add_step_data(o=observations, a=actions, a_p=action_probs, i=indeces, r=rewards, d=terminated, t=truncated)
            data_collector.check_save_sequence()
            terminated_indeces, returns, terminated_envs = data_collector.check_done_and_return()

            new_indeces = bandits.update_and_get_new_indeces(terminated_indeces, returns)
            indeces[terminated_envs] = new_indeces

            losses = learner.check_and_update(data_collector)
            actor.pull_weights()

            num_frames += self.num_envs
            metric.add_return(returns, num_frames)
            metric.add_losses(losses, num_frames)

            print(f"Frames: {num_frames}/{self.num_frames}", end='\r')

        self.env.close()