class Training:
    def __init__(self, config):
        self.max_frames = config['max_frames']
        self.trained_frames = config['trained_frames']
        self.num_envs = config['num_envs']


    def run(self, environments, data_collector, metric, bandits, learner, actor):
        next_observations, infos = environments.reset()
        indeces = bandits.get_all_indeces(self.num_envs)
        
        while self.trained_frames < self.max_frames:
            observations = next_observations
            policy = actor.calculate_policy(observations, indeces)
            actions, action_probs = actor.get_action(policy, stochastic=True, random=False)

            converted_actions = environments.convert_actions(actions, infos)
            next_observations, rewards, terminated, truncated, infos = environments.step(converted_actions)

            data_collector.add_step_data(o=observations, a=actions, a_p=action_probs, i=indeces, r=rewards, d=terminated, t=truncated)
            data_collector.check_save_sequence()
            terminated_indeces, returns, terminated_envs = data_collector.check_done_and_return()

            new_indeces = bandits.update_and_get_new_indeces(terminated_indeces, returns)
            indeces[terminated_envs] = new_indeces

            losses = learner.check_and_update(data_collector)
            actor.pull_weights()

            self.trained_frames += self.num_envs
            metric.add_return(returns, self.trained_frames)
            metric.add_losses(losses, self.trained_frames)

            print(f"Frames: {self.trained_frames}/{self.max_frames}", end='\r')

        environments.close()