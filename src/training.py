import json
import torch
import datetime
from configs import configs


class Training:
    def __init__(self, env_name, load_run, wandb_id, train_parameters, abbreviation_dict):
        self.env_name = env_name
        self.config = self.get_config(self.env_name, load_run, wandb_id, train_parameters)
        self.log_dir = self.get_log_dir(self.config, self.env_name, abbreviation_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.played_frames = self.config['played_frames']
        self.max_frames = self.config['max_frames']
        self.num_envs = self.config['num_envs']


    def get_config(self, env_name, load_run, wandb_id, train_parameters):
        if load_run is not None:
            run_path = f'../runs/{env_name}/{load_run}/hyperparameters.json'
            with open(run_path, "r") as file:
                config = json.load(file)
            config['load_run'] = load_run
            config['wandb_id'] = wandb_id
            if train_parameters is not None:
                config['max_frames'] = config['played_frames'] + train_parameters['max_frames']
            return config
        config = {key: train_parameters[key] if key in train_parameters else value for key, value in configs[env_name].items()}
        if config['lr_finder']:
            config['max_frames'] = (config['per_min_frames'] / config['sample_reuse'] +
                                    config['num_envs'] * (config['sequence_length'] + config['bootstrap_length']) +
                                    config['warmup_steps'] * config['update_frequency'] * config['num_envs'])
        return config


    def get_log_dir(self, config, env_name, abbreviation_dict):
        if config['load_run'] is None:
            abbreviated_parameters = {abbreviation_dict[key]: value for key, value in config.items() if key in abbreviation_dict}
            parameter_string = ','.join([f'{key}{value}' for key, value in abbreviated_parameters.items()])
            timestamp = datetime.datetime.now().strftime('%b%d-%H-%M-%S')
            log_dir = f'../runs/{env_name}/{parameter_string}_{timestamp}'
            if abbreviation_dict['add_on'] is not None: log_dir += '_' + abbreviation_dict['add_on']
            if config['lr_finder']: log_dir += '_lr'
        else:
            log_dir = f'../runs/{env_name}/{config["load_run"]}'
        return log_dir
    

    def run(self, environments, data_collector, metric, bandits, learner, actor):
        next_observations, infos = environments.reset()
        indeces = bandits.get_all_indeces(self.num_envs)
        
        while self.played_frames < self.max_frames:
            observations = next_observations
            actor.pull_weights(learner)
            actions, action_probs = actor.get_actions(observations, indeces, training=True)
            converted_actions = environments.convert_actions(actions, infos)
            next_observations, rewards, terminated, truncated, infos = environments.step(converted_actions)

            data_collector.add_data(o=observations, a=actions, a_p=action_probs, i=indeces, r=rewards, d=terminated, t=truncated)
            terminated_indeces, returns, terminated_envs = data_collector.check_done_and_return()
            new_indeces, index_data = bandits.update_and_get_data(data_collector, terminated_indeces, returns, terminated_envs)
            indeces[terminated_envs] = new_indeces
            
            losses = learner.check_and_update(data_collector)

            self.played_frames += self.num_envs
            metric.add_return(data_collector, returns, terminated_envs, self.played_frames)
            metric.add_index_data(index_data, self.played_frames)
            metric.add_losses(losses, self.played_frames)
            print(f"Frames: {self.played_frames}/{self.max_frames}", end='\r')

        # save model and optimizer again?
        # save all other needed parameters again?
        data_collector.save_data_collector()
        environments.close()
        metric.close_writer()