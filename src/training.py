import json
import torch
import datetime
from configs import configs

from timeit import default_timer as dt
import numpy as np


class Training:
    def __init__(self, env_name, load_run, wandb_id=None, train_parameters=None, abbreviation_dict=None):
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

        training_started = False
        run_steps = []
        bandit_steps = []
        env_steps = []
        dc_steps = []
        metric_t_steps = []
        actor_t_steps = []
        learner_t_steps = []
        conv_a_t_steps = []
        
        while self.played_frames < self.max_frames:

            start = dt()
            observations = next_observations
            actor.pull_weights(learner)
            before_actor = dt()
            actions, action_probs = actor.get_actions(observations, indeces, training=True)
            actor_t = dt() - before_actor
            before_conv_a = dt()
            converted_actions = environments.convert_actions(actions, infos)
            conv_a_t = dt() - before_conv_a
            before_env = dt()
            next_observations, rewards, terminated, truncated, infos = environments.step(converted_actions)
            env = dt() - before_env

            before_dc = dt()
            data_collector.add_data(o=observations, a=actions, a_p=action_probs, i=indeces, r=rewards, d=terminated, t=truncated)
            train_indeces, val_indeces, train_returns, val_returns, train_envs, val_envs = data_collector.check_done_and_return()
            dc = dt() - before_dc
            before_bandit = dt()
            new_train_indeces, new_val_indeces, index_data = bandits.update_and_get_data(data_collector, train_indeces, train_returns, train_envs, val_envs)
            bandit = dt() - before_bandit
            indeces[train_envs] = new_train_indeces
            indeces[val_envs] = new_val_indeces
            
            before_learner = dt()
            losses, targets = learner.check_and_update(data_collector)
            learner_t = dt() - before_learner

            self.played_frames += self.num_envs
            before_metric = dt()
            metric.add_train_return(train_returns, self.played_frames)
            metric.add_val_return(val_returns, val_envs, self.played_frames)
            metric.add_index_data(index_data, self.played_frames)
            metric.add_targets(targets, self.played_frames)
            metric.add_losses(losses, self.played_frames)
            metric_t = dt() - before_metric
            print(f"Frames: {self.played_frames}/{self.max_frames}", end='\r')


            end = dt()
            run_step = end - start

            run_steps.append(run_step)
            bandit_steps.append(bandit)
            env_steps.append(env)
            dc_steps.append(dc)
            metric_t_steps.append(metric_t)
            actor_t_steps.append(actor_t)
            learner_t_steps.append(learner_t)
            conv_a_t_steps.append(conv_a_t)

            if not training_started and learner.update_count >= 5:
                training_started = True
                run_steps = []
                bandit_steps = []
                env_steps = []
                dc_steps = []
                metric_t_steps = []
                actor_t_steps = []
                learner_t_steps = []
                conv_a_t_steps = []
            
            sum = np.sum([bandit, env, dc, metric_t, actor_t, learner_t, conv_a_t])
            mean_sum = np.sum([np.mean(bandit_steps), np.mean(env_steps), np.mean(dc_steps), np.mean(metric_t_steps), np.mean(actor_t_steps), np.mean(learner_t_steps), np.mean(conv_a_t_steps)])

            print(f"Frames: {self.played_frames}/{self.max_frames}")
            print(f"Type:\tSec\tMean\tPerc\tMean")
            print(f"Step:\t{run_step:.4f}\t{np.mean(run_steps):.4f}\t{np.mean(run_steps):.4f}\t{np.mean(run_steps):.4f}")
            print(f"Bandit:\t{bandit:.4f}\t{np.mean(bandit_steps):.4f}\t{bandit/run_step:.4f}\t{np.mean(bandit_steps)/np.mean(run_steps):.4f}")
            print(f"Env:\t{env:.4f}\t{np.mean(env_steps):.4f}\t{env/run_step:.4f}\t{np.mean(env_steps)/np.mean(run_steps):.4f}")
            print(f"DC:\t{dc:.4f}\t{np.mean(dc_steps):.4f}\t{dc/run_step:.4f}\t{np.mean(dc_steps)/np.mean(run_steps):.4f}")
            print(f"Metric:\t{metric_t:.4f}\t{np.mean(metric_t_steps):.4f}\t{metric_t/run_step:.4f}\t{np.mean(metric_t_steps)/np.mean(run_steps):.4f}")
            print(f"Actor:\t{actor_t:.4f}\t{np.mean(actor_t_steps):.4f}\t{actor_t/run_step:.4f}\t{np.mean(actor_t_steps)/np.mean(run_steps):.4f}")
            print(f"Learn:\t{learner_t:.4f}\t{np.mean(learner_t_steps):.4f}\t{learner_t/run_step:.4f}\t{np.mean(learner_t_steps)/np.mean(run_steps):.4f}")
            print(f"Conv_a:\t{conv_a_t:.4f}\t{np.mean(conv_a_t_steps):.4f}\t{conv_a_t/run_step:.4f}\t{np.mean(conv_a_t_steps)/np.mean(run_steps):.4f}")
            print(f"Sum:\t{sum:.4f}\t{mean_sum:.4f}\t{sum/run_step:.4f}\t{mean_sum/np.mean(run_steps):.4f}")


        # save model and optimizer again?
        # save all other needed parameters again?
        data_collector.save_data_collector()
        environments.close()
        metric.close_writer()

        