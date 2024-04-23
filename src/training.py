import json
import datetime
from configs import configs

from timeit import default_timer as dt
import numpy as np


class Training:
    def __init__(self, env_name, train_parameters, run_name_dict):
        self.env_name = env_name
        self.config = self.get_config(self.env_name, train_parameters)
        self.log_dir = self.get_log_dir(self.config, self.env_name, run_name_dict)
        self.hyperparams_file = f'{self.log_dir}/hyperparameters.json'
        self.played_frames = self.config['played_frames']
        self.train_frames = self.config['train_frames']
        self.num_envs = self.config['num_envs']


    def get_config(self, env_name, train_parameters):
        config = {key: train_parameters[key] if key in train_parameters else value for key, value in configs[env_name].items()}
        if config['load_run'] is not None:
            run_path = f"../runs/{env_name}/{config['load_run']}/hyperparameters.json"
            with open(run_path, "r") as file:
                config = json.load(file)
            config['load_run'] = train_parameters['load_run']
            config['train_frames'] = config['played_frames'] + train_parameters['train_frames']
            return config
        if config['lr_finder']:
            config['train_frames'] = (config['per_min_frames'] / config['sample_reuse'] +
                                    config['num_envs'] * (config['sequence_length'] + config['bootstrap_length']) +
                                    config['warmup_steps'] * config['update_frequency'] * config['num_envs'])
        return config


    def get_log_dir(self, config, env_name, run_name_dict):
        if config['load_run'] is None:
            abbreviated_parameters = {run_name_dict[key]: value for key, value in config.items() if key in run_name_dict}
            parameter_string = ','.join([f'{key}{value}' for key, value in abbreviated_parameters.items()])
            if run_name_dict['prefix'] != '': parameter_string = run_name_dict['prefix'] + ',' + parameter_string
            timestamp = datetime.datetime.now().strftime('%b%d-%H-%M-%S')
            log_dir = f'../runs/{env_name}/{parameter_string},{timestamp}'
            if config['lr_finder']: log_dir += '_lr'
            if run_name_dict['suffix'] != '': log_dir += ',' + run_name_dict['suffix']
        else:
            log_dir = f'../runs/{env_name}/{config["load_run"]}'
        return log_dir
    

    def save_everything(self, played_frames, learner, bandits, data_collector, training=False):
        if (not training or (learner.update_count + 1) % self.config['d_push'] == 0) and not self.config['lr_finder']:
            learner.save_state()
            data_collector.save_data_collector()
            bandits.save_bandits()
            self.config['played_frames'] = played_frames
            with open(self.hyperparams_file, 'w') as file:
                json.dump(self.config, file)
    

    def update_mean(self, old_mean, new_value, n):
        return old_mean + (new_value - old_mean) / (n + 1)
    


    def run(self, environments, data_collector, metric, bandits, learner, actor):
        next_observations, infos = environments.reset()
        indeces = bandits.get_all_indeces(self.num_envs)

        training_started = False
        mean_steps = 0
        mean_run = 0
        mean_bandit = 0
        mean_env = 0
        mean_dc = 0
        mean_metric = 0
        mean_actor = 0
        mean_learner = 0
        mean_conv_a = 0
        mean_saving = 0
        mean_sum = 0
        
        while self.played_frames < self.train_frames:

            start = dt()
            observations = next_observations.copy()
            before_actor = dt()
            actor.pull_weights(learner)
            actions, action_probs = actor.get_actions(observations, indeces, training=True)
            actor_t = dt() - before_actor
            before_conv_a = dt()
            converted_actions = environments.convert_actions(actions, infos)
            conv_a = dt() - before_conv_a
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
            print(f"Frames: {self.played_frames}/{self.train_frames}", end='\r')

            before_saving = dt()
            self.save_everything(self.played_frames, learner, bandits, data_collector, training=True)
            saving = dt() - before_saving


            end = dt()
            run_step = end - start
            sum = np.sum([bandit, env, dc, metric_t, actor_t, learner_t, conv_a, saving])

            if not training_started and learner.update_count >= 5:
                training_started = True
                mean_steps = 0
                mean_run = 0
                mean_bandit = 0
                mean_env = 0
                mean_dc = 0
                mean_metric = 0
                mean_actor = 0
                mean_learner = 0
                mean_conv_a = 0
                mean_saving = 0
                mean_sum = 0
            
            mean_steps += 1
            mean_run = self.update_mean(mean_run, run_step, mean_steps)
            mean_bandit = self.update_mean(mean_bandit, bandit, mean_steps)
            mean_env = self.update_mean(mean_env, env, mean_steps)
            mean_dc = self.update_mean(mean_dc, dc, mean_steps)
            mean_metric = self.update_mean(mean_metric, metric_t, mean_steps)
            mean_actor = self.update_mean(mean_actor, actor_t, mean_steps)
            mean_learner = self.update_mean(mean_learner, learner_t, mean_steps)
            mean_conv_a = self.update_mean(mean_conv_a, conv_a, mean_steps)
            mean_saving = self.update_mean(mean_saving, saving, mean_steps)
            mean_sum = self.update_mean(mean_sum, sum, mean_steps)

            print(f"Frames: {self.played_frames}/{self.train_frames}")
            print(f"Type:\tSec\tMean\tPerc\tMean")
            print(f"Step:\t{run_step:.4f}\t{mean_run:.4f}\t{run_step/mean_run:.4f}\t{mean_run/mean_run:.4f}")
            print(f"Bandit:\t{bandit:.4f}\t{mean_bandit:.4f}\t{bandit/run_step:.4f}\t{mean_bandit/mean_run:.4f}")
            print(f"Env:\t{env:.4f}\t{mean_env:.4f}\t{env/run_step:.4f}\t{mean_env/mean_run:.4f}")
            print(f"DC:\t{dc:.4f}\t{mean_dc:.4f}\t{dc/run_step:.4f}\t{mean_dc/mean_run:.4f}")
            print(f"Metric:\t{metric_t:.4f}\t{mean_metric:.4f}\t{metric_t/run_step:.4f}\t{mean_metric/mean_run:.4f}")
            print(f"Actor:\t{actor_t:.4f}\t{mean_actor:.4f}\t{actor_t/run_step:.4f}\t{mean_actor/mean_run:.4f}")
            print(f"Learn:\t{learner_t:.4f}\t{mean_learner:.4f}\t{learner_t/run_step:.4f}\t{mean_learner/mean_run:.4f}")
            print(f"Conv_a:\t{conv_a:.4f}\t{mean_conv_a:.4f}\t{conv_a/run_step:.4f}\t{mean_conv_a/mean_run:.4f}")
            print(f"Saving:\t{saving:.4f}\t{mean_saving:.4f}\t{saving/run_step:.4f}\t{mean_saving/mean_run:.4f}")
            print(f"Sum:\t{sum:.4f}\t{mean_sum:.4f}\t{sum/run_step:.4f}\t{mean_sum/mean_run:.4f}")


        self.save_everything(self.played_frames, learner, bandits, data_collector)
        environments.close()
        metric.close_writer()

        print(f"Frames: {self.played_frames}/{self.train_frames}")
        print(f"Type:\tSec\tMean\tPerc\tMean")
        print(f"Step:\t{run_step:.4f}\t{mean_run:.4f}\t{run_step/mean_run:.4f}\t{mean_run/mean_run:.4f}")
        print(f"Bandit:\t{bandit:.4f}\t{mean_bandit:.4f}\t{bandit/run_step:.4f}\t{mean_bandit/mean_run:.4f}")
        print(f"Env:\t{env:.4f}\t{mean_env:.4f}\t{env/run_step:.4f}\t{mean_env/mean_run:.4f}")
        print(f"DC:\t{dc:.4f}\t{mean_dc:.4f}\t{dc/run_step:.4f}\t{mean_dc/mean_run:.4f}")
        print(f"Metric:\t{metric_t:.4f}\t{mean_metric:.4f}\t{metric_t/run_step:.4f}\t{mean_metric/mean_run:.4f}")
        print(f"Actor:\t{actor_t:.4f}\t{mean_actor:.4f}\t{actor_t/run_step:.4f}\t{mean_actor/mean_run:.4f}")
        print(f"Learn:\t{learner_t:.4f}\t{mean_learner:.4f}\t{learner_t/run_step:.4f}\t{mean_learner/mean_run:.4f}")
        print(f"Conv_a:\t{conv_a:.4f}\t{mean_conv_a:.4f}\t{conv_a/run_step:.4f}\t{mean_conv_a/mean_run:.4f}")
        print(f"Saving:\t{saving:.4f}\t{mean_saving:.4f}\t{saving/run_step:.4f}\t{mean_saving/mean_run:.4f}")
        print(f"Sum:\t{sum:.4f}\t{mean_sum:.4f}\t{sum/run_step:.4f}\t{mean_sum/mean_run:.4f}")