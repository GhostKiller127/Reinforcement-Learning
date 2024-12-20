import os
import json
import yaml
import asyncio
import datetime
import numpy as np
from configs import configs
from yaml import dump, Dumper
from timeit import default_timer as dt



class Training:
    def __init__(self, env_name, train_parameters, run_name_dict):
        self.env_name = env_name
        self.config = self.get_config(self.env_name, train_parameters)
        self.log_dir = self.get_log_dir(self.config, self.env_name, run_name_dict)
        self.env_configs = self.get_env_configs()
        self.hyperparams_file = f'{self.log_dir}/hyperparameters.json'

#region init

    def get_config(self, env_name, train_parameters):
        def update_config(config, train_parameters, configs):
            for key, value in configs.items():
                if key in train_parameters and isinstance(value, dict):
                    config[key] = update_config({}, train_parameters[key], value)
                else:
                    config[key] = train_parameters.get(key, value)
            return config

        config = update_config({}, train_parameters, configs[env_name])

        if config['load_run'] is not None:
            run_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../runs/{env_name}/{config['load_run']}/hyperparameters.json"))
            with open(run_path, "r") as file:
                config = json.load(file)
            config['load_run'] = train_parameters['load_run']
            config['train_frames'] = config['played_frames'] + train_parameters['train_frames']
            return config
        if config['lr_finder']:
            config['train_frames'] = (config['per_min_frames'] / config['sample_reuse'] +
                                    (config['train_envs'] + config['val_envs']) * (config['sequence_length'] + config['bootstrap_length']) +
                                    config['warmup_steps'] * 2 * config['update_frequency'] * (config['train_envs'] + config['val_envs']))
        return config


    def get_log_dir(self, config, env_name, run_name_dict):
        if config['load_run'] is None:
            def generate_run_string(run_name_dict, configs):
                run_string = ""
                for key, value in run_name_dict.items():
                    if key in configs and isinstance(value, str):
                        run_string += f"{value}{configs[key]},"
                    elif isinstance(value, dict) and key in configs:
                        run_string += generate_run_string(value, configs[key])
                return run_string
            run_name = generate_run_string(run_name_dict, config).rstrip(',')
            
            if run_name_dict['prefix'] != '':
                run_name = run_name_dict['prefix'] + ',' + run_name
            if run_name_dict['timestamp']:
                run_name += ',' + datetime.datetime.now().strftime('%b%d-%H-%M-%S')
            if config['lr_finder']:
                run_name += ',lr'
            if run_name_dict['suffix'] != '':
                run_name += ',' + run_name_dict['suffix']
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../runs/{env_name}/{run_name}"))
        else:
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../runs/{env_name}/{config['load_run']}"))
        return log_dir
    
    
    def get_env_configs(self):
        if self.env_name == 'Crypto-v0':
            if self.config['load_run'] is None:
                configs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../Bybit/src/configs.yaml"))
            else:
                configs_path = os.path.join(self.log_dir, "configs.yaml")
            with open(configs_path, 'r') as configs_file:
                env_configs = yaml.safe_load(configs_file)
            return env_configs
    

    async def save_everything(self, learner, bandits, data_collector, environments, metric, training=False):
        if (not training or (learner.update_count + 1) % self.config['d_push'] == 0) and not self.config['lr_finder']:
            print('Saving...')
            await learner.save_states()
            data_collector.save_data_collector()
            bandits.save_bandits()
            environments.save_environments()
            metric.save_state()
            with open(self.hyperparams_file, 'w') as file:
                json.dump(self.config, file)
    

    def update_mean(self, old_mean, new_value, n):
        return old_mean + (new_value - old_mean) / (n + 1)

#endregion
#region run

    async def run(self, environments, data_collector, metric, bandits, learner, actor, stop):
        next_observations, next_infos = environments.reset()
        indeces = bandits.get_all_indeces()

        training_started = False
        mean_steps = 0
        mean_run = 0
        mean_bandit = 0
        mean_env = 0
        mean_dc = 0
        mean_metric = 0
        mean_actor = 0
        mean_learner = 0
        mean_saving = 0
        mean_sum = 0
        
        while self.config['played_frames'] < self.config['train_frames']:
            start = dt()
            observations = next_observations.copy()
            observations = environments.preprocess_observations(observations)
            infos = next_infos.copy()
            before_actor = dt()
            await actor.pull_weights(learner)
            actions, action_probs = actor.get_actions(observations, indeces, training=True)
            actor_t = dt() - before_actor
            before_env = dt()
            next_observations, rewards, terminated, truncated, next_infos = environments.step(actions, infos)
            env = dt() - before_env

            before_dc = dt()
            data_collector.add_data(o=observations, a=actions, a_p=action_probs, i=indeces, r=rewards, d=terminated, t=truncated, infos=infos)
            train_indeces, val_indeces, train_returns, val_returns, train_envs, val_envs = data_collector.check_done_and_return()
            dc = dt() - before_dc
            before_bandit = dt()
            new_train_indeces, new_val_indeces, index_data = bandits.update_and_get_data(data_collector, train_indeces, train_returns, train_envs, val_envs)
            bandit = dt() - before_bandit
            indeces[train_envs] = new_train_indeces
            indeces[val_envs] = new_val_indeces
            
            before_learner = dt()
            losses, targets = learner.check_and_update(data_collector, environments)
            learner_t = dt() - before_learner

            self.config['played_frames'] += self.config['train_envs']
            before_metric = dt()
            metric.add_observations(observations, self.config['played_frames'])
            metric.add_infos(infos, self.config['played_frames'])
            metric.add_train_return(train_returns, self.config['played_frames'])
            metric.add_val_return(val_returns, val_envs, self.config['played_frames'])
            metric.add_index_data(index_data, self.config['played_frames'])
            metric.add_targets(targets, self.config['played_frames'])
            metric.add_losses(losses, self.config['played_frames'])
            metric_t = dt() - before_metric
            print(f"Frames: {self.config['played_frames']}/{self.config['train_frames']}", end='\r')

            before_saving = dt()
            await self.save_everything(learner, bandits, data_collector, environments, metric, training=True)
            saving = dt() - before_saving

#endregion
#region metrics

            end = dt()
            run_step = end - start
            sum = np.sum([bandit, env, dc, metric_t, actor_t, learner_t, saving])

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
            mean_saving = self.update_mean(mean_saving, saving, mean_steps)
            mean_sum = self.update_mean(mean_sum, sum, mean_steps)

            print(f"Frames: {self.config['played_frames']}/{self.config['train_frames']}")
            print(f"Type:\tSec\tMean\tPerc\tMean")
            print(f"Sum:\t{sum:.4f}\t{mean_sum:.4f}\t{sum/run_step:.4f}\t{mean_sum/mean_run:.4f}")
            print(f"Env:\t{env:.4f}\t{mean_env:.4f}\t{env/run_step:.4f}\t{mean_env/mean_run:.4f}")
            print(f"Learn:\t{learner_t:.4f}\t{mean_learner:.4f}\t{learner_t/run_step:.4f}\t{mean_learner/mean_run:.4f}")
            print(f"Actor:\t{actor_t:.4f}\t{mean_actor:.4f}\t{actor_t/run_step:.4f}\t{mean_actor/mean_run:.4f}")
            print(f"Bandit:\t{bandit:.4f}\t{mean_bandit:.4f}\t{bandit/run_step:.4f}\t{mean_bandit/mean_run:.4f}")
            print(f"Metric:\t{metric_t:.4f}\t{mean_metric:.4f}\t{metric_t/run_step:.4f}\t{mean_metric/mean_run:.4f}")
            print(f"Saving:\t{saving:.4f}\t{mean_saving:.4f}\t{saving/run_step:.4f}\t{mean_saving/mean_run:.4f}")
            print(f"DC:\t{dc:.4f}\t{mean_dc:.4f}\t{dc/run_step:.4f}\t{mean_dc/mean_run:.4f}")

            await asyncio.sleep(0)
            if stop[0]:
                break

#endregion
#region finish

        await self.save_everything(learner, bandits, data_collector, environments, metric)
        environments.close()
        before_upload = dt()
        metric.close_writer()
        upload = dt() - before_upload
        upload_time = datetime.timedelta(seconds=upload)

        print(f"Frames: {self.config['played_frames']}/{self.config['train_frames']}")
        print(f"Type:\tSec\tMean\tPerc\tMean")
        print(f"Sum:\t{sum:.4f}\t{mean_sum:.4f}\t{sum/run_step:.4f}\t{mean_sum/mean_run:.4f}")
        print(f"Env:\t{env:.4f}\t{mean_env:.4f}\t{env/run_step:.4f}\t{mean_env/mean_run:.4f}")
        print(f"Learn:\t{learner_t:.4f}\t{mean_learner:.4f}\t{learner_t/run_step:.4f}\t{mean_learner/mean_run:.4f}")
        print(f"Actor:\t{actor_t:.4f}\t{mean_actor:.4f}\t{actor_t/run_step:.4f}\t{mean_actor/mean_run:.4f}")
        print(f"Bandit:\t{bandit:.4f}\t{mean_bandit:.4f}\t{bandit/run_step:.4f}\t{mean_bandit/mean_run:.4f}")
        print(f"Metric:\t{metric_t:.4f}\t{mean_metric:.4f}\t{metric_t/run_step:.4f}\t{mean_metric/mean_run:.4f}")
        print(f"Saving:\t{saving:.4f}\t{mean_saving:.4f}\t{saving/run_step:.4f}\t{mean_saving/mean_run:.4f}")
        print(f"DC:\t{dc:.4f}\t{mean_dc:.4f}\t{dc/run_step:.4f}\t{mean_dc/mean_run:.4f}")
        print(f"Upload:\t{str(upload_time)}")

#endregion