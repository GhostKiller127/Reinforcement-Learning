import torch
import json
from configs import configs
from actor import Actor
from learner import Learner
from bandit import Bandits
from training import Training
from data_collector import DataCollector
from metric import Metric


def get_config(env_name, load_run, train_parameters):
    if load_run is None:
        return {key: train_parameters[key] if key in train_parameters else value for key, value in configs[env_name].items()}
    else:
        run_path = f'runs/{env_name}/{load_run}/hyperparameters.json'
        with open(run_path, "r") as file:
            config = json.load(file)
        config['load_run'] = load_run
        config['max_frames'] = config['trained_frames'] + train_parameters['max_frames']
        return config
        

# env_name = "CartPole-v1"
env_name = "LunarLander-v2"
# env_name = "LaserHockey-v0"

load_run = None
# load_run = 'env16,bs4,ss20,bb1,lr0.0003,d_o4,d_i20_Oct19-01-15-27'

# if load_run is specified only max_frames will be used
train_parameters = {'max_frames': 10000000,
                   'num_envs': 16,
                   'batch_size': 4,
                   'sequence_length': 20,
                   'bootstrap_length': 1,
                   'learning_rate': 3e-4,
                   'd_push': 4,
                   'd_pull': 20}

config = get_config(env_name, load_run, train_parameters)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_collector = DataCollector(config)
metric = Metric(config, env_name, train_parameters)
bandits = Bandits(config, metric)
learner = Learner(config, metric, device)
actor = Actor(config, metric, device)
training = Training(config, env_name)

training.run(actor, learner, bandits, data_collector, metric)
