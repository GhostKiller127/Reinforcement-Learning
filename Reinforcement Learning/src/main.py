import torch
from configs import configs
from actor import Actor
from learner import Learner
from bandit import Bandits
from training import Training
from data_collector import DataCollector
from metric import Metric

# env_name = "CartPole-v1"
env_name = "LunarLander-v2"
# env_name = "LaserHockey-v0"

test_parameters = {'num_envs': 32,
                   'batch_size': 8,
                   'sequence_length': 20,
                   'bootstrap_length': 1,
                   'learning_rate': 2e-4,
                   'd_push': 4,
                   'd_pull': 20}

config = {key: test_parameters[key] if key in test_parameters else value for key, value in configs[env_name].items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_collector = DataCollector(config)
metric = Metric(config, env_name, test_parameters)
bandits = Bandits(config)
learner = Learner(config, metric, device)
actor = Actor(config, metric, device)
training = Training(config, env_name)

training.run(actor, learner, bandits, data_collector, metric)