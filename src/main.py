from environments import Environments
from data_collector import DataCollector
from metric import Metric
from bandit import Bandits
from learner import Learner
from actor import Actor
from training import Training


env_name = 'CartPole-v1'
# env_name = 'LunarLander-v2'
# env_name = 'LaserHockey-v0'

# load_run = None
# wandb_id = None
load_run = 'n256,b16,s100,bb100,d_t1000,g0.995,lr0.01_Apr04-01-15-30'
wandb_id = 'lkb3vpdu'

# if load_run is specified only max_frames will be used
train_parameters = {'max_frames': 1000000,
                    'per_buffer_size': 1000000,
                    'per_min_frames': 100000,
                    'lr_finder': False}

abbreviation_dict = {
                     'num_envs': 'n',
                     'batch_size': 'b',
                     'sequence_length': 's',
                     'bootstrap_length': 'bb',
                     'd_target': 'd_t',
                     'discount': 'g',
                     'learning_rate': 'lr',
                     'add_on': None}


training = Training(env_name, load_run, wandb_id, train_parameters, abbreviation_dict)
environments = Environments(training)
data_collector = DataCollector(training)
metric = Metric(training)
bandits = Bandits(training)
learner = Learner(training)
actor = Actor(training)

training.run(environments, data_collector, metric, bandits, learner, actor)
