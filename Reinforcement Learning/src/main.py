from environments import Environments
from data_collector import DataCollector
from metric import Metric
from bandit import Bandits
from learner import Learner
from actor import Actor
from training import Training


# env_name = 'CartPole-v1'
env_name = 'LunarLander-v2'
# env_name = 'LaserHockey-v0'

load_run = None
# load_run = 'n256,bs64,s20,lr0.001_Oct25-14-41-00_lr'

# if load_run is specified only max_frames will be used
train_parameters = {'max_frames': 10000000,
                    'lr_finder': False}

abbreviation_dict = {
                     'batch_size': 'b',
                     'sequence_length': 's',
                     'bootstrap_length': 'bb',
                     'discount': 'd',
                     # 'update_frequency': 'up',
                     # 'reward_scaling_1': 'r1-',
                     # 'reward_scaling_2': 'r2-',
                     'learning_rate': 'lr',
                     'add_on': None}


training = Training(env_name, load_run, train_parameters, abbreviation_dict)
environments = Environments(training)
data_collector = DataCollector(training)
metric = Metric(training)
bandits = Bandits(training)
learner = Learner(training)
actor = Actor(training)

training.run(environments, data_collector, metric, bandits, learner, actor)
