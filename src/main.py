from environments import Environments
from data_collector import DataCollector
from metric import Metric
from bandit import Bandits
from learner import Learner
from actor import Actor
from training import Training


# env_name = 'CartPole-v1'
# env_name = 'LunarLander-v2'
env_name = 'LaserHockey-v0'

load_run = None
wandb_id = None
# load_run = 'n256,b16,s100,bb100,d_t1000,g0.997,lr0.001_Apr06-02-03-50'
# wandb_id = 'w55asotf'

# if load_run is specified only max_frames will be used
train_parameters = {'max_frames': 100000,
                    'per_buffer_size': 200000,
                    'per_min_frames': 100,
                    'lr_finder': False}

abbreviation_dict = {
                     'num_envs': 'n',
                    #  'batch_size': 'b',
                    #  'sequence_length': 's',
                    #  'bootstrap_length': 'bb',
                     'd_target': 'd_t',
                    #  'discount': 'g',
                    #  'learning_rate': 'lr',
                     'reward_scaling_1': 'r1s',
                     'reward_scaling_2': 'r2s',
                     'v_loss_scaling': 'v',
                     'q_loss_scaling': 'q',
                     'p_loss_scaling': 'p',
                     'add_on': None}


training = Training(env_name, load_run, wandb_id, train_parameters, abbreviation_dict)
environments = Environments(training)
data_collector = DataCollector(training)
metric = Metric(training)
bandits = Bandits(training)
learner = Learner(training)
actor = Actor(training)

training.run(environments, data_collector, metric, bandits, learner, actor)
