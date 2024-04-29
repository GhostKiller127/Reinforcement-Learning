from environments import Environments
from data_collector import DataCollector
from metric import Metric
# from bandit import Bandits
from bandit_jax import Bandits2 as Bandits
# from learner import Learner
from learner_jax import Learner
# from actor import Actor
from actor_jax import Actor
from training import Training


# env_name = 'CartPole-v1'
env_name = 'LunarLander-v2'
# env_name = 'LaserHockey-v0'

# if you want to continue a run, 'load_run' and 'train_frames' need to be specified. the rest will be overwritten.
train_parameters = {
                    # 'load_run': 'save_load_test,,bandit_jax',
                    # 'train_frames': 100000,
                    # 'per_buffer_size': 100000,
                    # 'per_min_frames': 10000,
                    # 'architecture': 'dense_jax',
                    # 'observation_length': 1,
                    # 'metrics': False,
                    # 'bandits': False,
                    # 'lr_finder': True,
                    }

run_name_dict = {
    'prefix': 'd64,s151,w1,d7',
    # 'prefix': 'save_load_test',
    'suffix': 'bandit_jax',
    'timestamp': False,
    # 'num_envs': 'n',
    # 'batch_size': 'b',
    # 'observation_length': 'obs',
    # 'sequence_length': 's',
    # 'bootstrap_length': 'bb',
    # 'd_target': 'd_t',
    # 'discount': 'g',
    # 'learning_rate': 'lr',
    # 'weight_decay': 'w',
    # 'reward_scaling_1': 'r1s',
    # 'reward_scaling_2': 'r2s',
    # 'v_loss_scaling': 'v',
    # 'q_loss_scaling': 'q',
    # 'p_loss_scaling': 'p',
    }


training = Training(env_name, train_parameters, run_name_dict)
environments = Environments(training)
data_collector = DataCollector(training)
metric = Metric(training)
bandits = Bandits(training)
learner = Learner(training)
actor = Actor(training)

training.run(environments, data_collector, metric, bandits, learner, actor)
