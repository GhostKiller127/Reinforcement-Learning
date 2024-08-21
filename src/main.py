import itertools
import traceback
from environments import Environments
from data_collector import DataCollector
from metric import Metric
# from bandit import Bandits
# from bandit_jax import Bandits
from bandit_jax_new import Bandits
# from learner import Learner
from learner_jax import Learner
# from actor import Actor
from actor_jax import Actor
from training import Training


# env_name = 'CartPole-v1'
env_name = 'LunarLander-v2'
# env_name = 'LaserHockey-v0'
# env_name = 'Crypto-v0'

# if you want to continue a run, 'load_run' and 'train_frames' need to be specified. the rest will be overwritten
# if you want to run multiple parameters, put them in a list
train_parameters = {
                    # "load_run": 'dense,rng27,bs64,s10,b10,e0.995,g0.99,lr0.0001,y10.6,y21.2,x0.6,v1.0,q5.0,p5.0',
                    "jax_seed": 27,
                    # "jax_seed": [42, 69, 420, 1337, 2070],
                    # "train_frames": 5000000,
                    # "per_buffer_size": 100000,
                    # "per_min_frames": 10000,
                    # "batch_size": 64,
                    # "update_frequency": 2,
                    # "observation_length": 4,
                    # "architecture": 'dense_jax',
                    # "metrics": False,
                    # "bandits": False,
                    # "lr_finder": True,
                    # "parameters": {
                    #     "S5": {
                    #         "n_layers": 4,
                    #         "d_model": 32,
                    #         "ssm_size": 32,
                    #         "blocks": 4,
                    #         "decoder_dim": 64,
                    #         }},
                    # "bandit_params": {
                    #       "d": 9,
                    #       },
                    }

run_name_dict = {
    "prefix": 'S5',
    "suffix": '',
    "timestamp": False,
    # "parameters": {
    #     "S5": {
    #         "n_layers": '',
    #         "d_model": '',
    #         "ssm_size": '',
    #         "blocks": '',
    #         }},
    "jax_seed": 'rng',
    "batch_size": 'bs',
    # "reset_interval": 'ri',
    # "reset_percentage": 'rp',
    # "replay_ratio": 'rr',
    # "observation_length": 'o',
    "sequence_length": 's',
    "bootstrap_length": 'b',
    "ema_coefficient": 'e',
    # "d_target": 'd',
    "discount": 'g',
    # "learning_rate": 'lr',
    # "weight_decay": 'w',
    "reward_scaling_y1": 'y1:',
    "reward_scaling_y2": 'y2:',
    "reward_scaling_x": 'x:',
    "v_loss_scaling": 'v',
    "q_loss_scaling": 'q',
    "p_loss_scaling": 'p',
    # "bandit_params": {
    #     "width_": 'w',
    #     "size": 's',
        # "d": 'd',
        # },
    }


def generate_combinations(param_dict):
    if isinstance(param_dict, dict):
        keys, values = zip(*param_dict.items())
        for combination in itertools.product(*[generate_combinations(v) for v in values]):
            yield dict(zip(keys, combination))
    elif isinstance(param_dict, list):
        for item in param_dict:
            yield from generate_combinations(item)
    else:
        yield param_dict


param_combinations = list(generate_combinations(train_parameters))

for i, combination in enumerate(param_combinations):
    try:
        training = Training(env_name, combination, run_name_dict)
        environments = Environments(training)
        data_collector = DataCollector(training)
        metric = Metric(training, i)
        bandits = Bandits(training)
        learner = Learner(training)
        actor = Actor(training)

        training.run(environments, data_collector, metric, bandits, learner, actor)
    except Exception as e:
        training.save_everything(learner, bandits, data_collector)
        environments.close()
        metric.close_writer()
        traceback.print_exc()
