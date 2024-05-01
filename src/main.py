import itertools
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


env_name = 'CartPole-v1'
# env_name = 'LunarLander-v2'
# env_name = 'LaserHockey-v0'

# if you want to continue a run, 'load_run' and 'train_frames' need to be specified. the rest will be overwritten
# if you want to run multiple parameters, put them in a list
train_parameters = {
                    # "load_run": 'inner_keys_test,dd128',
                    # "jax_seed": [27, 42, 69, 420, 1337, 2070],
                    "train_frames": 100000,
                    "per_buffer_size": 100000,
                    "per_min_frames": 10000,
                    "architecture": 'dense_jax',
                    "observation_length": 1,
                    "parameters": {
                        "S5": {
                            "n_layers": 4,
                            "d_model": 32,
                            "ssm_size": 32,
                            "blocks": 4,
                            "decoder_dim": [64, 128],
                            }},
                    "bandit_params": {
                          "width_": 1,
                          "size": 151,
                          "d": [3],
                          },
                    "metrics": False,
                    # "bandits": False,
                    # "lr_finder": True,
                    }

run_name_dict = {
    "prefix": 'inner_keys_test',
    "suffix": '',
    "timestamp": False,
    # "jax_seed": 'rng',
    # "num_envs": 'n',
    # "batch_size": 'b',
    # "observation_length": 'obs',
    # "sequence_length": 's',
    # "bootstrap_length": 'bb',
    # "d_target": 'd_t',
    # "discount": 'g',
    # "learning_rate": 'lr',
    # "weight_decay": 'w',
    # "reward_scaling_1": 'r1s',
    # "reward_scaling_2": 'r2s',
    # "v_loss_scaling": 'v',
    # "q_loss_scaling": 'q',
    # "p_loss_scaling": 'p',
    "parameters": {
        "S5": {
            # "n_layers": 4,
            # "d_model": 32,
            # "ssm_size": 32,
            # "blocks": 4,
            "decoder_dim": 'dd',
            }},
    # "bandit_params": {
    #     "d": 'd'},
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

for combination in param_combinations:
    training = Training(env_name, combination, run_name_dict)
    environments = Environments(training)
    data_collector = DataCollector(training)
    metric = Metric(training)
    bandits = Bandits(training)
    learner = Learner(training)
    actor = Actor(training)

    training.run(environments, data_collector, metric, bandits, learner, actor)
