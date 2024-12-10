import os
import signal
import asyncio
import warnings
import itertools
import traceback
from numpy import ComplexWarning
warnings.filterwarnings("ignore", category=ComplexWarning)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
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

#region configs

# env_name = 'CartPole-v1'
# env_name = 'LunarLander-v2'
# env_name = 'LaserHockey-v0'
env_name = 'Crypto-v0'

# if you want to continue a run, 'load_run' and 'train_frames' need to be specified. the rest will be overwritten
# Add more configuration dictionaries as needed
configs = [
    {
        "train_parameters": {
            # "load_run": 'S5,2,128,128,8,g0.98,y0.4,y0.5,x500.0,y0.4,x500.0,par,s100,f1,15min',
            "train_frames": 100000000,
            # "per_min_frames": 100000,
            # "metrics": False,
        },
        "run_name": {
            "prefix": 'S5',
            "suffix": 'par,s100,f0.55,15min',
            # "prefix": 'dense',
            # "suffix": 'test',
            "timestamp": False,
            "parameters": {
                "S5": {
                    "n_layers": '',
                    "d_model": '',
                    "decoder_dim": '',
                    "blocks": '',
                    "dropout": 'd',
                }},
                # "dense_jax": {
                #     "hidden_dim": 'd',
                # }},
            # "per_priority_exponent": 'p',
            # "importance_sampling_exponent": 'i',
            # "bootstrap_length": 'b',
            "discount": 'g',
            "reward_scaling_y1": 'y',
            "reward_scaling_y2": 'y',
            "reward_scaling_x": 'x',
            "cumulative_reward_scaling_y": 'y',
            "cumulative_reward_scaling_x": 'x',
            # "v_loss_scaling": 'v',
            # "q_loss_scaling": 'q',
            # "p_loss_scaling": 'p',
        }
    },
    # {
    #     "train_parameters": {
    #         "train_frames": 200000,
    #     },
    #     "run_name": {
    #         "prefix": 'dense',
    #         "suffix": 'test_2',
    #         "timestamp": False,
    #         "parameters": {
    #             "dense_jax": {
    #                 "hidden_dim": 'd',
    #             }},
    #         "per_priority_exponent": 'p',
    #         "importance_sampling_exponent": 'i',
    #         "v_loss_scaling": 'v',
    #         "q_loss_scaling": 'q',
    #         "p_loss_scaling": 'p',
    #     }
    # }
]

#endregion
#region run

async def run_training():
    stop = [False]
    def signal_handler():
        stop[0] = True

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    for i, config in enumerate(configs):
        try:
            training = Training(env_name, config['train_parameters'], config['run_name'])
            environments = Environments(training)
            data_collector = DataCollector(training)
            metric = Metric(training, i)
            bandits = Bandits(training)
            learner = await Learner(training).initialize()
            actor = Actor(training)

            await training.run(environments, data_collector, metric, bandits, learner, actor, stop)

        except Exception as e:
            print(f"\nError occurred during training iteration {i}:")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            environments.close()
            metric.close_writer()


if __name__ == "__main__":
    asyncio.run(run_training())