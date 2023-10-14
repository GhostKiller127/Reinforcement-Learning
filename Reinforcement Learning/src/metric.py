import os
import datetime
import json
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class Metric:
    def __init__(self, config, env_name, test_parameters):
        abbreviation_dict = {"num_steps": 's',
                             "num_envs": 'n',
                             "batch_size": 's',
                             "sequence_length": 'ss',
                             "bootstrap_length": 'bb',
                             "d_push": 'out',
                             "d_pull": 'in'}
        test_parameters_abbreviated = self.replace_keys(test_parameters, abbreviation_dict)
        parameter_string = ','.join([f'{key}{value}' for key, value in test_parameters_abbreviated.items()])
        timestamp = datetime.datetime.now().strftime("%b%d-%H-%M-%S")
        run_name = f"{parameter_string}_{timestamp}"

        self.log_dir = f"runs/{env_name}/{run_name}"
        self.writer = SummaryWriter(log_dir=self.log_dir)

        hyperparams_file = os.path.join(self.log_dir, 'hyperparameters.json')
        with open(hyperparams_file, 'w') as file:
            json.dump(config, file)

        self.metrics = {
            'accuracy/agent1': 0,
            'accuracy/agent2': 0,
            'loss/train': 0,
            'loss/val': 0
        }
    
    def replace_keys(self, first_dict, second_dict):
        return {second_dict[key]: value if key in second_dict else value for key, value in first_dict.items()}

    def add_scalars(self, data, step):
        for name, value in data.items():
            self.writer.add_scalar(name, value, global_step=step)
    
    def add_losses(self, train_loss, val_loss, step):
        self.writer.add_scalar('loss/train', train_loss, global_step=step)
        self.writer.add_scalar('loss/val', val_loss, global_step=step)

    def add_return(self, returns, step):
        if returns is None:
            return
        self.writer.add_scalar('return', np.mean(returns), global_step=step)

    def close_writer(self):
        self.writer.flush()
        self.writer.close()