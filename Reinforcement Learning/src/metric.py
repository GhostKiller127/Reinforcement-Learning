import os
import datetime
import json
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class Metric:
    def __init__(self, config, env_name, test_parameters):
        abbreviation_dict = {'num_envs': 'env',
                             'batch_size': 'bs',
                             'sequence_length': 'ss',
                             'bootstrap_length': 'bb',
                             'learning_rate': 'lr',
                             'd_push': 'd_o',
                             'd_pull': 'd_i'}
        test_parameters_abbreviated = self.replace_keys(test_parameters, abbreviation_dict)
        parameter_string = ','.join([f'{key}{value}' for key, value in test_parameters_abbreviated.items()])
        timestamp = datetime.datetime.now().strftime('%b%d-%H-%M-%S')
        self.log_dir = f'runs/{env_name}/{parameter_string}_{timestamp}'
        self.writer = SummaryWriter(log_dir=self.log_dir)

        hyperparams_file = os.path.join(self.log_dir, 'hyperparameters.json')
        with open(hyperparams_file, 'w') as file:
            json.dump(config, file)
    

    def replace_keys(self, first_dict, second_dict):
        return {second_dict[key]: value if key in second_dict else value for key, value in first_dict.items()}


    def add_losses(self, losses, num_frames):
        if losses is None:
            return
        for loss in losses:
            self.writer.add_scalar('loss/loss1', loss, global_step=num_frames)
            self.writer.add_scalar('loss/loss2', loss, global_step=num_frames)
            self.writer.add_scalar('v_loss/v_loss1', loss, global_step=num_frames)
            self.writer.add_scalar('v_loss/v_loss2', loss, global_step=num_frames)
            self.writer.add_scalar('q_loss/q_loss1', loss, global_step=num_frames)
            self.writer.add_scalar('q_loss/q_loss2', loss, global_step=num_frames)
            self.writer.add_scalar('p_loss/p_loss1', loss, global_step=num_frames)
            self.writer.add_scalar('p_loss/p_loss2', loss, global_step=num_frames)
            self.writer.add_scalar('gradient_norm/gradient_norm1', loss, global_step=num_frames)
            self.writer.add_scalar('gradient_norm/gradient_norm2', loss, global_step=num_frames)


    def add_return(self, returns, num_frames):
        if returns is None:
            return
        self.writer.add_scalar('_return', np.mean(returns), global_step=num_frames)


    def close_writer(self):
        self.writer.flush()
        self.writer.close()