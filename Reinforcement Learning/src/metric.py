import json
import shutil
import datetime
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class Metric:
    def __init__(self, config, env_name):
        self.config = config
        if config['load_run'] is None:
            abbreviation_dict = {'num_envs': 'env',
                                'batch_size': 'bs',
                                'sequence_length': 'ss',
                                'bootstrap_length': 'bb',
                                'learning_rate': 'lr',
                                'd_push': 'd_o',
                                'd_pull': 'd_i'}
            abbreviated_parameters = self.replace_keys(abbreviation_dict)
            parameter_string = ','.join([f'{key}{value}' for key, value in abbreviated_parameters.items()])
            timestamp = datetime.datetime.now().strftime('%b%d-%H-%M-%S')
            self.log_dir = f'runs/{env_name}/{parameter_string}_{timestamp}'
        else:
            self.log_dir = f'runs/{env_name}/{config["load_run"]}'
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.hyperparams_file = f'{self.log_dir}/hyperparameters.json'
        with open(self.hyperparams_file, 'w') as file:
            json.dump(config, file)
    

    def replace_keys(self, abbreviation_dict):
        return {abbreviation_dict[key]: value for key, value in self.config.items() if key in abbreviation_dict}


    def add_losses(self, losses, trained_frames):
        if losses is None:
            return
        for loss in losses:
            self.writer.add_scalar('loss/loss1', loss, global_step=trained_frames)
            self.writer.add_scalar('loss/loss2', loss, global_step=trained_frames)
            self.writer.add_scalar('v_loss/v_loss1', loss, global_step=trained_frames)
            self.writer.add_scalar('v_loss/v_loss2', loss, global_step=trained_frames)
            self.writer.add_scalar('q_loss/q_loss1', loss, global_step=trained_frames)
            self.writer.add_scalar('q_loss/q_loss2', loss, global_step=trained_frames)
            self.writer.add_scalar('p_loss/p_loss1', loss, global_step=trained_frames)
            self.writer.add_scalar('p_loss/p_loss2', loss, global_step=trained_frames)
            self.writer.add_scalar('gradient_norm/gradient_norm1', loss, global_step=trained_frames)
            self.writer.add_scalar('gradient_norm/gradient_norm2', loss, global_step=trained_frames)
        self.config['trained_frames'] = trained_frames
        with open(self.hyperparams_file, 'w') as file:
            json.dump(self.config, file)


    def add_return(self, returns, trained_frames):
        if returns is None:
            return
        self.writer.add_scalar('_return', np.mean(returns), global_step=trained_frames)


    def close_writer(self):
        self.writer.flush()
        self.writer.close()