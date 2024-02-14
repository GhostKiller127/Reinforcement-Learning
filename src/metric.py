import json
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class Metric:
    def __init__(self, training_class):
        self.config = training_class.config
        self.log_dir = training_class.log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.hyperparams_file = f'{self.log_dir}/hyperparameters.json'
        with open(self.hyperparams_file, 'w') as file:
            json.dump(self.config, file)
    

    def add_return(self, data_collector, returns, terminated_envs, played_frames):
        if returns is None:
            return
        self.writer.add_scalar('_return', np.mean(returns), global_step=played_frames)
        for _, env in enumerate(terminated_envs):
            if env == self.config['num_envs'] - 2 and data_collector.frame_count >= self.config['per_min_frames']:
                self.writer.add_scalar('_return/stochastic', returns[_], global_step=played_frames)
            if env == self.config['num_envs'] - 1 and data_collector.frame_count >= self.config['per_min_frames']:
                self.writer.add_scalar('_return/argmax', returns[_], global_step=played_frames)

    
    def add_index_data(self, index_data, played_frames):
        if index_data is None:
            return
        tau1, tau2, epsilon, n1, n2, n3, w1, w2, w3 = index_data
        # self.writer.add_histogram('bandit index count/tau1', n1, global_step=played_frames)
        # self.writer.add_histogram('bandit index count/tau2', n2, global_step=played_frames)
        # self.writer.add_histogram('bandit index count/epsilon', n3, global_step=played_frames)
        # self.writer.add_histogram('bandit index weight/tau1', w1, global_step=played_frames)
        # self.writer.add_histogram('bandit index weight/tau2', w2, global_step=played_frames)
        # self.writer.add_histogram('bandit index weight/epsilon', w3, global_step=played_frames)
        self.writer.add_scalar('bandit/max_tau1', tau1, global_step=played_frames)
        self.writer.add_scalar('bandit/max_tau2', tau2, global_step=played_frames)
        self.writer.add_scalar('bandit/max_epsilon', epsilon, global_step=played_frames)


    def add_losses(self, losses, played_frames):
        if losses is None:
            return
        l1, l2, v_l1, v_l2, q_l1, q_l2, p_l1, p_l2, norm1, norm2, lr = losses
        self.writer.add_scalar('loss/loss1', l1, global_step=played_frames)
        self.writer.add_scalar('loss/loss2', l2, global_step=played_frames)
        self.writer.add_scalar('v_loss/v_loss1', v_l1, global_step=played_frames)
        self.writer.add_scalar('v_loss/v_loss2', v_l2, global_step=played_frames)
        self.writer.add_scalar('q_loss/q_loss1', q_l1, global_step=played_frames)
        self.writer.add_scalar('q_loss/q_loss2', q_l2, global_step=played_frames)
        self.writer.add_scalar('p_loss/p_loss1', p_l1, global_step=played_frames)
        self.writer.add_scalar('p_loss/p_loss2', p_l2, global_step=played_frames)
        self.writer.add_scalar('gradient_norm/gradient_norm1', norm1, global_step=played_frames)
        self.writer.add_scalar('gradient_norm/gradient_norm2', norm2, global_step=played_frames)
        self.writer.add_scalar('learning_rate/learning_rate', lr, global_step=played_frames)
        self.config['played_frames'] = played_frames
        with open(self.hyperparams_file, 'w') as file:
            json.dump(self.config, file)


    def close_writer(self):
        self.writer.flush()
        self.writer.close()