import os
import wandb
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class Metric:
    def __init__(self, training_class, i=0):
        self.config = training_class.config
        self.log_dir = training_class.log_dir
        self.env_name = training_class.env_name
        if self.config['metrics']:
            self.initialize_wandb(i)
        self.writer = SummaryWriter(log_dir=self.log_dir, max_queue=1000000000, flush_secs=300)
    

    def initialize_wandb(self, i):
        self.run_name = '/'.join(self.log_dir.split('/')[-1:])
        if self.config['load_run'] is None:
            self.config['wandb_id'] = wandb.util.generate_id()
        if i == 0:
            wandb.tensorboard.patch(root_logdir=self.log_dir)
        wandb.init(project=self.env_name,
                   name=self.run_name,
                   id=self.config['wandb_id'],
                   resume='allow',
                   dir='../',
                   config=self.config)
        os.remove('../wandb/debug.log')
        os.remove('../wandb/debug-internal.log')

    
    def add_train_return(self, train_returns, played_frames):
        if train_returns.size != 0 and self.config['metrics']:
            self.writer.add_scalar('_return/train', np.mean(train_returns), global_step=played_frames)

    
    def add_val_return(self, val_returns, val_envs, played_frames):
        if self.config['metrics']:
            stochastic_returns = [val_returns[i] for i, env in enumerate(val_envs) if env < (self.config['num_envs'] - self.config['val_envs'] // 2)]
            greedy_returns = [val_returns[i] for i, env in enumerate(val_envs) if env >= (self.config['num_envs'] - self.config['val_envs'] // 2)]
            if stochastic_returns:
                self.writer.add_scalar('_return/val_stochastic', np.mean(stochastic_returns), global_step=played_frames)
            if greedy_returns:
                self.writer.add_scalar('_return/val_greedy', np.mean(greedy_returns), global_step=played_frames)

    
    def add_index_data(self, index_data, played_frames):
        if index_data is not None and self.config['metrics']:
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


    def add_targets(self, targets, played_frames):
        if targets is not None and self.config['metrics']:
            rt1, rt2, vt1, vt2, pt1, pt2 = targets
            self.writer.add_histogram('retrace targets/rt1', rt1, global_step=played_frames, bins=20)
            self.writer.add_histogram('retrace targets/rt2', rt2, global_step=played_frames, bins=20)
            self.writer.add_histogram('vtrace targets/vt1', vt1, global_step=played_frames, bins=20)
            self.writer.add_histogram('vtrace targets/vt2', vt2, global_step=played_frames, bins=20)
            self.writer.add_histogram('policy targets/pt1', pt1, global_step=played_frames, bins=20)
            self.writer.add_histogram('policy targets/pt2', pt2, global_step=played_frames, bins=20)


    def add_losses(self, losses, played_frames):
        if losses is not None and self.config['metrics']:
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


    def close_writer(self):
        self.writer.flush()
        self.writer.close()
        wandb.finish()