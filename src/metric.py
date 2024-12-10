import os
import json
import math
import wandb
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

#region init

class Metric:
    def __init__(self, training_class, i=0):
        self.config = training_class.config
        self.log_dir = training_class.log_dir
        self.env_name = training_class.env_name
        if self.config['metrics']:
            self.initialize_wandb(i)
        self.writer = SummaryWriter(log_dir=self.log_dir, max_queue=1e10, flush_secs=30)

        self.counters = {
            'observations': 0,
            'train_return': 0,
            'train_val_stochastic': 0,
            'train_val_greedy': 0,
            'val_stochastic': 0,
            'val_greedy': 0,

            'cum_train_hist': 0,
            'cum_train_val_stoch_hist': 0,
            'cum_val_stoch_hist': 0,
            'cum_train_val_greedy_hist': 0,
            'cum_val_greedy_hist': 0,

            'act_freq_train': 0,
            'act_freq_train_val_stoch': 0,
            'act_freq_train_val_greedy': 0,
            'act_freq_val_stoch': 0,
            'act_freq_val_greedy': 0,

            'act_lens_train': 0,
            'act_lens_train_val_stoch': 0,
            'act_lens_train_val_greedy': 0,
            'act_lens_val_stoch': 0,
            'act_lens_val_greedy': 0,

            'act_rews_train': 0,
            'act_rews_train_val_stoch': 0,
            'act_rews_train_val_greedy': 0,
            'act_rews_val_stoch': 0,
            'act_rews_val_greedy': 0,

            'index_data': 0,
            'targets': 0,
            'losses': 0
        }
        if self.config['load_run'] is not None:
            self.load_state()


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

#endregion
#region helper

    def save_state(self):
        state_path = os.path.join(self.log_dir, 'metric_state.json')
        with open(state_path, 'w') as f:
            json.dump(self.counters, f)


    def load_state(self):
        state_path = os.path.join(self.log_dir, 'metric_state.json')
        with open(state_path, 'r') as f:
            self.counters = json.load(f)


    def should_log(self, counter):
        log_interval = math.ceil(np.sqrt(counter / 200))
        return counter % log_interval == 0


    def split_and_process(self, arr):
        splits = [
            arr[:self.config['train_envs']],  # train
            arr[self.config['train_envs']:self.config['train_envs'] + (self.config['val_envs'] // 4)],  # train_val_stochastic
            arr[self.config['train_envs'] + (self.config['val_envs'] // 4):-(self.config['val_envs'] // 2)],  # val_stochastic
            arr[-(self.config['val_envs'] // 2):-(self.config['val_envs'] // 4)],  # train_val_greedy
            arr[-(self.config['val_envs'] // 4):]  # val_greedy
        ]
        return [np.concatenate(split) for split in splits]
    

    def split_returns(self, val_returns, val_envs):
        if self.env_name == 'Crypto-v0':
            splits = [
                (lambda e: e < (self.config['train_envs'] + self.config['val_envs'] // 4)),  # train_val_stochastic
                (lambda e: (self.config['train_envs'] + self.config['val_envs'] // 4) <= e < (self.config['train_envs'] + self.config['val_envs'] // 2)),  # val_stochastic
                (lambda e: (self.config['train_envs'] + self.config['val_envs'] // 2) <= e < (self.config['train_envs'] + (3/2) * (self.config['val_envs'] // 2))),  # train_val_greedy
                (lambda e: e >= (self.config['train_envs'] + (3/2) * (self.config['val_envs'] // 2)))  # val_greedy
            ]
        else:
            splits = [
                (lambda e: e < (self.config['train_envs'] + self.config['val_envs'] // 2)),  # val_stochastic
                (lambda e: e >= (self.config['train_envs'] + self.config['val_envs'] // 2))  # val_greedy
            ]
        
        return [[val_returns[i] for i, env in enumerate(val_envs) if condition(env)] 
                for condition in splits]


    def close_writer(self):
        self.writer.flush()
        self.writer.close()
        wandb.finish()

#endregion
#region observations

    def add_observations(self, observations, played_frames):
        if self.config['metrics'] and self.env_name not in ['Crypto-v0']:
            self.counters['observations'] += 1
            if self.should_log(self.counters['observations']):
                self.writer.add_histogram('observations', observations, global_step=played_frames, bins=30)
                self.writer.add_scalar('observations/mean', np.mean(observations), global_step=played_frames)
                self.writer.add_scalar('observations/std', np.std(observations), global_step=played_frames)

#endregion
#region infos

    def add_infos(self, infos, played_frames):
        if self.config['metrics'] and self.env_name == 'Crypto-v0':
            cum_reward_history = []
            action_frequencies = []
            action_lengths = []
            action_rewards = []
            for info in infos:
                action_frequencies.append(info.get('action_frequency', np.empty(0)))
                action_lengths.append(info.get('action_lengths', np.empty(0)))
                action_rewards.append(info.get('action_rewards', np.empty(0)))
                if info['action_length'] != 0:
                    if info['data_version'] == 'sequential':
                        cum_reward_history.append(info['history']['cumulative_reward_history'][-1])
                    elif info['data_version'] == 'parallel':
                        cum_reward_history.append(info['history']['cumulative_reward_history'][-info['action_length']:].flatten())
                else:
                    cum_reward_history.append(np.empty(0))

            train_hist, train_val_stoch_hist, val_stoch_hist, train_val_greedy_hist, val_greedy_hist = self.split_and_process(cum_reward_history)
            train_freq, train_val_stoch_freq, val_stoch_freq, train_val_greedy_freq, val_greedy_freq = self.split_and_process(action_frequencies)
            train_lens, train_val_stoch_lens, val_stoch_lens, train_val_greedy_lens, val_greedy_lens = self.split_and_process(action_lengths)
            train_rews, train_val_stoch_rews, val_stoch_rews, train_val_greedy_rews, val_greedy_rews = self.split_and_process(action_rewards)
#endregion
#region cumulative reward

            if train_hist.size != 0:
                self.counters['cum_train_hist'] += 1
                if self.should_log(self.counters['cum_train_hist']):
                    self.writer.add_histogram('cumulative reward history/train', train_hist, global_step=played_frames, bins=30)
                    self.writer.add_scalar('cumulative reward history/train_mean', np.mean(train_hist), global_step=played_frames)
                    self.writer.add_scalar('cumulative reward history/train_std', np.std(train_hist), global_step=played_frames)

            if train_val_stoch_hist.size != 0:
                self.counters['cum_train_val_stoch_hist'] += 1
                if self.should_log(self.counters['cum_train_val_stoch_hist']):
                    self.writer.add_histogram('cumulative reward history/train_val_stochastic', train_val_stoch_hist, global_step=played_frames, bins=30)
                    self.writer.add_scalar('cumulative reward history/train_val_stochastic_mean', np.mean(train_val_stoch_hist), global_step=played_frames)
                    self.writer.add_scalar('cumulative reward history/train_val_stochastic_std', np.std(train_val_stoch_hist), global_step=played_frames)

            if val_stoch_hist.size != 0:
                self.counters['cum_val_stoch_hist'] += 1
                if self.should_log(self.counters['cum_val_stoch_hist']):
                    self.writer.add_histogram('cumulative reward history/val_stochastic', val_stoch_hist, global_step=played_frames, bins=30)
                    self.writer.add_scalar('cumulative reward history/val_stochastic_mean', np.mean(val_stoch_hist), global_step=played_frames)
                    self.writer.add_scalar('cumulative reward history/val_stochastic_std', np.std(val_stoch_hist), global_step=played_frames)

            if train_val_greedy_hist.size != 0:
                self.counters['cum_train_val_greedy_hist'] += 1
                if self.should_log(self.counters['cum_train_val_greedy_hist']):
                    self.writer.add_histogram('cumulative reward history/train_val_greedy', train_val_greedy_hist, global_step=played_frames, bins=30)
                    self.writer.add_scalar('cumulative reward history/train_val_greedy_mean', np.mean(train_val_greedy_hist), global_step=played_frames)
                    self.writer.add_scalar('cumulative reward history/train_val_greedy_std', np.std(train_val_greedy_hist), global_step=played_frames)

            if val_greedy_hist.size != 0:
                self.counters['cum_val_greedy_hist'] += 1
                if self.should_log(self.counters['cum_val_greedy_hist']):
                    self.writer.add_histogram('cumulative reward history/val_greedy', val_greedy_hist, global_step=played_frames, bins=30)
                    self.writer.add_scalar('cumulative reward history/val_greedy_mean', np.mean(val_greedy_hist), global_step=played_frames)
                    self.writer.add_scalar('cumulative reward history/val_greedy_std', np.std(val_greedy_hist), global_step=played_frames)

#endregion
#region action frequency

            if train_freq.size != 0:
                self.counters['act_freq_train'] += 1
                if self.should_log(self.counters['act_freq_train']):
                    self.writer.add_scalar('action frequency/train', np.mean(train_freq), global_step=played_frames)
        
            if train_val_stoch_freq.size != 0:
                self.counters['act_freq_train_val_stoch'] += 1
                if self.should_log(self.counters['act_freq_train_val_stoch']):
                    self.writer.add_scalar('action frequency/train_val_stochastic', np.mean(train_val_stoch_freq), global_step=played_frames)
            
            if val_stoch_freq.size != 0:
                self.counters['act_freq_val_stoch'] += 1
                if self.should_log(self.counters['act_freq_val_stoch']):
                    self.writer.add_scalar('action frequency/val_stochastic', np.mean(val_stoch_freq), global_step=played_frames)
            
            if train_val_greedy_freq.size != 0:
                self.counters['act_freq_train_val_greedy'] += 1
                if self.should_log(self.counters['act_freq_train_val_greedy']):
                    self.writer.add_scalar('action frequency/train_val_greedy', np.mean(train_val_greedy_freq), global_step=played_frames)
            
            if val_greedy_freq.size != 0:
                self.counters['act_freq_val_greedy'] += 1
                if self.should_log(self.counters['act_freq_val_greedy']):
                    self.writer.add_scalar('action frequency/val_greedy', np.mean(val_greedy_freq), global_step=played_frames)

#endregion
#region action length

            if train_lens.size != 0:
                self.counters['act_lens_train'] += 1
                if self.should_log(self.counters['act_lens_train']):
                    self.writer.add_scalar('action length/train', np.mean(train_lens), global_step=played_frames)
            
            if train_val_stoch_lens.size != 0:
                self.counters['act_lens_train_val_stoch'] += 1
                if self.should_log(self.counters['act_lens_train_val_stoch']):
                    self.writer.add_scalar('action length/train_val_stochastic', np.mean(train_val_stoch_lens), global_step=played_frames)

            if val_stoch_lens.size != 0:
                self.counters['act_lens_val_stoch'] += 1
                if self.should_log(self.counters['act_lens_val_stoch']):
                    self.writer.add_scalar('action length/val_stochastic', np.mean(val_stoch_lens), global_step=played_frames)

            if train_val_greedy_lens.size != 0:
                self.counters['act_lens_train_val_greedy'] += 1
                if self.should_log(self.counters['act_lens_train_val_greedy']):
                    self.writer.add_scalar('action length/train_val_greedy', np.mean(train_val_greedy_lens), global_step=played_frames)

            if val_greedy_lens.size != 0:
                self.counters['act_lens_val_greedy'] += 1
                if self.should_log(self.counters['act_lens_val_greedy']):
                    self.writer.add_scalar('action length/val_greedy', np.mean(val_greedy_lens), global_step=played_frames)

#endregion
#region action reward

            if train_rews.size != 0:
                self.counters['act_rews_train'] += 1
                if self.should_log(self.counters['act_rews_train']):
                    self.writer.add_scalar('action reward/train', np.mean(train_rews), global_step=played_frames)

            if train_val_stoch_rews.size != 0:
                self.counters['act_rews_train_val_stoch'] += 1
                if self.should_log(self.counters['act_rews_train_val_stoch']):
                    self.writer.add_scalar('action reward/train_val_stochastic', np.mean(train_val_stoch_rews), global_step=played_frames)

            if val_stoch_rews.size != 0:
                self.counters['act_rews_val_stoch'] += 1
                if self.should_log(self.counters['act_rews_val_stoch']):
                    self.writer.add_scalar('action reward/val_stochastic', np.mean(val_stoch_rews), global_step=played_frames)
        
            if train_val_greedy_rews.size != 0:
                self.counters['act_rews_train_val_greedy'] += 1
                if self.should_log(self.counters['act_rews_train_val_greedy']):
                    self.writer.add_scalar('action reward/train_val_greedy', np.mean(train_val_greedy_rews), global_step=played_frames)

            if val_greedy_rews.size != 0:
                self.counters['act_rews_val_greedy'] += 1
                if self.should_log(self.counters['act_rews_val_greedy']):
                    self.writer.add_scalar('action reward/val_greedy', np.mean(val_greedy_rews), global_step=played_frames)

#endregion
#region returns

    def add_train_return(self, train_returns, played_frames):
        if train_returns.size != 0 and self.config['metrics']:
            self.counters['train_return'] += 1
            if self.should_log(self.counters['train_return']):
                self.writer.add_scalar('_return/train', np.mean(train_returns), global_step=played_frames)

    
    def add_val_return(self, val_returns, val_envs, played_frames):
        if self.config['metrics']:
            returns = self.split_returns(val_returns, val_envs)

            if self.env_name == 'Crypto-v0':
                train_val_stoch, val_stoch, train_val_greedy, val_greedy = returns

                if train_val_stoch:
                    self.counters['train_val_stochastic'] += 1
                    if self.should_log(self.counters['train_val_stochastic']):
                        self.writer.add_scalar('_return/train_val_stochastic', np.mean(train_val_stoch), global_step=played_frames)
                
                if train_val_greedy:
                    self.counters['train_val_greedy'] += 1
                    if self.should_log(self.counters['train_val_greedy']):
                        self.writer.add_scalar('_return/train_val_greedy', np.mean(train_val_greedy), global_step=played_frames)

            else:
                val_stoch, val_greedy = returns

            if val_stoch:
                self.counters['val_stochastic'] += 1
                if self.should_log(self.counters['val_stochastic']):
                    self.writer.add_scalar('_return/val_stochastic', np.mean(val_stoch), global_step=played_frames)
            
            if val_greedy:
                self.counters['val_greedy'] += 1
                if self.should_log(self.counters['val_greedy']):
                    self.writer.add_scalar('_return/val_greedy', np.mean(val_greedy), global_step=played_frames)

#endregion
#region index data

    def add_index_data(self, index_data, played_frames):
        if index_data is not None and self.config['metrics']:
            self.counters['index_data'] += 1
            if self.should_log(self.counters['index_data']):
                tau1, tau2, epsilon, n1, n2, n3, w1, w2, w3 = index_data
                self.writer.add_scalar('bandit/max_tau1', tau1, global_step=played_frames)
                self.writer.add_scalar('bandit/max_tau2', tau2, global_step=played_frames)
                self.writer.add_scalar('bandit/max_epsilon', epsilon, global_step=played_frames)
                # self.writer.add_histogram('bandit index count/tau1', n1, global_step=played_frames)
                # self.writer.add_histogram('bandit index count/tau2', n2, global_step=played_frames)
                # self.writer.add_histogram('bandit index count/epsilon', n3, global_step=played_frames)
                # self.writer.add_histogram('bandit index weight/tau1', w1, global_step=played_frames)
                # self.writer.add_histogram('bandit index weight/tau2', w2, global_step=played_frames)
                # self.writer.add_histogram('bandit index weight/epsilon', w3, global_step=played_frames)

#endregion
#region targets

    def add_targets(self, targets, played_frames):
        if targets is not None and self.config['metrics']:
            self.counters['targets'] += 1
            if self.should_log(self.counters['targets']):
                rt1, rt2, vt1, vt2, pt1, pt2 = targets
                self.writer.add_histogram('retrace targets/rt1', rt1, global_step=played_frames, bins=30)
                self.writer.add_histogram('retrace targets/rt2', rt2, global_step=played_frames, bins=30)
                self.writer.add_histogram('vtrace targets/vt1', vt1, global_step=played_frames, bins=30)
                self.writer.add_histogram('vtrace targets/vt2', vt2, global_step=played_frames, bins=30)
                self.writer.add_histogram('policy targets/pt1', pt1, global_step=played_frames, bins=30)
                self.writer.add_histogram('policy targets/pt2', pt2, global_step=played_frames, bins=30)
                self.writer.add_scalar('retrace targets/rt1_mean', np.mean(rt1), global_step=played_frames)
                self.writer.add_scalar('retrace targets/rt1_std', np.std(rt1), global_step=played_frames)
                self.writer.add_scalar('retrace targets/rt2_mean', np.mean(rt2), global_step=played_frames)
                self.writer.add_scalar('retrace targets/rt2_std', np.std(rt2), global_step=played_frames)
                self.writer.add_scalar('vtrace targets/vt1_mean', np.mean(vt1), global_step=played_frames)
                self.writer.add_scalar('vtrace targets/vt1_std', np.std(vt1), global_step=played_frames)
                self.writer.add_scalar('vtrace targets/vt2_mean', np.mean(vt2), global_step=played_frames)
                self.writer.add_scalar('vtrace targets/vt2_std', np.std(vt2), global_step=played_frames)
                self.writer.add_scalar('policy targets/pt1_mean', np.mean(pt1), global_step=played_frames)
                self.writer.add_scalar('policy targets/pt1_std', np.std(pt1), global_step=played_frames)
                self.writer.add_scalar('policy targets/pt2_mean', np.mean(pt2), global_step=played_frames)
                self.writer.add_scalar('policy targets/pt2_std', np.std(pt2), global_step=played_frames)

#endregion
#region losses

    def add_losses(self, losses, played_frames):
        if losses is not None and self.config['metrics']:
            self.counters['losses'] += 1
            if self.should_log(self.counters['losses']):
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

#endregion