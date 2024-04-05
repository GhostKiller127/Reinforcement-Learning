import os
import numpy as np
import torch
import torch.nn.functional as F
from architectures import DenseModel



class Learner:
    def __init__(self, training_class):
        self.update_count = 0
        self.step_count = 0
        self.device = training_class.device
        self.config = training_class.config
        self.log_dir = f'{training_class.log_dir}/models'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config['mixed_precision'])
        if self.config['architecture'] == 'dense':
            self.learner1 = DenseModel(self.config['dense_params'], self.device)
            self.learner2 = DenseModel(self.config['dense_params'], self.device)
            self.target1 = DenseModel(self.config['dense_params'], self.device)
            self.target2 = DenseModel(self.config['dense_params'], self.device)
        for param1, param2 in zip(self.target1.parameters(), self.target2.parameters()):
            param1.requires_grad_(False)
            param2.requires_grad_(False)
        initial_lr, end_lr, gamma = self.get_scheduler_args()
        self.optimizer1 = torch.optim.AdamW(self.learner1.parameters(), lr=initial_lr, eps=self.config['adamw_epsilon'])
        self.optimizer2 = torch.optim.AdamW(self.learner2.parameters(), lr=initial_lr, eps=self.config['adamw_epsilon'])
        self.scheduler1 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer1, gamma=(end_lr/initial_lr)**gamma)
        self.scheduler2 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer2, gamma=(end_lr/initial_lr)**gamma)
        if self.config['load_run'] is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.save_weights_and_data_collector()
        else:
            self.load_weights()
        self.update_target_weights()

    
    def get_scheduler_args(self):
        if self.config['load_run'] is not None:
            self.config['warmup_steps'] = 0
            return self.config["learning_rate"], self.config["learning_rate"], 0
        if self.config["lr_finder"]:
            return 1e-8, 1e-2, 1 / self.config['warmup_steps']
        self.config['warmup_steps'] = int(self.config['warmup_steps'] / 4)
        return self.config["learning_rate"] / 1e3, self.config["learning_rate"], 1 / self.config['warmup_steps']


    def save_weights_and_data_collector(self, data_collector=None):
        if self.update_count % self.config['d_push'] == 0:
            torch.save(self.learner1.state_dict(), f'{self.log_dir}/learner1.pth')
            torch.save(self.learner2.state_dict(), f'{self.log_dir}/learner2.pth')
            torch.save(self.optimizer1.state_dict(), f'{self.log_dir}/optimizer1.pth')
            torch.save(self.optimizer2.state_dict(), f'{self.log_dir}/optimizer2.pth')
            if data_collector is not None and not self.config['lr_finder']:
                data_collector.save_data_collector()


    def load_weights(self):
        self.learner1.load_state_dict(torch.load(f'{self.log_dir}/learner1.pth'))
        self.learner2.load_state_dict(torch.load(f'{self.log_dir}/learner2.pth'))
        self.optimizer1.load_state_dict(torch.load(f'{self.log_dir}/optimizer1.pth'))
        self.optimizer2.load_state_dict(torch.load(f'{self.log_dir}/optimizer2.pth'))


    def update_target_weights(self):
        if self.update_count % self.config['d_target'] == 0:
            self.target1.load_state_dict(self.learner1.state_dict())
            self.target2.load_state_dict(self.learner2.state_dict())


    def calculate_values(self, batch):
        v1, a1 = self.learner1(batch['o'])
        v2, a2 = self.learner2(batch['o'])
        v1_, a1_ = self.target1(batch['o'])
        v2_, a2_ = self.target2(batch['o'])

        i = torch.tensor(batch['i'], dtype=torch.float32).to(self.device)
        a = torch.tensor(batch['a'], dtype=torch.int64).to(self.device).unsqueeze(2)
        tau1 = i[:,:,0].unsqueeze(2)
        tau2 = i[:,:,1].unsqueeze(2)
        epsilon = i[:,:,2].unsqueeze(2)
        softmax_1 = F.softmax(a1 * tau1, dim=2)
        softmax_2 = F.softmax(a2 * tau2, dim=2)
        softmax_1_ = F.softmax(a1_ * tau1, dim=2)
        softmax_2_ = F.softmax(a2_ * tau2, dim=2)
        policy1 = epsilon * softmax_1 + (1 - epsilon) * softmax_2.detach()
        policy2 = epsilon * softmax_1.detach() + (1 - epsilon) * softmax_2
        policy_ = epsilon * softmax_1_ + (1 - epsilon) * softmax_2_
        p1 = policy1.gather(dim=2, index=a)
        p2 = policy2.gather(dim=2, index=a)

        a1 = a1.gather(dim=2, index=a).add_(torch.sum(policy1.detach() * a1, dim=2).unsqueeze(2))
        a2 = a2.gather(dim=2, index=a).add_(torch.sum(policy2.detach() * a2, dim=2).unsqueeze(2))
        q1 = a1 + v1.detach()
        q2 = a2 + v2.detach()
        a1_.add_(torch.sum(policy_ * a1_, dim=2).unsqueeze(2))
        a2_.add_(torch.sum(policy_ * a2_, dim=2).unsqueeze(2))
        q1_ = a1_ + v1_
        q2_ = a2_ + v2_
        q1_m = torch.sum(policy1.detach() * q1_, dim=2).unsqueeze(2)
        q2_m = torch.sum(policy2.detach() * q2_, dim=2).unsqueeze(2)
        q1_ = q1_.gather(dim=2, index=a)
        q2_ = q2_.gather(dim=2, index=a)

        return v1, v2, q1, q2, p1, p2, v1_, v2_, q1_, q2_, q1_m, q2_m
    

    def scale_value1(self, x):
        x_log = np.log(np.abs(x) + 1) * self.config['reward_scaling_1']
        x = np.where(np.sign(x) > 0, x_log * 2, x_log * -1)
        return x

    def scale_value2(self, x):
        x = (np.sign(x) * ((np.abs(x) + 1.)**(1/2) - 1.) + 0.001 * x) * self.config['reward_scaling_2']
        return x

    def invert_scale1(self, h):
        h = h / self.config['reward_scaling_1']
        h_ = np.where(np.sign(h) > 0, np.abs(h) / 2, np.abs(h))
        h_ = np.sign(h) * (np.exp(h_) - 1)
        return h_

    def invert_scale2(self, h):
        h = h / self.config['reward_scaling_2']
        h = np.sign(h) * ((((1 + 4*0.001*(np.abs(h) + 1 + 0.001))**(1/2) - 1) / (2*0.001))**2 - 1)
        return h
    

    def calculate_retrace_targets(self, q1, q2, q1_, q2_, q1_m, q2_m, p, batch):
        q1 = q1.detach().cpu().numpy().squeeze()
        q2 = q2.detach().cpu().numpy().squeeze()
        q1_ = q1_.cpu().numpy().squeeze()
        q2_ = q2_.cpu().numpy().squeeze()
        q1_ = self.invert_scale1(q1_)
        q2_ = self.invert_scale2(q2_)
        q1_m = q1_m.cpu().numpy().squeeze()
        q2_m = q2_m.cpu().numpy().squeeze()
        q1_m = self.invert_scale1(q1_m)
        q2_m = self.invert_scale2(q2_m)
        p = p.detach().cpu().numpy()
        c = np.minimum(p / batch['a_p'], self.config['c_clip']).squeeze()
        
        c = np.lib.stride_tricks.sliding_window_view(c[:, :-2], (self.config['batch_size'], self.config['bootstrap_length'])).squeeze()
        c = np.transpose(c, (1, 0, 2)).copy()
        c[:,:,0] = 1
        c = np.cumprod(c, axis=2)
        gamma = self.config['discount'] * np.ones((self.config['batch_size'], self.config['sequence_length'], self.config['bootstrap_length']))
        gamma[:,:,0] = 1
        gamma = np.cumprod(gamma, axis=2)

        mask = np.logical_or(batch['d'], batch['t'])[:, :-2]
        mask = np.lib.stride_tricks.sliding_window_view(mask, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))
        mask = np.cumsum(mask, axis=2)
        mask_e = np.array(mask, dtype=bool)
        mask_t = np.roll(mask_e, shift=1, axis=2)
        mask_t[:,:,0] = False
        mask_e = np.invert(mask_e)
        mask_t = np.invert(mask_t)

        td1_t = batch['r'][:, :-2] - q1_[:, :-2]
        td2_t = batch['r'][:, :-2] - q2_[:, :-2]
        td1_e = self.config['discount'] * q1_m[:, 1:-1]
        td2_e = self.config['discount'] * q2_m[:, 1:-1]
        td1_t = np.lib.stride_tricks.sliding_window_view(td1_t, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))
        td2_t = np.lib.stride_tricks.sliding_window_view(td2_t, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))
        td1_e = np.lib.stride_tricks.sliding_window_view(td1_e, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))
        td2_e = np.lib.stride_tricks.sliding_window_view(td2_e, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))

        td1 = td1_t * mask_t + td1_e * mask_e
        td2 = td2_t * mask_t + td2_e * mask_e
        rt1 = q1_[:, :self.config['sequence_length']] + np.sum(gamma * c * td1, axis=2)
        rt2 = q2_[:, :self.config['sequence_length']] + np.sum(gamma * c * td2, axis=2)

        rt1 = self.scale_value1(rt1)
        rt2 = self.scale_value2(rt2)
        rtd1 = rt1 - q1[:, :self.config['sequence_length']]
        rtd2 = rt2 - q2[:, :self.config['sequence_length']]
        rt1 = torch.tensor(rt1).to(self.device).unsqueeze(2)
        rt2 = torch.tensor(rt2).to(self.device).unsqueeze(2)
        return rt1, rt2, rtd1, rtd2
    
    
    def calculate_vtrace_targets(self, v1_, v2_, p, batch):
        v1_ = v1_.cpu().numpy().squeeze()
        v2_ = v2_.cpu().numpy().squeeze()
        v1_ = self.invert_scale1(v1_)
        v2_ = self.invert_scale2(v2_)
        p = p.detach().cpu().numpy()
        c = np.minimum(p / batch['a_p'], self.config['c_clip']).squeeze()
        rho_t = np.minimum(p / batch['a_p'], self.config['rho_clip']).squeeze()

        rho = np.lib.stride_tricks.sliding_window_view(rho_t[:, :-1], (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))
        c = np.lib.stride_tricks.sliding_window_view(c[:, :-1], (self.config['batch_size'], self.config['bootstrap_length'])).squeeze()
        c = np.transpose(c, (1, 0, 2)).copy()
        c = np.roll(c, shift=1, axis=2)
        c[:,:,0] = 1
        c = np.cumprod(c, axis=2)

        gamma = self.config['discount'] * np.ones((self.config['batch_size'], self.config['sequence_length'] + 1, self.config['bootstrap_length']))
        gamma[:,:,0] = 1
        gamma = np.cumprod(gamma, axis=2)

        mask = np.logical_or(batch['d'], batch['t'])[:, :-1]
        mask = np.lib.stride_tricks.sliding_window_view(mask, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))
        mask = np.cumsum(mask, axis=2)
        mask_e = np.array(mask, dtype=bool)
        mask_t = np.roll(mask_e, shift=1, axis=2)
        mask_t[:,:,0] = False
        mask_e = np.invert(mask_e)
        mask_t = np.invert(mask_t)

        td1_t = batch['r'][:, :-1] - v1_[:, :-1]
        td2_t = batch['r'][:, :-1] - v2_[:, :-1]
        td1_e = self.config['discount'] * v1_[:, 1:]
        td2_e = self.config['discount'] * v2_[:, 1:]
        td1_t = np.lib.stride_tricks.sliding_window_view(td1_t, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))
        td2_t = np.lib.stride_tricks.sliding_window_view(td2_t, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))
        td1_e = np.lib.stride_tricks.sliding_window_view(td1_e, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))
        td2_e = np.lib.stride_tricks.sliding_window_view(td2_e, (self.config['batch_size'], self.config['bootstrap_length'])).squeeze().transpose((1, 0, 2))

        td1 = td1_t * mask_t + td1_e * mask_e
        td2 = td2_t * mask_t + td2_e * mask_e
        vt1 = v1_[:, :self.config['sequence_length'] + 1] + np.sum(gamma * c * rho * td1, axis=2)
        vt2 = v2_[:, :self.config['sequence_length'] + 1] + np.sum(gamma * c * rho * td2, axis=2)

        pt1 = (rho_t[:, :self.config['sequence_length']] * 
               (batch['r'][:, :self.config['sequence_length']] + self.config['discount'] * vt1[:, 1:] - v1_[:, :self.config['sequence_length']]))
        pt2 = (rho_t[:, :self.config['sequence_length']] * 
               (batch['r'][:, :self.config['sequence_length']] + self.config['discount'] * vt2[:, 1:] - v2_[:, :self.config['sequence_length']]))

        vt1 = self.scale_value1(vt1)
        vt2 = self.scale_value2(vt2)
        pt1 = self.scale_value1(pt1)
        pt2 = self.scale_value2(pt2)
        vt1 = torch.tensor(vt1[:,:-1]).to(self.device).unsqueeze(2)
        vt2 = torch.tensor(vt2[:,:-1]).to(self.device).unsqueeze(2)
        pt1 = torch.tensor(pt1).to(self.device).unsqueeze(2)
        pt2 = torch.tensor(pt2).to(self.device).unsqueeze(2)
        return vt1, vt2, pt1, pt2


    def calculate_losses(self, v1, v2, q1, q2, p1, p2, rt1, rt2, vt1, vt2, pt1, pt2):
        v_loss1 = torch.mean((vt1 - v1[:,:self.config['sequence_length']])**2)
        v_loss2 = torch.mean((vt2 - v2[:,:self.config['sequence_length']])**2)
        q_loss1 = torch.mean((rt1 - q1[:,:self.config['sequence_length']])**2)
        q_loss2 = torch.mean((rt2 - q2[:,:self.config['sequence_length']])**2)
        p_loss1 = torch.mean(pt1 * -torch.log(p1[:,:self.config['sequence_length']] + 1e-6))
        p_loss2 = torch.mean(pt2 * -torch.log(p2[:,:self.config['sequence_length']] + 1e-6))
        return v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2
    

    def update_weights(self, data_collector, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            loss1 = ((self.config['v_loss_scaling'] * v_loss1 + self.config['q_loss_scaling'] * q_loss1 + self.config['p_loss_scaling'] * p_loss1) /
                    (self.config['v_loss_scaling'] + self.config['q_loss_scaling'] + self.config['p_loss_scaling']))
            loss2 = ((self.config['v_loss_scaling'] * v_loss2 + self.config['q_loss_scaling'] * q_loss2 + self.config['p_loss_scaling'] * p_loss2) /
                    (self.config['v_loss_scaling'] + self.config['q_loss_scaling'] + self.config['p_loss_scaling']))
        self.scaler.scale(loss1).backward()
        self.scaler.scale(loss2).backward()
        # loss1.backward()
        # loss2.backward()
        gradient_norm1 = torch.nn.utils.clip_grad_norm_(self.learner1.parameters(), self.config['adamw_clip_norm'])
        gradient_norm2 = torch.nn.utils.clip_grad_norm_(self.learner2.parameters(), self.config['adamw_clip_norm'])
        self.scaler.step(self.optimizer1)
        self.scaler.step(self.optimizer2)
        # self.optimizer1.step()
        # self.optimizer2.step()
        self.scaler.update()
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        learning_rate = self.optimizer1.param_groups[0]['lr']
        if self.update_count < self.config['warmup_steps']:
            self.scheduler1.step()
            self.scheduler2.step()
        self.update_count += 1
        self.save_weights_and_data_collector(data_collector)
        self.update_target_weights()
        return loss1, loss2, gradient_norm1, gradient_norm2, learning_rate


    def check_and_update(self, data_collector):
        losses = None
        if self.step_count % self.config['update_frequency'] == 0 and data_collector.frame_count >= self.config['per_min_frames']:
            batched_sequences, sequence_indeces = data_collector.load_batched_sequences()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                v1, v2, q1, q2, p1, p2, v1_, v2_, q1_, q2_, q1_m, q2_m = self.calculate_values(batched_sequences)
                rt1, rt2, rtd1, rtd2 = self.calculate_retrace_targets(q1, q2, q1_, q2_, q1_m, q2_m, p1, batched_sequences)
                vt1, vt2, pt1, pt2 = self.calculate_vtrace_targets(v1_, v2_, p1, batched_sequences)
                v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2 = self.calculate_losses(v1, v2, q1, q2, p1, p2, rt1, rt2, vt1, vt2, pt1, pt2)
            loss1, loss2, gradient_norm1, gradient_norm2, learning_rate = self.update_weights(data_collector, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2)
            losses = loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, gradient_norm1, gradient_norm2, learning_rate
            data_collector.update_priorities(rtd1, rtd2, sequence_indeces)
        self.step_count += 1
        if losses is None:
            return None
        else:
            return tuple(loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in losses)
