import os
import numpy as np
import torch
import torch.nn.functional as F
from architectures import DenseModel
from architectures import TransformerModel


class Learner:
    def __init__(self, training_class):
        self.update_count = 0
        self.step_count = 0
        self.device = training_class.device
        self.config = training_class.config
        self.log_dir = f'{training_class.log_dir}/models'
        if self.config['architecture_params']['architecture'] == 'dense':
            self.learner1 = DenseModel(self.config, self.device)
            self.learner2 = DenseModel(self.config, self.device)
        if self.config['architecture_params']['architecture'] == 'transformer':
            self.learner1 = TransformerModel(self.config, self.device)
            self.learner2 = TransformerModel(self.config, self.device)
        initial_lr, end_lr, gamma = self.get_scheduler_args()
        self.optimizer1 = torch.optim.AdamW(self.learner1.parameters(), lr=initial_lr, eps=self.config['adamw_epsilon'])
        self.optimizer2 = torch.optim.AdamW(self.learner2.parameters(), lr=initial_lr, eps=self.config['adamw_epsilon'])
        self.scheduler1 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer1, gamma=(end_lr/initial_lr)**gamma)
        self.scheduler2 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer2, gamma=(end_lr/initial_lr)**gamma)
        if self.config['load_run'] is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.push_weights()
        else:
            self.pull_weights()

    
    def get_scheduler_args(self):
        if self.config['load_run'] is not None:
            self.config['warmup_steps'] = 0
            return self.config["learning_rate"], self.config["learning_rate"], 0
        if self.config["lr_finder"]:
            return 1e-8, 1e-2, 1 / self.config['warmup_steps']
        self.config['warmup_steps'] = int(self.config['warmup_steps'] / 4)
        return self.config["learning_rate"] / 100, self.config["learning_rate"], 1 / self.config['warmup_steps']


    def push_weights(self):
        if self.update_count % self.config['d_push'] == 0:
            torch.save(self.learner1.state_dict(), f'{self.log_dir}/learner1.pth')
            torch.save(self.learner2.state_dict(), f'{self.log_dir}/learner2.pth')
            torch.save(self.optimizer1.state_dict(), f'{self.log_dir}/optimizer1.pth')
            torch.save(self.optimizer2.state_dict(), f'{self.log_dir}/optimizer2.pth')


    def pull_weights(self):
        self.learner1.load_state_dict(torch.load(f'{self.log_dir}/learner1.pth'))
        self.learner2.load_state_dict(torch.load(f'{self.log_dir}/learner2.pth'))
        self.optimizer1.load_state_dict(torch.load(f'{self.log_dir}/optimizer1.pth'))
        self.optimizer2.load_state_dict(torch.load(f'{self.log_dir}/optimizer2.pth'))


    def calculate_values(self, batch):
        v1, a1 = self.learner1(batch['o'])
        v2, a2 = self.learner2(batch['o'])

        i = torch.tensor(batch['i'], dtype=torch.float32).to(self.device)
        a = torch.tensor(batch['a'], dtype=torch.int64).to(self.device).unsqueeze(2)
        tau1 = i[:,:,0].unsqueeze(2)
        tau2 = i[:,:,1].unsqueeze(2)
        epsilon = i[:,:,2].unsqueeze(2)
        softmax_1 = F.softmax(a1 * tau1, dim=2)
        softmax_2 = F.softmax(a2 * tau2, dim=2)
        policy1 = epsilon * softmax_1 + (1 - epsilon) * softmax_2.detach()
        policy2 = epsilon * softmax_1.detach() + (1 - epsilon) * softmax_2
        p1 = policy1.gather(dim=2, index=a)
        p2 = policy2.gather(dim=2, index=a)

        a1 = a1.gather(dim=2, index=a) + torch.sum(policy1.detach() * a1, dim=2).unsqueeze(2)
        a2 = a2.gather(dim=2, index=a) + torch.sum(policy2.detach() * a2, dim=2).unsqueeze(2)
        q1 = a1 + v1.detach()
        q2 = a2 + v2.detach()
        return v1, v2, q1, q2, p1, p2
    

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


    def calculate_retrace_targets(self, q1, q2, p, batch):
        q1_ = q1.detach().cpu().numpy().squeeze()
        q2_ = q2.detach().cpu().numpy().squeeze()
        q1 = self.invert_scale1(q1_)
        q2 = self.invert_scale2(q2_)
        p = p.detach().cpu().numpy()
        c = np.minimum(p / batch['a_p'], self.config['c_clip']).squeeze()

        rt1 = np.zeros((self.config['batch_size'], self.config['sequence_length']))
        rt2 = np.zeros((self.config['batch_size'], self.config['sequence_length']))
        rtd1 = np.zeros((self.config['batch_size'], self.config['sequence_length']))
        rtd2 = np.zeros((self.config['batch_size'], self.config['sequence_length']))
        for b in range(self.config['batch_size']):
            for t in range(self.config['sequence_length']):
                c_tk = 1
                td1, td2 = 0, 0
                for k in range(self.config['bootstrap_length']):
                    if k >= 1:
                        c_tk *= c[b,t+k]
                    if batch['d'][b,t+k] or batch['t'][b,t+k]:
                        td1 += self.config['discount']**k * c_tk * (batch['r'][b,t+k] - q1[b,t+k])
                        td2 += self.config['discount']**k * c_tk * (batch['r'][b,t+k] - q2[b,t+k])
                        break
                    td1 += self.config['discount']**k * c_tk * (batch['r'][b,t+k] + self.config['discount'] * q1[b,t+k+1] - q1[b,t+k])
                    td2 += self.config['discount']**k * c_tk * (batch['r'][b,t+k] + self.config['discount'] * q2[b,t+k+1] - q2[b,t+k])
                rt1[b,t] = q1[b,t] + td1
                rt2[b,t] = q2[b,t] + td2
        rt1 = self.scale_value1(rt1)
        rt2 = self.scale_value2(rt2)
        rtd1 = rt1 - q1_[:,: - self.config['bootstrap_length'] - 1]
        rtd2 = rt2 - q2_[:,: - self.config['bootstrap_length'] - 1]
        rt1 = torch.tensor(rt1).to(self.device).unsqueeze(2)
        rt2 = torch.tensor(rt2).to(self.device).unsqueeze(2)
        return rt1, rt2, rtd1, rtd2
    

    def calculate_vtrace_targets(self, v1, v2, p, batch):
        v1 = v1.detach().cpu().numpy().squeeze()
        v2 = v2.detach().cpu().numpy().squeeze()
        v1 = self.invert_scale1(v1)
        v2 = self.invert_scale2(v2)
        p = p.detach().cpu().numpy()
        c = np.minimum(p / batch['a_p'], self.config['c_clip']).squeeze()
        rho = np.minimum(p / batch['a_p'], self.config['rho_clip']).squeeze()
        
        vt1 = np.zeros((self.config['batch_size'], self.config['sequence_length'] + 1))
        vt2 = np.zeros((self.config['batch_size'], self.config['sequence_length'] + 1))
        pt1 = np.zeros((self.config['batch_size'], self.config['sequence_length']))
        pt2 = np.zeros((self.config['batch_size'], self.config['sequence_length']))
        for b in range(self.config['batch_size']):
            for t in range(self.config['sequence_length'] + 1):
                c_tk = 1
                td1, td2 = 0, 0
                for k in range(self.config['bootstrap_length']):
                    if k >= 1:
                        c_tk *= c[b,t+k-1]
                    if batch['d'][b,t+k] or batch['t'][b,t+k]:
                        td1 += self.config['discount']**k * c_tk * rho[b,t+k] * (batch['r'][b,t+k] - v1[b,t+k])
                        td2 += self.config['discount']**k * c_tk * rho[b,t+k] * (batch['r'][b,t+k] - v2[b,t+k])
                        break
                    td1 += self.config['discount']**k * c_tk * rho[b,t+k] * (batch['r'][b,t+k] + self.config['discount'] * v1[b,t+k+1] - v1[b,t+k])
                    td2 += self.config['discount']**k * c_tk * rho[b,t+k] * (batch['r'][b,t+k] + self.config['discount'] * v2[b,t+k+1] - v2[b,t+k])
                vt1[b,t] = v1[b,t] + td1
                vt2[b,t] = v2[b,t] + td2
                if t > 0:
                    pt1[b,t-1] = rho[b,t-1] * (batch['r'][b,t-1] + self.config['discount'] * vt1[b,t] - v1[b,t-1])
                    pt2[b,t-1] = rho[b,t-1] * (batch['r'][b,t-1] + self.config['discount'] * vt2[b,t] - v2[b,t-1])
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
        v_loss1 = torch.mean((vt1 - v1[:,:-self.config['bootstrap_length']-1])**2)
        v_loss2 = torch.mean((vt2 - v2[:,:-self.config['bootstrap_length']-1])**2)
        q_loss1 = torch.mean((rt1 - q1[:,:-self.config['bootstrap_length']-1])**2)
        q_loss2 = torch.mean((rt2 - q2[:,:-self.config['bootstrap_length']-1])**2)
        p_loss1 = torch.mean(pt1 * -torch.log(p1[:,:-self.config['bootstrap_length']-1] + 1e-6))
        p_loss2 = torch.mean(pt2 * -torch.log(p2[:,:-self.config['bootstrap_length']-1] + 1e-6))
        return v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2
    

    def update_weights(self, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2):
        loss1 = ((self.config['v_loss_scaling'] * v_loss1 + self.config['q_loss_scaling'] * q_loss1 + self.config['p_loss_scaling'] * p_loss1) /
                 (self.config['v_loss_scaling'] + self.config['q_loss_scaling'] + self.config['p_loss_scaling']))
        loss2 = ((self.config['v_loss_scaling'] * v_loss2 + self.config['q_loss_scaling'] * q_loss2 + self.config['p_loss_scaling'] * p_loss2) /
                 (self.config['v_loss_scaling'] + self.config['q_loss_scaling'] + self.config['p_loss_scaling']))
        loss1.backward()
        loss2.backward()
        gradient_norm1 = torch.nn.utils.clip_grad_norm_(self.learner1.parameters(), self.config['adamw_clip_norm'])
        gradient_norm2 = torch.nn.utils.clip_grad_norm_(self.learner2.parameters(), self.config['adamw_clip_norm'])
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        learning_rate = self.optimizer1.param_groups[0]['lr']
        if self.update_count < self.config['warmup_steps']:
            self.scheduler1.step()
            self.scheduler2.step()
        self.update_count += 1
        self.push_weights()
        return loss1, loss2, gradient_norm1, gradient_norm2, learning_rate


    def check_and_update(self, data_collector):
        if self.config['per_experience_replay']:
            if self.step_count % self.config['update_frequency'] == 0 and data_collector.frame_count >= self.config['per_min_frames']:
                batched_sequences, sequence_indeces = data_collector.load_per_batched_sequences()
            else:
                batched_sequences = []
            self.step_count += 1
        else:
            batched_sequences = data_collector.load_batched_sequences()
        losses = None
        for batch in batched_sequences:
            v1, v2, q1, q2, p1, p2 = self.calculate_values(batch)
            rt1, rt2, rtd1, rtd2 = self.calculate_retrace_targets(q1, q2, p1, batch)
            vt1, vt2, pt1, pt2 = self.calculate_vtrace_targets(v1, v2, p1, batch)
            if self.config['per_experience_replay']:
                data_collector.update_priorities(rtd1, rtd2, sequence_indeces)
            v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2 = self.calculate_losses(v1, v2, q1, q2, p1, p2, rt1, rt2, vt1, vt2, pt1, pt2)
            loss1, loss2, gradient_norm1, gradient_norm2, learning_rate = self.update_weights(v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2)
            losses = loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, gradient_norm1, gradient_norm2, learning_rate
        return losses