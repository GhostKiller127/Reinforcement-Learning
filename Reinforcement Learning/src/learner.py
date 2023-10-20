import os
import torch
import torch.nn.functional as F
from architectures import DenseModel
from architectures import TransformerModel


class Learner:
    def __init__(self, config, metric, device):
        self.count = 0
        self.device = device
        self.config = config
        self.log_dir = f'{metric.log_dir}/models'
        if config['architecture_params']['architecture'] == 'dense':
            self.learner1 = DenseModel(config, device)
            self.learner2 = DenseModel(config, device)
        if config['architecture_params']['architecture'] == 'transformer':
            self.learner1 = TransformerModel(config, device)
            self.learner2 = TransformerModel(config, device)
        initial_lr, end_lr, gamma = self.get_scheduler_args()
        self.optimizer1 = torch.optim.AdamW(self.learner1.parameters(), lr=initial_lr)
        self.optimizer2 = torch.optim.AdamW(self.learner2.parameters(), lr=initial_lr)
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
        self.config['warmup_steps'] = int(self.config['warmup_steps'] / 2)
        return self.config["learning_rate"] / 100, self.config["learning_rate"], 1 / self.config['warmup_steps']


    def push_weights(self):
        if self.count % self.config['d_push'] == 0:
            torch.save(self.learner1.state_dict(), f'{self.log_dir}/learner1.pth')
            torch.save(self.learner2.state_dict(), f'{self.log_dir}/learner2.pth')
            torch.save(self.optimizer1.state_dict(), f'{self.log_dir}/optimizer1.pth')
            torch.save(self.optimizer2.state_dict(), f'{self.log_dir}/optimizer2.pth')


    def pull_weights(self):
        self.learner1.load_state_dict(torch.load(f'{self.log_dir}/learner1.pth'))
        self.learner2.load_state_dict(torch.load(f'{self.log_dir}/learner1.pth'))
        self.optimizer1.load_state_dict(torch.load(f'{self.log_dir}/optimizer1.pth'))
        self.optimizer2.load_state_dict(torch.load(f'{self.log_dir}/optimizer2.pth'))


    def calculate_values(self, batch):
        v1, a1 = self.learner1(batch['o'])
        v2, a2 = self.learner2(batch['o'])

        batch['i'] = torch.tensor(batch['i'], dtype=torch.float32).to(self.device)
        tau1 = batch['i'][:,:,0].unsqueeze(2)
        tau2 = batch['i'][:,:,1].unsqueeze(2)
        epsilon = batch['i'][:,:,2].unsqueeze(2)
        softmax_1 = F.softmax(a1 * tau1, dim=2)
        softmax_2 = F.softmax(a2 * tau2, dim=2)
        policy1 = epsilon * softmax_1 + (1 - epsilon) * softmax_2.detach()
        policy2 = epsilon * softmax_1.detach() + (1 - epsilon) * softmax_2
        p1 = policy1.gather(dim=2, index=batch['a'].unsqueeze(2))
        p2 = policy2.gather(dim=2, index=batch['a'].unsqueeze(2))

        a1 = a1.gather(dim=2, index=batch['a'].unsqueeze(2)) + torch.sum(policy1.detach() * a1, dim=2).unsqueeze(2)
        a2 = a2.gather(dim=2, index=batch['a'].unsqueeze(2)) + torch.sum(policy2.detach() * a2, dim=2).unsqueeze(2)
        q1 = a1 + v1.detach()
        q2 = a2 + v2.detach()
        return v1, v2, q1, q2, p1, p2
    

    def shape_reward(self, r1, r2):
        if not torch.is_tensor(r1):
            r1 = torch.tensor(r1, dtype=torch.float32).to(self.device)
            r2 = torch.tensor(r2, dtype=torch.float32).to(self.device)
        r_log = torch.log(torch.abs(r1) + 1.) / self.config["reward_scaling_1"]
        r1 = torch.where(torch.sign(r1) > 0, r_log * 2, r_log * -1)
        r2 = (torch.sign(r2) * ((torch.abs(r2) + 1.)**(1/4) - 1.) + 0.001 * r2) / self.config["reward_scaling_2"]
        return r1, r2


    def calculate_retrace_targets(self, q1, q2, p, batch):
        r1, r2 = self.shape_reward(batch['r'], batch['r'])
        c = torch.minimum(p / batch['a_p'], torch.tensor(self.config['c_clip']).to(self.device)).squeeze()

        rt1 = torch.zeros((self.config['batch_size'], self.config['sequence_length'])).to(self.device)
        rt2 = torch.zeros((self.config['batch_size'], self.config['sequence_length'])).to(self.device)
        for b in range(self.config['batch_size']):
            for t in range(self.config['sequence_length']):
                c_tk = 1
                target1, target2 = 0, 0
                for k in range(self.config['bootstrap_length']):
                    if k >= 1:
                        c_tk *= c[b,t+k]
                    if batch['d'][b,t+k] or batch['t'][b,t+k]:
                        target1 += self.config['discount']**k * c_tk * (r1[b,t+k] - q1[b,t+k,0])
                        target2 += self.config['discount']**k * c_tk * (r2[b,t+k] - q2[b,t+k,0])
                        break
                    target1 += self.config['discount']**k * c_tk * (r1[b,t+k] + self.config['discount'] * q1[b,t+k+1,0] - q1[b,t+k,0])
                    target2 += self.config['discount']**k * c_tk * (r2[b,t+k] + self.config['discount'] * q2[b,t+k+1,0] - q2[b,t+k,0])
                rt1[b,t] = q1[b,t,0] + target1
                rt2[b,t] = q2[b,t,0] + target2
        return rt1.unsqueeze(2), rt2.unsqueeze(2)
    

    def calculate_vtrace_targets(self, v1, v2, p, batch):
        r1, r2 = self.shape_reward(batch['r'], batch['r'])
        c = torch.minimum(p / batch['a_p'], torch.tensor(self.config['c_clip']).to(self.device)).squeeze()
        rho = torch.minimum(p / batch['a_p'], torch.tensor(self.config['rho_clip']).to(self.device)).squeeze()
        
        vt1 = torch.zeros((self.config['batch_size'], self.config['sequence_length'] + 1)).to(self.device)
        vt2 = torch.zeros((self.config['batch_size'], self.config['sequence_length'] + 1)).to(self.device)
        pt1 = torch.zeros((self.config['batch_size'], self.config['sequence_length'])).to(self.device)
        pt2 = torch.zeros((self.config['batch_size'], self.config['sequence_length'])).to(self.device)
        for b in range(self.config['batch_size']):
            for t in range(self.config['sequence_length'] + 1):
                c_tk = 1
                target1, target2 = 0, 0
                for k in range(self.config['bootstrap_length']):
                    if k >= 1:
                        c_tk *= c[b,t+k-1]
                    if batch['d'][b,t+k] or batch['t'][b,t+k]:
                        target1 += self.config['discount']**k * c_tk * rho[b,t+k] * (r1[b,t+k] - v1[b,t+k,0])
                        target2 += self.config['discount']**k * c_tk * rho[b,t+k] * (r2[b,t+k] - v2[b,t+k,0])
                        break
                    target1 += self.config['discount']**k * c_tk * rho[b,t+k] * (r1[b,t+k] + self.config['discount'] * v1[b,t+k+1,0] - v1[b,t+k,0])
                    target2 += self.config['discount']**k * c_tk * rho[b,t+k] * (r2[b,t+k] + self.config['discount'] * v2[b,t+k+1,0] - v2[b,t+k,0])
                vt1[b,t] = v1[b,t,0] + target1
                vt2[b,t] = v2[b,t,0] + target2

                if t > 0:
                    pt1[b,t-1] = rho[b,t-1] * (batch['r'][b,t-1] + self.config['discount'] * vt1[b,t] - v1[b,t-1])
                    pt2[b,t-1] = rho[b,t-1] * (batch['r'][b,t-1] + self.config['discount'] * vt2[b,t] - v2[b,t-1])
        return vt1[:,:-1].unsqueeze(2), vt2[:,:-1].unsqueeze(2), pt1.unsqueeze(2), pt2.unsqueeze(2)


    def calculate_losses(self, v1, v2, q1, q2, p1, p2, rt1, rt2, vt1, vt2, pt1, pt2):
        v_loss1 = torch.mean((vt1 - v1[:,:-self.config['bootstrap_length']-1])**2)
        v_loss2 = torch.mean((vt2 - v2[:,:-self.config['bootstrap_length']-1])**2)
        q_loss1 = torch.mean((rt1 - q1[:,:-self.config['bootstrap_length']-1])**2)
        q_loss2 = torch.mean((rt2 - q2[:,:-self.config['bootstrap_length']-1])**2)
        p_loss1 = torch.mean(pt1 * -torch.log(p1[:,:-self.config['bootstrap_length']-1] + 1e-6))
        p_loss2 = torch.mean(pt2 * -torch.log(p2[:,:-self.config['bootstrap_length']-1] + 1e-6))
        return v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2
    

    def update_weights(self, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2):
        loss1 = (self.config['v_loss_scaling'] * v_loss1 + self.config['q_loss_scaling'] * q_loss1 + self.config['p_loss_scaling'] * p_loss1) / (self.config['v_loss_scaling'] + self.config['q_loss_scaling'] + self.config['p_loss_scaling'])
        loss2 = (self.config['v_loss_scaling'] * v_loss2 + self.config['q_loss_scaling'] * q_loss2 + self.config['p_loss_scaling'] * p_loss2) / (self.config['v_loss_scaling'] + self.config['q_loss_scaling'] + self.config['p_loss_scaling'])
        loss1.backward()
        loss2.backward()
        gradient_norm1 = torch.nn.utils.clip_grad_norm_(self.learner1.parameters(), self.config['adamw_clip_norm'])
        gradient_norm2 = torch.nn.utils.clip_grad_norm_(self.learner2.parameters(), self.config['adamw_clip_norm'])
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        if self.count < self.config['warmup_steps']:
            self.scheduler1.step()
            self.scheduler2.step()
        self.count += 1
        self.push_weights()
        return loss1, loss2, gradient_norm1, gradient_norm2


    def check_and_update(self, data_collector):
        batched_sequences = data_collector.load_batched_sequences()
        losses = None
        for batch in batched_sequences:
            v1, v2, q1, q2, p1, p2 = self.calculate_values(batch)
            rt1, rt2 = self.calculate_retrace_targets(q1.detach(), q2.detach(), p1.detach(), batch)
            vt1, vt2, pt1, pt2 = self.calculate_vtrace_targets(v1.detach(), v2.detach(), p1.detach(), batch)
            v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2 = self.calculate_losses(v1, v2, q1, q2, p1, p2, rt1, rt2, vt1, vt2, pt1, pt2)
            loss1, loss2, gradient_norm1, gradient_norm2 = self.update_weights(v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2)
            losses = loss1, loss2, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2, gradient_norm1, gradient_norm2
        return losses