import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from architectures import DenseModel
from architectures import TransformerModel


class Learner:
    def __init__(self, config, metric, device):
        self.count = 0
        self.device = device
        self.config = config
        self.log_dir = f'models/{metric.run_name}'
        if config['architecture_params']['architecture'] == 'dense':
            self.learner1 = DenseModel(config, device)
            self.learner2 = DenseModel(config, device)
        if config['architecture_params']['architecture'] == 'transformer':
            self.learner1 = TransformerModel(config, device)
            self.learner2 = TransformerModel(config, device)
        self.optimizer1 = optim.AdamW(self.learner1.parameters(), lr=1e-4) 
        self.optimizer2 = optim.AdamW(self.learner2.parameters(), lr=1e-4) 
        os.makedirs(self.log_dir, exist_ok=True)
        self.push_weights()


    def push_weights(self):
        if self.count % self.config['d_push'] == 0:
            torch.save(self.learner1.state_dict(), f'{self.log_dir}/learner1.pth')
            torch.save(self.learner2.state_dict(), f'{self.log_dir}/learner2.pth')
    

    def calculate_values(self, observations, actions, indices):
        v1, a1 = self.learner1(observations)
        v2, a2 = self.learner2(observations)

        indices = torch.tensor(indices, dtype=torch.float32).to(self.device)
        tau1 = indices[:,:,0].unsqueeze(2)
        tau2 = indices[:,:,1].unsqueeze(2)
        epsilon = indices[:,:,2].unsqueeze(2)
        softmax_1 = F.softmax(a1 * tau1, dim=2)
        softmax_2 = F.softmax(a2 * tau2, dim=2)
        policy1 = epsilon * softmax_1 + (1 - epsilon) * softmax_2.detach()
        policy2 = epsilon * softmax_1.detach() + (1 - epsilon) * softmax_2
        p1 = policy1.gather(dim=2, index=actions.unsqueeze(2))
        p2 = policy2.gather(dim=2, index=actions.unsqueeze(2))

        a1 = a1.gather(dim=2, index=actions.unsqueeze(2)) + torch.sum(policy1.detach() * a1, dim=2).unsqueeze(2)
        a2 = a2.gather(dim=2, index=actions.unsqueeze(2)) + torch.sum(policy2.detach() * a2, dim=2).unsqueeze(2)
        q1 = a1 + v1.detach()
        q2 = a2 + v2.detach()
        return v1, v2, q1, q2, p1, p2
    

    def calculate_retrace_targets(self, q1, q2, p, a_p, r):
        r = torch.tensor(r, dtype=torch.float32).to(self.device).unsqueeze(2)
        c = torch.minimum(p / a_p, torch.tensor(self.config['c_clip']).to(self.device))
        td1 = r[:,:-1] + self.config['discount'] * q1[:,1:] - q1[:,:-1]
        td2 = r[:,:-1] + self.config['discount'] * q2[:,1:] - q2[:,:-1]

        rt1 = torch.zeros((self.config['batch_size'], self.config['sequence_length'])).to(self.device)
        rt2 = torch.zeros((self.config['batch_size'], self.config['sequence_length'])).to(self.device)
        for b in range(self.config['batch_size']):
            for t in range(self.config['sequence_length']):
                c_tk = 1
                target1, target2 = 0, 0
                for k in range(self.config['bootstrap_length']):
                    if k >= 1:
                        c_tk *= c[b,t+k]
                    target1 += self.config['discount']**k * c_tk * td1[b,t+k]
                    target2 += self.config['discount']**k * c_tk * td2[b,t+k]
                rt1[b,t] = q1[b,t] + target1
                rt2[b,t] = q2[b,t] + target2
        return rt1.unsqueeze(2), rt2.unsqueeze(2)
    

    def calculate_vtrace_targets(self, v1, v2, p, a_p, r):
        r = torch.tensor(r, dtype=torch.float32).to(self.device).unsqueeze(2)
        c = torch.minimum(p / a_p, torch.tensor(self.config['c_clip']).to(self.device))
        rho = torch.minimum(p / a_p, torch.tensor(self.config['rho_clip']).to(self.device))
        td1 = r[:,:-1] + self.config['discount'] * v1[:,1:] - v1[:,:-1]
        td2 = r[:,:-1] + self.config['discount'] * v2[:,1:] - v2[:,:-1]
        
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
                    target1 += self.config['discount']**k * c_tk * rho[b,t+k] * td1[b,t+k]
                    target2 += self.config['discount']**k * c_tk * rho[b,t+k] * td2[b,t+k]
                vt1[b,t] = v1[b,t] + target1
                vt2[b,t] = v2[b,t] + target2

                if t < self.config['sequence_length']:
                    pt1[b,t] = rho[b,t] * (r[b,t] + self.config['discount'] * vt1[b,t+1] - v1[b,t])
                    pt2[b,t] = rho[b,t] * (r[b,t] + self.config['discount'] * vt2[b,t+1] - v2[b,t])
        return vt1[:,:-1].unsqueeze(2), vt2[:,:-1].unsqueeze(2), pt1.unsqueeze(2), pt2.unsqueeze(2)


    def calculate_losses(self, v1, v2, q1, q2, p1, p2, rt1, rt2, vt1, vt2, pt1, pt2):
        v_loss1 = torch.mean((vt1 - v1[:,:-self.config['bootstrap_length']-1])**2)
        v_loss2 = torch.mean((vt2 - v2[:,:-self.config['bootstrap_length']-1])**2)
        q_loss1 = torch.mean((rt1 - q1[:,:-self.config['bootstrap_length']-1])**2)
        q_loss2 = torch.mean((rt2 - q2[:,:-self.config['bootstrap_length']-1])**2)
        p_loss1 = torch.mean(pt1 * -torch.log(p1[:,:-self.config['bootstrap_length']-1]))
        p_loss2 = torch.mean(pt2 * -torch.log(p2[:,:-self.config['bootstrap_length']-1]))
        return v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2
    

    def update_weights(self, v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2):
        loss1 = self.config['v_loss_scaling'] * v_loss1 + self.config['q_loss_scaling'] * q_loss1 + self.config['p_loss_scaling'] * p_loss1
        loss2 = self.config['v_loss_scaling'] * v_loss2 + self.config['q_loss_scaling'] * q_loss2 + self.config['p_loss_scaling'] * p_loss2
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        loss1.backward()
        loss2.backward()
        self.optimizer1.step()
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer2.step()
        self.count += 1
        self.push_weights()
        return loss1 + loss2, self.count


    def check_and_update(self, data_collector):
        batched_sequences = data_collector.load_batched_sequences()
        losses = []
        for batch in batched_sequences:
            v1, v2, q1, q2, p1, p2 = self.calculate_values(batch['o'], batch['a'], batch['i'])
            rt1, rt2 = self.calculate_retrace_targets(q1.detach(), q2.detach(), p1.detach(), batch['a_p'], batch['r'])
            vt1, vt2, pt1, pt2 = self.calculate_vtrace_targets(v1.detach(), v2.detach(), p1.detach(), batch['a_p'], batch['r'])
            v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2 = self.calculate_losses(v1, v2, q1, q2, p1, p2, rt1, rt2, vt1, vt2, pt1, pt2)
            loss, count = self.update_weights(v_loss1, v_loss2, q_loss1, q_loss2, p_loss1, p_loss2)
            losses.append([loss, count])
        return losses