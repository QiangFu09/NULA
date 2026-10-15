"""Network and historical optimizer updates used in the ICML 2024 experiments.

The momentum-buffer and noise conventions are retained; see REPRODUCIBILITY.md.
"""

# Adapted from SeiruGanki/Uniform-in-time-propagation-of-chaos-for-kinetic-mean-field-Langevin-dynamics.
# See THIRD_PARTY_NOTICES.md.
import torch
from torch import nn

init_stdv = 0.1


class Net(nn.Module):
    def __init__(self, N=512, cut_off=20, activation_type="Tanh", input_dim=1000):
        super(Net, self).__init__()
        self.particle_num = N

        if activation_type == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_type == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == "Tanh":
            self.activation = nn.Tanh()

        self.fc1 = nn.Linear(input_dim, N)
        self.fc2 = nn.Linear(N, 1, bias=False)

        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc1.bias, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)

        self.cut_off = cut_off

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)

        if self.cut_off != None:
            self.fc2.weight.data = torch.clamp(self.fc2.weight.data, min=-self.cut_off, max=self.cut_off)

        x = self.fc2(x) / self.particle_num

        return x.squeeze()

class NULAOptimizer(torch.optim.Optimizer):
    def __init__(self, params, N, lr=1e-2, lr_2=1e-1, momentum=True, gamma=10, weight_decay=0.01, noise_scale=0):
        defaults = dict(lr=lr, lr_2=lr_2, gamma=gamma, weight_decay=weight_decay, noise_scale=noise_scale,
                        momentum=momentum)
        self.N = N
        super(NULAOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            noise_scale = group['noise_scale']
            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                noise_1 = torch.randn_like(p.data) * 0.001

                grad = p.grad.data * self.N
                param_state = self.state[p]

                if group['momentum'] == True:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.randn_like(p.data) * init_stdv

                    buf = param_state['momentum_buffer']
                    noise_2 = torch.randn_like(buf.data) * 0.001
                    grad.add_(p.data, alpha=weight_decay)
                    p.data.add_(buf, alpha=0.0001)
                    p.data.add_(grad, alpha=-0.02)
                    p.data.add_(noise_1)
                    buf = buf * 0.99
                    buf.add_(grad, alpha=-0.02)
                    buf.add_(noise_2)

                else:
                    if weight_decay != 0:
                        grad.add_(p.data, alpha=weight_decay)

                    p.data.add_(grad, alpha=-lr)

                    if noise_scale != 0:
                        noise = torch.randn_like(p.data) * noise_scale * (lr / 2)
                        p.data.add_(noise)

        return loss

class EMNULAOptimizer(torch.optim.Optimizer):
    def __init__(self, params, N, lr=1e-2, lr_2=1e-1, momentum=True, gamma=10, weight_decay=0.01, noise_scale=0):
        defaults = dict(lr=lr, lr_2=lr_2, gamma=gamma, weight_decay=weight_decay, noise_scale=noise_scale,
                        momentum=momentum)
        self.N = N
        super(EMNULAOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            lr_2 = group['lr_2']

            weight_decay = group['weight_decay']
            noise_scale = group['noise_scale']

            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad.data * self.N

                param_state = self.state[p]

                if group['momentum'] == True:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.randn_like(p.data) * init_stdv

                    buf = param_state['momentum_buffer']

                    if gamma != 0:
                        buf.add_(buf.data, alpha=-gamma * lr_2)

                    buf.add_(-grad * lr_2)

                    if weight_decay != 0:
                        buf.add_(p.data, alpha=-weight_decay * lr_2)

                    if noise_scale != 0:
                        noise = torch.randn_like(p.data) * 0.001
                        buf.data.add_(noise)

                    p.data.add_(buf, alpha=lr)
                else:
                    if weight_decay != 0:
                        grad.add_(p.data, alpha=weight_decay)

                    p.data.add_(grad, alpha=-lr)

                    if noise_scale != 0:
                        noise = torch.randn_like(p.data) * noise_scale * (lr / 2)
                        p.data.add_(noise)

        return loss
