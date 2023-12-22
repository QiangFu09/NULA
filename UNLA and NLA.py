#Code of UNLA and NLA
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np
import pickle

init_std = 0.1
init_stdv = 0.1
n = 100
d = 1000
torch.manual_seed(0)
a = torch.rand(d)

def smooth_truncation(x, R):
    # return np.exp(-(x ** 2-R**2))/(np.exp(-(x ** 2-R**2)) + 1)
    return (torch.exp(x/(0.5*R))/(torch.exp(x/(0.5*R)) + 1) - 0.5) * R * 2


class Net(nn.Module):
    def __init__(self, N=512, cut_off=20, activation_type=None):
        super(Net, self).__init__()
        self.particle_num = N

        if activation_type == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_type == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == "Tanh":
            self.activation = nn.Tanh()

        self.fc1 = nn.Linear(1000, N)
        self.fc2 = nn.Linear(N, 1, bias=False)

        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc1.bias, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)

        self.cut_off = cut_off

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        # restrict the weights of fc2 to be within [0, cutoff]

        if self.cut_off != None:
            self.fc2.weight.data = torch.clamp(self.fc2.weight.data, min=-self.cut_off, max=self.cut_off)

        x = self.fc2(x) / self.particle_num

        return x.squeeze()

class FullMomentumGradientDescentWithNoisyAndWeightDecay(torch.optim.Optimizer):
    def __init__(self, params, N, lr=1e-2, lr_2=1e-1, momentum=True, gamma=10, weight_decay=0.01, noise_scale=0):
        defaults = dict(lr=lr, lr_2=lr_2, gamma=gamma, weight_decay=weight_decay, noise_scale=noise_scale,
                        momentum=momentum)
        self.N = N
        super(FullMomentumGradientDescentWithNoisyAndWeightDecay, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            #lr_2 = group['lr_2']
            weight_decay = group['weight_decay']
            noise_scale = group['noise_scale']
            #noise_1, noise_2 = noisegenerateor(gamma, lr, 784)
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
                        # buf = param_state['momentum_buffer'] = torch.clone(grad).detach()

                    buf = param_state['momentum_buffer']
                    noise_2 = torch.randn_like(buf.data) * 0.001
                    grad.add_(weight_decay, p.data)
                    p.data.add_(0.0001, buf)
                    p.data.add_(-0.02, grad)
                    #p.data.add_(-weight_decay * 0.01, p.data)
                    p.data.add_(noise_1)
                    buf = buf * 0.99
                    buf.add_(-0.02, grad)
                    buf.add_(noise_2)
                    
                else:
                    if weight_decay != 0:
                        grad.add_(weight_decay, p.data)

                    p.data.add_(-lr, grad)

                    if noise_scale != 0:
                        noise = torch.randn_like(p.data) * noise_scale * (lr ** 1 / 2)
                        p.data.add_(noise)

        return loss


def Gaussian_func(x):
    v = x - a
    d = x.shape[1]
    y = torch.sum(v ** 2, dim = 1)
    #y.detach().numpy()
    return torch.exp(-y / (2 * d))
input_data = torch.rand(n, d)
target_data = Gaussian_func(input_data)

dataset = TensorDataset(input_data, target_data)

def trainloss(x, y):
    return (x - y) ** 2 / 2
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

weight_decay = 1e-4
sigma = 0.001
gamma = 1
lr = 0.01 # KL lr = 0.5  MSE 0.1

cut_off = 10

activation_type = 'Tanh' # relu as default
loss_type = 'MSE'
output_result = True
#momentum = True
test_var = 'momentum'
test_values = [True, False] 
test_time_per_objective = 1
result_trainloss_set = []
result_L2_trainloss_set = []
result_testacc_set = []
for sd in [0, 1, 2, 3, 4]:
    torch.manual_seed(sd)
    input_data = torch.rand(n, d)
    target_data = Gaussian_func(input_data)
    dataset = TensorDataset(input_data, target_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
    for particle_num in [256, 512, 1024, 2048]:
        for test_value in test_values:
            globals()[test_var] = test_value
            momentum = test_value
            trainloss_current = []
            L2_trainloss_current = []
            testacc_current = []

            for test_number in tqdm(range(test_time_per_objective)):
                # Define the loss function and optimizer
                net = Net(N=particle_num, cut_off=cut_off, activation_type=activation_type).to(device)

                if loss_type == 'MSE':
                    criterion = nn.MSELoss()
                elif loss_type == 'KL':
                    criterion = nn.CrossEntropyLoss()

                optimizer = FullMomentumGradientDescentWithNoisyAndWeightDecay(net.parameters(), particle_num,
                                                                               momentum=momentum, lr=lr, lr_2=lr,
                                                                               weight_decay=weight_decay, noise_scale=sigma,
                                                                               gamma=gamma)

                # Train the network
                T = 500
                epochs = 10000

                # epochs = 10000

                train_losses = []
                test_accs = []
                L2_train_losses = []

                for epoch in range(1, epochs + 1):
                    #             for param_group in optimizer.param_groups:
                    #                 param_group['lr_2'] = 1

                    train_loss = 0.0
                    net.train()

                    optimizer.zero_grad()
                    output = net(data)
                    loss = trainloss(output, target).mean()

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * data.size(0)

                    train_loss /= len(train_loader.dataset)
                    train_losses.append(train_loss)
                    squared_sum = 0
                    for param in net.parameters():
                        squared_sum += torch.sum(param.data ** 2) / particle_num

                    L2_trian_loss = train_loss + squared_sum.item() * weight_decay
                    L2_train_losses.append(L2_trian_loss)

                    if (epoch == 1 or epoch % 1000 == 0) and output_result:
                        print('Epoch: {} Train Loss: {:.6f} L2-Train Loss: {:.6f}'.format(
                            epoch, train_loss, L2_trian_loss))

                trainloss_current.append(train_losses)
                L2_trainloss_current.append(L2_train_losses)

            result_trainloss_set.append(trainloss_current)
            result_L2_trainloss_set.append(L2_trainloss_current)

            avg_trainloss_cur = np.mean(np.array(trainloss_current)[:, -500:])
            avg_L2_trainloss = np.mean(np.array(L2_trainloss_current)[:, -500:])

            print('Finish test %s=%d' % (test_var, globals()[test_var]), 'cutoff = %.2f' % (cut_off),
                  'number of particle=%.3e' % ((particle_num)),
                  'avg_trainloss=%.3e' % (avg_trainloss_cur),
                  'avg_L2_trainloss=%.3e' % (avg_L2_trainloss))
            if momentum==True:
                with open("UNLA" + str(particle_num) + "seed" + str(sd), "wb") as fp:
                    pickle.dump(train_losses, fp)
            else:
                with open("NLA" + str(particle_num) + "seed" + str(sd), "wb") as fp:
                    pickle.dump(train_losses, fp)
        