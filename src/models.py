import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'  # no GPU training as it causes resource conflicts with Environment todo


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, num_fc_1, num_fc_2, num_fc_last, is_actor=True):
        super(Network, self).__init__()
        self.is_actor = is_actor

        self.fc1 = nn.Linear(state_dim * 2, num_fc_1)
        if self.is_actor:
            self.fc2 = nn.Linear(num_fc_1, num_fc_2)
        else:
            self.fc2 = nn.Linear(num_fc_1 + action_dim * 2, num_fc_2)
        self.fc3 = nn.Linear(num_fc_2, num_fc_last)
        self.nonlin = 'relu'
        self.nonlin_layer = getattr(F, self.nonlin)
        self.nonlin_layer_last = F.tanh

    def forward(self, state, action=None):
        x = self.nonlin_layer(self.fc1(state))
        if not self.is_actor:
            x = torch.cat([x, action], dim=1)  # concatenate with action
        x = self.nonlin_layer(self.fc2(x))
        x = self.fc3(x)
        if self.is_actor:
            x = self.nonlin_layer_last(x)  # tanh for actor
        # else:
        #     x = self.nonlin_layer(x)
        return x

    def reset_parameters(self):
        self.fc1.weight.data.kaiming_normal_(mode='fan_in', nonlinearity=self.nonlin, gain=1)
        self.fc1.bias.data.kaiming_normal_(mode='fan_in', nonlinearity=self.nonlin, gain=1)
        self.fc2.weight.data.kaiming_normal_(mode='fan_in', nonlinearity=self.nonlin, gain=1)
        self.fc2.bias.data.kaiming_normal_(mode='fan_in', nonlinearity=self.nonlin, gain=1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)


class Actor(Network):
    def __init__(self, state_dim, action_dim, num_fc_1, num_fc_2):
        super(Actor, self).__init__(state_dim, action_dim, num_fc_1, num_fc_2, action_dim, is_actor=True)


class Critic(Network):
    def __init__(self, state_dim, action_dim, num_fc_1, num_fc_2):
        super(Critic, self).__init__(state_dim, action_dim, num_fc_1, num_fc_2, 1, is_actor=False)
