import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'  # no GPU training as it causes resource conflicts with Environment todo


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, num_fc_1, num_fc_2):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(state_dim, num_fc_1)
        self.fc2 = nn.Linear(num_fc_1, num_fc_2)
        self.fc3_actor = nn.Linear(num_fc_2, action_dim)
        self.fc3_critic = nn.Linear(num_fc_2, 1)
        self.nonlin_layer = F.leaky_relu
        self.nonlin_layer_value = F.tanh

    def forward(self, x):
        x = self.nonlin_layer(self.fc1(x))
        x = self.nonlin_layer(self.fc2(x))
        action = self.nonlin_layer(self.fc3_actor(x))
        value = self.nonlin_layer_value(self.fc3_critic(x))
        return action, value
