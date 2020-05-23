# -*- coding: utf-8 -*-
import const
from src import models, buffer, noise

from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam

torch.random.manual_seed(const.random_seed)  # todo


class DRLAgent:
    model_path = const.file_path_model

    def __init__(self, num_states: int = const.state_size,
                 num_actions: int = const.action_size,
                 memory_size: int = const.memory_size,
                 gamma: float = const.gamma,
                 batch_size: int = const.batch_size,
                 tau: float = const.tau,
                 model_learning_rate: float = const.model_learning_rate,
                 num_fc_1: int = const.num_fc_1,
                 num_fc_2: int = const.num_fc_2
                 ):

        # agent params
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory = buffer.ReplayBuffer(memory_size)  # todo shared buffer for both agents
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.start_policy_training_iter = batch_size  # start training after: buffer_size >= batch_size
        self.policy_freq = const.policy_freq
        self.t = 0
        self.noise = noise.OrnsteinUhlenbeckActionNoise()
        self.glie = noise.GLIE()

        # model params
        self.model_learning_rate = model_learning_rate
        self.num_fc_1 = num_fc_1  # todo actor & critic
        self.num_fc_2 = num_fc_2
        self.with_bn = const.with_bn

        self.actor = models.Actor(state_dim=self.num_states, action_dim=self.num_actions,
                                  num_fc_1=self.num_fc_1, num_fc_2=self.num_fc_2, with_bn=self.with_bn)
        self.actor_target = models.Actor(state_dim=self.num_states, action_dim=self.num_actions,
                                         num_fc_1=self.num_fc_1, num_fc_2=self.num_fc_2, with_bn=self.with_bn)
        self.actor_opt = Adam(self.actor.parameters(), lr=self.model_learning_rate, weight_decay=0)

        self.critic = models.Critic(state_dim=self.num_states, action_dim=self.num_actions,
                                    num_fc_1=self.num_fc_1, num_fc_2=self.num_fc_2, with_bn=self.with_bn)
        self.critic_target = models.Critic(state_dim=self.num_states, action_dim=self.num_actions,
                                           num_fc_1=self.num_fc_1, num_fc_2=self.num_fc_2, with_bn=self.with_bn)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.model_learning_rate, weight_decay=0)

        self._model_summary(self.actor, title='Actor')
        self._model_summary(self.critic, title='Critic')

        # for plots
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.noise_list = []

    def reset(self):
        self.noise.reset()

    def act(self, states):
        s_tensor = torch.Tensor(states).reshape(1, -1)  # shape (1, 2) to use in BN
        self.actor.eval()
        with torch.no_grad():
            a = self.actor.forward(s_tensor).detach().numpy()
        self.actor.train()
        curr_noise = self.glie.get_eps() * self.noise()
        self.noise_list.append(curr_noise)
        a = a.flatten()  # shape (2,)
        actions = a + curr_noise  # result in shape (2,)
        actions = np.clip(actions, - const.max_action, const.max_action)
        return actions

    def do_stuff(self, state, action, reward, next_state, done, agent_index):
        self.t += 1
        self.memory.append((state, action, reward, next_state, done))  # memorize

        # Train agent after collecting sufficient data
        if self.t >= self.start_policy_training_iter:
            self._train(agent_index)

    def load(self):
        const.myprint('Loading model from:', self.model_path)
        # load with architecture
        checkpoint = torch.load(self.model_path)
        self.actor = models.Actor(state_dim=checkpoint['num_states'],
                                  action_dim=checkpoint['num_actions'],
                                  num_fc_1=checkpoint['num_fc_1'],
                                  num_fc_2=checkpoint['num_fc_2'],
                                  with_bn=const.with_bn)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic = models.Critic(state_dim=checkpoint['num_states'],
                                    action_dim=checkpoint['num_actions'],
                                    num_fc_1=checkpoint['num_fc_1'],
                                    num_fc_2=checkpoint['num_fc_2'],
                                    with_bn=const.with_bn)
        self.critic.load_state_dict(checkpoint['critic'])

        # change mode (to use only for inference)
        self.actor.eval()
        self.critic.eval()

    def save(self):
        const.myprint('Saving model to:', self.model_path)
        # save with architecture
        checkpoint = {'num_states': self.num_states,
                      'num_actions': self.num_actions,
                      'num_fc_1': self.num_fc_1,
                      'num_fc_2': self.num_fc_2,
                      'actor': self.actor.state_dict(),
                      'critic': self.critic.state_dict()
                      }
        torch.save(checkpoint, self.model_path)

    def set_model_path(self, i):
        p = self.model_path
        self.model_path = Path(p.parent, 'model_' + str(i) + p.suffix)

    def _model_summary(self, model, title='Model'):
        const.myprint("model_summary --> " + title)
        const.myprint()
        const.myprint("Layer_name" + "\t" * 7 + "Number of Parameters")
        const.myprint("=" * 100)
        model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
        layer_name = [child for child in model.children()]
        j = 0
        total_params = 0
        for i in layer_name:
            param = 0
            try:
                bias = (i.bias is not None)
            except:
                bias = False
            if not bias:
                param = model_parameters[j].numel() + model_parameters[j + 1].numel()
                j = j + 2
            else:
                param = model_parameters[j].numel()
                j = j + 1
            const.myprint(str(i) + "\t" * 3 + str(param))
            total_params += param
        const.myprint("=" * 100)
        const.myprint(f"Total Params:{total_params}")

    def _train(self, agent_index):
        # Sample a minibatch from the replay memory
        states_batch, action_batch, reward_batch, next_states_batch, not_done_batch = self.memory.sample(self.batch_size)

        # Compute Q targets for next states
        next_actions_batch = self.actor_target(next_states_batch)  # shape (batch_size, 2)

        # fill in missing action with the other agent's real action
        if agent_index == 0:
            next_actions_batch = torch.cat([next_actions_batch, action_batch[:, 2:]], dim=1)
        else:
            next_actions_batch = torch.cat([action_batch[:, :2], next_actions_batch], dim=1)

        q_values_next_target = self.critic_target(next_states_batch, next_actions_batch)

        # Compute target y
        targets_batch = reward_batch + not_done_batch * self.gamma * q_values_next_target

        # Compute loss for critic
        expected_batch = self.critic(states_batch, action_batch)
        critic_loss = mse_loss(expected_batch, targets_batch)

        # Update critic by minimizing the loss
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self.critic_loss_list.append((self.t, critic_loss.detach().numpy().mean()))

        # Delayed policy updates
        if self.t % self.policy_freq == 0:

            # Compute loss for actor
            actions_predicted = self.actor(states_batch)

            # fill in missing action with the other agent's real action
            if agent_index == 0:
                actions_predicted = torch.cat([actions_predicted, action_batch[:, 2:]], dim=1)
            else:
                actions_predicted = torch.cat([action_batch[:, :2], actions_predicted], dim=1)

            actor_loss = - self.critic(states_batch, actions_predicted).mean()

            # Update the actor policy using the sampled policy gradient
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.actor_loss_list.append((self.t, actor_loss.detach().numpy().mean()))

            # Update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

            # Update epsilon noise value
            self.glie.update_eps()
