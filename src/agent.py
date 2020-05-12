# -*- coding: utf-8 -*-
import const
from src import models, buffer

from pathlib import Path
import numpy as np
import torch

torch.random.manual_seed(const.random_seed)  # todo


class DRLAgent:
    model_path = const.file_path_model

    def __init__(self, num_states: int = const.state_size,
                 num_actions: int = const.action_size,
                 memory_size: int = const.memory_size,
                 gamma: float = const.gamma,
                 batch_size: int = const.batch_size,
                 expl_noise: int = const.expl_noise,
                 model_learning_rate: float = const.model_learning_rate,
                 num_fc_actor: int = const.num_fc_actor,
                 num_fc_critic: int = const.num_fc_critic
                 ):

        # agent params
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory = buffer.ReplayBuffer(memory_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.expl_noise = expl_noise
        self.start_policy_training_iter = batch_size  # start training after: buffer_size >= batch_size

        # model params
        self.model_learning_rate = model_learning_rate
        self.num_fc_actor = num_fc_actor
        self.num_fc_critic = num_fc_critic

        self.policy = models.TD3(state_dim=self.num_states,
                                 action_dim=self.num_actions,
                                 max_action=const.max_action,
                                 discount=self.gamma,
                                 num_fc_actor=self.num_fc_actor,
                                 num_fc_critic=self.num_fc_critic,
                                 learning_rate=self.model_learning_rate)

        self._model_summary(self.policy.actor, title='Actor')
        self._model_summary(self.policy.critic, title='Critic')

    def act(self, states, t):
        # Select action randomly or according to policy
        if t < self.start_policy_training_iter:
            action = np.random.randn(const.num_agents, self.num_actions)
        else:
            action = (
                    self.policy.select_action(np.array(states[0]))  # todo [0]
                    + np.random.normal(0, const.max_action * self.expl_noise, size=self.num_actions)
            )
        actions = action.reshape(1, -1)
        actions = np.clip(actions, - const.max_action, const.max_action)  # all actions between -1 and 1

        return actions

    def do_stuff(self, state, action, reward, next_state, done, t):
        self.memory.append((state, action, reward, next_state, done))  # memorize

        # Train agent after collecting sufficient data
        if t >= self.start_policy_training_iter:
            self.policy.train(self.memory, self.batch_size)

    def load(self):
        const.myprint('Loading model from:', self.model_path)
        # load with architecture
        checkpoint = torch.load(self.model_path)
        self.policy = models.TD3(state_dim=checkpoint['num_states'],
                                 action_dim=checkpoint['num_actions'],
                                 max_action=const.max_action,
                                 discount=checkpoint['gamma'],
                                 num_fc_actor=checkpoint['num_fc_actor'],
                                 num_fc_critic=checkpoint['num_fc_critic'],
                                 learning_rate=checkpoint['learning_rate'])
        self.policy.critic.load_state_dict(checkpoint['critic'])
        self.policy.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.policy.actor.load_state_dict(checkpoint['actor'])
        self.policy.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])

        # change mode (to use only for inference)
        self.policy.actor.eval()

    def save(self):
        const.myprint('Saving model to:', self.model_path)
        # save with architecture
        checkpoint = {'num_states': self.num_states,
                      'num_actions': self.num_actions,
                      'gamma': self.gamma,
                      'num_fc_actor': self.num_fc_actor,
                      'num_fc_critic': self.num_fc_critic,
                      'learning_rate': self.model_learning_rate,
                      'critic': self.policy.critic.state_dict(),
                      'critic_optimizer': self.policy.critic_optimizer.state_dict(),
                      'actor': self.policy.actor.state_dict(),
                      'actor_optimizer': self.policy.actor_optimizer.state_dict()
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
