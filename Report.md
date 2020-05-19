# Problem Setting

The problem of playing tennis can be formulated as a Reinforcement Learning (**RL**) problem, where an Agent learns through interaction with Environment to achieve their goal - maintaining the ball in the air for as many time steps as possible. The Agent can observe the state which the Environment is in, and takes actions that affect this state. In turn, the Environment gives the Agent feedback ("rewards") based on the actions.

This setting can be formulated as Markov Decision Process (**MDP**), where:

* Action space `A`, which is continuous with 2 dimensions (corresponding to move and jump motions)
* State space `S`, which is defined to be 24-dimensional (eg. position and velocity of the ball and racket)
* The transition to the next state `s_{t+1}` and the resulting reward `r_{t+1}` are defined by the Environment and depend only on the current state `s_t` and the Agent's chosen action `a_t` ("one-step dynamics")
* Discount rate `gamma`, which is used by the Agent to prioritize current rewards over future rewards.

The Agent's goal is to find the optimal _policy_ `pi: S -> A` that maximizes the expected discounted return `J` (weighted sum of all future rewards per episode). Thus, the optimal policy is a function that gives the best possible action for each Environment state. In order to achieve this goal, the Agent can utilize the estimated values of `(s, a)` pairs by learning the so-called _action-value function_ `Q(s, a)`.

In our case, both state and action space are continuous and high-dimensional, which means that Deep Neural Networks (**DNN**s) can be used to represent the action-value function `Q` as well as policy `pi`. The dependency of both functions on the network weights is denoted by `Q_θ` and `pi_φ`, where `θ` and `φ` are the respective DNN weights.

Well-established RL algorithms for discrete action spaces are eg. Deep Q-Networks (**DQN**) and Double Deep Q-Networks (**Double DQN**). Both are based on a Q-Learning algorithm, which updates the Q-values iteratively as follows:

`Q_new(s_t, a_t) := Q(s_t, a_t) + alpha * (r_{t+1} + gamma * max(Q(s_{t+1}, a)) - Q(s_t, a_t))`

In other words, `<Q_new> = <Q_old> + alpha * (<target> - <Q_old>)`

In Q-Learning, the Q-values are utilized by the Agent to choose his actions with "epsilon-greedy policy", where for each state the best action according to `Q` is chosen with probability `(1 - epsilon)`, and a random action - with probability `epsilon`. Traditionally, `epsilon` is decayed after each episode during training.


# Learning Algorithm

Generally, RL algorithms can be divided into 3 categories:

| Category        | Description     | Examples  |
| :-------------: | :-------------: | :-----: |
| Value-Based (_Critic-Only_) | Use a DNN to learn action-value function `Q` and deriving epsilon-greedy policy `pi` based on `Q` | [DQN](https://www.nature.com/articles/nature14236), [Double DQN](https://arxiv.org/abs/1509.06461) |
| Policy-Based (_Actor-Only_) | Use a DNN to directly learn policy `pi` | [PPO](https://arxiv.org/abs/1707.06347), [TRPO](https://arxiv.org/abs/1502.05477) |
| _Actor-Critic_ |  Use `Q` as a baseline for learning `pi` | [DDPG](https://arxiv.org/abs/1509.02971), [ACKTR](https://arxiv.org/abs/1708.05144), [SAC](https://arxiv.org/abs/1801.01290) |


Actor-Critic methods combine the advantages of the other two categories of methods. In this case, an **Actor** represents the learned policy `pi` and is used to select actions, while the **Critic** represents the action-value function `Q` and evaluates `pi` by gathering experience and learning action values. Actor's training is then based on the learned action values.

In this project, an Actor-Critic method called "Deep Deterministic Policy Gradient" (**DDPG**) was used (see this [paper](https://arxiv.org/abs/1509.02971)). In case of playing tennis, a Multi-Agent RL setting was used to train each player separately. Thus, both Agents observed the common (concatenated) Environment state and knew each other's actions, but each learns to act according to his own reward signal.

The nature of interaction between agents can be either cooperative (all agents must maximize a shared return), competitive (agents have conflicting goals), or mixed (like in our tennis case), and the learning algorithm should be able to tackle this type of interaction.


## DDPG

DDPG is a model-free off-policy Actor-Critic algorithm which addresses well-known problems of overestimation bias and high variance.

_Overestimation bias_ is a natural result of function approximation errors in Q-Learning, where the maximization of a noisy value estimate induces a consistent overestimation. The effect of these errors is further accumulated, because Q-values for the current state are updated using the Q-value estimate of a subsequent state (see formula above). This accumulated error causes _high variance_, where any bad states can be estimated with too high Q-value, resulting in suboptimal policy updates and even divergence.

While it is an established fact that methods applied to discrete action spaces are susceptible to overestimation bias, it was shown that the problem occurs likewise in continuous action spaces (with Actor-Critic methods).

In order to solve these two major problems, DDPG algorithm builds upon DQN and introduces and puts together different techniques:

1) Experience replay buffer for minimizing correlations between samples.

2) Target networks, a common approach in DQN methods, which turns out to be critical for variance reduction by reducing the accumulation of errors. 

3) Policy updates are delayed until the value estimate has converged. This technique couples value function `Q` and policy `pi` more effectively.


#### Experience replay buffer

It is a notable fact that training DNNs in RL settings is instable due to correlations in sequences of Environment observations. Traditionally, this instability is overcome by letting the Agent re-learn from its long-passed experience.

Hence, the Agent maintains a replay buffer of capacity `M` where he stores his previous experience in form of tuples `(s_t, a_t, r_{t+1}, s_{t+1})`. Every now and then, the Agent samples a mini-batch of tuples randomly from the buffer and uses these to update `Q`. Thus, the sequence correlation gets eliminated, and the learned policy is more robust.


#### Target networks

To solve a problem of "moving target" (correlations between action-values and target values), we don't change the DNN weights during training step, because they are used for estimating next best action. We achieve this by maintaining two DNNs - one is used for training, the other ("_target network_") is fixed and only updated with current weights after each `d` steps. 

Thus, target networks are frozen copies of Actor and Critic network `Q` and `pi`, correspondingly. These are used for estimating the target value as follows:

`<target> = r_{t+1} + gamma * Q_target(s_{t+1}, a_{t+1})`, where `a_{t+1} = pi_target(s_{t+1})`

The weights `θ'` of a target network `Q_target` are either updated periodically to exactly match the weights `θ` of the current network `Q`, or by some proportion `τ` at each time step: 

`θ' := τ * θ + (1 − τ) * θ'`

This update can be applied while sampling random mini-batches of transitions from an experience replay buffer.


#### Delayed Policy Updates

Because policy updates on high-error states lead to divergent behavior, the Actor `pi_φ` should be updated at a lower frequency than the Critic (eg. each `d` iterations), so that the Q-value error is minimized before the policy update. 

At the same time, each `d` iterations, both target Actor and Critic `pi_target`, `Q_target` are updated using the same proportion `τ` of their current networks.


#### Ornstein-Uhlenbeck Noise

A major challenge of learning in continuous action spaces is exploration. An advantage of off-policy algorithms such as DDPG is that we can treat the problem of exploration independently from the learning algorithm. We constructed an exploration policy `pi_target` by adding noise sampled from a noise process `N` to our actor policy:

`pi_target(s_t) = pi_φ(s_t) + N`

`N` can be chosen to suit the environment. Like in original DDPG paper, we used an Ornstein-Uhlenbeck process to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia (for implementation see [here](https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py)).


In order to encourage more exploration in the beginning of training, we use a linearly decayed noise factor `epsilon`.


#### Algorithm

1. Initialize: `N` - number of training episodes, replay memory with capacity `M`, mini-batch size `B`, `gamma` - discount rate; policy and target update frequency `d`, `τ` - proportion of current weights in a target weight update; parameters of the Ornstein-Uhlenbeck noise for action selection (`σ`, `θ`, `dt`); linear `epsilon` decay policy

2. Initialize two identical Agents with their own Actor and Critic networks

3. (For each Agent) Initialize weights and biases according to [He scheme](https://arxiv.org/abs/1502.01852): Actor `pi_φ`, Critic `Q_θ`; create their copies: target Actor `pi_target` and target Critic `Q_target`

4. For each episode out of `N`:

   4.1. While not done:

      4.1.1. Observe `s_t`

      4.1.2. Choose `a_t` using current policy: `a_t ~ pi_φ(s_t) + epsilon * noise`, where `noise ~ OUNoise(σ, θ, dt)`. Concatenate the resulting actions from both Agents
   
      4.1.3. Take action `a_t`, observe reward `r_{t+1}` and next state `s_{t+1}` of the Environment 
   
      4.1.4. Store tuple `(s_t, a_t, r_{t+1}, s_{t+1}, done_{t+1})` in Agents' replay memory, where `done_{t+1} = 1` if the episode ended at timestep `t+1`, else `0`. Note that all states and actions in this tuple are the concatenated ones coming from both Agents, but the reward value belongs to the corresponding agent
   
      4.1.5. For each Agent:
      
      * Sample random mini-batch of size `B` from memory
      
      * Compute target action `a' = pi_target(s_{t+1}) + epsilon * noise`, where `noise ~ OUNoise(σ, θ, dt)`. Concatenate it with another Agent's action in order to use as input for Critic
      
      * `<target> = r_{t+1} + gamma * (1 - done_{t+1}) * Q_target(s_{t+1}, a')`
   
      * Perform gradient descent on Critic's weights w.r.t. `<target>` with MSE as loss function
   
      * Every `d` steps: update current Actor `pi_φ` by the deterministic policy gradient w.r.t. `Q_θ` and update all target networks:
      
      `θ_target := τ * θ + (1 − τ) * θ_target`
      
      `φ_target := τ * φ + (1 − τ) * φ_target`
      
      * Every `d` steps: update `epsilon`


## Implementation

The interaction between the Agent and the Environment is implemented in `algo.py`. The Agent's internal logic is placed in `agent.py`, including action selection according to a current policy, using replay buffer and a call to train the policy.

`models.py` contains the Actor and Critic DNNs with the required weight and bias initialization. 

The Actor network architecture is sequential with 3 linear neuron layers and RELU as an activation function. State vector (a concatenation of Env state as seen by each Agent) is fed directly to the input layer, and the last layer's activation function is `tanh`. While the number of neurons in the first and second hidden layer is configurable (see variables `num_fc_1` and `num_fc_2` in `const.py`), the output layer has the dimensionality of the action space. Thus, Actor outputs the action only for the respective agent.

The Critic network receives the concatenated states and actions vector as input. However, the actions are fed to the second hidden layer, whereas the states are fed to the input layer, as described in DDPG paper. Critic network has the same architecture as Actor, except that the output layer has 1 neuron without activation function.

Finally, Grid Search is implemented in `hyperparameter_search.py` in order to select the best Agent solving the given Environment. The script also documents what hyperparameter values were tested so far. Best resulting hyperparameters are already listed in `const.py`.


# Hyperparameter optimization

As mentioned above, the current best hyperparameters of the algorithm found by the Grid Search are the following:

`N = 1500`

`M = 200000`

`gamma = 0.99`

`B = 128`

`τ = 0.06`

`d = 1`

`model_learning_rate = 0.001`

`num_fc_actor = 256`

`num_fc_critic = 128`


# Future Work

Other (single-agent) Actor-Critic RL algorithms can be implemented such as A3C, T3D or SAC.

Furthermore, prioritized replay buffer may be utilized to sample important or rare experience tuples more often, proportionally to the received reward.

Finally, some algorithms specifically developed for Multi-Agent RL setting can be considered, such as [MADDPG](https://arxiv.org/abs/1706.02275). It was shown to outperform DDPG on several Multi-Agent Particle Environments.


This method considers action policies of other agents and is able to successfully learn policies that require complex multiagent coordination. Utilizing an ensemble of policies for each agent leads to more robust multi-agent policies. 
