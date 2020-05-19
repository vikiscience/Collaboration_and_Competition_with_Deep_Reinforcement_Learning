[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: output/score_ref.png
[report]: Report.md

# Collaboration and Competition with Deep Reinforcement Learning

### Introduction

This project is based on the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

Current best result is: +0.719

<!--- Env solved in 1356 episodes, avg_score: 0.509 --->

![image2]

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in a preferred folder, and unzip (or decompress) the file. 

### Instructions

Agent training is done using Python 3.6 and PyTorch 0.4.0.

1. Follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install the working environment.

2. In case that PyTorch v.0.4.0 fails to install via pip (eg. for Windows 10), do it manually:

   `conda install pytorch=0.4.0 -c pytorch`

3. Change the variable `file_name_env` in `const.py` to point at the downloaded Environment accordingly, eg. use the path to `Tennis.exe` for Windows 10 (for other OS, see [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control.ipynb))

4. To test if the installation worked, try:

   `python main.py`

    This will print the Environment info. 

5. To train your own agent with hyperparameter values specified in `const.py`, use the following command:

   `python main.py -e train`
   
   This will produce the following files:

   * `models/model_1.npy` and `models/model_2.npy` with saved network weights for each agent
   * `output/score.png` (plot of **scores** for each episode)
   * `output/loss.png` (plot of losses for each agent's actor and critic networks)
   * `output/actions.png` (plot of actions from replay buffer of the first agent)
   * `output/noise.png` (plot of noise applied to actions of the first agent)

6. To see the **reference** agent provided in this repo in action, execute:

   `python main.py -e test -r`

7. For other command line options, refer to: 

   `python main.py --help`
   
### Model

The problem of agent collaboration and competition was solved by utilizing Deep Reinforcement Learning setting, particularly by implementing a MADDPG algorithm.

For model details, see [report].
