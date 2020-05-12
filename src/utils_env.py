import const

from unityagents import UnityEnvironment


class Environment(UnityEnvironment):
    def __init__(self):
        super().__init__(file_name=const.file_name_env)

    def get_info(self):
        # get the default brain
        brain_name = self.brain_names[0]
        brain = self.brains[brain_name]

        # reset the environment
        env_info = self.reset(train_mode=True)[brain_name]

        # number of agents in the environment
        # number of agents
        num_agents = len(env_info.agents)
        print('Number of agents:', num_agents)

        # number of actions
        action_size = brain.vector_action_space_size
        print('Number of actions:', action_size)

        # examine the state space
        states = env_info.vector_observations
        state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
        print('The state for the first agent looks like:', states[0])

        return state_size, action_size
