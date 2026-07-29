import copy
from hemac import HeMAC_v0


class HeMACEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.env = HeMAC_v0.parallel_env(**self.args)
        self.n_agents = len(self.env.possible_agents)
        self.share_observation_space = [
            self.env.observation_space(agent) for agent in self.env.unwrapped.agents
        ]
        self.observation_space = [
            self.env.observation_space(agent) for agent in self.env.unwrapped.agents
        ]
        self.action_space = [
            self.env.action_space(agent) for agent in self.env.unwrapped.agents
        ]

    def step(self, actions):
        return obs, state, rewards, dones, info, available_actions

    def reset(self):
        return obs, state, available_actions

    def seed(self, seed):
        pass

    def render(self):
        pass

    def close(self):
        self.env.close()
