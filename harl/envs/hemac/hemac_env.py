import copy

import numpy as np
from gymnasium import spaces
from hemac import HeMAC_v0


class HeMACEnv:
    def __init__(self, args):
        # should be discrete
        # observers and provisioners should be discrete (hardcoded in env)
        assert args["drone_config"]["discrete_action_space"]
        self.discrete = True

        self.args = copy.deepcopy(args)
        self.env = HeMAC_v0.parallel_env(**self.args)
        self.n_agents = len(self.env.possible_agents)

        self._raw_observation_spaces = [
            self.env.observation_space(agent)
            for agent in self.env.unwrapped.possible_agents
        ]
        self._max_obs_dim = max(
            space.shape[0] for space in self._raw_observation_spaces
        )
        self.observation_space = [
            spaces.Box(
                low=-10000.0,
                high=10000.0,
                shape=(self._max_obs_dim,),
                dtype=np.float32,
            )
            for _ in self.env.unwrapped.possible_agents
        ]
        self.share_observation_space = [
            spaces.Box(
                low=-10000.0,
                high=10000.0,
                shape=(self._max_obs_dim,),
                dtype=np.float32,
            )
            for _ in self.env.unwrapped.possible_agents
        ]
        self.action_space = [
            self.env.action_space(agent) for agent in self.env.unwrapped.possible_agents
        ]

    def step(self, actions):
        actions = self._format_actions(actions)
        actions = self.wrap(actions)
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        observation_list = [
            observations[agent] for agent in self.env.unwrapped.possible_agents
        ]
        observation_list = self._pad_observations(observation_list)
        # rewards already include global reward
        individual_rewards = [
            [rewards[agent]] for agent in self.env.unwrapped.possible_agents
        ]
        dones = {
            agent: terminations[agent] or truncations[agent]
            for agent in self.env.unwrapped.possible_agents
        }
        return (
            observation_list,
            observation_list,
            individual_rewards,
            self.unwrap(dones),
            self.unwrap(infos),
            self.get_available_actions(),
        )

    def _format_actions(self, actions):
        action_array = np.asarray(actions)
        if action_array.ndim == 2 and action_array.shape[1] == 1:
            action_array = action_array.reshape(-1)
        return action_array

    def reset(self, seed=None):
        observations, infos = self.env.reset(seed=seed)
        observation_list = [
            observations[agent] for agent in self.env.unwrapped.possible_agents
        ]
        # TODO: synthesize a real state
        # state = [observations[agent] for agent in self.env.unwrapped.possible_agents]
        available_actions = self.get_available_actions()
        observation_list = self._pad_observations(observation_list)
        return observation_list, observation_list, available_actions

    def _pad_observations(self, observations):
        padded = []
        for obs in observations:
            obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
            if obs_array.shape[0] < self._max_obs_dim:
                pad_width = self._max_obs_dim - obs_array.shape[0]
                obs_array = np.pad(obs_array, (0, pad_width), mode="constant")
            padded.append(obs_array)
        return padded

    def get_available_actions(self):
        assert self.discrete
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_available_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_available_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def seed(self, seed):
        self.env.reset(seed=seed)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def wrap(self, l):
        dictionary = {}
        for i, agent in enumerate(self.env.unwrapped.possible_agents):
            dictionary[agent] = l[i]
        return dictionary

    def unwrap(self, d):
        return [d[agent] for agent in self.env.unwrapped.possible_agents]
