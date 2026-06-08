import copy
import numpy as np
from gymnasium import spaces
from hemac import HeMAC_v0


class HeMACEnv:
    def __init__(self, args):

        # should be discrete
        # observers and provisioners should be discrete (hardcoded in env)
        assert args["drone_config"]["discrete_action_space"]

        self.args = copy.deepcopy(args)

        self.discrete = True
        self.env = HeMAC_v0.parallel_env(**self.args)
        self.agents = list(self.env.possible_agents)
        self.n_agents = len(self.agents)
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 25
            self.args["max_cycles"] = 26

        self.cur_step = 0
        self._raw_observation_spaces = [
            self.env.observation_spaces[agent] for agent in self.agents
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
            for _ in self.agents
        ]
        # TODO: replace env.state() with a richer global state for the critic.
        self.share_observation_space = [
            spaces.Box(
                low=0.0,
                high=1.0,
                shape=self.env.state_space.shape,
                dtype=np.float32,
            )
            for _ in range(self.n_agents)
        ]
        self.action_space = [self.env.action_spaces[agent] for agent in self.agents]

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        assert self.discrete
        action_dict = self.wrap(self._format_actions(actions))

        observations, rewards, terminations, truncations, infos = self.env.step(
            action_dict
        )
        self.cur_step += 1
        if self.cur_step == self.max_cycles:
            truncations = {agent: True for agent in self.agents}
            for agent in self.agents:
                infos[agent]["bad_transition"] = True
        dones = {
            agent: terminations[agent] or truncations[agent] for agent in self.agents
        }
        share_observations = [self._get_state() for _ in range(self.n_agents)]

        total_reward = sum(rewards.values())
        team_rewards = [[total_reward]] * self.n_agents
        return (
            self._pad_observations(self.unwrap(observations)),
            share_observations,
            team_rewards,
            self.unwrap(dones),
            self.unwrap(infos),
            self.get_avail_actions(),
        )

    def reset(self, seed=None):
        """Returns initial observations and states"""
        observations_dict, _ = self.env.reset(seed=seed)
        self.agents = list(self.env.possible_agents)
        self.cur_step = 0
        observations = [observations_dict[agent] for agent in self.agents]
        share_observations = [self._get_state() for _ in range(self.n_agents)]
        available_actions = self.get_avail_actions()
        return self._pad_observations(observations), share_observations, available_actions

    def get_avail_actions(self):
        assert self.discrete
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def _get_state(self):
        return np.asarray(self.env.state(), dtype=np.float32)

    def _pad_observations(self, observations):
        padded = []
        for obs in observations:
            obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
            if obs_array.shape[0] < self._max_obs_dim:
                pad_width = self._max_obs_dim - obs_array.shape[0]
                obs_array = np.pad(obs_array, (0, pad_width), mode="constant")
            padded.append(obs_array)
        return padded

    def _format_actions(self, actions):
        action_array = np.asarray(actions)
        if action_array.ndim == 2 and action_array.shape[1] == 1:
            action_array = action_array.reshape(-1)
        return action_array

    def seed(self, seed):
        self.env.reset(seed=seed)

    def wrap(self, l):
        dictionary = {}
        for i, agent in enumerate(self.agents):
            dictionary[agent] = l[i]
        return dictionary

    def unwrap(self, dictionary):
        return [dictionary[agent] for agent in self.agents]
