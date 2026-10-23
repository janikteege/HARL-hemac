import copy
from ctypes import ArgumentError

import gymnasium
import numpy as np
from hemac import HeMAC_v0


class HeMACEnv:
    def __init__(self, args):
        # should be discrete
        # observers and provisioners should be discrete (hardcoded in env)
        assert args["drone_config"]["discrete_action_space"]
        self.discrete = True

        self.args = copy.deepcopy(args)
        self.state_type = self.args.pop("state_type", "EP")
        self._pending_seed = None
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
            gymnasium.spaces.Box(
                low=-10000.0,
                high=10000.0,
                shape=(self._max_obs_dim,),
                dtype=np.float32,
            )
            for _ in self.env.unwrapped.possible_agents
        ]
        state_space = self.get_state_space()
        self.share_observation_space = [state_space for _ in self.env.possible_agents]
        self.action_space = [
            self.env.action_space(agent) for agent in self.env.unwrapped.possible_agents
        ]

    def print_action(self, action):
        if action == 0:
            print("Charge")
        elif action == 1:
            print("Top Right Full")
        elif action == 2:
            print("Bottom Right Full")
        elif action == 3:
            print("Top Left Full")
        elif action == 4:
            print("Bottom Left Full")
        # NOTE: add actions for more fine grained control
        elif action == 5:
            print("Top Right Light")
        elif action == 6:
            print("Bottom Right Light")
        elif action == 7:
            print("Top Left Light")
        elif action == 8:
            print("Bottom Left Light")
        elif action == 9:
            print("Nothing")

    def print_drone_obs(self, obs):
        print(f"To Goal (x,y): {obs[0]}, {obs[1]}")
        print(f"Communication valid: {bool(obs[2])}, age: {obs[3]}")
        print(f"Charge Level: {obs[4]}")
        print(f"To Base (x,y): {obs[5]}, {obs[6]}")
        print(f"Distances: R{obs[7]} U{obs[8]} L{obs[9]} D{obs[10]}")
        print(f"Relative Position1 (x,y): {obs[11]} {obs[12]}")
        print(f"Relative Position2 (x,y): {obs[13]} {obs[14]}")

    def step(self, actions):
        actions = self._format_actions(actions)
        actions = self.wrap(actions)
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        for agent in self.env.unwrapped.possible_agents:
            if truncations[agent] and not terminations[agent]:
                infos[agent]["bad_transition"] = True

        # print("------------Drone0----------")
        # self.print_action(actions["drone_0"])
        # self.print_drone_obs(observations["drone_0"])
        # print("------------Drone1----------")
        # self.print_action(actions["drone_1"])
        # self.print_drone_obs(observations["drone_1"])

        observation_list = [
            observations[agent] for agent in self.env.unwrapped.possible_agents
        ]
        observation_list = self._pad_observations(observation_list)
        state_observations = [
            self.get_state_observations(agent_id) for agent_id in range(self.n_agents)
        ]

        agents = self.env.unwrapped.possible_agents
        individual_rewards = [[rewards[agent]] for agent in agents]

        dones = {
            agent: terminations[agent] or truncations[agent]
            for agent in self.env.unwrapped.possible_agents
        }
        return (
            observation_list,
            state_observations,
            individual_rewards,
            self.unwrap(dones),
            self.unwrap(infos),
            self.get_available_actions(),
        )

    def get_state_observations(self, focal_agent_id):
        state = self.env.state()
        if self.state_type == "EP":
            return state
        elif self.state_type == "FP":
            # one-hot encoding of the agent
            focal_agent = np.zeros(len(self.env.possible_agents), dtype=np.float32)
            focal_agent[focal_agent_id] = 1.0
            return np.append(state, focal_agent)
        raise ArgumentError("state_type has to be either EP or FP")

    def get_state_space(self):
        """Calculate the shared observation space for the global state."""
        state_space = self.env.state_space
        if self.state_type == "EP":
            return state_space
        elif self.state_type == "FP":
            low = np.concatenate(
                (state_space.low, np.zeros(self.n_agents, dtype=np.float32))
            )
            high = np.concatenate(
                (state_space.high, np.ones(self.n_agents, dtype=np.float32))
            )
            return gymnasium.spaces.Box(low, high, dtype=np.float32)

        raise ValueError("state_type has to be either EP or FP")

    def _format_actions(self, actions):
        action_array = np.asarray(actions)
        if action_array.ndim == 2 and action_array.shape[1] == 1:
            action_array = action_array.reshape(-1)
        return action_array

    def reset(self, seed=None):
        effective_seed = seed if seed is not None else self._pending_seed
        observations, infos = self.env.reset(seed=effective_seed)
        self._pending_seed = None

        observation_list = [
            observations[agent] for agent in self.env.unwrapped.possible_agents
        ]
        state_observations = [
            self.get_state_observations(agent_id) for agent_id in range(self.n_agents)
        ]
        available_actions = self.get_available_actions()
        observation_list = self._pad_observations(observation_list)
        return observation_list, state_observations, available_actions

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
        max_action_count = max(self.action_space, key=lambda space: space.n).n
        for agent_id in range(self.n_agents):
            avail_actions.append(
                self.get_available_agent_actions(agent_id, pad_to=max_action_count)
            )
        return avail_actions

    def get_available_agent_actions(self, agent_id, pad_to=None) -> list[int]:
        """Returns the available actions for agent_id"""
        action_count = self.action_space[agent_id].n
        available_actions = [1] * action_count
        if pad_to is not None:
            assert action_count <= pad_to, "pad_to has to be larger than action count"
            available_actions.extend([0] * (pad_to - action_count))
        return available_actions

    def seed(self, seed):
        self._pending_seed = seed

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
