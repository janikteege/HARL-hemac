import copy
import numpy as np
from gymnasium import spaces
from hemac import HeMAC_v0
from hemac.environment.drone import Drone
from hemac.environment.observer import Observer


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
            self.env.observation_space(agent) for agent in self.agents
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
        self.share_observation_space = [self.get_state_space()]
        self.action_space = [self.env.action_space(agent) for agent in self.agents]

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
        share_observations = [self._get_state_curated()]

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
        share_observations = [self._get_state_curated()]
        available_actions = self.get_avail_actions()
        return (
            self._pad_observations(observations),
            share_observations,
            available_actions,
        )

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

    def _get_state_curated(self):
        # UNUSED
        hemac_env = self.env.unwrapped.env
        # self.env.state() just returns [0,0] for hemac
        # that is useless, so we use our own global state
        state_list = []

        # EPISODE PROGRESS
        state_list.append(
            hemac_env.num_frames / hemac_env.max_cycles
        )  # Normalized timestep

        # POI Info
        for goal in hemac_env.goals:  # there is env.number_of_POIs
            state_list.extend([goal.x, goal.y, int(goal.detected)])

        # BASE/HOME POSITION
        base = hemac_env.world.base
        state_list.extend([base.centerx, base.centery])

        # OBSERVER COMMUNICATION (what all drones see)
        state_list.extend(hemac_env.world.observer_communication)

        # DRONE STATES (for each drone)
        for agent in hemac_env.agents_list:
            if isinstance(agent, Drone):
                state_list.extend([agent.x, agent.y])  # Position
                state_list.extend([agent.vx, agent.vy])  # Speed
                state_list.append(
                    agent.charge_level / agent.max_charge
                )  # Normalized charge
                state_list.append(agent.carried_targets)  # Targets being carried

        # MAYBE OBSERVER STATE
        for agent in hemac_env.agents_list:
            if isinstance(agent, Observer):
                state_list.extend([agent.x, agent.y])  # Position
                state_list.append(int(agent.goal_in_view))  # sees POI
                state_list.append(
                    agent.orientation
                )  # rad, orientation angle, speed is constant

        # ALSO PROVISIONER STATE

        # EPISODE STATUS
        state_list.append(float(hemac_env.collided))  # Collision flag
        state_list.append(float(hemac_env.terminate))  # Termination flag

        # GLOBAL REWARD
        state_list.append(hemac_env.global_reward)

        return np.asarray(state_list, dtype=np.float32)

    def get_state_space(self):
        """Calculate the shared observation space for the global state."""

        hemac_env = self.env.unwrapped.env

        # Count components
        n_pois = hemac_env.number_of_POIs
        n_drones = sum(1 for agent in hemac_env.agents_list if isinstance(agent, Drone))
        n_observers = sum(
            1 for agent in hemac_env.agents_list if isinstance(agent, Observer)
        )

        # Calculate total state size
        state_size = (
            1  # Episode progress
            + 3 * n_pois  # POI: (x, y, detected)
            + 2  # Base position
            + 2  # Observer communication
            + 6 * n_drones  # Drones: (x, y, vx, vy, charge, targets)
            + 4 * n_observers  # Observers: (x, y, sees_poi, orientation)
            + 2  # Collision, terminate flags
            + 1  # Global reward
        )

        # Return Box space
        # Values are typically unbounded or very large
        shared_obs_space = spaces.Box(
            low=-10000, high=10000, shape=(state_size,), dtype=np.float32
        )

        return shared_obs_space

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
