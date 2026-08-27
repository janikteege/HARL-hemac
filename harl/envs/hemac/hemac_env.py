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
        state_space = self.get_state_space()
        self.share_observation_space = [
            state_space for _ in self.env.unwrapped.possible_agents
        ]
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
        print(f"Charge Level: {obs[2]}")
        print(f"To Base (x,y): {obs[3]}, {obs[4]}")
        print(f"Distances: R{obs[5]} U{obs[6]} L{obs[7]} D{obs[8]}")
        print(f"Relative Position1 (x,y): {obs[9]} {obs[10]}")
        print(f"Relative Position2 (x,y): {obs[11]} {obs[12]}")

    def print_state_observation(self, state):
        """Print the global state observation in a human-readable format."""
        state = np.asarray(state).reshape(-1)
        idx = 0

        # EPISODE PROGRESS
        print("------------ Episode ------------")
        print(f"Progress: {state[idx]:.3f}")
        idx += 1

        # POI INFO
        hemac_env = self.env.unwrapped.env
        print("------------ POIs ------------")

        for i in range(hemac_env.number_of_POIs):
            x = state[idx]
            y = state[idx + 1]
            detected = state[idx + 2]

            print(f"POI {i}: Position=({x:.3f}, {y:.3f}), Detected={bool(detected)}")

            idx += 3

        # BASE POSITION
        print("------------ Base ------------")
        print(f"Position: ({state[idx]:.3f}, {state[idx + 1]:.3f})")
        idx += 2

        # OBSERVER COMMUNICATION
        print("------------ Observer Communication ------------")
        print(f"Goal Position: ({state[idx]:.3f}, {state[idx + 1]:.3f})")
        idx += 2

        # DRONES
        print("------------ Drones ------------")

        for agent_key in self.env.unwrapped.possible_agents:
            agent = hemac_env.agents_list[hemac_env.agent_name_mapping[agent_key]]

            if isinstance(agent, Drone):
                agent_type = state[idx]
                x = state[idx + 1]
                y = state[idx + 2]
                vx = state[idx + 3]
                vy = state[idx + 4]
                charge = state[idx + 5]
                targets = state[idx + 6]

                print(f"{agent_key}:")
                print(f"  Type: {int(agent_type)}")
                print(f"  Position: ({x:.3f}, {y:.3f})")
                print(f"  Velocity: ({vx:.3f}, {vy:.3f})")
                print(f"  Charge: {charge:.3f}")
                print(f"  Carried Targets: {targets:.3f}")

                idx += 7

        # OBSERVERS
        print("------------ Observers ------------")

        for agent_key in self.env.unwrapped.possible_agents:
            agent = hemac_env.agents_list[hemac_env.agent_name_mapping[agent_key]]

            if isinstance(agent, Observer):
                agent_type = state[idx]
                x = state[idx + 1]
                y = state[idx + 2]
                goal_in_view = state[idx + 3]
                orientation = state[idx + 4]

                print(f"{agent_key}:")
                print(f"  Type: {int(agent_type)}")
                print(f"  Position: ({x:.3f}, {y:.3f})")
                print(f"  Goal in View: {bool(goal_in_view)}")
                print(f"  Orientation: {orientation:.3f}")

                idx += 5

        # EPISODE STATUS
        print("------------ Status ------------")
        print(f"Collision: {bool(state[idx])}")
        print(f"Terminated: {bool(state[idx + 1])}")
        idx += 2

        # Sanity check
        if idx != len(state):
            print(f"WARNING: {len(state) - idx} unused state values remain.")

    def step(self, actions):
        actions = self._format_actions(actions)
        actions = self.wrap(actions)
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

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
        state = self.get_state_observations(observations)
        # self.print_state_observation(state)
        state_observations = [state for _ in self.env.unwrapped.possible_agents]

        agents = self.env.unwrapped.possible_agents
        global_reward = self.env.unwrapped.env.global_reward
        # HeMAC adds the global reward to every individual reward. Count it once
        # while retaining all agent-specific safety penalties.
        individual_reward = sum(rewards[agent] - global_reward for agent in agents)
        team_reward = global_reward + individual_reward
        team_rewards = [[team_reward] for _ in agents]

        dones = {
            agent: terminations[agent] or truncations[agent]
            for agent in self.env.unwrapped.possible_agents
        }
        return (
            observation_list,
            state_observations,
            team_rewards,
            self.unwrap(dones),
            self.unwrap(infos),
            self.get_available_actions(),
        )

    def get_state_observations(self, observations):
        # information about all the agents and targets
        # position relative to origin
        # encoding of agent types
        # last actions
        # information about the scenario, ie number of obstacles, position of the base
        # road network graph
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
            state_list.extend(
                [
                    goal.x / hemac_env.area.width,
                    goal.y / hemac_env.area.height,
                    int(goal.detected),
                ]
            )

        # BASE/HOME POSITION
        base = hemac_env.world.base
        state_list.extend(
            [base.centerx / hemac_env.area.width, base.centery / hemac_env.area.height]
        )

        # OBSERVER COMMUNICATION (what all drones see)
        goal_x_norm = hemac_env.world.observer_communication[0] / hemac_env.area.width
        goal_y_norm = hemac_env.world.observer_communication[1] / hemac_env.area.height
        state_list.append(goal_x_norm)
        state_list.append(goal_y_norm)

        # DRONE STATES (for each drone)
        for agent_key in self.env.unwrapped.possible_agents:
            agent = hemac_env.agents_list[hemac_env.agent_name_mapping[agent_key]]
            if isinstance(agent, Drone):
                state_list.append(0)  # 0 for agent type drone
                state_list.extend(
                    [agent.x / hemac_env.area.width, agent.y / hemac_env.area.height]
                )  # Position
                state_list.extend(
                    [agent.vx / agent.max_speed, agent.vy / agent.max_speed]
                )  # Speed
                state_list.append(
                    agent.charge_level / agent.max_charge
                )  # Normalized charge
                state_list.append(
                    agent.carried_targets / agent.carrying_capacity
                )  # Targets being carried

        # MAYBE OBSERVER STATE
        for agent_key in self.env.unwrapped.possible_agents:
            agent = hemac_env.agents_list[hemac_env.agent_name_mapping[agent_key]]
            if isinstance(agent, Observer):
                state_list.append(1)  # 1 for agent type observer
                state_list.extend(
                    [agent.x / hemac_env.area.width, agent.y / hemac_env.area.height]
                )  # Position
                state_list.append(int(agent.goal_in_view))  # sees POI
                state_list.append(
                    agent.orientation / (2 * np.pi)
                )  # rad, orientation angle, speed is constant

        # ALSO PROVISIONER STATE

        # NUMBER OF OBSTACLES

        # EPISODE STATUS
        state_list.append(float(hemac_env.collided))  # Collision flag
        state_list.append(float(hemac_env.terminate))  # Termination flag

        # GLOBAL REWARD
        # state_list.append(hemac_env.global_reward)

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
            + 7 * n_drones  # Drones: (type, x, y, vx, vy, charge, targets)
            + 5 * n_observers  # Observers: (type, x, y, sees_poi, orientation)
            + 2  # Collision, terminate flags
            # + 1  # Global reward
        )

        # Return Box space
        shared_obs_space = spaces.Box(
            low=-1, high=1, shape=(state_size,), dtype=np.float32
        )

        return shared_obs_space

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
        state = self.get_state_observations(observations)
        state_observations = [state for _ in self.env.unwrapped.possible_agents]
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
