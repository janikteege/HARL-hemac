import numpy as np
import pytest
from hemac.environment.HeMAC import HeMAC

from harl.envs.hemac.hemac_env import HeMACEnv


@pytest.fixture
def env(state_type):
    assert state_type in ["EP", "FP"]
    args = {
        "time_factor": 0.5,
        "state_type": state_type,
        "area_size": [500, 500],
        "render_ratio": 1.0,
        "n_observers": 1,
        "n_drones": 1,
        "n_provisioners": 0,
        "min_obstacles": 1,
        "max_obstacles": 1,
        "rescuing_targets": False,
        "observer_comm_range": 300,
        "patrol_config": {
            "benchmark": True,
            "area": [[100, 100], [250, 100], [820, 480], [600, 800], [100, 620]],
        },
        "poi_config": [{"speed": 1.0, "dimension": [8, 8], "spawn_mode": "random"}],
        "drone_config": {
            "drones_starting_pos": [],
            "drone_ui_dimension": 16,
            "drone_max_speed": 10,
            "drone_max_charge": 9999,
            "discrete_action_space": True,
        },
        "drone_sensor": {"model": "RoundCamera", "params": {"sensing_range": 30}},
        "observer_sensor": {
            "model": "ForwardFacingCamera",
            "params": {
                "hfov": 0.5235987755982988,
                "sensing_range": 100,
            },
        },
        "provisioner_sensor": {
            "model": "ForwardFacingCamera",
            "params": {
                "hfov": 1.5707963267948966,
                "sensing_range": 30,
            },
        },
    }
    env = HeMACEnv(args=args)
    yield env
    env.close()


class TestHemacEnv:
    """Test the wrapper around HeMAC to work with HARL."""

    @pytest.mark.parametrize("state_type", ["EP"])
    def test_state_observations_ep(self, env):
        """Test if EP State observations from reset look correct."""
        _, state_obs, _ = env.reset(seed=13)

        for space, state in zip(env.share_observation_space, state_obs, strict=True):
            assert space.contains(state)

        # NOTE: HARL will also only use the first element in EP State Type Case
        state_observations = state_obs[0]
        hemac_raw = env.env.unwrapped.env
        assert isinstance(hemac_raw, HeMAC)
        assert (hemac_raw.state() == state_observations).all()

    @pytest.mark.parametrize("state_type", ["FP"])
    def test_state_observations_fp(self, env):
        """Test if FP State observations from reset look correct."""
        _, state_observations, _ = env.reset(seed=13)

        for space, state in zip(
            env.share_observation_space, state_observations, strict=True
        ):
            assert space.contains(state)

        hemac_raw = env.env.unwrapped.env
        assert isinstance(hemac_raw, HeMAC)
        # verify one-hot encoding
        for index in range(env.n_agents):
            one_hot = np.zeros(env.n_agents)
            one_hot[index] = 1.0

            focal_state_observations = state_observations[index]
            assert (
                hemac_raw.state() == focal_state_observations[: -env.n_agents]
            ).all()

            assert (one_hot == focal_state_observations[-env.n_agents :]).all
