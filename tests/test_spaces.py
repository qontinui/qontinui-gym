"""Tests for qontinui_gym.spaces module."""

from pathlib import Path

import numpy as np
from gymnasium import spaces

from qontinui_gym.config.loader import load_qontinui_config
from qontinui_gym.spaces.action_space import (
    ActionSpaceConfig,
    ActionSpaceMode,
    build_action_space,
    decode_action,
    get_action_metadata,
)
from qontinui_gym.spaces.observation_space import (
    ObservationSpaceConfig,
    build_observation,
    build_observation_space,
    get_observation_metadata,
)


class TestActionSpaceConfig:
    """Tests for ActionSpaceConfig."""

    def test_default_config(self) -> None:
        """Test default action space config."""
        config = ActionSpaceConfig()
        assert config.mode == ActionSpaceMode.HYBRID
        assert config.include_coordinates is True
        assert config.include_text_input is True
        assert config.screen_width == 1920
        assert config.screen_height == 1080

    def test_discrete_factory(self) -> None:
        """Test discrete config factory method."""
        config = ActionSpaceConfig.discrete()
        assert config.mode == ActionSpaceMode.DISCRETE

    def test_hybrid_factory(self) -> None:
        """Test hybrid config factory method."""
        config = ActionSpaceConfig.hybrid()
        assert config.mode == ActionSpaceMode.HYBRID


class TestBuildActionSpace:
    """Tests for build_action_space function."""

    def test_build_discrete_space(self, sample_config_file: Path) -> None:
        """Test building discrete action space."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig.discrete()

        space = build_action_space(config, action_config)

        assert isinstance(space, spaces.Discrete)
        assert space.n == 3  # 3 workflows in sample config

    def test_build_hybrid_space(self, sample_config_file: Path) -> None:
        """Test building hybrid action space."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig.hybrid()

        space = build_action_space(config, action_config)

        assert isinstance(space, spaces.Dict)
        assert "action_type" in space.spaces
        assert "workflow_idx" in space.spaces
        assert "coordinates" in space.spaces

    def test_space_sample_discrete(self, sample_config_file: Path) -> None:
        """Test sampling from discrete space."""
        config = load_qontinui_config(sample_config_file)
        space = build_action_space(config, ActionSpaceConfig.discrete())

        for _ in range(10):
            action = space.sample()
            assert isinstance(action, int | np.integer)
            assert 0 <= action < space.n

    def test_space_sample_hybrid(self, sample_config_file: Path) -> None:
        """Test sampling from hybrid space."""
        config = load_qontinui_config(sample_config_file)
        space = build_action_space(config, ActionSpaceConfig.hybrid())

        for _ in range(10):
            action = space.sample()
            assert isinstance(action, dict)
            assert "action_type" in action
            assert "workflow_idx" in action

    def test_exclude_workflows(self, sample_config_file: Path) -> None:
        """Test excluding workflows."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig(
            mode=ActionSpaceMode.DISCRETE,
            exclude_workflows=["Login"],
        )

        space = build_action_space(config, action_config)
        assert space.n == 2  # 3 - 1 excluded = 2

    def test_include_categories(self, sample_config_file: Path) -> None:
        """Test filtering by category."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig(
            mode=ActionSpaceMode.DISCRETE,
            include_categories=["Authentication"],
        )

        space = build_action_space(config, action_config)
        assert space.n == 2  # Only Auth category workflows


class TestDecodeAction:
    """Tests for decode_action function."""

    def test_decode_discrete_action(self, sample_config_file: Path) -> None:
        """Test decoding discrete action."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig.discrete()

        action_type, params = decode_action(0, config, action_config)
        assert action_type == "workflow"
        assert params["workflow_name"] == "Login"

        action_type, params = decode_action(1, config, action_config)
        assert action_type == "workflow"
        assert params["workflow_name"] == "Logout"

    def test_decode_hybrid_workflow_action(self, sample_config_file: Path) -> None:
        """Test decoding hybrid workflow action."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig.hybrid()

        action = {"action_type": 0, "workflow_idx": 2}
        action_type, params = decode_action(action, config, action_config)

        assert action_type == "workflow"
        assert params["workflow_name"] == "Navigate Home"

    def test_decode_click_action(self, sample_config_file: Path) -> None:
        """Test decoding click action."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig.hybrid()

        action = {
            "action_type": 1,  # click
            "coordinates": np.array([100.0, 200.0]),
        }
        action_type, params = decode_action(action, config, action_config)

        assert action_type == "click"
        assert params["x"] == 100.0
        assert params["y"] == 200.0

    def test_decode_scroll_action(self, sample_config_file: Path) -> None:
        """Test decoding scroll action."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig.hybrid()

        action = {
            "action_type": 4,  # scroll
            "scroll": {"direction": 1, "clicks": 3},
        }
        action_type, params = decode_action(action, config, action_config)

        assert action_type == "scroll"
        assert params["direction"] == "down"
        assert params["clicks"] == 4  # 0-indexed to 1-indexed

    def test_decode_wait_action(self, sample_config_file: Path) -> None:
        """Test decoding wait action."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig.hybrid()

        action = {
            "action_type": 7,  # wait
            "wait_duration": np.array([5.0]),
        }
        action_type, params = decode_action(action, config, action_config)

        assert action_type == "wait"
        assert params["duration"] == 5.0


class TestGetActionMetadata:
    """Tests for get_action_metadata function."""

    def test_metadata_content(self, sample_config_file: Path) -> None:
        """Test metadata content."""
        config = load_qontinui_config(sample_config_file)
        action_config = ActionSpaceConfig.hybrid()

        metadata = get_action_metadata(config, action_config)

        assert metadata["mode"] == "hybrid"
        assert metadata["num_workflows"] == 3
        assert "Login" in metadata["workflow_names"]
        assert "workflow" in metadata["action_types"]
        assert metadata["screen_size"] == (1920, 1080)


class TestObservationSpaceConfig:
    """Tests for ObservationSpaceConfig."""

    def test_default_config(self) -> None:
        """Test default observation space config."""
        config = ObservationSpaceConfig()
        assert config.include_screenshot is True
        assert config.include_state_info is True
        assert config.screenshot_width == 224
        assert config.screenshot_height == 224


class TestBuildObservationSpace:
    """Tests for build_observation_space function."""

    def test_build_default_space(self, sample_config_file: Path) -> None:
        """Test building default observation space."""
        config = load_qontinui_config(sample_config_file)
        obs_config = ObservationSpaceConfig()

        space = build_observation_space(config, obs_config)

        assert isinstance(space, spaces.Dict)
        assert "screenshot" in space.spaces
        assert "state" in space.spaces

    def test_screenshot_shape(self, sample_config_file: Path) -> None:
        """Test screenshot shape in observation space."""
        config = load_qontinui_config(sample_config_file)
        obs_config = ObservationSpaceConfig(
            screenshot_width=320,
            screenshot_height=240,
        )

        space = build_observation_space(config, obs_config)
        screenshot_space = space["screenshot"]

        assert screenshot_space.shape == (240, 320, 3)

    def test_space_sample(self, sample_config_file: Path) -> None:
        """Test sampling from observation space."""
        config = load_qontinui_config(sample_config_file)
        space = build_observation_space(config, ObservationSpaceConfig())

        obs = space.sample()

        assert isinstance(obs, dict)
        assert "screenshot" in obs
        assert obs["screenshot"].shape == (224, 224, 3)  # Default size


class TestBuildObservation:
    """Tests for build_observation function."""

    def test_build_observation_with_screenshot(
        self,
        sample_config_file: Path,
        sample_screenshot: np.ndarray,
    ) -> None:
        """Test building observation with screenshot."""
        config = load_qontinui_config(sample_config_file)
        obs_config = ObservationSpaceConfig()

        obs = build_observation(
            config=config,
            obs_config=obs_config,
            screenshot=sample_screenshot,
            current_state_ids=["state-1"],
        )

        assert "screenshot" in obs
        assert "state" in obs
        # Screenshot gets resized to config dimensions (default 224x224)
        assert obs["screenshot"].shape == (224, 224, 3)

    def test_build_observation_without_screenshot(
        self,
        sample_config_file: Path,
    ) -> None:
        """Test building observation without screenshot."""
        config = load_qontinui_config(sample_config_file)
        obs_config = ObservationSpaceConfig()

        obs = build_observation(
            config=config,
            obs_config=obs_config,
            screenshot=None,
            current_state_ids=["state-1"],
        )

        assert "screenshot" in obs
        # Should be zeros when no screenshot provided
        assert np.all(obs["screenshot"] == 0)


class TestGetObservationMetadata:
    """Tests for get_observation_metadata function."""

    def test_metadata_content(self, sample_config_file: Path) -> None:
        """Test metadata content."""
        config = load_qontinui_config(sample_config_file)
        obs_config = ObservationSpaceConfig()

        metadata = get_observation_metadata(config, obs_config)

        assert "screenshot_shape" in metadata
        assert "num_states" in metadata
        assert metadata["num_states"] == 3
