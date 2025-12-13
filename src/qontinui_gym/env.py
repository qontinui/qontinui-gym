"""Main Gymnasium environment for Qontinui visual automation.

This module provides the QontinuiEnv class, which wraps the qontinui-runner
to provide a standard Gymnasium interface for RL algorithms.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.core import RenderFrame

from qontinui_gym.client.runner_client import RunnerClient
from qontinui_gym.config.loader import QontinuiConfig, load_qontinui_config
from qontinui_gym.spaces.action_space import (
    ActionSpaceConfig,
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

logger = logging.getLogger(__name__)


class QontinuiEnv(gym.Env[dict[str, Any], int | dict[str, Any]]):
    """Gymnasium environment for visual GUI automation with Qontinui.

    This environment connects to a running qontinui-runner instance
    and allows RL agents to interact with GUI applications through
    workflows, clicks, and other actions.

    Supports:
    - Hybrid action space (discrete workflow selection + parameterized actions)
    - Multi-modal observations (screenshot + state info)
    - Customizable through action/observation space configs

    Example:
        env = QontinuiEnv(
            config_path="/path/to/automation.json",
            runner_host="localhost",
            runner_port=9876,
            max_episode_steps=100,
        )

        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

    Note:
        The default reward is 0. Use reward wrappers or the reward
        system (qontinui_gym.rewards) to define meaningful rewards.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1,
    }

    def __init__(
        self,
        config_path: str,
        runner_host: str | None = None,
        runner_port: int | None = None,
        action_space_config: ActionSpaceConfig | None = None,
        observation_space_config: ObservationSpaceConfig | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 100,
        monitor: int | str | None = None,
        auto_load_config: bool = True,
        capture_screenshots: bool = True,
    ):
        """Initialize the Qontinui environment.

        Args:
            config_path: Path to QontinuiConfig JSON file
            runner_host: Runner host (default: auto-detect)
            runner_port: Runner port (default: 9876)
            action_space_config: Configuration for action space
            observation_space_config: Configuration for observation space
            render_mode: "human" or "rgb_array"
            max_episode_steps: Maximum steps per episode
            monitor: Monitor to use (index or position like "left")
            auto_load_config: Automatically load config to runner on reset
            capture_screenshots: Capture screenshots for observations
        """
        super().__init__()

        # Store configuration
        self.config_path = config_path
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.monitor = monitor
        self.auto_load_config = auto_load_config
        self.capture_screenshots = capture_screenshots

        # Load QontinuiConfig
        logger.info(f"Loading config from: {config_path}")
        self.config = load_qontinui_config(config_path)

        # Initialize runner client
        self.client = RunnerClient(host=runner_host, port=runner_port)

        # Build spaces from config
        self._action_config = action_space_config or ActionSpaceConfig()
        self._obs_config = observation_space_config or ObservationSpaceConfig()

        self.action_space = build_action_space(self.config, self._action_config)
        self.observation_space = build_observation_space(self.config, self._obs_config)

        # Episode state
        self._current_step = 0
        self._episode_start_time = 0.0
        self._previous_state_ids: set[str] = set()
        self._current_state_ids: set[str] = set()
        self._previous_state_names: set[str] = set()
        self._current_state_names: set[str] = set()
        self._last_screenshot: npt.NDArray[np.uint8] | None = None
        self._action_history: list[int] = []
        self._config_loaded = False

        # Metadata for logging
        self._action_metadata = get_action_metadata(self.config, self._action_config)
        self._obs_metadata = get_observation_metadata(self.config, self._obs_config)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed (for reproducibility)
            options: Optional reset options:
                - initial_states: List of state names to navigate to
                - goal_state: Goal state for goal-conditioned RL

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        options = options or {}

        # Ensure runner is connected
        if not self._ensure_runner_connected():
            raise ConnectionError(
                f"Cannot connect to qontinui-runner at {self.client.base_url}"
            )

        # Load config if needed
        if self.auto_load_config and not self._config_loaded:
            self._load_config_to_runner()

        # Reset episode state
        self._current_step = 0
        self._episode_start_time = time.time()
        self._previous_state_ids = set()
        self._current_state_ids = set()
        self._previous_state_names = set()
        self._current_state_names = set()
        self._action_history = []

        # Navigate to initial states if specified
        initial_states = options.get("initial_states")
        if initial_states is None:
            # Use config's initial states
            initial_states = [s.name for s in self.config.initial_states]

        if initial_states:
            result = self.client.go_to_state(
                state_names=initial_states,
                monitor=self.monitor,
            )
            if result.success:
                self._current_state_names = set(initial_states)
                self._current_state_ids = {
                    s.id for s in self.config.states if s.name in initial_states
                }

        # Capture initial screenshot
        if self.capture_screenshots:
            self._last_screenshot = self.client.take_screenshot(monitor=self.monitor)

        # Build observation
        observation = self._build_observation()

        # Build info
        info = {
            "config_name": self.config.metadata.name,
            "available_workflows": self._action_metadata["workflow_names"],
            "current_state_ids": list(self._current_state_ids),
            "current_state_names": list(self._current_state_names),
            "goal_state": options.get("goal_state"),
        }

        return observation, info

    def step(
        self,
        action: int | dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute an action and return the result.

        Args:
            action: Action from the action space (int for discrete, dict for hybrid)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)

        Note:
            The default reward is 0. Use reward wrappers to define rewards.
        """
        self._current_step += 1

        # Store previous state
        self._previous_state_ids = self._current_state_ids.copy()
        self._previous_state_names = self._current_state_names.copy()
        screenshot_before = self._last_screenshot

        # Decode and execute action
        action_type, action_params = decode_action(
            action, self.config, self._action_config
        )

        result = self._execute_action(action_type, action_params)

        # Capture new screenshot
        if self.capture_screenshots:
            self._last_screenshot = self.client.take_screenshot(monitor=self.monitor)

        # Update action history
        if isinstance(action, int):
            self._action_history.append(action)
        elif isinstance(action, dict) and "workflow_idx" in action:
            self._action_history.append(action["workflow_idx"])

        # Build observation
        observation = self._build_observation()

        # Compute reward (default: 0, use wrappers for custom rewards)
        reward = 0.0

        # Check termination
        terminated = self._is_terminated()
        truncated = self._current_step >= self.max_episode_steps

        # Build info dict with all information needed for reward computation
        info = {
            # State information
            "previous_state_ids": list(self._previous_state_ids),
            "current_state_ids": list(self._current_state_ids),
            "previous_state_names": list(self._previous_state_names),
            "current_state_names": list(self._current_state_names),
            # Action information
            "action_type": action_type,
            "action_params": action_params,
            "action_success": result.success,
            "action_duration_ms": result.duration_ms,
            "action_error": result.error,
            "workflow_name": result.workflow_name,
            # Visual information
            "screenshot_before": screenshot_before,
            "screenshot_after": self._last_screenshot,
            # Episode tracking
            "step": self._current_step,
            "episode_time": time.time() - self._episode_start_time,
            # For goal-conditioned RL (set by options in reset)
            "goal_state": None,
            "goal_achieved": False,
        }

        return observation, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render the current state.

        Returns:
            Screenshot as numpy array (for rgb_array mode) or None
        """
        if self.render_mode == "rgb_array":
            if self._last_screenshot is not None:
                return self._last_screenshot  # type: ignore[return-value]
            return self.client.take_screenshot(monitor=self.monitor)  # type: ignore[return-value]
        elif self.render_mode == "human":
            # For human mode, could display in a window
            # For now, just return the screenshot
            return self._last_screenshot  # type: ignore[return-value]
        return None

    def close(self) -> None:
        """Clean up resources."""
        self.client.close()

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _ensure_runner_connected(self) -> bool:
        """Ensure the runner is connected and healthy."""
        try:
            return self.client.health()
        except Exception as e:
            logger.error(f"Failed to connect to runner: {e}")
            return False

    def _load_config_to_runner(self) -> None:
        """Load the config file to the runner."""
        try:
            success = self.client.load_config(self.config_path)
            if success:
                self._config_loaded = True
                logger.info("Config loaded to runner")
            else:
                logger.warning("Failed to load config to runner")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def _execute_action(
        self,
        action_type: str,
        params: dict[str, Any],
    ) -> Any:
        """Execute an action and return the result."""
        from qontinui_gym.client.runner_client import WorkflowResult

        try:
            if action_type == "workflow":
                workflow_name = params.get("workflow_name", "")
                if workflow_name:
                    result = self.client.run_workflow(
                        workflow_name=workflow_name,
                        monitor=self.monitor,
                    )
                    # Update state based on result
                    self._update_state_after_action()
                    return result
                return WorkflowResult(
                    success=False,
                    workflow_name="",
                    error="No workflow name provided",
                )

            elif action_type == "go_to_state":
                state_name = params.get("state_name", "")
                if state_name:
                    go_result = self.client.go_to_state(
                        state_names=[state_name],
                        monitor=self.monitor,
                    )
                    if go_result.success:
                        self._current_state_names = {state_name}
                        state = self.config.get_state_by_name(state_name)
                        if state:
                            self._current_state_ids = {state.id}
                    return WorkflowResult(
                        success=go_result.success,
                        workflow_name=f"go_to_state:{state_name}",
                        duration_ms=go_result.duration_ms,
                        error=go_result.error,
                    )
                return WorkflowResult(
                    success=False,
                    workflow_name="go_to_state",
                    error="No state name provided",
                )

            elif action_type in ("click", "double_click", "right_click"):
                # Direct click actions would need additional runner support
                # For now, return failure
                return WorkflowResult(
                    success=False,
                    workflow_name=action_type,
                    error="Direct click actions not yet implemented",
                )

            elif action_type == "wait":
                duration = params.get("duration", 1.0)
                time.sleep(duration)
                return WorkflowResult(
                    success=True,
                    workflow_name="wait",
                    duration_ms=int(duration * 1000),
                )

            else:
                return WorkflowResult(
                    success=False,
                    workflow_name=action_type,
                    error=f"Unknown action type: {action_type}",
                )

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return WorkflowResult(
                success=False,
                workflow_name=params.get("workflow_name", action_type),
                error=str(e),
            )

    def _update_state_after_action(self) -> None:
        """Update current state by querying the runner."""
        try:
            states = self.client.list_states()
            active_states = [s for s in states if s.get("active", False)]
            self._current_state_ids = {s.get("id", "") for s in active_states}
            self._current_state_names = {s.get("name", "") for s in active_states}
        except Exception:
            pass  # Keep previous state if query fails

    def _build_observation(self) -> dict[str, Any]:
        """Build observation from current state."""
        return build_observation(
            config=self.config,
            obs_config=self._obs_config,
            screenshot=self._last_screenshot,
            current_state_ids=list(self._current_state_ids),
            state_confidences=None,  # Could be populated from state detection
            available_workflow_indices=None,  # All available by default
            step=self._current_step,
            action_history=self._action_history,
        )

    def _is_terminated(self) -> bool:
        """Check if episode should terminate.

        Override this method or use termination wrappers for custom logic.
        Default: terminate if any final state is reached.
        """
        final_state_ids = {s.id for s in self.config.final_states}
        final_state_names = {s.name for s in self.config.final_states}

        if self._current_state_ids & final_state_ids:
            return True
        if self._current_state_names & final_state_names:
            return True

        return False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_config(self) -> QontinuiConfig:
        """Get the loaded QontinuiConfig."""
        return self.config

    def get_action_metadata(self) -> dict[str, Any]:
        """Get metadata about the action space."""
        return self._action_metadata

    def get_observation_metadata(self) -> dict[str, Any]:
        """Get metadata about the observation space."""
        return self._obs_metadata

    def get_workflow_names(self) -> list[str]:
        """Get list of available workflow names."""
        names: list[str] = self._action_metadata["workflow_names"]
        return names

    def get_state_names(self) -> list[str]:
        """Get list of state names."""
        return [s.name for s in self.config.states]
