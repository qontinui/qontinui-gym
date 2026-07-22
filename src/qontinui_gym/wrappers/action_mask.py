"""Action masking wrapper for invalid action handling.

Provides action masks based on the current state, compatible
with algorithms that support action masking (like MaskablePPO).
"""

from __future__ import annotations

from typing import Any, SupportsFloat, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

# Qontinui environments always expose dict observations and either a plain
# workflow index or a hybrid action dict (see QontinuiEnv in env.py).
_ObsType: TypeAlias = dict[str, Any]
_ActType: TypeAlias = int | dict[str, Any]


class ActionMaskWrapper(gym.Wrapper[_ObsType, _ActType, _ObsType, _ActType]):
    """Mask invalid actions based on current state.

    Uses state machine transitions to determine which workflows
    are valid from the current state. Compatible with Stable-Baselines3
    MaskablePPO and similar algorithms.

    Example:
        env = QontinuiEnv(config_path="app.json")
        env = ActionMaskWrapper(env)

        obs, info = env.reset()
        mask = env.action_masks()  # Get valid actions
        # Use mask with MaskablePPO or similar
    """

    def __init__(
        self,
        env: gym.Env[_ObsType, _ActType],
        default_all_valid: bool = True,
    ):
        """Initialize action mask wrapper.

        Args:
            env: Base environment
            default_all_valid: If True, all actions valid when state unknown
        """
        super().__init__(env)
        self.default_all_valid = default_all_valid
        self._current_valid_actions: npt.NDArray[np.int8] | None = None
        self._num_actions = self._get_num_actions()

    def _get_num_actions(self) -> int:
        """Get number of actions from action space."""
        action_space = self.env.action_space
        if isinstance(action_space, spaces.Discrete):
            return int(action_space.n)
        elif isinstance(action_space, spaces.Dict):
            if "workflow_idx" in action_space.spaces:
                workflow_space = action_space["workflow_idx"]
                if isinstance(workflow_space, spaces.Discrete):
                    return int(workflow_space.n)
        return 1

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset and update action mask.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self._update_action_mask(obs, info)
        info["action_mask"] = self._current_valid_actions
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step and update action mask.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_action_mask(obs, info)
        info["action_mask"] = self._current_valid_actions
        return obs, reward, terminated, truncated, info

    def _update_action_mask(
        self,
        observation: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        """Update mask based on current states and available transitions."""
        # Check if observation has available_actions
        if "available_actions" in observation:
            self._current_valid_actions = observation["available_actions"].copy()
            return

        # Check info for valid actions
        if "valid_workflow_indices" in info:
            mask = np.zeros(self._num_actions, dtype=np.int8)
            for idx in info["valid_workflow_indices"]:
                if 0 <= idx < self._num_actions:
                    mask[idx] = 1
            self._current_valid_actions = mask
            return

        # Default: all actions valid
        if self.default_all_valid:
            self._current_valid_actions = np.ones(self._num_actions, dtype=np.int8)
        else:
            self._current_valid_actions = np.zeros(self._num_actions, dtype=np.int8)

    def action_masks(self) -> npt.NDArray[np.int8]:
        """Return current action mask.

        This method is called by SB3 MaskablePPO to get valid actions.

        Returns:
            Binary mask array (1 = valid, 0 = invalid)
        """
        if self._current_valid_actions is None:
            return np.ones(self._num_actions, dtype=np.int8)
        return self._current_valid_actions


class TerminateOnInvalidWrapper(gym.Wrapper[_ObsType, _ActType, _ObsType, _ActType]):
    """Terminate episode on invalid action.

    Useful for training agents to only take valid actions.

    Example:
        env = TerminateOnInvalidWrapper(env, penalty=-10.0)
    """

    def __init__(
        self,
        env: gym.Env[_ObsType, _ActType],
        penalty: float = -1.0,
    ):
        """Initialize termination wrapper.

        Args:
            env: Base environment
            penalty: Reward penalty for invalid action
        """
        super().__init__(env)
        self.penalty = penalty

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step with invalid action termination.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if action was invalid
        if not info.get("action_success", True):
            # reward is SupportsFloat, need to convert for arithmetic
            reward = float(reward) + self.penalty
            terminated = True
            info["invalid_action_termination"] = True

        return obs, reward, terminated, truncated, info
