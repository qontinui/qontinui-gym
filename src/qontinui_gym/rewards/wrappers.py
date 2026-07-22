"""Gymnasium-compatible wrappers for reward functions.

This module provides wrappers that integrate qontinui-gym's reward
system with standard Gymnasium environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsFloat, TypeAlias

import gymnasium as gym
from gymnasium import Wrapper

from qontinui_gym.rewards.base import ActionResult, StepInfo

if TYPE_CHECKING:
    from qontinui_gym.rewards.composers import ComposedReward

# Qontinui environments always expose dict observations and either a plain
# workflow index or a hybrid action dict (see QontinuiEnv in env.py).
_ObsType: TypeAlias = dict[str, Any]
_ActType: TypeAlias = int | dict[str, Any]


class QontinuiRewardWrapper(Wrapper[_ObsType, _ActType, _ObsType, _ActType]):
    """Gymnasium wrapper that applies a qontinui-gym reward function.

    This wrapper intercepts the reward from the base environment
    and replaces it with a reward computed by the provided
    reward function.

    Example:
        base_env = QontinuiEnv(config_path="app.json")

        reward = (RewardBuilder()
            .goal_reaching(["checkout"], reward=100.0)
            .step_penalty(-0.01)
            .build())

        env = QontinuiRewardWrapper(base_env, reward_function=reward)

        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        env: gym.Env[_ObsType, _ActType],
        reward_function: ComposedReward,
        add_original_reward: bool = True,
    ):
        """Initialize reward wrapper.

        Args:
            env: Base Gymnasium environment
            reward_function: Reward function to apply
            add_original_reward: Include original reward in info dict
        """
        super().__init__(env)
        self.reward_function = reward_function
        self.add_original_reward = add_original_reward
        self._step_count = 0
        self._episode_time = 0.0

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute action and compute custom reward.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Build StepInfo from environment info
        step_info = self._build_step_info(action, info, obs)

        # Compute custom reward
        custom_reward = self.reward_function(step_info)

        # Add metadata to info
        if self.add_original_reward:
            info["original_reward"] = original_reward
        info["reward_components"] = self.reward_function.get_statistics()

        self._step_count += 1

        return obs, custom_reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment and reward function.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.reward_function.reset()
        self._step_count = 0
        self._episode_time = 0.0
        return obs, info

    def _build_step_info(
        self,
        action: Any,
        info: dict[str, Any],
        obs: Any,
    ) -> StepInfo:
        """Build StepInfo from environment step results.

        Args:
            action: Action taken
            info: Info dict from environment
            obs: Observation from environment

        Returns:
            StepInfo for reward computation
        """
        # Extract state information from info dict
        previous_state_ids = frozenset(info.get("previous_state_ids", []))
        current_state_ids = frozenset(info.get("current_state_ids", []))
        previous_state_names = frozenset(info.get("previous_state_names", []))
        current_state_names = frozenset(info.get("current_state_names", []))

        # Build action result
        action_result = ActionResult(
            success=info.get("action_success", True),
            action_type=info.get("action_type", "workflow"),
            workflow_name=info.get("workflow_name"),
            duration_ms=info.get("action_duration_ms", 0),
            error=info.get("action_error"),
            states_reached=tuple(info.get("states_reached", [])),
        )

        return StepInfo(
            previous_state_ids=previous_state_ids,
            current_state_ids=current_state_ids,
            previous_state_names=previous_state_names,
            current_state_names=current_state_names,
            action_type=info.get("action_type", "workflow"),
            action_params=info.get("action_params", {}),
            action_result=action_result,
            screenshot_before=info.get("screenshot_before"),
            screenshot_after=info.get("screenshot_after"),
            pattern_matches=tuple(info.get("pattern_matches", [])),
            step_count=self._step_count,
            episode_time=info.get("episode_time", 0.0),
            goal_state=info.get("goal_state"),
            goal_achieved=info.get("goal_achieved", False),
            metadata=info.get("metadata", {}),
        )


class RewardScaleWrapper(Wrapper[_ObsType, _ActType, _ObsType, _ActType]):
    """Simple wrapper to scale rewards by a constant factor.

    Example:
        env = RewardScaleWrapper(env, scale=0.01)
    """

    def __init__(self, env: gym.Env[_ObsType, _ActType], scale: float = 1.0):
        """Initialize reward scale wrapper.

        Args:
            env: Base environment
            scale: Scale factor for rewards
        """
        super().__init__(env)
        self.scale = scale

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward) * self.scale, terminated, truncated, info


class RewardClipWrapper(Wrapper[_ObsType, _ActType, _ObsType, _ActType]):
    """Wrapper to clip rewards to a range.

    Example:
        env = RewardClipWrapper(env, min_reward=-1.0, max_reward=1.0)
    """

    def __init__(
        self,
        env: gym.Env[_ObsType, _ActType],
        min_reward: float = -float("inf"),
        max_reward: float = float("inf"),
    ):
        """Initialize reward clip wrapper.

        Args:
            env: Base environment
            min_reward: Minimum reward value
            max_reward: Maximum reward value
        """
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        clipped = max(self.min_reward, min(self.max_reward, float(reward)))
        info["unclipped_reward"] = reward
        return obs, clipped, terminated, truncated, info
