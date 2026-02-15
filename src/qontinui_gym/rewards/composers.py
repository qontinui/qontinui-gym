"""Reward composition utilities for Qontinui environments.

This module provides tools for combining multiple reward components:
- ComposedReward: Combine components with various strategies
- RewardBuilder: Fluent API for building reward functions
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from qontinui_gym.rewards.base import RewardComponent, StepInfo
from qontinui_gym.rewards.components import (
    ActionSuccessReward,
    InvalidActionPenalty,
    StateProgressReward,
    StateReachReward,
    StateTransitionReward,
    StateVisitCountReward,
    StepPenalty,
)

if TYPE_CHECKING:
    from qontinui_gym.config.loader import QontinuiConfig


class CompositionMode(Enum):
    """How to combine multiple reward components."""

    SUM = "sum"  # Simple sum of all components
    WEIGHTED_SUM = "weighted_sum"  # Sum with explicit weights (weights in components)
    PRODUCT = "product"  # Multiply all components
    MIN = "min"  # Take minimum (pessimistic)
    MAX = "max"  # Take maximum (optimistic)
    MEAN = "mean"  # Average of all components
    CUSTOM = "custom"  # User-provided function


@dataclass
class RunningStats:
    """Running statistics for reward normalization."""

    mean: float = 0.0
    var: float = 1.0
    count: int = 0
    epsilon: float = 1e-8

    def update(self, value: float) -> None:
        """Update running statistics with new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.var += delta * delta2

    def normalize(self, value: float) -> float:
        """Normalize value using running statistics."""
        std = float(np.sqrt(self.var / max(1, self.count))) + self.epsilon
        return (value - self.mean) / std


class ComposedReward:
    """Compose multiple reward components into a single reward function.

    This is the main class for building complex reward functions by
    combining simpler components.

    Example:
        reward = ComposedReward([
            StateReachReward({"goal"}, reward_per_state=10.0),
            StepPenalty(-0.01),
            StateProgressReward(state_distances, gamma=0.99),
        ], mode=CompositionMode.SUM)

        # Use in environment
        r = reward(step_info)
    """

    def __init__(
        self,
        components: Sequence[RewardComponent],
        mode: CompositionMode = CompositionMode.SUM,
        custom_combiner: Callable[[list[float]], float] | None = None,
        clip_range: tuple[float, float] | None = None,
        normalize: bool = False,
    ):
        """Initialize composed reward.

        Args:
            components: List of reward components to combine
            mode: How to combine component rewards
            custom_combiner: Custom function for CUSTOM mode
            clip_range: Optional (min, max) to clip final reward
            normalize: Whether to normalize rewards using running stats
        """
        self.components = list(components)
        self.mode = mode
        self.custom_combiner = custom_combiner
        self.clip_range = clip_range
        self.normalize = normalize
        self._running_stats = RunningStats() if normalize else None
        self._config: QontinuiConfig | None = None

    def initialize(self, config: QontinuiConfig) -> None:
        """Initialize all components with configuration."""
        self._config = config
        for component in self.components:
            if hasattr(component, "initialize"):
                component.initialize(config)

    def __call__(self, step_info: StepInfo) -> float:
        """Compute combined reward for this step.

        Args:
            step_info: Information about the current step

        Returns:
            Combined reward value
        """
        if not self.components:
            return 0.0

        rewards = [c.compute(step_info) for c in self.components]

        # Combine based on mode
        if self.mode == CompositionMode.SUM:
            total = sum(rewards)
        elif self.mode == CompositionMode.WEIGHTED_SUM:
            total = sum(rewards)  # Weights already applied in components
        elif self.mode == CompositionMode.PRODUCT:
            total = float(np.prod(rewards))
        elif self.mode == CompositionMode.MIN:
            total = min(rewards)
        elif self.mode == CompositionMode.MAX:
            total = max(rewards)
        elif self.mode == CompositionMode.MEAN:
            total = float(np.mean(rewards))
        elif self.mode == CompositionMode.CUSTOM and self.custom_combiner:
            total = self.custom_combiner(rewards)
        else:
            total = sum(rewards)

        # Optional normalization
        if self.normalize and self._running_stats:
            self._running_stats.update(total)
            total = self._running_stats.normalize(total)

        # Optional clipping
        if self.clip_range:
            total = float(np.clip(total, self.clip_range[0], self.clip_range[1]))

        return total

    def reset(self) -> None:
        """Reset all components for new episode."""
        for component in self.components:
            component.reset()

    def get_components(self) -> list[RewardComponent]:
        """Get all component reward functions."""
        return self.components

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics for all components."""
        return {
            "mode": self.mode.value,
            "num_components": len(self.components),
            "components": [
                c.get_statistics() if hasattr(c, "get_statistics") else {"name": str(c)}
                for c in self.components
            ],
            "normalize": self.normalize,
            "clip_range": self.clip_range,
        }


class RewardBuilder:
    """Fluent builder for creating composed rewards.

    Provides a convenient API for building reward functions step by step.

    Example:
        reward = (RewardBuilder()
            .goal_reaching(["checkout_complete"], reward=100.0, terminal=True)
            .progress_shaping(state_distances, gamma=0.99)
            .step_penalty(-0.001)
            .exploration_bonus(beta=0.1)
            .build())
    """

    def __init__(self) -> None:
        """Initialize reward builder."""
        self._components: list[RewardComponent] = []
        self._mode = CompositionMode.SUM
        self._clip_range: tuple[float, float] | None = None
        self._normalize = False
        self._custom_combiner: Callable[[list[float]], float] | None = None

    def goal_reaching(
        self,
        target_states: set[str] | list[str],
        reward: float = 1.0,
        terminal: bool = False,
        once_per_episode: bool = True,
        weight: float = 1.0,
    ) -> RewardBuilder:
        """Add goal-reaching reward component.

        Args:
            target_states: States that give reward when reached
            reward: Reward per target state reached
            terminal: Mark episode terminal on reach
            once_per_episode: Only reward first time reaching each state
            weight: Component weight
        """
        self._components.append(
            StateReachReward(
                target_states=set(target_states),
                reward_per_state=reward,
                terminal_on_reach=terminal,
                once_per_episode=once_per_episode,
                weight=weight,
            )
        )
        return self

    def progress_shaping(
        self,
        state_distances: dict[str, int],
        gamma: float = 0.99,
        weight: float = 1.0,
    ) -> RewardBuilder:
        """Add potential-based progress shaping.

        Args:
            state_distances: Map of state_id -> distance to goal
            gamma: Discount factor for shaping
            weight: Component weight
        """
        self._components.append(
            StateProgressReward(
                state_distances=state_distances,
                gamma=gamma,
                weight=weight,
            )
        )
        return self

    def step_penalty(
        self,
        penalty: float = -0.001,
        weight: float = 1.0,
    ) -> RewardBuilder:
        """Add step penalty for efficiency.

        Args:
            penalty: Penalty per step (should be negative)
            weight: Component weight
        """
        self._components.append(StepPenalty(penalty=penalty, weight=weight))
        return self

    def exploration_bonus(
        self,
        beta: float = 1.0,
        weight: float = 1.0,
    ) -> RewardBuilder:
        """Add count-based exploration bonus.

        Args:
            beta: Scale factor for bonus
            weight: Component weight
        """
        self._components.append(StateVisitCountReward(beta=beta, weight=weight))
        return self

    def transition_reward(
        self,
        success_reward: float = 0.1,
        failure_penalty: float = -0.1,
        bonus_transitions: dict[tuple[str, str], float] | None = None,
        weight: float = 1.0,
    ) -> RewardBuilder:
        """Add transition success/failure rewards.

        Args:
            success_reward: Reward for successful transition
            failure_penalty: Penalty for failed action
            bonus_transitions: Extra rewards for specific transitions
            weight: Component weight
        """
        self._components.append(
            StateTransitionReward(
                reward_on_transition=success_reward,
                penalty_on_failure=failure_penalty,
                bonus_transitions=bonus_transitions,
                weight=weight,
            )
        )
        return self

    def action_success(
        self,
        reward: float = 0.1,
        only_on_state_change: bool = False,
        weight: float = 1.0,
    ) -> RewardBuilder:
        """Add reward for successful actions.

        Args:
            reward: Reward for success
            only_on_state_change: Only reward when state changes
            weight: Component weight
        """
        self._components.append(
            ActionSuccessReward(
                reward=reward,
                only_on_state_change=only_on_state_change,
                weight=weight,
            )
        )
        return self

    def invalid_action_penalty(
        self,
        penalty: float = -0.5,
        weight: float = 1.0,
    ) -> RewardBuilder:
        """Add penalty for invalid/failed actions.

        Args:
            penalty: Penalty for invalid action
            weight: Component weight
        """
        self._components.append(InvalidActionPenalty(penalty=penalty, weight=weight))
        return self

    def custom(self, component: RewardComponent) -> RewardBuilder:
        """Add a custom reward component.

        Args:
            component: Custom RewardComponent instance
        """
        self._components.append(component)
        return self

    def with_mode(self, mode: CompositionMode) -> RewardBuilder:
        """Set composition mode.

        Args:
            mode: How to combine component rewards
        """
        self._mode = mode
        return self

    def with_clipping(self, min_val: float, max_val: float) -> RewardBuilder:
        """Clip final reward to range.

        Args:
            min_val: Minimum reward value
            max_val: Maximum reward value
        """
        self._clip_range = (min_val, max_val)
        return self

    def with_normalization(self) -> RewardBuilder:
        """Enable running normalization of rewards."""
        self._normalize = True
        return self

    def with_custom_combiner(
        self,
        combiner: Callable[[list[float]], float],
    ) -> RewardBuilder:
        """Set custom reward combination function.

        Args:
            combiner: Function that takes list of rewards and returns combined
        """
        self._custom_combiner = combiner
        self._mode = CompositionMode.CUSTOM
        return self

    def build(self) -> ComposedReward:
        """Build the composed reward function.

        Returns:
            ComposedReward ready to use
        """
        return ComposedReward(
            components=self._components,
            mode=self._mode,
            custom_combiner=self._custom_combiner,
            clip_range=self._clip_range,
            normalize=self._normalize,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def default_reward(
    goal_states: set[str] | None = None,
    step_penalty: float = -0.001,
    goal_reward: float = 10.0,
) -> ComposedReward:
    """Create a default reward function with sensible settings.

    Args:
        goal_states: Optional target states for goal reward
        step_penalty: Per-step penalty
        goal_reward: Reward for reaching goal states

    Returns:
        ComposedReward with default components
    """
    builder = RewardBuilder().step_penalty(step_penalty).transition_reward()

    if goal_states:
        builder = builder.goal_reaching(goal_states, reward=goal_reward)

    return builder.build()


def sparse_reward(
    goal_states: set[str],
    goal_reward: float = 1.0,
) -> ComposedReward:
    """Create a sparse reward function (only goal reward).

    Args:
        goal_states: Target states that give reward
        goal_reward: Reward for reaching goal

    Returns:
        ComposedReward with only goal reward
    """
    return RewardBuilder().goal_reaching(goal_states, reward=goal_reward).build()


def dense_reward(
    goal_states: set[str],
    state_distances: dict[str, int],
    goal_reward: float = 10.0,
    step_penalty: float = -0.001,
    shaping_weight: float = 1.0,
) -> ComposedReward:
    """Create a dense reward function with progress shaping.

    Args:
        goal_states: Target states
        state_distances: Map of state_id -> distance to goal
        goal_reward: Reward for reaching goal
        step_penalty: Per-step penalty
        shaping_weight: Weight for progress shaping

    Returns:
        ComposedReward with shaping
    """
    return (
        RewardBuilder()
        .goal_reaching(goal_states, reward=goal_reward)
        .progress_shaping(state_distances, weight=shaping_weight)
        .step_penalty(step_penalty)
        .build()
    )
