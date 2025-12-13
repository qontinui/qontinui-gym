"""Base classes and protocols for reward functions.

This module defines the core abstractions for the reward system:
- StepInfo: Immutable information about each environment step
- RewardComponent: Protocol for atomic reward components
- RewardFunction: Protocol for complete reward functions
- BaseRewardComponent: Abstract base class with common functionality
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from qontinui_gym.config.loader import QontinuiConfig


@dataclass(frozen=True)
class PatternMatch:
    """Immutable pattern match result from state detection."""

    pattern_name: str
    similarity: float
    region: tuple[int, int, int, int]  # x, y, width, height
    state_id: str | None = None


@dataclass(frozen=True)
class ActionResult:
    """Immutable result of executing an action."""

    success: bool
    action_type: str = "workflow"
    workflow_name: str | None = None
    duration_ms: int = 0
    error: str | None = None
    states_reached: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepInfo:
    """Immutable information about a single environment step.

    Contains all information from qontinui that reward functions might need.
    This is passed to reward components for computing rewards.

    Attributes:
        previous_state_ids: State IDs active before the action
        current_state_ids: State IDs active after the action
        previous_state_names: State names active before the action
        current_state_names: State names active after the action
        action_type: Type of action taken (workflow, click, etc.)
        action_params: Parameters for the action
        action_result: Result of executing the action
        screenshot_before: Screenshot before action (optional)
        screenshot_after: Screenshot after action (optional)
        pattern_matches: Pattern matches from state detection
        step_count: Current step in the episode
        episode_time: Time elapsed in episode (seconds)
        goal_state: Target state for goal-conditioned RL (optional)
        goal_achieved: Whether goal was reached
        metadata: Additional custom data
    """

    # State information
    previous_state_ids: frozenset[str] = field(default_factory=frozenset)
    current_state_ids: frozenset[str] = field(default_factory=frozenset)
    previous_state_names: frozenset[str] = field(default_factory=frozenset)
    current_state_names: frozenset[str] = field(default_factory=frozenset)

    # Action information
    action_type: str = "workflow"
    action_params: dict[str, Any] = field(default_factory=dict)
    action_result: ActionResult = field(
        default_factory=lambda: ActionResult(success=True)
    )

    # Visual information (optional)
    screenshot_before: npt.NDArray[np.uint8] | None = None
    screenshot_after: npt.NDArray[np.uint8] | None = None
    pattern_matches: tuple[PatternMatch, ...] = ()

    # Episode tracking
    step_count: int = 0
    episode_time: float = 0.0

    # Goal-conditioned RL support
    goal_state: str | None = None
    goal_achieved: bool = False

    # Custom metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def state_changed(self) -> bool:
        """Check if state changed during this step."""
        return self.current_state_ids != self.previous_state_ids

    @property
    def transition_success(self) -> bool:
        """Check if the action executed successfully."""
        return self.action_result.success

    @property
    def workflow_name(self) -> str | None:
        """Get the workflow name if action was a workflow."""
        return self.action_result.workflow_name


@runtime_checkable
class RewardComponent(Protocol):
    """Protocol for atomic reward components.

    Reward components are composable building blocks that compute
    partial rewards based on specific aspects of the step.

    Implement this protocol to create custom reward components.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this reward component."""
        ...

    def compute(self, step_info: StepInfo) -> float:
        """Compute the reward contribution from this component.

        Args:
            step_info: Information about the current step

        Returns:
            Reward value (can be any float, will be composed with others)
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (called at episode start)."""
        ...


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for complete reward functions.

    A RewardFunction composes multiple components and produces
    the final reward value for a step.
    """

    def __call__(self, step_info: StepInfo) -> float:
        """Compute the total reward for this step."""
        ...

    def reset(self) -> None:
        """Reset all internal state (called at episode start)."""
        ...

    def get_components(self) -> list[RewardComponent]:
        """Get all component reward functions."""
        ...

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics for logging/debugging."""
        ...


class BaseRewardComponent(ABC):
    """Abstract base class for reward components with common functionality.

    Provides:
    - Name management
    - Weight scaling
    - Statistics tracking (call count, total reward, average)

    Subclasses should implement _compute_impl() instead of compute().

    Example:
        class MyReward(BaseRewardComponent):
            def _compute_impl(self, step_info: StepInfo) -> float:
                return 1.0 if step_info.state_changed else 0.0
    """

    def __init__(self, name: str | None = None, weight: float = 1.0):
        """Initialize the reward component.

        Args:
            name: Component name (defaults to class name)
            weight: Weight multiplier for the reward
        """
        self._name = name or self.__class__.__name__
        self.weight = weight
        self._call_count = 0
        self._total_reward = 0.0
        self._config: QontinuiConfig | None = None

    @property
    def name(self) -> str:
        """Get the component name."""
        return self._name

    def initialize(self, config: QontinuiConfig) -> None:
        """Initialize with configuration (optional override).

        Called once when the reward function is attached to an environment.

        Args:
            config: The environment's QontinuiConfig
        """
        self._config = config

    @abstractmethod
    def _compute_impl(self, step_info: StepInfo) -> float:
        """Implementation of reward computation (override this).

        Args:
            step_info: Information about the current step

        Returns:
            Raw reward value (before weight scaling)
        """
        pass

    def compute(self, step_info: StepInfo) -> float:
        """Compute weighted reward with statistics tracking.

        Args:
            step_info: Information about the current step

        Returns:
            Weighted reward value
        """
        raw_reward = self._compute_impl(step_info)
        weighted = raw_reward * self.weight
        self._call_count += 1
        self._total_reward += weighted
        return weighted

    def reset(self) -> None:
        """Reset internal state for new episode."""
        self._call_count = 0
        self._total_reward = 0.0

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about this component's rewards.

        Returns:
            Dict with name, call_count, total_reward, average_reward
        """
        return {
            "name": self.name,
            "weight": self.weight,
            "call_count": self._call_count,
            "total_reward": self._total_reward,
            "average_reward": (
                self._total_reward / self._call_count if self._call_count > 0 else 0.0
            ),
        }


class ConstantReward(BaseRewardComponent):
    """Simple constant reward (useful for testing)."""

    def __init__(self, value: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.value = value

    def _compute_impl(self, step_info: StepInfo) -> float:
        return self.value
