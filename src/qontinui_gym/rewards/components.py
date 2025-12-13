"""Built-in reward components for Qontinui environments.

This module provides a library of commonly useful reward components:
- State-based rewards (reaching states, transitions, progress)
- Action-based rewards (step penalty, duration, success/failure)
- Visual rewards (pattern matching, screen changes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qontinui_gym.rewards.base import BaseRewardComponent, StepInfo

if TYPE_CHECKING:
    pass


# =============================================================================
# State-Based Rewards
# =============================================================================


class StateReachReward(BaseRewardComponent):
    """Reward for reaching specific target states.

    Common for goal-conditioned RL and sparse reward settings.

    Example:
        reward = StateReachReward(
            target_states={"checkout_complete", "order_confirmed"},
            reward_per_state=10.0,
            terminal_on_reach=True,
        )
    """

    def __init__(
        self,
        target_states: set[str],
        reward_per_state: float = 1.0,
        terminal_on_reach: bool = False,
        once_per_episode: bool = True,
        **kwargs: Any,
    ):
        """Initialize state reach reward.

        Args:
            target_states: Set of target state names or IDs
            reward_per_state: Reward for reaching each target state
            terminal_on_reach: Mark episode as terminal when reached
            once_per_episode: Only reward first time reaching each state
        """
        super().__init__(**kwargs)
        self.target_states = target_states
        self.reward_per_state = reward_per_state
        self.terminal_on_reach = terminal_on_reach
        self.once_per_episode = once_per_episode
        self._reached_states: set[str] = set()

    def _compute_impl(self, step_info: StepInfo) -> float:
        # Check both state IDs and names
        current = step_info.current_state_ids | step_info.current_state_names
        newly_reached = current & self.target_states

        if self.once_per_episode:
            newly_reached = newly_reached - self._reached_states
            self._reached_states |= newly_reached

        return len(newly_reached) * self.reward_per_state

    def reset(self) -> None:
        super().reset()
        self._reached_states.clear()


class StateTransitionReward(BaseRewardComponent):
    """Reward for successful state transitions.

    Encourages the agent to learn valid transition sequences.

    Example:
        reward = StateTransitionReward(
            reward_on_transition=0.1,
            penalty_on_failure=-0.1,
            bonus_transitions={("login", "dashboard"): 1.0},
        )
    """

    def __init__(
        self,
        reward_on_transition: float = 0.1,
        penalty_on_failure: float = -0.1,
        bonus_transitions: dict[tuple[str, str], float] | None = None,
        **kwargs: Any,
    ):
        """Initialize state transition reward.

        Args:
            reward_on_transition: Reward for any successful transition
            penalty_on_failure: Penalty for failed action
            bonus_transitions: Extra rewards for specific (from, to) pairs
        """
        super().__init__(**kwargs)
        self.reward_on_transition = reward_on_transition
        self.penalty_on_failure = penalty_on_failure
        self.bonus_transitions = bonus_transitions or {}

    def _compute_impl(self, step_info: StepInfo) -> float:
        if not step_info.transition_success:
            return self.penalty_on_failure

        reward = 0.0

        # Base reward for state change
        if step_info.state_changed:
            reward += self.reward_on_transition

            # Check for bonus transitions
            for prev in step_info.previous_state_names:
                for curr in step_info.current_state_names:
                    key = (prev, curr)
                    if key in self.bonus_transitions:
                        reward += self.bonus_transitions[key]

        return reward


class StateVisitCountReward(BaseRewardComponent):
    """Exploration bonus based on state visit counts.

    Implements count-based exploration: r = beta / sqrt(N(s))

    Example:
        reward = StateVisitCountReward(beta=1.0)
    """

    def __init__(
        self,
        beta: float = 1.0,
        decay: float = 1.0,
        **kwargs: Any,
    ):
        """Initialize visit count reward.

        Args:
            beta: Scale factor for exploration bonus
            decay: Decay factor for counts (1.0 = no decay)
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.decay = decay
        self._visit_counts: dict[str, float] = {}

    def _compute_impl(self, step_info: StepInfo) -> float:
        # Apply decay to all counts
        if self.decay < 1.0:
            for key in self._visit_counts:
                self._visit_counts[key] *= self.decay

        # Get current states
        current = step_info.current_state_names | step_info.current_state_ids
        if not current:
            return 0.0

        total_bonus = 0.0
        for state in current:
            count = self._visit_counts.get(state, 0.0) + 1.0
            self._visit_counts[state] = count
            total_bonus += self.beta / np.sqrt(count)

        return total_bonus / len(current)  # Average over current states

    def reset(self) -> None:
        super().reset()
        self._visit_counts.clear()


class StateProgressReward(BaseRewardComponent):
    """Reward for making progress toward a goal state.

    Uses potential-based shaping: R = gamma * phi(s') - phi(s)
    where phi(s) = -distance_to_goal

    Requires state distances to be precomputed.

    Example:
        distances = compute_state_distances(config, ["goal"])
        reward = StateProgressReward(
            state_distances=distances,
            gamma=0.99,
        )
    """

    def __init__(
        self,
        state_distances: dict[str, int],
        gamma: float = 0.99,
        max_distance: int = 100,
        **kwargs: Any,
    ):
        """Initialize progress reward.

        Args:
            state_distances: Mapping of state_id -> distance to goal
            gamma: Discount factor for potential shaping
            max_distance: Distance to use for unknown/unreachable states
        """
        super().__init__(**kwargs)
        self.state_distances = state_distances
        self.gamma = gamma
        self.max_distance = max_distance
        self._prev_potential: float | None = None

    def _get_potential(self, state_ids: frozenset[str]) -> float:
        """Compute potential function phi(s) = -distance_to_goal."""
        if not state_ids:
            return -self.max_distance

        min_dist = self.max_distance
        for state_id in state_ids:
            dist = self.state_distances.get(state_id, -1)
            if dist >= 0:
                min_dist = min(min_dist, dist)

        return -min_dist  # Negative so closer = higher potential

    def _compute_impl(self, step_info: StepInfo) -> float:
        curr_potential = self._get_potential(step_info.current_state_ids)

        if self._prev_potential is None:
            self._prev_potential = self._get_potential(step_info.previous_state_ids)

        # Potential-based shaping: R = gamma * phi(s') - phi(s)
        reward = self.gamma * curr_potential - self._prev_potential
        self._prev_potential = curr_potential

        return reward

    def reset(self) -> None:
        super().reset()
        self._prev_potential = None


# =============================================================================
# Action-Based Rewards
# =============================================================================


class StepPenalty(BaseRewardComponent):
    """Small penalty per step to encourage efficiency.

    Standard technique to prevent infinite episode length.

    Example:
        reward = StepPenalty(penalty=-0.001)
    """

    def __init__(self, penalty: float = -0.001, **kwargs: Any):
        super().__init__(**kwargs)
        self.penalty = penalty

    def _compute_impl(self, step_info: StepInfo) -> float:
        return self.penalty


class ActionDurationReward(BaseRewardComponent):
    """Reward based on action execution duration.

    Can encourage faster executions or penalize slow ones.

    Example:
        reward = ActionDurationReward(
            target_duration=1.0,
            penalty_per_second=-0.1,
        )
    """

    def __init__(
        self,
        target_duration: float = 1.0,
        penalty_per_second: float = -0.1,
        **kwargs: Any,
    ):
        """Initialize duration reward.

        Args:
            target_duration: Target duration in seconds
            penalty_per_second: Penalty per second over target
        """
        super().__init__(**kwargs)
        self.target_duration = target_duration
        self.penalty_per_second = penalty_per_second

    def _compute_impl(self, step_info: StepInfo) -> float:
        duration_seconds = step_info.action_result.duration_ms / 1000.0
        excess = max(0.0, duration_seconds - self.target_duration)
        return excess * self.penalty_per_second


class InvalidActionPenalty(BaseRewardComponent):
    """Penalty for failed or invalid actions.

    Example:
        reward = InvalidActionPenalty(penalty=-0.5)
    """

    def __init__(self, penalty: float = -0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.penalty = penalty

    def _compute_impl(self, step_info: StepInfo) -> float:
        if not step_info.transition_success:
            return self.penalty
        return 0.0


class ActionSuccessReward(BaseRewardComponent):
    """Reward for successful action execution.

    Example:
        reward = ActionSuccessReward(reward=0.1, only_on_state_change=True)
    """

    def __init__(
        self,
        reward: float = 0.1,
        only_on_state_change: bool = False,
        **kwargs: Any,
    ):
        """Initialize success reward.

        Args:
            reward: Reward for successful action
            only_on_state_change: Only reward if state actually changed
        """
        super().__init__(**kwargs)
        self.reward = reward
        self.only_on_state_change = only_on_state_change

    def _compute_impl(self, step_info: StepInfo) -> float:
        if not step_info.transition_success:
            return 0.0

        if self.only_on_state_change and not step_info.state_changed:
            return 0.0

        return self.reward


# =============================================================================
# Visual Rewards
# =============================================================================


class PatternMatchReward(BaseRewardComponent):
    """Reward based on pattern matching quality.

    Useful for learning to interact with specific UI elements.

    Example:
        reward = PatternMatchReward(
            target_patterns={"login_button", "submit_button"},
            similarity_threshold=0.8,
            reward_per_match=0.1,
        )
    """

    def __init__(
        self,
        target_patterns: set[str] | None = None,
        similarity_threshold: float = 0.8,
        reward_per_match: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize pattern match reward.

        Args:
            target_patterns: Specific patterns to reward (None = all)
            similarity_threshold: Minimum similarity for reward
            reward_per_match: Reward per high-quality match
        """
        super().__init__(**kwargs)
        self.target_patterns = target_patterns
        self.similarity_threshold = similarity_threshold
        self.reward_per_match = reward_per_match

    def _compute_impl(self, step_info: StepInfo) -> float:
        if not step_info.pattern_matches:
            return 0.0

        matches = step_info.pattern_matches
        if self.target_patterns:
            matches = tuple(
                m for m in matches if m.pattern_name in self.target_patterns
            )

        high_quality = [m for m in matches if m.similarity >= self.similarity_threshold]
        return len(high_quality) * self.reward_per_match


class ScreenChangeReward(BaseRewardComponent):
    """Reward for causing meaningful screen changes.

    Uses image difference to detect meaningful interactions.

    Example:
        reward = ScreenChangeReward(
            change_threshold=0.05,
            reward_on_change=0.05,
        )
    """

    def __init__(
        self,
        change_threshold: float = 0.05,
        reward_on_change: float = 0.05,
        **kwargs: Any,
    ):
        """Initialize screen change reward.

        Args:
            change_threshold: Minimum change ratio to trigger reward
            reward_on_change: Reward when change detected
        """
        super().__init__(**kwargs)
        self.change_threshold = change_threshold
        self.reward_on_change = reward_on_change

    def _compute_impl(self, step_info: StepInfo) -> float:
        if step_info.screenshot_before is None or step_info.screenshot_after is None:
            return 0.0

        # Compute normalized difference
        diff = np.abs(
            step_info.screenshot_before.astype(np.float32)
            - step_info.screenshot_after.astype(np.float32)
        )
        change_ratio = np.mean(diff) / 255.0

        if change_ratio >= self.change_threshold:
            return self.reward_on_change
        return 0.0


# =============================================================================
# Goal-Conditioned Rewards
# =============================================================================


class GoalConditionedReward(BaseRewardComponent):
    """Sparse reward for goal-conditioned RL.

    Returns reward only when the goal state is reached.

    Example:
        reward = GoalConditionedReward(goal_reward=10.0)
    """

    def __init__(
        self,
        goal_reward: float = 1.0,
        use_step_info_goal: bool = True,
        fixed_goal: str | None = None,
        **kwargs: Any,
    ):
        """Initialize goal-conditioned reward.

        Args:
            goal_reward: Reward for reaching the goal
            use_step_info_goal: Use goal from step_info (for varying goals)
            fixed_goal: Fixed goal state (overrides step_info)
        """
        super().__init__(**kwargs)
        self.goal_reward = goal_reward
        self.use_step_info_goal = use_step_info_goal
        self.fixed_goal = fixed_goal

    def _compute_impl(self, step_info: StepInfo) -> float:
        # Determine goal
        if self.fixed_goal:
            goal = self.fixed_goal
        elif self.use_step_info_goal and step_info.goal_state:
            goal = step_info.goal_state
        else:
            return 0.0

        # Check if goal reached
        current = step_info.current_state_names | step_info.current_state_ids
        if goal in current:
            return self.goal_reward

        return 0.0
