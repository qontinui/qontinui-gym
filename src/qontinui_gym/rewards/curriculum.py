"""Curriculum learning support for reward functions.

This module provides tools for curriculum learning where the reward
function adapts as the agent's skill improves:
- CurriculumSchedule: Defines progression through curriculum levels
- CurriculumReward: Adapts reward based on curriculum level
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qontinui_gym.rewards.base import BaseRewardComponent, RewardComponent, StepInfo


class CurriculumSchedule(ABC):
    """Abstract base class for curriculum schedules.

    A curriculum schedule determines when to advance to harder
    reward configurations based on training progress.
    """

    @abstractmethod
    def get_current_level(self, episode: int, total_steps: int) -> int:
        """Get current curriculum level.

        Args:
            episode: Current episode number
            total_steps: Total steps taken across all episodes

        Returns:
            Current curriculum level (0-indexed)
        """
        pass

    @abstractmethod
    def get_max_level(self) -> int:
        """Get maximum curriculum level."""
        pass

    def update(self, episode_return: float) -> None:
        """Update curriculum based on episode performance.

        Override for performance-based curricula.

        Args:
            episode_return: Total return from the episode
        """
        pass


class LinearCurriculum(CurriculumSchedule):
    """Linear progression through curriculum levels.

    Advances to next level after fixed number of episodes.

    Example:
        schedule = LinearCurriculum(num_levels=5, episodes_per_level=1000)
    """

    def __init__(
        self,
        num_levels: int,
        episodes_per_level: int,
    ):
        """Initialize linear curriculum.

        Args:
            num_levels: Total number of curriculum levels
            episodes_per_level: Episodes to train at each level
        """
        self.num_levels = num_levels
        self.episodes_per_level = episodes_per_level

    def get_current_level(self, episode: int, total_steps: int) -> int:
        return min(episode // self.episodes_per_level, self.num_levels - 1)

    def get_max_level(self) -> int:
        return self.num_levels - 1


class StepBasedCurriculum(CurriculumSchedule):
    """Curriculum that advances based on total training steps.

    Example:
        schedule = StepBasedCurriculum(
            num_levels=3,
            steps_per_level=100000,
        )
    """

    def __init__(
        self,
        num_levels: int,
        steps_per_level: int,
    ):
        """Initialize step-based curriculum.

        Args:
            num_levels: Total number of curriculum levels
            steps_per_level: Steps to train at each level
        """
        self.num_levels = num_levels
        self.steps_per_level = steps_per_level

    def get_current_level(self, episode: int, total_steps: int) -> int:
        return min(total_steps // self.steps_per_level, self.num_levels - 1)

    def get_max_level(self) -> int:
        return self.num_levels - 1


class PerformanceBasedCurriculum(CurriculumSchedule):
    """Curriculum that advances based on agent performance.

    Monitors episode returns and advances when success rate exceeds
    a threshold over a window of recent episodes.

    Example:
        schedule = PerformanceBasedCurriculum(
            num_levels=5,
            advancement_threshold=0.8,
            window_size=100,
        )
    """

    def __init__(
        self,
        num_levels: int,
        advancement_threshold: float = 0.8,
        success_threshold: float = 0.0,
        window_size: int = 100,
    ):
        """Initialize performance-based curriculum.

        Args:
            num_levels: Total number of curriculum levels
            advancement_threshold: Success rate to advance (0-1)
            success_threshold: Minimum return to count as success
            window_size: Episodes to consider for success rate
        """
        self.num_levels = num_levels
        self.advancement_threshold = advancement_threshold
        self.success_threshold = success_threshold
        self.window_size = window_size
        self._current_level = 0
        self._episode_returns: list[float] = []

    def update(self, episode_return: float) -> None:
        """Update with episode return, potentially advancing level."""
        self._episode_returns.append(episode_return)

        if len(self._episode_returns) >= self.window_size:
            recent = self._episode_returns[-self.window_size :]
            successes = sum(1 for r in recent if r > self.success_threshold)
            success_rate = successes / len(recent)

            if success_rate >= self.advancement_threshold:
                self._current_level = min(
                    self._current_level + 1, self.num_levels - 1
                )
                # Clear history when advancing
                self._episode_returns = []

    def get_current_level(self, episode: int, total_steps: int) -> int:
        return self._current_level

    def get_max_level(self) -> int:
        return self.num_levels - 1


class CurriculumReward(BaseRewardComponent):
    """Reward that adapts based on curriculum level.

    Allows defining different reward functions for different levels,
    enabling gradual difficulty progression.

    Example:
        # Easy: Dense rewards, then progressively sparser
        schedule = LinearCurriculum(num_levels=3, episodes_per_level=1000)
        level_rewards = {
            0: dense_reward_fn,   # Easy: lots of shaping
            1: medium_reward_fn,  # Medium: some shaping
            2: sparse_reward_fn,  # Hard: sparse rewards only
        }
        reward = CurriculumReward(schedule, level_rewards)
    """

    def __init__(
        self,
        schedule: CurriculumSchedule,
        level_rewards: dict[int, RewardComponent],
        default_reward: RewardComponent | None = None,
        **kwargs: Any,
    ):
        """Initialize curriculum reward.

        Args:
            schedule: Curriculum schedule determining current level
            level_rewards: Mapping of level -> reward component
            default_reward: Fallback reward if level not in mapping
        """
        super().__init__(**kwargs)
        self.schedule = schedule
        self.level_rewards = level_rewards
        self.default_reward = default_reward
        self._current_episode = 0
        self._total_steps = 0

    def set_episode(self, episode: int) -> None:
        """Set current episode number.

        Should be called at the start of each episode.

        Args:
            episode: Current episode number
        """
        self._current_episode = episode

    def update_episode_return(self, episode_return: float) -> None:
        """Update curriculum with episode return.

        Should be called at the end of each episode.

        Args:
            episode_return: Total return from the episode
        """
        self.schedule.update(episode_return)

    def _compute_impl(self, step_info: StepInfo) -> float:
        level = self.schedule.get_current_level(
            self._current_episode, self._total_steps
        )
        self._total_steps += 1

        reward_fn = self.level_rewards.get(level, self.default_reward)
        if reward_fn:
            return reward_fn.compute(step_info)
        return 0.0

    def reset(self) -> None:
        super().reset()
        # Reset all level rewards
        for reward_fn in self.level_rewards.values():
            reward_fn.reset()
        if self.default_reward:
            self.default_reward.reset()

    def get_current_level(self) -> int:
        """Get the current curriculum level."""
        return self.schedule.get_current_level(
            self._current_episode, self._total_steps
        )

    def get_statistics(self) -> dict[str, Any]:
        stats = super().get_statistics()
        stats["curriculum_level"] = self.get_current_level()
        stats["max_level"] = self.schedule.get_max_level()
        return stats


class MultiGoalCurriculum(BaseRewardComponent):
    """Curriculum with progressively harder goals.

    Starts with nearby goals and gradually increases difficulty
    by targeting more distant states.

    Example:
        # Goals ordered by difficulty
        goals = [
            ["easy_state"],           # Level 0
            ["medium_state"],         # Level 1
            ["hard_state"],           # Level 2
            ["final_goal_state"],     # Level 3
        ]
        reward = MultiGoalCurriculum(
            schedule=LinearCurriculum(4, 500),
            goal_levels=goals,
            goal_reward=10.0,
        )
    """

    def __init__(
        self,
        schedule: CurriculumSchedule,
        goal_levels: list[list[str]],
        goal_reward: float = 1.0,
        **kwargs: Any,
    ):
        """Initialize multi-goal curriculum.

        Args:
            schedule: Curriculum schedule
            goal_levels: List of goal state sets per level
            goal_reward: Reward for reaching current level's goal
        """
        super().__init__(**kwargs)
        self.schedule = schedule
        self.goal_levels = goal_levels
        self.goal_reward = goal_reward
        self._current_episode = 0
        self._total_steps = 0
        self._reached_goals: set[str] = set()

    def set_episode(self, episode: int) -> None:
        """Set current episode number."""
        self._current_episode = episode
        self._reached_goals.clear()

    def _compute_impl(self, step_info: StepInfo) -> float:
        level = self.schedule.get_current_level(
            self._current_episode, self._total_steps
        )
        self._total_steps += 1

        # Get current level's goals
        if level >= len(self.goal_levels):
            level = len(self.goal_levels) - 1
        current_goals = set(self.goal_levels[level])

        # Check if any goal reached
        current = step_info.current_state_names | step_info.current_state_ids
        newly_reached = (current & current_goals) - self._reached_goals
        self._reached_goals |= newly_reached

        return len(newly_reached) * self.goal_reward

    def reset(self) -> None:
        super().reset()
        self._reached_goals.clear()
