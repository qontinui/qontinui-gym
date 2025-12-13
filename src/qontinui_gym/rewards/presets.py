"""Pre-configured reward functions for common use cases.

This module provides ready-to-use reward configurations for
typical RL scenarios with GUI automation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qontinui_gym.rewards.base import RewardComponent
from qontinui_gym.rewards.components import (
    GoalConditionedReward,
    StateReachReward,
    StepPenalty,
)
from qontinui_gym.rewards.composers import (
    ComposedReward,
    CompositionMode,
    RewardBuilder,
)
from qontinui_gym.rewards.curriculum import (
    CurriculumReward,
    LinearCurriculum,
    PerformanceBasedCurriculum,
)
from qontinui_gym.rewards.intrinsic import CuriosityReward, NoveltyReward

if TYPE_CHECKING:
    pass


def goal_conditioned_preset(
    goal_states: set[str] | list[str],
    state_distances: dict[str, int] | None = None,
    goal_reward: float = 10.0,
    step_penalty: float = -0.001,
    shaping_weight: float = 1.0,
) -> ComposedReward:
    """Standard goal-conditioned reward with optional progress shaping.

    Good starting point for navigation tasks where the agent needs
    to reach specific target states.

    Args:
        goal_states: Target states that give terminal reward
        state_distances: Optional distances for progress shaping
        goal_reward: Reward for reaching goal
        step_penalty: Per-step penalty for efficiency
        shaping_weight: Weight for progress shaping

    Returns:
        Configured ComposedReward
    """
    builder = (
        RewardBuilder()
        .goal_reaching(goal_states, reward=goal_reward, terminal=True)
        .step_penalty(step_penalty)
    )

    if state_distances:
        builder = builder.progress_shaping(state_distances, weight=shaping_weight)

    return builder.build()


def exploration_preset(
    beta: float = 1.0,
    novelty_bonus: float = 0.5,
    step_penalty: float = -0.001,
    include_curiosity: bool = True,
) -> ComposedReward:
    """Reward focused on exploration of state space.

    Good for discovering application capabilities without
    specific goals.

    Args:
        beta: Scale for visit count exploration bonus
        novelty_bonus: Scale for novelty-based exploration
        step_penalty: Per-step penalty
        include_curiosity: Include curiosity-based reward

    Returns:
        Configured ComposedReward
    """
    builder = (
        RewardBuilder()
        .exploration_bonus(beta=beta)
        .step_penalty(step_penalty)
        .custom(NoveltyReward(novelty_bonus=novelty_bonus))
    )

    if include_curiosity:
        builder = builder.custom(CuriosityReward(beta=0.3))

    return builder.build()


def task_completion_preset(
    task_states: dict[str, float],
    penalty_states: dict[str, float] | None = None,
    step_penalty: float = -0.001,
) -> ComposedReward:
    """Multi-task reward with different rewards per target state.

    Good for applications with multiple possible goals where
    different outcomes have different values.

    Args:
        task_states: Mapping of state_name -> reward
        penalty_states: Optional mapping of state_name -> penalty
        step_penalty: Per-step penalty

    Returns:
        Configured ComposedReward
    """
    components: list[RewardComponent] = [StepPenalty(penalty=step_penalty)]

    # Add rewards for each task state
    for state_name, reward in task_states.items():
        components.append(
            StateReachReward(
                target_states={state_name},
                reward_per_state=reward,
            )
        )

    # Add penalties for error states
    if penalty_states:
        for state_name, penalty in penalty_states.items():
            components.append(
                StateReachReward(
                    target_states={state_name},
                    reward_per_state=penalty,  # penalty should be negative
                )
            )

    return ComposedReward(components, mode=CompositionMode.SUM)


def sparse_preset(
    goal_states: set[str] | list[str],
    goal_reward: float = 1.0,
) -> ComposedReward:
    """Pure sparse reward - only reward for reaching goals.

    Useful for benchmarking and testing pure exploration
    capabilities without any reward shaping.

    Args:
        goal_states: Target states
        goal_reward: Reward for reaching any goal

    Returns:
        Configured ComposedReward
    """
    return ComposedReward(
        [GoalConditionedReward(goal_reward=goal_reward, fixed_goal=None)],
        mode=CompositionMode.SUM,
    )


def curriculum_preset(
    goal_levels: list[list[str]],
    rewards_per_level: list[float] | None = None,
    episodes_per_level: int = 1000,
    step_penalty: float = -0.001,
) -> ComposedReward:
    """Curriculum learning with progressively harder goals.

    Starts with easier goals and gradually increases difficulty
    as the agent improves.

    Args:
        goal_levels: List of goal state sets, ordered by difficulty
        rewards_per_level: Optional rewards per level (default: all 10.0)
        episodes_per_level: Episodes at each difficulty level
        step_penalty: Per-step penalty

    Returns:
        Configured ComposedReward with curriculum
    """
    num_levels = len(goal_levels)

    if rewards_per_level is None:
        rewards_per_level = [10.0] * num_levels

    schedule = LinearCurriculum(
        num_levels=num_levels,
        episodes_per_level=episodes_per_level,
    )

    # Create reward for each level
    level_rewards: dict[int, RewardComponent] = {}
    for level, (goals, reward) in enumerate(zip(goal_levels, rewards_per_level)):
        level_rewards[level] = StateReachReward(
            target_states=set(goals),
            reward_per_state=reward,
            terminal_on_reach=True,
        )

    curriculum = CurriculumReward(
        schedule=schedule,
        level_rewards=level_rewards,
    )

    return ComposedReward(
        [curriculum, StepPenalty(penalty=step_penalty)],
        mode=CompositionMode.SUM,
    )


def adaptive_curriculum_preset(
    goal_levels: list[list[str]],
    success_threshold: float = 0.0,
    advancement_rate: float = 0.8,
    window_size: int = 100,
    step_penalty: float = -0.001,
) -> ComposedReward:
    """Adaptive curriculum that advances based on performance.

    Only advances to harder goals when success rate exceeds threshold.

    Args:
        goal_levels: List of goal state sets, ordered by difficulty
        success_threshold: Minimum return to count as success
        advancement_rate: Required success rate to advance (0-1)
        window_size: Episodes to consider for success rate
        step_penalty: Per-step penalty

    Returns:
        Configured ComposedReward with adaptive curriculum
    """
    num_levels = len(goal_levels)

    schedule = PerformanceBasedCurriculum(
        num_levels=num_levels,
        advancement_threshold=advancement_rate,
        success_threshold=success_threshold,
        window_size=window_size,
    )

    level_rewards_adaptive: dict[int, RewardComponent] = {}
    for level, goals in enumerate(goal_levels):
        level_rewards_adaptive[level] = StateReachReward(
            target_states=set(goals),
            reward_per_state=10.0,
            terminal_on_reach=True,
        )

    curriculum = CurriculumReward(
        schedule=schedule,
        level_rewards=level_rewards_adaptive,
    )

    return ComposedReward(
        [curriculum, StepPenalty(penalty=step_penalty)],
        mode=CompositionMode.SUM,
    )


def balanced_preset(
    goal_states: set[str] | list[str],
    state_distances: dict[str, int] | None = None,
    goal_reward: float = 10.0,
    step_penalty: float = -0.001,
    exploration_weight: float = 0.1,
    transition_reward: float = 0.1,
) -> ComposedReward:
    """Balanced reward combining multiple signals.

    A general-purpose reward that combines goal-reaching,
    exploration, and transition rewards.

    Args:
        goal_states: Target states
        state_distances: Optional distances for shaping
        goal_reward: Reward for reaching goal
        step_penalty: Per-step penalty
        exploration_weight: Weight for exploration bonus
        transition_reward: Reward for successful transitions

    Returns:
        Configured ComposedReward
    """
    builder = (
        RewardBuilder()
        .goal_reaching(goal_states, reward=goal_reward, terminal=True)
        .step_penalty(step_penalty)
        .exploration_bonus(beta=exploration_weight)
        .transition_reward(success_reward=transition_reward, failure_penalty=-0.1)
    )

    if state_distances:
        builder = builder.progress_shaping(state_distances, weight=0.5)

    return builder.build()
