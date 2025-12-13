"""Reward functions and utilities for Qontinui environments.

This module provides a composable reward system for RL research:
- Base classes and protocols for custom rewards
- Built-in reward components (state-based, action-based, visual)
- Composition utilities for combining rewards
- Intrinsic motivation rewards (curiosity, novelty)
- Curriculum learning support
- Logging and analysis tools
"""

from qontinui_gym.rewards.base import (
    ActionResult,
    BaseRewardComponent,
    PatternMatch,
    RewardComponent,
    RewardFunction,
    StepInfo,
)
from qontinui_gym.rewards.components import (
    ActionDurationReward,
    InvalidActionPenalty,
    PatternMatchReward,
    ScreenChangeReward,
    StateReachReward,
    StateTransitionReward,
    StateVisitCountReward,
    StepPenalty,
)
from qontinui_gym.rewards.composers import (
    ComposedReward,
    CompositionMode,
    RewardBuilder,
)
from qontinui_gym.rewards.wrappers import QontinuiRewardWrapper

__all__ = [
    # Base
    "StepInfo",
    "PatternMatch",
    "ActionResult",
    "RewardComponent",
    "RewardFunction",
    "BaseRewardComponent",
    # Components
    "StateReachReward",
    "StateTransitionReward",
    "StateVisitCountReward",
    "StepPenalty",
    "ActionDurationReward",
    "InvalidActionPenalty",
    "PatternMatchReward",
    "ScreenChangeReward",
    # Composers
    "CompositionMode",
    "ComposedReward",
    "RewardBuilder",
    # Wrappers
    "QontinuiRewardWrapper",
]
