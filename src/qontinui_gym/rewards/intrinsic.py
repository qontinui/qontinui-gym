"""Intrinsic motivation rewards for exploration.

This module provides intrinsic reward components that encourage
exploration without relying on external rewards:
- Curiosity-based rewards
- Novelty-based rewards
- Random Network Distillation (RND)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from qontinui_gym.rewards.base import BaseRewardComponent, StepInfo


class CuriosityReward(BaseRewardComponent):
    """Curiosity-driven intrinsic reward (ICM-style).

    Rewards the agent for encountering states that are hard to predict.
    Uses forward dynamics prediction error as intrinsic reward.

    Note: This is a simplified implementation. Full ICM requires
    training forward and inverse dynamics models.

    Example:
        reward = CuriosityReward(beta=0.5)
    """

    def __init__(
        self,
        beta: float = 0.5,
        feature_dim: int = 64,
        history_size: int = 100,
        **kwargs: Any,
    ):
        """Initialize curiosity reward.

        Args:
            beta: Scale factor for curiosity bonus
            feature_dim: Dimension for simple feature representation
            history_size: Size of state history buffer
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.feature_dim = feature_dim
        self._state_history: deque[npt.NDArray[np.float32]] = deque(maxlen=history_size)

    def _extract_features(self, step_info: StepInfo) -> npt.NDArray[np.float32]:
        """Extract simple features from step info."""
        # Simple feature: hash of state IDs
        features = np.zeros(self.feature_dim, dtype=np.float32)

        for i, state_id in enumerate(sorted(step_info.current_state_ids)):
            idx = hash(state_id) % self.feature_dim
            features[idx] = 1.0

        return features

    def _compute_impl(self, step_info: StepInfo) -> float:
        features = self._extract_features(step_info)

        # Simple prediction error: distance to nearest historical state
        if not self._state_history:
            self._state_history.append(features)
            return self.beta  # Maximum curiosity for first state

        # Find minimum distance to any historical state
        min_dist = float("inf")
        for hist_features in self._state_history:
            dist = float(np.linalg.norm(features - hist_features))
            min_dist = min(min_dist, dist)

        self._state_history.append(features)

        # Normalize and scale
        curiosity = min(1.0, min_dist / np.sqrt(self.feature_dim))
        return float(self.beta * curiosity)

    def reset(self) -> None:
        super().reset()
        self._state_history.clear()


class NoveltyReward(BaseRewardComponent):
    """Novelty-based intrinsic reward.

    Rewards visiting novel state-action combinations not seen before.
    Uses a simple hash-based novelty detection.

    Example:
        reward = NoveltyReward(novelty_bonus=0.5, decay=0.99)
    """

    def __init__(
        self,
        novelty_bonus: float = 0.5,
        decay: float = 0.99,
        state_encoder: Callable[[StepInfo], tuple[Any, ...]] | None = None,
        **kwargs: Any,
    ):
        """Initialize novelty reward.

        Args:
            novelty_bonus: Maximum bonus for novel states
            decay: Decay factor for familiarity (1.0 = no decay)
            state_encoder: Custom function to encode state for hashing
        """
        super().__init__(**kwargs)
        self.novelty_bonus = novelty_bonus
        self.decay = decay
        self.state_encoder = state_encoder or self._default_encoder
        self._familiarity: dict[tuple[Any, ...], float] = {}

    def _default_encoder(self, step_info: StepInfo) -> tuple[Any, ...]:
        """Default state encoding using state IDs and action."""
        return (
            tuple(sorted(step_info.current_state_ids)),
            step_info.action_type,
        )

    def _compute_impl(self, step_info: StepInfo) -> float:
        state_key = self.state_encoder(step_info)

        # Apply decay to all familiarity values
        if self.decay < 1.0:
            for key in self._familiarity:
                self._familiarity[key] *= self.decay

        # Compute novelty
        familiarity = self._familiarity.get(state_key, 0.0)
        novelty = max(0.0, 1.0 - familiarity)

        # Update familiarity
        self._familiarity[state_key] = min(1.0, familiarity + 0.2)

        return self.novelty_bonus * novelty

    def reset(self) -> None:
        super().reset()
        self._familiarity.clear()


class RandomNetworkDistillationReward(BaseRewardComponent):
    """Random Network Distillation (RND) style intrinsic reward.

    Uses the prediction error of a randomly initialized fixed network
    as a measure of novelty. States that are hard to predict are novel.

    This is a simplified version that uses simple hash-based features
    instead of actual neural networks.

    Example:
        reward = RandomNetworkDistillationReward(scale=1.0)
    """

    def __init__(
        self,
        scale: float = 1.0,
        feature_dim: int = 128,
        num_targets: int = 8,
        learning_rate: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize RND reward.

        Args:
            scale: Scale factor for RND bonus
            feature_dim: Dimension of feature space
            num_targets: Number of random target functions
            learning_rate: How fast predictor learns (0-1)
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.feature_dim = feature_dim
        self.num_targets = num_targets
        self.learning_rate = learning_rate

        # Initialize random "networks" (just random projections)
        self._rng = np.random.RandomState(42)
        self._target_weights = self._rng.randn(num_targets, feature_dim)
        self._predictor_weights = np.zeros((num_targets, feature_dim))

    def _state_to_features(self, step_info: StepInfo) -> npt.NDArray[np.float32]:
        """Convert state to feature vector."""
        features = np.zeros(self.feature_dim, dtype=np.float32)

        # Encode state IDs
        for state_id in step_info.current_state_ids:
            for i, char in enumerate(state_id):
                idx = (ord(char) + i * 31) % self.feature_dim
                features[idx] += 1.0

        # Encode action
        action_hash = hash(step_info.action_type)
        features[action_hash % self.feature_dim] += 1.0

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm

        return features

    def _compute_impl(self, step_info: StepInfo) -> float:
        features = self._state_to_features(step_info)

        # Compute target outputs (fixed random projection)
        target_output = np.tanh(self._target_weights @ features)

        # Compute predictor outputs
        predictor_output = np.tanh(self._predictor_weights @ features)

        # Prediction error is the intrinsic reward
        error = float(np.mean((target_output - predictor_output) ** 2))

        # Update predictor (move toward target)
        gradient = predictor_output - target_output
        self._predictor_weights -= self.learning_rate * np.outer(gradient, features)

        return self.scale * error

    def reset(self) -> None:
        super().reset()
        # Don't reset predictor weights - RND learns across episodes


class StateEntropyReward(BaseRewardComponent):
    """Reward based on state visitation entropy.

    Encourages diverse state visitation by rewarding states that
    increase the entropy of the visitation distribution.

    Example:
        reward = StateEntropyReward(scale=0.1)
    """

    def __init__(
        self,
        scale: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize entropy reward.

        Args:
            scale: Scale factor for entropy bonus
        """
        super().__init__(**kwargs)
        self.scale = scale
        self._visit_counts: dict[frozenset[str], int] = {}
        self._total_visits: int = 0

    def _compute_impl(self, step_info: StepInfo) -> float:
        state_key = step_info.current_state_ids

        # Update counts
        self._visit_counts[state_key] = self._visit_counts.get(state_key, 0) + 1
        self._total_visits += 1

        # Compute entropy bonus (reward for increasing entropy)
        if self._total_visits < 2:
            return self.scale

        # Probability of current state before this visit
        count = self._visit_counts[state_key]
        p_before = (
            (count - 1) / (self._total_visits - 1) if self._total_visits > 1 else 0
        )
        p_after = count / self._total_visits

        # Information gain from this visit
        if p_before > 0:
            info_before = -p_before * np.log(p_before + 1e-10)
            info_after = -p_after * np.log(p_after + 1e-10)
            info_gain = float(info_after - info_before)
        else:
            # New state - high information gain
            info_gain = float(-np.log(1 / self._total_visits))

        return self.scale * max(0.0, info_gain)

    def reset(self) -> None:
        super().reset()
        self._visit_counts.clear()
        self._total_visits = 0
