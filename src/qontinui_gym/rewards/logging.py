"""Reward logging and analysis utilities.

This module provides tools for logging reward information during
training for later analysis and debugging.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RewardLogEntry:
    """Log entry for a single step."""

    episode: int
    step: int
    total_reward: float
    component_rewards: dict[str, float]
    state_ids: list[str]
    action_type: str
    action_success: bool
    timestamp: float = 0.0


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""

    episode: int
    total_return: float
    num_steps: int
    mean_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    success: bool = False
    final_states: list[str] = field(default_factory=list)


class RewardLogger:
    """Logger for reward information during training.

    Tracks step-by-step rewards and computes episode statistics
    for analysis and debugging.

    Example:
        logger = RewardLogger(log_path="rewards.json")

        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)

                logger.log_step(
                    reward=reward,
                    components=info.get("reward_components", {}),
                    state_ids=info.get("current_state_ids", []),
                    action_type=info.get("action_type", ""),
                    action_success=info.get("action_success", True),
                )

                done = terminated or truncated

            logger.end_episode()

        logger.save()
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        log_frequency: int = 1,
        max_entries: int = 100000,
    ):
        """Initialize reward logger.

        Args:
            log_path: Path to save log file (JSON)
            log_frequency: Log every N steps (1 = all steps)
            max_entries: Maximum log entries to keep in memory
        """
        self.log_path = Path(log_path) if log_path else None
        self.log_frequency = log_frequency
        self.max_entries = max_entries

        self._logs: list[RewardLogEntry] = []
        self._episode_stats: list[EpisodeStats] = []
        self._current_episode = 0
        self._current_step = 0
        self._episode_rewards: list[float] = []
        self._episode_states: list[str] = []
        self._component_totals: dict[str, float] = defaultdict(float)

    def log_step(
        self,
        reward: float,
        components: dict[str, Any] | None = None,
        state_ids: list[str] | None = None,
        action_type: str = "",
        action_success: bool = True,
        timestamp: float = 0.0,
    ) -> None:
        """Log a single step.

        Args:
            reward: Total reward for this step
            components: Component statistics dict
            state_ids: Current state IDs
            action_type: Type of action taken
            action_success: Whether action succeeded
            timestamp: Time in episode
        """
        self._episode_rewards.append(reward)

        if state_ids:
            self._episode_states = state_ids

        # Extract component rewards
        component_rewards: dict[str, float] = {}
        if components and "components" in components:
            for comp in components["components"]:
                name = comp.get("name", "unknown")
                avg = comp.get("average_reward", 0.0)
                component_rewards[name] = avg
                self._component_totals[name] += avg

        # Log entry (respecting frequency)
        if self._current_step % self.log_frequency == 0:
            if len(self._logs) < self.max_entries:
                entry = RewardLogEntry(
                    episode=self._current_episode,
                    step=self._current_step,
                    total_reward=reward,
                    component_rewards=component_rewards,
                    state_ids=state_ids or [],
                    action_type=action_type,
                    action_success=action_success,
                    timestamp=timestamp,
                )
                self._logs.append(entry)

        self._current_step += 1

    def end_episode(self, success: bool = False) -> EpisodeStats:
        """End current episode and compute statistics.

        Args:
            success: Whether episode was successful

        Returns:
            Statistics for the completed episode
        """
        if not self._episode_rewards:
            stats = EpisodeStats(
                episode=self._current_episode,
                total_return=0.0,
                num_steps=0,
                mean_reward=0.0,
                std_reward=0.0,
                max_reward=0.0,
                min_reward=0.0,
                success=success,
                final_states=self._episode_states,
            )
        else:
            rewards = np.array(self._episode_rewards)
            stats = EpisodeStats(
                episode=self._current_episode,
                total_return=float(np.sum(rewards)),
                num_steps=len(rewards),
                mean_reward=float(np.mean(rewards)),
                std_reward=float(np.std(rewards)),
                max_reward=float(np.max(rewards)),
                min_reward=float(np.min(rewards)),
                success=success,
                final_states=self._episode_states,
            )

        self._episode_stats.append(stats)

        # Reset for next episode
        self._current_episode += 1
        self._current_step = 0
        self._episode_rewards = []
        self._episode_states = []

        return stats

    def save(self, path: str | Path | None = None) -> None:
        """Save logs to JSON file.

        Args:
            path: Override path (uses log_path if None)
        """
        save_path = Path(path) if path else self.log_path
        if not save_path:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "logs": [asdict(entry) for entry in self._logs],
            "episode_stats": [asdict(stats) for stats in self._episode_stats],
            "component_totals": dict(self._component_totals),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def get_summary(self) -> dict[str, Any]:
        """Get overall training summary.

        Returns:
            Dict with aggregated statistics
        """
        if not self._episode_stats:
            return {"num_episodes": 0}

        returns = [s.total_return for s in self._episode_stats]
        successes = [s.success for s in self._episode_stats]

        return {
            "num_episodes": len(self._episode_stats),
            "total_steps": sum(s.num_steps for s in self._episode_stats),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "max_return": float(np.max(returns)),
            "min_return": float(np.min(returns)),
            "success_rate": sum(successes) / len(successes),
            "return_trend": float(np.polyfit(range(len(returns)), returns, 1)[0])
            if len(returns) > 1
            else 0.0,
            "component_totals": dict(self._component_totals),
        }

    def get_recent_stats(self, n: int = 100) -> dict[str, Any]:
        """Get statistics for recent episodes.

        Args:
            n: Number of recent episodes to consider

        Returns:
            Dict with recent statistics
        """
        if not self._episode_stats:
            return {"num_episodes": 0}

        recent = self._episode_stats[-n:]
        returns = [s.total_return for s in recent]
        successes = [s.success for s in recent]

        return {
            "num_episodes": len(recent),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "success_rate": sum(successes) / len(successes),
        }

    def clear(self) -> None:
        """Clear all logged data."""
        self._logs.clear()
        self._episode_stats.clear()
        self._current_episode = 0
        self._current_step = 0
        self._episode_rewards.clear()
        self._episode_states.clear()
        self._component_totals.clear()
