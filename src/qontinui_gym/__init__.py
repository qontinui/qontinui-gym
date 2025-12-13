"""Gymnasium environments for visual GUI automation with Qontinui.

This package provides Gymnasium-compatible environments that interface with
the Qontinui visual automation platform, enabling reinforcement learning
research on GUI automation tasks.

Example usage:
    from qontinui_gym import QontinuiEnv
    from qontinui_gym.rewards import RewardBuilder

    env = QontinuiEnv(config_path="automation.json")
    reward_fn = (RewardBuilder()
        .goal_reaching(["checkout_complete"], reward=100.0)
        .step_penalty(-0.001)
        .build())

    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
"""

__version__ = "0.1.0"

from qontinui_gym.env import QontinuiEnv
from qontinui_gym.spaces import (
    ActionSpaceConfig,
    ObservationSpaceConfig,
    build_action_space,
    build_observation_space,
)

__all__ = [
    "QontinuiEnv",
    "ActionSpaceConfig",
    "ObservationSpaceConfig",
    "build_action_space",
    "build_observation_space",
]
