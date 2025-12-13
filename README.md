# qontinui-gym

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0+-green.svg)](https://gymnasium.farama.org/)

Gymnasium environments for reinforcement learning with visual GUI automation.

## Overview

qontinui-gym provides a standard [Gymnasium](https://gymnasium.farama.org/) interface for training RL agents on visual GUI automation tasks. It connects to [qontinui-runner](https://github.com/qontinui/qontinui-runner) to execute workflows and navigate application states through visual pattern matching.

**Key capabilities:**

- **Hybrid action spaces** — Discrete workflow selection combined with parameterized actions (click coordinates, text input, scroll)
- **Multi-modal observations** — Screenshots + state machine information for rich environment representation
- **Composable reward system** — Build custom rewards from reusable components with built-in shaping, exploration bonuses, and curriculum learning
- **Framework agnostic** — Works with Stable-Baselines3, RLlib, CleanRL, or any Gymnasium-compatible library

## Installation

```bash
pip install qontinui-gym
```

With optional dependencies:

```bash
# For Stable-Baselines3 integration
pip install qontinui-gym[sb3]

# For computer vision features
pip install qontinui-gym[vision]

# All optional dependencies
pip install qontinui-gym[all]
```

## Quick Start

```python
from qontinui_gym import QontinuiEnv

# Create environment
env = QontinuiEnv(
    config_path="automation.json",  # QontinuiConfig exported from qontinui-web
    runner_host="localhost",
    runner_port=9876,
    max_episode_steps=100,
)

# Standard Gymnasium loop
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## Custom Rewards

qontinui-gym provides a powerful, composable reward system:

```python
from qontinui_gym import QontinuiEnv
from qontinui_gym.rewards import RewardBuilder, QontinuiRewardWrapper

# Create environment
base_env = QontinuiEnv(config_path="automation.json")

# Build custom reward function
reward_fn = (
    RewardBuilder()
    .goal_reaching(["checkout_complete"], reward=100.0, terminal=True)
    .step_penalty(-0.001)
    .exploration_bonus(beta=0.5)
    .build()
)

# Wrap environment with reward function
env = QontinuiRewardWrapper(base_env, reward_function=reward_fn)
```

### Reward Components

**State-based:**
- `StateReachReward` — Reward for reaching target states
- `StateTransitionReward` — Reward for successful transitions
- `StateVisitCountReward` — Exploration bonus based on visit counts
- `StateProgressReward` — Potential-based shaping toward goals

**Action-based:**
- `StepPenalty` — Per-step penalty for efficiency
- `ActionDurationReward` — Reward based on execution time
- `InvalidActionPenalty` — Penalty for failed actions

**Intrinsic motivation:**
- `CuriosityReward` — Prediction error as reward
- `NoveltyReward` — Bonus for novel states
- `StateEntropyReward` — Encourage diverse state visitation

### Reward Presets

```python
from qontinui_gym.rewards.presets import (
    goal_conditioned_preset,
    exploration_preset,
    curriculum_preset,
)

# Goal-conditioned with progress shaping
reward = goal_conditioned_preset(
    goal_states={"checkout_complete"},
    state_distances=compute_state_distances(config, ["checkout_complete"]),
)

# Exploration-focused
reward = exploration_preset(beta=1.0, novelty_bonus=0.5)

# Curriculum learning with progressive difficulty
reward = curriculum_preset(
    goal_levels=[["easy_goal"], ["medium_goal"], ["hard_goal"]],
    episodes_per_level=1000,
)
```

## Spaces

### Action Space

**Discrete mode** — Simple workflow selection:

```python
from qontinui_gym.spaces import ActionSpaceConfig

env = QontinuiEnv(
    config_path="automation.json",
    action_space_config=ActionSpaceConfig.discrete(),
)
# Action space: Discrete(num_workflows)
```

**Hybrid mode** (default) — Workflows + parameterized actions:

```python
env = QontinuiEnv(
    config_path="automation.json",
    action_space_config=ActionSpaceConfig.hybrid(),
)
# Action space: Dict({
#     "action_type": Discrete(N),
#     "workflow_idx": Discrete(W),
#     "coordinates": Box([0,0], [W,H]),
#     "text": Text(256),
#     "state_idx": Discrete(S),
# })
```

### Observation Space

```python
from qontinui_gym.spaces import ObservationSpaceConfig

# Multi-modal observations (default)
config = ObservationSpaceConfig()

# Custom screenshot size
config = ObservationSpaceConfig(screenshot_width=320, screenshot_height=240)
```

## Wrappers

```python
from qontinui_gym.wrappers import FrameStackWrapper, ActionMaskWrapper

# Stack frames for temporal information
env = FrameStackWrapper(env, num_frames=4)

# Action masking for invalid actions (SB3 MaskablePPO compatible)
env = ActionMaskWrapper(env)
```

## Stable-Baselines3 Integration

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from qontinui_gym import QontinuiEnv
from qontinui_gym.rewards import RewardBuilder, QontinuiRewardWrapper

def make_env():
    env = QontinuiEnv(config_path="automation.json")
    reward = RewardBuilder().goal_reaching(["goal"]).step_penalty(-0.001).build()
    return QontinuiRewardWrapper(env, reward)

env = DummyVecEnv([make_env])
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Architecture

```
RL Algorithm (PPO, DQN, etc.)
         ↓ Gymnasium API
    qontinui-gym
         ↓ HTTP
    qontinui-runner (port 9876)
         ↓
    GUI Automation (qontinui library)
```

## Requirements

- Python 3.10+
- [qontinui-runner](https://github.com/qontinui/qontinui-runner) running and accessible
- A QontinuiConfig JSON file (exported from qontinui-web)

## License

MIT
