# qontinui-gym

Gymnasium environments for visual GUI automation with [Qontinui](https://github.com/qontinui).

## Overview

qontinui-gym provides a standard [Gymnasium](https://gymnasium.farama.org/) interface for reinforcement learning research on GUI automation tasks. It connects to the qontinui-runner application to execute visual automation workflows.

## Features

- **Standard Gymnasium API**: Compatible with any RL framework (Stable-Baselines3, RLlib, CleanRL, etc.)
- **Hybrid Action Space**: Discrete workflow selection + parameterized actions (coordinates, text, scroll)
- **Multi-modal Observations**: Screenshots + state machine information
- **Composable Reward System**: Build custom rewards from reusable components
- **Curriculum Learning**: Progressive difficulty through curriculum schedules
- **Intrinsic Motivation**: Curiosity and novelty-based exploration rewards

## Installation

```bash
pip install qontinui-gym
```

With optional dependencies:
```bash
# For computer vision features
pip install qontinui-gym[vision]

# For Stable-Baselines3 integration
pip install qontinui-gym[sb3]

# All optional dependencies
pip install qontinui-gym[all]
```

## Quick Start

```python
from qontinui_gym import QontinuiEnv

# Create environment
env = QontinuiEnv(
    config_path="automation.json",
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

# Build custom reward
reward_fn = (RewardBuilder()
    .goal_reaching(["checkout_complete"], reward=100.0, terminal=True)
    .step_penalty(-0.001)
    .exploration_bonus(beta=0.5)
    .build())

# Wrap environment
env = QontinuiRewardWrapper(base_env, reward_function=reward_fn)
```

### Reward Components

**State-based:**
- `StateReachReward` - Reward for reaching target states
- `StateTransitionReward` - Reward for successful transitions
- `StateVisitCountReward` - Exploration bonus based on visit counts
- `StateProgressReward` - Potential-based shaping toward goals

**Action-based:**
- `StepPenalty` - Per-step penalty for efficiency
- `ActionDurationReward` - Reward based on execution time
- `InvalidActionPenalty` - Penalty for failed actions

**Intrinsic motivation:**
- `CuriosityReward` - Prediction error as reward
- `NoveltyReward` - Bonus for novel states
- `StateEntropyReward` - Encourage diverse state visitation

### Presets

```python
from qontinui_gym.rewards.presets import (
    goal_conditioned_preset,
    exploration_preset,
    curriculum_preset,
)

# Goal-conditioned with shaping
reward = goal_conditioned_preset(
    goal_states={"checkout_complete"},
    state_distances=compute_state_distances(config, ["checkout_complete"]),
)

# Exploration-focused
reward = exploration_preset(beta=1.0, novelty_bonus=0.5)

# Curriculum learning
reward = curriculum_preset(
    goal_levels=[["easy"], ["medium"], ["hard"]],
    episodes_per_level=1000,
)
```

## Spaces

### Action Space

**Discrete mode:**
```python
from qontinui_gym.spaces import ActionSpaceConfig

config = ActionSpaceConfig.discrete()
# Action space: Discrete(num_workflows)
```

**Hybrid mode (default):**
```python
config = ActionSpaceConfig.hybrid()
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

# Multi-modal (default)
config = ObservationSpaceConfig()

# Visual only
config = ObservationSpaceConfig.visual_only()

# State only (no screenshot)
config = ObservationSpaceConfig.state_only()
```

## Wrappers

```python
from qontinui_gym.wrappers import FrameStackWrapper, ActionMaskWrapper

# Stack frames for temporal information
env = FrameStackWrapper(env, num_frames=4)

# Action masking for invalid actions
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
    reward = RewardBuilder().goal_reaching(["goal"]).build()
    return QontinuiRewardWrapper(env, reward)

env = DummyVecEnv([make_env])
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Architecture

```
RL Algorithm
    ↓ (Gymnasium API)
qontinui-gym
    ↓ (HTTP)
qontinui-runner (Tauri app, port 9876)
    ↓
GUI Automation (qontinui library)
```

## Requirements

- Python 3.10+
- qontinui-runner running and accessible
- A valid QontinuiConfig JSON file (exported from qontinui-web)

## License

MIT
