"""Stable-Baselines3 integration example for qontinui-gym.

This example demonstrates how to train RL agents using
Stable-Baselines3 with qontinui-gym environments.

Requirements:
    pip install stable-baselines3
    pip install qontinui-gym[sb3]
"""

from typing import Any

import numpy as np


def create_env(config_path: str, goal_states: list[str] | None = None) -> Any:
    """Create a wrapped qontinui-gym environment for training.

    Args:
        config_path: Path to QontinuiConfig JSON file
        goal_states: Optional goal states for reward

    Returns:
        Wrapped Gymnasium environment
    """
    from qontinui_gym import QontinuiEnv
    from qontinui_gym.config.loader import compute_state_distances
    from qontinui_gym.rewards import QontinuiRewardWrapper, RewardBuilder
    from qontinui_gym.wrappers import ActionMaskWrapper, FrameStackWrapper

    # Create base environment
    env = QontinuiEnv(
        config_path=config_path,
        max_episode_steps=100,
        capture_screenshots=True,
    )

    # Build reward function
    if goal_states:
        config = env.get_config()
        state_distances = compute_state_distances(config, goal_states)

        reward_fn = (
            RewardBuilder()
            .goal_reaching(goal_states, reward=10.0, terminal=True)
            .progress_shaping(state_distances, gamma=0.99)
            .step_penalty(-0.001)
            .exploration_bonus(beta=0.1)
            .build()
        )

        env = QontinuiRewardWrapper(env, reward_function=reward_fn)

    # Add wrappers
    env = FrameStackWrapper(env, num_frames=4)
    env = ActionMaskWrapper(env)

    return env


def train_ppo(
    config_path: str, goal_states: list[str], total_timesteps: int = 10000
) -> Any:
    """Train a PPO agent on qontinui-gym.

    Args:
        config_path: Path to QontinuiConfig
        goal_states: Goal states for reward
        total_timesteps: Training steps
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
        )
    except ImportError:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3")
        return

    print("Creating environment...")
    env = DummyVecEnv([lambda: create_env(config_path, goal_states)])

    print("Creating PPO agent...")
    model = PPO(
        "MultiInputPolicy",  # For Dict observation space
        env,
        verbose=1,
        tensorboard_log="./tb_logs/qontinui_ppo/",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
    )

    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    print("Saving model...")
    model.save("qontinui_ppo_agent")

    print("Done! Model saved to qontinui_ppo_agent.zip")
    return model


def train_dqn(
    config_path: str, goal_states: list[str], total_timesteps: int = 10000
) -> Any:
    """Train a DQN agent on qontinui-gym.

    Note: DQN requires discrete action space. Use ActionSpaceConfig.discrete()
    when creating the environment.

    Args:
        config_path: Path to QontinuiConfig
        goal_states: Goal states for reward
        total_timesteps: Training steps
    """
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3")
        return

    from qontinui_gym import QontinuiEnv
    from qontinui_gym.config.loader import compute_state_distances
    from qontinui_gym.rewards import QontinuiRewardWrapper, RewardBuilder
    from qontinui_gym.spaces import ActionSpaceConfig

    # Create environment with discrete action space for DQN
    def make_env() -> QontinuiRewardWrapper:
        env = QontinuiEnv(
            config_path=config_path,
            max_episode_steps=100,
            action_space_config=ActionSpaceConfig.discrete(),
        )

        config = env.get_config()
        state_distances = compute_state_distances(config, goal_states)

        reward_fn = (
            RewardBuilder()
            .goal_reaching(goal_states, reward=10.0, terminal=True)
            .progress_shaping(state_distances)
            .step_penalty(-0.001)
            .build()
        )

        return QontinuiRewardWrapper(env, reward_function=reward_fn)

    print("Creating environment...")
    env = DummyVecEnv([make_env])

    print("Creating DQN agent...")
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs/qontinui_dqn/",
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
    )

    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    print("Saving model...")
    model.save("qontinui_dqn_agent")

    return model


def evaluate_agent(model_path: str, config_path: str, num_episodes: int = 10) -> None:
    """Evaluate a trained agent.

    Args:
        model_path: Path to saved model
        config_path: Path to QontinuiConfig
        num_episodes: Number of evaluation episodes
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("stable-baselines3 not installed.")
        return

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    env = create_env(config_path)

    print(f"Evaluating for {num_episodes} episodes...")
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    print(f"\nResults over {num_episodes} episodes:")
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Mean length: {mean_length:.1f} +/- {std_length:.1f}")

    env.close()


if __name__ == "__main__":
    print("Stable-Baselines3 Integration Example")
    print("=" * 40)
    print()
    print("This example requires:")
    print("  1. qontinui-runner running on localhost:9876")
    print("  2. A valid QontinuiConfig JSON file")
    print("  3. stable-baselines3 installed")
    print()
    print("Usage:")
    print("  # Train PPO")
    print('  train_ppo("config.json", ["goal_state"], total_timesteps=10000)')
    print()
    print("  # Evaluate")
    print('  evaluate_agent("qontinui_ppo_agent", "config.json")')
