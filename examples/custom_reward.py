"""Custom reward function example for qontinui-gym.

This example demonstrates how to use the reward system to define
custom reward functions for RL training.
"""

from qontinui_gym import QontinuiEnv
from qontinui_gym.config.loader import compute_state_distances
from qontinui_gym.rewards import (
    ComposedReward,
    QontinuiRewardWrapper,
    RewardBuilder,
    StateReachReward,
    StateVisitCountReward,
    StepPenalty,
)
from qontinui_gym.rewards.presets import goal_conditioned_preset


def example_builder_api() -> None:
    """Example using the RewardBuilder fluent API."""
    print("=== RewardBuilder API ===\n")

    config_path = "path/to/your/automation-config.json"

    # Create base environment
    base_env = QontinuiEnv(
        config_path=config_path,
        max_episode_steps=100,
    )

    # Define goal states
    goal_states = {"checkout_complete", "order_confirmed"}

    # Build reward function using fluent API
    reward_fn = (
        RewardBuilder()
        .goal_reaching(goal_states, reward=100.0, terminal=True)
        .step_penalty(-0.001)
        .exploration_bonus(beta=0.5)
        .transition_reward(success_reward=0.1, failure_penalty=-0.2)
        .build()
    )

    # Wrap environment with reward function
    env = QontinuiRewardWrapper(base_env, reward_function=reward_fn)

    # Run episode
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.4f}, total={total_reward:.4f}")

        if terminated or truncated:
            break

    print(f"\nFinal total reward: {total_reward:.4f}")
    print(f"Reward components: {info.get('reward_components', {})}")

    env.close()


def example_with_shaping() -> None:
    """Example with potential-based reward shaping."""
    print("\n=== Reward Shaping ===\n")

    config_path = "path/to/your/automation-config.json"

    base_env = QontinuiEnv(config_path=config_path, max_episode_steps=100)
    config = base_env.get_config()

    # Compute distances from all states to goal
    goal_states = ["checkout_complete"]
    state_distances = compute_state_distances(config, goal_states)

    print(f"State distances to goal: {state_distances}")

    # Build shaped reward
    reward_fn = (
        RewardBuilder()
        .goal_reaching(goal_states, reward=10.0, terminal=True)
        .progress_shaping(state_distances, gamma=0.99)
        .step_penalty(-0.001)
        .build()
    )

    env = QontinuiRewardWrapper(base_env, reward_function=reward_fn)

    obs, info = env.reset()
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: shaped_reward={reward:.4f}")
        if terminated or truncated:
            break

    env.close()


def example_manual_composition() -> None:
    """Example manually composing reward components."""
    print("\n=== Manual Composition ===\n")

    config_path = "path/to/your/automation-config.json"

    base_env = QontinuiEnv(config_path=config_path, max_episode_steps=100)

    # Create individual components
    components = [
        StateReachReward(
            target_states={"login_success"},
            reward_per_state=5.0,
            weight=1.0,
        ),
        StateReachReward(
            target_states={"checkout_complete"},
            reward_per_state=20.0,
            terminal_on_reach=True,
            weight=1.0,
        ),
        StepPenalty(penalty=-0.01, weight=1.0),
        StateVisitCountReward(beta=0.1, weight=1.0),
    ]

    # Compose manually
    reward_fn = ComposedReward(components)

    env = QontinuiRewardWrapper(base_env, reward_function=reward_fn)

    obs, info = env.reset()
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 10 == 0:
            stats = reward_fn.get_statistics()
            print(f"Step {step}: reward={reward:.4f}")
            for comp in stats["components"]:
                print(f"  {comp['name']}: avg={comp['average_reward']:.4f}")
        if terminated or truncated:
            break

    env.close()


def example_preset() -> None:
    """Example using preset reward configurations."""
    print("\n=== Preset Rewards ===\n")

    config_path = "path/to/your/automation-config.json"

    base_env = QontinuiEnv(config_path=config_path, max_episode_steps=100)
    config = base_env.get_config()

    goal_states = {"checkout_complete"}
    state_distances = compute_state_distances(config, list(goal_states))

    # Use preset
    reward_fn = goal_conditioned_preset(
        goal_states=goal_states,
        state_distances=state_distances,
        goal_reward=10.0,
        step_penalty=-0.001,
        shaping_weight=0.5,
    )

    env = QontinuiRewardWrapper(base_env, reward_function=reward_fn)

    obs, info = env.reset()
    total = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break

    print(f"Total reward with preset: {total:.4f}")
    env.close()


if __name__ == "__main__":
    print("Note: These examples require a running qontinui-runner")
    print("and a valid config file. Update the config_path variables.\n")

    # Uncomment to run examples:
    # example_builder_api()
    # example_with_shaping()
    # example_manual_composition()
    # example_preset()

    print("Examples ready to run - update config paths and uncomment.")
