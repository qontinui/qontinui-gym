"""Basic usage example for qontinui-gym.

This example demonstrates how to create a QontinuiEnv and interact
with it using the standard Gymnasium API.

Requirements:
    - qontinui-runner must be running on localhost:9876
    - A valid QontinuiConfig JSON file
"""

from qontinui_gym import QontinuiEnv


def main() -> None:
    # Create environment
    # Replace with your actual config path
    config_path = "path/to/your/automation-config.json"

    env = QontinuiEnv(
        config_path=config_path,
        runner_host="localhost",
        runner_port=9876,
        max_episode_steps=50,
        render_mode="rgb_array",  # or "human"
    )

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Available workflows: {env.get_workflow_names()}")
    print(f"States: {env.get_state_names()}")

    # Reset environment
    observation, info = env.reset()
    print(f"\nInitial info: {info}")

    # Run episode with random actions
    total_reward = 0.0
    num_steps = 0

    for step in range(50):
        # Sample random action
        action = env.action_space.sample()

        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        num_steps += 1

        print(
            f"Step {step}: "
            f"action={info.get('workflow_name', 'N/A')}, "
            f"success={info.get('action_success', False)}, "
            f"states={info.get('current_state_names', [])}"
        )

        if terminated or truncated:
            print(f"\nEpisode ended after {num_steps} steps")
            print(f"Total reward: {total_reward}")
            break

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
