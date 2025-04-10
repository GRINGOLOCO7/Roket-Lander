import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import time
from gym.wrappers import RecordVideo

# Import your custom environment
from rocket_env import RocketEnv

# Set parameters (should match those used during training)
task = 'landing'  # 'hover' or 'landing'
rocket_type = 'starship'  # 'starship' or 'falcon'
model_path = f"ppo_rocket_{task}_{rocket_type}"  # Path to your saved model

# Register the environment
gym.register(
    id='Rocket-v0',
    entry_point='rocket_env:RocketEnv',
    kwargs={'task': task, 'rocket_type': rocket_type, 'render_mode': 'human'}
)

def run_trained_model(save_video=False, num_episodes=5, sleep_time=0.01):
    """
    Run the trained model in the environment

    Args:
        save_video: Whether to save video recordings of episodes
        num_episodes: Number of episodes to run
        sleep_time: Time to sleep between steps for better visualization
    """
    print(f"Loading model from {model_path}...")
    # Load the trained model
    model = PPO.load(model_path)

    # Create environment
    if save_video:
        # Wrap environment with RecordVideo
        env = RecordVideo(
            gym.make('Rocket-v0', task=task, rocket_type=rocket_type, render_mode='rgb_array'),
            video_folder=f"videos/{task}_{rocket_type}",
            episode_trigger=lambda x: True  # Record every episode
        )
    else:
        env = gym.make('Rocket-v0', task=task, rocket_type=rocket_type, render_mode='human')

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"\nStarting episode {episode+1}/{num_episodes}")

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            # Small delay for better visualization
            time.sleep(sleep_time)

            # Print some info occasionally
            if step_count % 50 == 0:
                print(f"Step {step_count}, Current reward: {reward:.2f}, Total reward: {total_reward:.2f}")

            # Optional: Print environment info
            if 'status' in info:
                print(f"Status: {info['status']}")

        print(f"Episode {episode+1} finished after {step_count} steps with total reward: {total_reward:.2f}")

    env.close()

def analyze_model_performance(num_episodes=20):
    """
    Run the model for multiple episodes and analyze its performance
    """
    print(f"Loading model from {model_path} for analysis...")
    model = PPO.load(model_path)

    # No rendering for faster evaluation
    env = gym.make('Rocket-v0', task=task, rocket_type=rocket_type, render_mode=None)

    rewards = []
    success_count = 0
    step_counts = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        success = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            # Check if episode ended with success (this depends on your environment's implementation)
            if terminated and 'status' in info and info['status'] == 'success':
                success = True

        rewards.append(total_reward)
        step_counts.append(step_count)
        if success:
            success_count += 1

        print(f"Episode {episode+1}/{num_episodes}: Reward={total_reward:.2f}, Steps={step_count}, Success={success}")

    # Calculate statistics
    success_rate = success_count / num_episodes * 100
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_steps = np.mean(step_counts)

    print("\nPerformance Analysis:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Episode Length: {avg_steps:.1f} steps")

    env.close()

    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_steps': avg_steps,
        'rewards': rewards
    }

if __name__ == "__main__":
    print(f"Running trained model for {task} task with {rocket_type} rocket type")

    # Choose which function to run
    mode = input("Choose mode (1: Visualize with rendering, 2: Analyze performance): ")

    if mode == "1":
        save_video = input("Save video? (y/n): ").lower() == 'y'
        num_episodes = int(input("Number of episodes to run: ") or "5")
        run_trained_model(save_video=save_video, num_episodes=num_episodes)
    else:
        num_episodes = int(input("Number of episodes for analysis: ") or "20")
        analyze_model_performance(num_episodes=num_episodes)