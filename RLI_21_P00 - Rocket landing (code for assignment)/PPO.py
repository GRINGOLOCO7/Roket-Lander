import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
# Import your custom environment
from rocket_env import RocketEnv

# Create save directories
task = 'landing'  # 'hover' or 'landing'
rocket_type = 'starship'  # 'starship' or 'falcon'

# Register the environment
gym.register(
    id='Rocket-v0',
    entry_point='rocket_env:RocketEnv',
    kwargs={'task': task, 'rocket_type': rocket_type, 'render_mode': None}  # No rendering during training
)

# Training
# --------

# Parallel environments for training
train_env = make_vec_env(
    'Rocket-v0',
    n_envs=4,
    env_kwargs={'task': task, 'rocket_type': rocket_type, 'render_mode': None}
)

# Create the PPO agent with better hyperparameters for continuous control
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    max_grad_norm=0.5
)

# Train the agent
print("Training the agent...")
model.learn(total_timesteps=100000)
model.save(f"ppo_rocket_{task}_{rocket_type}")

# Close training environment
train_env.close()

# Evaluation
# ----------

print("Evaluating the trained agent...")

# Create a single environment for evaluation with rendering
eval_env = gym.make('Rocket-v0', task=task, rocket_type=rocket_type, render_mode="human")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Manual evaluation loop with rendering
obs, _ = eval_env.reset()
cumulative_reward = 0
episode_count = 0
max_episodes = 5

print(f"Playing {max_episodes} episodes with the trained agent...")

while episode_count < max_episodes:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    cumulative_reward += reward

    # If episode is done
    if terminated or truncated:
        print(f"Episode {episode_count+1} finished with reward: {cumulative_reward}")
        episode_count += 1
        cumulative_reward = 0
        obs, _ = eval_env.reset()

        if episode_count >= max_episodes:
            break

# Close evaluation environment
eval_env.close()