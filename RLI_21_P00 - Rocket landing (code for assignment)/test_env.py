import gymnasium as gym
from rocket_env import RocketEnv

# Register the environment
gym.register(
    id='Rocket-v0',
    entry_point='rocket_env:RocketEnv',
    kwargs={'task': 'landing', 'rocket_type': 'falcon', 'render_mode': 'human'}
)

# Create and use the environment
env = gym.make('Rocket-v0')
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Your agent would select actions here
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()