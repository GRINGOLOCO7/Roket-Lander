import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import glob
from rocket import Rocket
import utils
import time
import torch.nn as nn
import shutil  # Added for directory operations

# Update device configuration for M3 MAX
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define the wrapper class to make Rocket compatible with gym
class RocketEnv(gym.Env):
    def __init__(self, task='landing', rocket_type='starship', max_steps=800):
        super(RocketEnv, self).__init__()
        
        # Initialize the rocket environment with path_to_bg_img set to None
        # This will prevent it from trying to load a background image
        self.rocket = Rocket(task=task, max_steps=max_steps, rocket_type=rocket_type, path_to_bg_img=None)
        
        # Define action and observation space
        # For Gym compatibility, we'll use a Discrete action space if that's what the Rocket expects
        if hasattr(self.rocket, 'action_table') and isinstance(self.rocket.action_table, list):
            # If the rocket uses an action table (discrete actions)
            self.action_space = spaces.Discrete(len(self.rocket.action_table))
            self.is_continuous_action = False
        else:
            # Use continuous action space otherwise
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.rocket.action_dims,), dtype=np.float32)
            self.is_continuous_action = True
            
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.rocket.state_dims,), dtype=np.float32)
        
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Parameters:
            seed (int, optional): The seed that is used to initialize the environment's PRNG.
            options (dict, optional): Additional options used to customize the reset behavior.
            
        Returns:
            observation: The initial observation of the space.
            info: Dictionary containing additional information.
        """
        # We ignore the seed parameter since the original Rocket class doesn't support it
        if seed is not None:
            # Optional: Set Python and NumPy seed if needed
            import random
            random.seed(seed)
            np.random.seed(seed)
            
        # Reset the environment
        observation = self.rocket.reset()
        info = {}  # Additional info dictionary required by Gymnasium API
        
        return observation, info
    
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        
        Parameters:
            action: The action to take
            
        Returns:
            observation: Next observation from the environment
            reward: Reward from the action taken
            terminated: Whether the episode has ended
            truncated: Whether the episode was truncated (e.g., due to time limit)
            info: Additional information
        """
        # Process the action based on the action space type
        if not self.is_continuous_action:
            # If using discrete action space, convert to integer
            action = int(action)  # Convert to int for indexing
        
        observation, reward, done, info = self.rocket.step(action)
        
        # Split 'done' into 'terminated' and 'truncated' for Gymnasium compatibility
        terminated = done and 'TimeLimit.truncated' not in info
        truncated = done and 'TimeLimit.truncated' in info
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        return self.rocket.render()

# Callback for saving models and plotting during training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.rewards = []
        self.episode_counter = 0
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Save the model and plot the rewards
        if self.n_calls % self.check_freq == 0:
            try:
                # Get the monitor's rewards
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                
                if len(x) > 0:
                    # Get the latest episode reward
                    if len(y) > 0:
                        self.rewards.append(y[-1])
                        self.episode_counter += 1
                        
                        # Plot the rewards
                        if self.episode_counter % 100 == 0:
                            plt.figure(figsize=(10, 6))
                            plt.plot(self.rewards)
                            plt.plot(utils.moving_avg(self.rewards, N=50))
                            plt.title('Training Rewards')
                            plt.xlabel('Episodes')
                            plt.ylabel('Rewards')
                            plt.legend(['Episode Reward', 'Moving Average (50)'])
                            plt.savefig(os.path.join(self.log_dir, f'rewards_{self.episode_counter}.png'))
                            plt.close()
                    
                    # Mean reward over last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                    # Save the best model
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(os.path.join(self.save_path, f"model_{self.episode_counter}"))
                    
                    # Regular checkpoints
                    if self.episode_counter % 1000 == 0:
                        checkpoint_path = os.path.join(self.log_dir, f"checkpoint_{self.episode_counter}")
                        self.model.save(checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                # In case of any error with the log data, just print and continue
                print(f"Warning: Error in callback: {str(e)}")
        
        return True

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.feature_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # Add normalization for better stability
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_net(observations)

def ensure_fresh_start(log_dir):
    """Ensure we have a fresh start by recreating the log directory"""
    try:
        # If the directory exists, remove it and its contents
        if os.path.exists(log_dir):
            print(f"Removing existing log directory: {log_dir}")
            shutil.rmtree(log_dir)
        
        # Create fresh directory
        print(f"Creating fresh log directory: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create best_model subdirectory
        best_model_path = os.path.join(log_dir, 'best_model')
        os.makedirs(best_model_path, exist_ok=True)
        
        print("Successfully prepared directories for training")
        return True
    except Exception as e:
        print(f"Error preparing directories: {str(e)}")
        return False

def main():
    task = 'landing'
    rocket_type = 'starship'
    max_steps = 800
    
    log_dir = "./sb3_starship_landing_logs/"
    
    # Ensure we have a fresh start with clean directories
    success = ensure_fresh_start(log_dir)
    if not success:
        print("Failed to prepare directories. Check permissions or try with different path.")
        return
    
    # TRY-EXCEPT block to catch any TensorBoard-related errors
    try:
        # Check for TensorBoard before starting
        import tensorboard
        print(f"TensorBoard is available (version: {tensorboard.__version__})")
    except ImportError:
        print("TensorBoard is not installed. Installing it now...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "tensorboard"])
            print("TensorBoard installed successfully")
        except Exception as e:
            print(f"Failed to install TensorBoard: {str(e)}")
            print("Continuing without TensorBoard logging...")
            # Set tensorboard_log to None to disable TensorBoard logging
            tensorboard_log = None
    
    # Environment setup with normalization
    env = RocketEnv(task=task, rocket_type=rocket_type, max_steps=max_steps)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        gamma=0.99
    )

    # Check if the action space is discrete or continuous
    is_discrete_action_space = isinstance(env.action_space, spaces.Discrete)

    # Custom network architecture - adjusted for discrete actions
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[256, 256, 128],  # Policy network
            vf=[256, 256, 128]   # Value network
        ),
        activation_fn=torch.nn.ReLU,
        normalize_images=False
    )
    
    # Define simple schedules instead of complex ones
    def linear_schedule(initial_value: float):
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func
    
    # Use constant entropy values instead of schedules
    entropy_value = 0.01  # Fixed entropy value

    # Improved PPO configuration with enhanced exploration
    # Use CPU for better compatibility as suggested by the warning
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(3e-4),  # Learning rate annealing
        n_steps=2048,
        batch_size=256,  # Increased batch size
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.2),  # Clip range annealing
        clip_range_vf=0.2,
        # Use a constant entropy coefficient instead of a schedule
        ent_coef=entropy_value,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,  # Disable SDE since we're using discrete actions
        target_kl=0.02,  # Slightly increased target KL for
    )
    # Enhanced callback with more monitoring and adaptive exploration
    class EnhancedTrainingCallback(SaveOnBestTrainingRewardCallback):
        def __init__(self, check_freq, log_dir, verbose=1, plateau_patience=10):
            super().__init__(check_freq, log_dir, verbose)
            self.training_start = time.time()
            self.last_time = self.training_start
            self.kl_divergences = []
            self.mean_rewards = []
            self.plateau_patience = plateau_patience
            self.plateau_counter = 0
            self.best_mean_reward_plateau = -np.inf
            
        def _on_step(self) -> bool:
            try:
                if self.n_calls % self.check_freq == 0:
                    # Track KL divergence
                    if hasattr(self.model, 'logger') and 'kl' in self.model.logger.name_to_value:
                        self.kl_divergences.append(self.model.logger.name_to_value['kl'])
                    
                    # Performance logging
                    current_time = time.time()
                    fps = self.check_freq / (current_time - self.last_time)
                    self.last_time = current_time
                    
                    # Get the monitor's rewards
                    try:
                        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                        
                        if len(x) > 0:
                            # Mean reward over last 100 episodes
                            mean_reward = np.mean(y[-100:])
                            self.mean_rewards.append(mean_reward)
                            
                            # Plateau detection and handling
                            if len(self.mean_rewards) > 5:  # Need some history
                                if mean_reward > self.best_mean_reward_plateau:
                                    self.best_mean_reward_plateau = mean_reward
                                    self.plateau_counter = 0
                                elif mean_reward <= self.best_mean_reward_plateau * 1.01:  # Allow 1% variation
                                    self.plateau_counter += 1
                                else:
                                    self.plateau_counter = 0
                                    
                                # If we're stuck on a plateau, temporarily increase exploration
                                if self.plateau_counter >= self.plateau_patience:
                                    if hasattr(self.model, 'ent_coef'):
                                        # Temporarily boost entropy for exploration
                                        self.model.ent_coef = 0.04  # Fixed value for plateau escape
                                        print(f"\nPlateau detected! Temporarily increasing entropy to {self.model.ent_coef:.4f} for exploration boost")
                                        # Reset after a while
                                        self.plateau_counter = 0
                    except Exception as e:
                        print(f"Warning: Error processing rewards: {str(e)}")
                    
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"FPS: {fps:.2f}")
                        if len(self.kl_divergences) > 0:
                            print(f"Recent KL divergence: {self.kl_divergences[-1]:.5f}")
                        if len(self.mean_rewards) > 0:
                            print(f"Current mean reward: {self.mean_rewards[-1]:.2f}")
                            if len(self.mean_rewards) > 5:
                                print(f"Plateau counter: {self.plateau_counter}/{self.plateau_patience}")
                    
                return super()._on_step()
            except Exception as e:
                print(f"Warning: Error in callback step: {str(e)}")
                return True  # Continue training despite errors

    # Create callback
    callback = EnhancedTrainingCallback(
        check_freq=1000,
        log_dir=log_dir,
        verbose=1,
        plateau_patience=10
    )

    # We're starting with a fresh training run, so no need to load existing models
    print("Starting fresh training run...")

    # Train model with error handling
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=16000000,
            callback=callback,
            tb_log_name="PPO_rocket_landing"
        )
        
        # Save final model and environment
        model.save(os.path.join(log_dir, "final_model"))
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        
        print("Training complete!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Try to save the model even if training failed
        try:
            print("Attempting to save partial progress...")
            model.save(os.path.join(log_dir, "partial_model"))
            env.save(os.path.join(log_dir, "partial_vec_normalize.pkl"))
            print("Partial progress saved.")
        except Exception as save_error:
            print(f"Failed to save partial progress: {str(save_error)}")
    
    # Print some ideas for improvement
    print("\nIdeas for improving landing precision:")
    print("1. Implement a more sophisticated reward function that increases reward exponentially as the rocket gets closer to the target")
    print("2. Add a reward component for maintaining proper orientation during descent")
    print("3. Penalize rapid changes in thruster control that could cause instability")
    print("4. Add curriculum learning - start with easier landing scenarios and gradually increase difficulty")
    print("5. Implement a reward shaping approach that rewards partial progress toward the target")
    print("6. Increase the penalty for excessive speed near the landing target")
    print("7. Add a reward component for fuel efficiency to discourage wasteful maneuvers")
    print("8. Use hierarchical reinforcement learning to separate high-level navigation from low-level control")

if __name__ == "__main__":
    main()