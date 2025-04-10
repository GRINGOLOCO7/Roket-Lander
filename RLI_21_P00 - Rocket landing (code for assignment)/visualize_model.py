import os
import numpy as np
import time
import argparse
import glob
import torch
import cv2  # Add explicit import of OpenCV
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from rocket_env import RocketEnv

# Update device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to the model file')
    parser.add_argument('--task', type=str, default='landing', choices=['landing', 'hover'], help='Task to perform')
    parser.add_argument('--rocket', type=str, default='starship', choices=['starship', 'falcon'], help='Rocket type')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--delay', type=float, default=0.01, help='Delay between steps for visualization')
    parser.add_argument('--save_frames', action='store_true', help='Save frames to create a video')
    return parser.parse_args()

def find_best_model(log_dir):
    """Find the best model in the log directory."""
    # First check for final model
    final_model_path = os.path.join(log_dir, "final_model.zip")
    if os.path.exists(final_model_path):
        return final_model_path
    
    # Then check for best model
    best_models = sorted(glob.glob(os.path.join(log_dir, "best_model", "model_*.zip")))
    if best_models:
        return best_models[-1]
    
    # Finally check for checkpoints
    checkpoints = sorted(glob.glob(os.path.join(log_dir, "checkpoint_*.zip")))
    if checkpoints:
        return checkpoints[-1]
    
    # Also check for partial models (in case training was interrupted)
    partial_model_path = os.path.join(log_dir, "partial_model.zip")
    if os.path.exists(partial_model_path):
        return partial_model_path
    
    return None

def main():
    args = parse_args()
    
    # Ensure output directory for saved frames
    if args.save_frames:
        frames_dir = "rocket_landing_frames"
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Frames will be saved to {frames_dir}")
    
    # Set up the environment
    log_dir = "./sb3_starship_landing_logs/"
    
    print(f"Creating environment with task={args.task}, rocket_type={args.rocket}")
    
    # Create environment with direct rendering
    env = RocketEnv(task=args.task, rocket_type=args.rocket, max_steps=800)
    
    # For rocket.py debug help
    if hasattr(env, 'rocket'):
        print(f"Rocket type: {env.rocket.rocket_type}")
        print(f"Rocket task: {env.rocket.task}")
        print(f"Viewport size: {env.rocket.viewport_w}x{env.rocket.viewport_h}")
    else:
        print("Warning: env does not have rocket attribute - using alternative environment structure")
    
    # Create monitor wrapper
    env = Monitor(env, log_dir)
    
    # Find the model to load
    model_path = args.model
    if model_path is None:
        model_path = find_best_model(log_dir)
        if model_path is None:
            raise ValueError(f"No model found in {log_dir}. Please specify a model path with --model.")
    
    print(f"Loading model from {model_path}")
    
    # Create a vector environment for the model
    vec_env = DummyVecEnv([lambda: env])
    
    # Load the normalization parameters if they exist
    vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print(f"Loading normalization parameters from {vec_normalize_path}")
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        # Don't update the normalization statistics during evaluation
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Load the model
    model = PPO.load(model_path, env=vec_env, device=device)
    
    # Run the model
    total_rewards = []
    success_count = 0
    
    for episode in range(args.episodes):
        print(f"\n=== Episode {episode+1}/{args.episodes} ===")
        
        # Reset environment
        obs = vec_env.reset()
        episode_reward = 0
        done = False
        step = 0
        frames = []
        
        while not done:
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take the action
            obs, reward, done, info = vec_env.step(action)
            
            # Extract done from the array
            done = done[0]
            
            episode_reward += reward[0]
            
            # Render the environment directly through the rocket
            if hasattr(env, 'rocket'):
                frame_0, frame_1 = env.rocket.render(
                    window_name=f'Rocket Landing - Episode {episode+1}',
                    wait_time=1,  # Just display, don't wait
                    with_trajectory=True,
                    with_camera_tracking=True
                )
                
                # Save frames if requested
                if args.save_frames:
                    frames.append(frame_1)
            else:
                # Fallback to environment render
                env.render()
            
            # Add a small delay for visualization
            time.sleep(args.delay)
            
            # Print step information
            if step % 20 == 0:
                print(f"Step: {step}, Reward: {reward[0]:.2f}, Total: {episode_reward:.2f}")
            
            step += 1
        
        # Determine if landing was successful
        success = False
        if 'is_success' in info[0]:
            success = info[0]['is_success']
        else:
            # Use several methods to determine success
            if hasattr(env, 'rocket'):
                # Direct check from rocket state
                success = env.rocket.check_landing_success(env.rocket.state)
                print(f"Landing check result: {success}")
            else:
                # Use a reward threshold to determine success
                success = episode_reward > 100
        
        # Display final image with longer duration
        cv2.waitKey(500)  # Show the final frame for 500ms
        
        # Save episode frames if requested
        if args.save_frames and frames:
            episode_dir = os.path.join("rocket_landing_frames", f"episode_{episode+1}")
            os.makedirs(episode_dir, exist_ok=True)
            
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(episode_dir, f"frame_{i:04d}.jpg"), frame)
            
            print(f"Saved {len(frames)} frames to {episode_dir}")
        
        if success:
            success_count += 1
            print(f"Episode {episode+1} - SUCCESS! Total reward: {episode_reward:.2f}")
        else:
            print(f"Episode {episode+1} - FAILED. Total reward: {episode_reward:.2f}")
        
        total_rewards.append(episode_reward)
    
    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"Average reward over {args.episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Success rate: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
    print(f"Rewards: {[f'{r:.1f}' for r in total_rewards]}")
    
    # Create a combined video from all successful episodes
    if args.save_frames:
        print("To create a video from frames, use:")
        print("ffmpeg -framerate 30 -pattern_type glob -i 'rocket_landing_frames/episode_X/frame_*.jpg' -c:v libx264 -pix_fmt yuv420p rocket_landing_episode_X.mp4")
    
    # Keep the last window open until a key is pressed
    print("\nPress any key to close the visualization...")
    cv2.waitKey(0)
    
    # Close all windows
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main() 