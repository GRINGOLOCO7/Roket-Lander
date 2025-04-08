import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from rocket import Rocket

class RocketEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, task='hover', rocket_type='falcon', max_steps=800):
        super(RocketEnv, self).__init__()
        self.task = task
        self.rocket_type = rocket_type
        self.max_steps = max_steps
        self.rocket = Rocket(max_steps=self.max_steps, task=self.task, rocket_type=self.rocket_type)


        # Observations are dictionaries with the agent's and the target's location.
        #self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=int)
        # Define the observation space based on the returned state from Rocket
        # The state is 8-dimensional according to the flatten method
        low = np.array([-np.inf] * 8, dtype=np.float32)
        high = np.array([np.inf] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


        # Define action space based on the action table
        self.action_space = spaces.Discrete(self.rocket.action_dims) # 3


        # target point
        #if task == 'hover':
        #    self._target_location = 0, 200, 50
        #elif task == 'landing':
        #    self._target_location = 0, self.H/2.0, 50
        #self._agent_location = self.rocket.create_random_state()


        # Set render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Window for rendering
        self.window = None
        self.clock = None


    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #self._agent_location = self.rocket.create_random_state()
        #if self.task == 'hover':
        #    self._target_location = 0, 200, 50
        #elif self.task == 'landing':
        #    self._target_location = 0, self.H/2.0, 50
        #observation = self._get_obs()
        # Reset the rocket
        observation = self.rocket.reset()

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        info = {}
        return observation, info

    def step(self, action):
        #observation, reward, terminated, info = self.rocket.action(action)
        # Take action in the rocket environment
        observation, reward, done, info = self.rocket.step(action)

        # Additional termination conditions could be added here
        terminated = done
        truncated = False

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _render_frame(self):
        """Render the current frame."""
        frame_0, frame_1 = self.rocket.render(window_name='Rocket Environment',
                                              wait_time=1,
                                              with_trajectory=True,
                                              with_camera_tracking=True)

        if self.render_mode == "human":
            # The rocket.render already displays the window
            return None
        else:
            # Return the rendered image
            return frame_1

    def close(self):
        """Close the environment."""
        cv2.destroyAllWindows()
        if hasattr(self.rocket, 'close'):
            self.rocket.close()

