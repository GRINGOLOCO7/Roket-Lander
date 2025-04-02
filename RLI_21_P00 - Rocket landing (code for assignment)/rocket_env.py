"""
+--------------------------------------------------------------------------------+
|  WARNING!!!                                                                    |
|  THIS IS JUST AN STUB FILE (TEMPLATE)                                          |
|  PROBABLY ALL LINES SHOULD BE CHANGED OR TOTALLY REPLACED IN ORDER TO GET A    |
|  WORKING FUNCTIONAL VERSION FOR YOUR ASSIGNMENT                                |
+--------------------------------------------------------------------------------+
"""

# import gymnasium as gym
# from gym import spaces
import numpy as np
from rocket import Rocket


class RocketEnv(): # gym.Env
    metadata = {'render.modes' : ['human','...']}
    def __init__(self):
        print("init")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=int)
        self.is_view = True
        self.rocket = Rocket(max_steps=800)
        self.memory = []

    def reset(self):
        mode = self.rocket.mode
        del self.rocket
        self.rocket = Rocket(max_steps=800, mode = mode)
        obs = self.rocket.observe()
        return obs

    def step(self, action):
        self.rocket.action(action)
        reward = self.rocket.evaluate() # ¿?
        done   = self.rocket.is_done()  # ¿?
        obs    = self.rocket.observe()  # ¿?
        return obs, reward, done, {'dist':self.rocket.distance, 'check':self.rocket.current_check, 'crash': not self.rocket.is_alive} # what the heck?

    def render(self, mode="human", close=False, msgs=[]):
        if self.is_view:
            self.rocket.view_(msgs)

    def set_view(self, flag):
        self.is_view = flag

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



if __name__ == "__main__":
    print("File __name__ is set to: {}" .format(__name__))
    # Using the gym library to create the environment
    # env = gym.make('your_environment_name-v0')
    from stable_baselines3.common.env_checker import check_env
    env = RocketEnv()
    print('CHECK_ENV','OK' if check_env(env) is None else 'ERROR')
    print(env.observation_space) 
    print(env.action_space) 
    print(type(env).__name__)
