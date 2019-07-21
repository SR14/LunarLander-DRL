import numpy as np
import gym

class LunarLander():
    def __init__(self, seed=False):

        # Lunar Lander Environment
        self.env = gym.make('LunarLanderContinuous-v2')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.env.reset()
        return state