import numpy as np

class GaussianNoise():
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_size):
        self.action_size = action_size

    def sample(self, scale, samples=1, target=False):
        """Update internal state and return it as a noise sample."""
        noise = np.random.normal(scale=scale, size=(samples, self.action_size))[0]
        if target:
            noise = np.clip(noise, -0.5, 0.5)
            return noise
        return noise