# buffer.py - replay buffer pro učení

from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)