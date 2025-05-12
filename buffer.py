"""
replay buffer

Třída ReplayBuffer uchovává pevně velkou frontu přechodů (state, action, reward, next_state, done)
a umožňuje náhodný sampling bez opakování.  Slouží k implementaci off-policy učení DQN.
"""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Fronta pro experience replay.

    Args:
        capacity: Maximální počet uložených přechodů. Jakmile je kapacita
            překročena, nejstarší položky se automaticky zahodí (FIFO).

    Attributes:
        buffer (Deque[Tuple]): Interní kruhová fronta s uloženými přechody.
    """

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        """
        Uloží přechod do bufferu.
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Náhodně vybere mini-batch přechodů ze zásobníku.

        Args:
            batch_size: Počet vrácených přechodů.

        Returns:
            Pět numpy polí: states, actions, rewards, next_states, dones.
        """
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))

    def __len__(self):
        """
        Aktuální počet uložených přechodů.
        """
        return len(self.buffer)