"""
Plně propojená Q-network použitá pro aproximaci akční hodnoty.

Architektura: 2 skryté vrstvy po 256 neuronech, aktivace ReLU.
"""

import torch.nn as nn

class QNetwork(nn.Module):
    """
    Aproximace akční hodnoty.

    Args:
        state_size: Dimenze vektoru pro pozorování prostředí.
        action_size: Počet možných akcí v prostředí.
    """

    def __init__(self, state_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        """
        Propaguje vstup jedním průchodem sítě.
        
        Args:
            x: vstupní tensory ve tvaru '(batch, state_size)'.
        """
        return self.model(x)
