import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.id = f"terrain-{type(self).__name__}"

    def initialize_weights(self):
        pass
