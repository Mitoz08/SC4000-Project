from torch import nn

class MLP(nn.Module):
    def __init__(self, layers = [128, 256, 64]):
        super().__init__()