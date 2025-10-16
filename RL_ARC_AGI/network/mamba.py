import torch.nn as nn
from mamba_ssm import Mamba

class MambaLayers(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_state,
                 num_layers):
        super(MambaLayers, self).__init__()
        self.layers = nn.ModuleList([Mamba(
                            d_model=d_model, # Model dimension d_model
                            d_state=d_state,  # SSM state expansion factor
                            d_conv=4,    # Local convolution width
                            expand=2,    # Block expansion factor
                            ) for i in range(num_layers)])
        self.activation = nn.GELU()
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x