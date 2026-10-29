import torch
import torch.nn as nn

from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.param = nn.Parameter(torch.empty(size=(out_features, in_features), device=device, dtype=dtype))
        std = (2/(self.in_features+self.out_features))**(0.5)
        torch.nn.init.trunc_normal_(self.param, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape or x.shape[-1] != self.in_features:
            raise ValueError
        return einsum(x, self.param, "... d_in, d_out d_in -> ... d_out")