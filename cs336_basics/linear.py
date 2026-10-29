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

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.d_model = embedding_dim
        self.vocab_size = num_embeddings
        self.embeddings = nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embeddings, std=1, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.empty(size=(d_model,), device=device, dtype=dtype))
        torch.nn.init.constant_(self.g, 1)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape or x.shape[-1] != self.d_model:
            raise ValueError
        in_dtype = x.dtype
        x = x.to(torch.float32)

        squared = torch.square(x)
        add_e = squared.add(self.eps)
        mean = torch.mean(add_e, dim=-1, keepdim=True)

        rsqrt = torch.rsqrt(mean)

        res = torch.multiply(x, rsqrt)
        res = torch.multiply(res, self.g)

        return res.to(in_dtype)