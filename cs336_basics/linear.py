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

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(size=(d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(size=(d_model, d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(size=(d_ff, d_model), device=device, dtype=dtype))
        std = (2/(self.d_model+self.d_ff))**(0.5)
        torch.nn.init.trunc_normal_(self.w1, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w2, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w3, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape or x.shape[-1] != self.d_model:
            raise ValueError
        gate = x @ self.w1.T
        silu = gate * torch.sigmoid(gate)

        up = x @ self.w3.T
        return (up * silu) @ self.w2.T


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        freqs = torch.arange(1, d_k//2+1, device=device)
        seqs = torch.arange(0, max_seq_len, device=device).unsqueeze(1)
        freqs = (theta ** ((2*freqs-2)/d_k)) ** -1  
        freqs = freqs * seqs
        self.r_cos =  torch.cos(freqs)
        self.r_sin =  torch.sin(freqs)
        self.register_buffer(name="cos", tensor=self.r_cos, persistent=False)
        self.register_buffer(name="sin", tensor=self.r_sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        q_even = x[..., 0::2]  # [q1, q3, q5, q7] (0-indexed: indices 0, 2, 4, 6)
        q_odd = x[..., 1::2]   # [q2, q4, q6, q8] (0-indexed: indices 1, 3, 5, 7)

        rotated_q = torch.stack([-q_odd, q_even], dim=-1).flatten(start_dim=-2)
        r_cos = self.r_cos[token_positions].repeat_interleave(2, dim=-1)
        r_sin = self.r_sin[token_positions].repeat_interleave(2, dim=-1)

        return r_cos * x + r_sin * rotated_q
