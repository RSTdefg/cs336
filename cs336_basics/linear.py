import torch
import torch.nn as nn

from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(size=(out_features, in_features), device=device, dtype=dtype))
        std = (2/(self.in_features+self.out_features))**(0.5)
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape or x.shape[-1] != self.in_features:
            raise ValueError
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.d_model = embedding_dim
        self.vocab_size = num_embeddings
        self.weight = nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(size=(d_model,), device=device, dtype=dtype))
        torch.nn.init.constant_(self.weight, 1)
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
        res = torch.multiply(res, self.weight)

        return res.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape or x.shape[-1] != self.d_model:
            raise ValueError
        gate = x @ self.w1.weight.T
        silu = gate * torch.sigmoid(gate)

        up = x @ self.w3.weight.T
        return (up * silu) @ self.w2.weight.T


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
        r_cos = self.cos[token_positions].repeat_interleave(2, dim=-1)
        r_sin = self.sin[token_positions].repeat_interleave(2, dim=-1)

        return r_cos * x + r_sin * rotated_q

def Softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.amax(x, dim=dim, keepdim=True)
    exp = torch.exp(x)
    sm = exp * torch.sum(exp, dim=dim, keepdim=True) ** -1
    return sm


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):

    d_k = K.shape[-1]

    scores: torch.Tensor = (Q @ torch.transpose(K, -1, -2)) / (d_k ** (1/2))
    if mask is not None:
        scores = scores.masked_fill(~mask, -float('inf'))

    attn = Softmax(scores, -1) @ V
    return attn 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,  max_seq_length: int, theta: int = None, device=None, positions=None):
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError
        super().__init__()
        self.num_heads = num_heads
        d_k = d_model // num_heads # assignment spec
        self.k_proj = Linear(d_model, d_model, device=device)
        self.q_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.output_proj = Linear(d_model, d_model, device=device)
        self.RoPE = None
        if theta:
            self.RoPE = RoPE(theta, d_k, max_seq_length, device)
        self.positions = positions


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_n = x.shape[-2]
        mask = torch.tril(torch.ones((seq_n, seq_n), device=x.device, dtype=torch.bool), 0)
        
        K = x @ self.k_proj.weight.T
        Q = x @ self.q_proj.weight.T
        V = x @ self.v_proj.weight.T

        Q = rearrange(Q, "... seq (head d_v) -> ... head seq d_v ", head = self.num_heads)
        K = rearrange(K, "... seq (head d_v) -> ... head seq d_v ", head = self.num_heads)
        V = rearrange(V, "... seq (head d_v) -> ... head seq d_v ", head = self.num_heads)
        pos = self.positions if self.positions is not None else torch.arange(0, seq_n, device=x.device)
        if self.RoPE is not None:
            Q = self.RoPE(Q, pos)
            K = self.RoPE(K, pos)

        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "... head seq d_v -> ... seq (head d_v)")

        return attn @ self.output_proj.weight.T


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int,  max_seq_length: int, d_ff, theta: int = None):
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError
        super().__init__()
        self.num_heads = num_heads
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_length, theta)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
