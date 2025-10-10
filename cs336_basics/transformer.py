import torch
from typing import Optional
import numpy as np
from jaxtyping import Float, Int, Bool
from einops import rearrange, einsum
import math

def get_weights(shape, init_std, device, dtype):
    return torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(shape, device=device, dtype=dtype), std=init_std, a=-3*init_std, b=3*init_std)) 

class Linear(torch.nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        w = torch.empty(d_out, d_in, device=device, dtype=dtype)
        std = np.sqrt(2 / (d_in + d_out))
        w = torch.nn.init.trunc_normal_(w, std=std, a=-3*std, b=3*std)
        self.weights = torch.nn.Parameter(w)

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> torch.Tensor:
        return torch.einsum("...i,ji->...j", x, self.weights) 

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, device=None, dtype=None):
        super().__init__()
        e = torch.empty(vocab_size, embedding_dim, device=device, dtype=dtype)
        e = torch.nn.init.trunc_normal_(e, std=1, a=-3, b=3)
        self.embeddings = torch.nn.Parameter(e)

    def forward(self, x: Int[torch.Tensor, "..."]) -> torch.Tensor:
        orig_shape = list(x.size())
        x_flat = x.flatten()
        result = self.embeddings[x_flat]
        embedding_dim = self.embeddings.shape[-1]
        return result.reshape(orig_shape + [embedding_dim])

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        gain = torch.empty(d_model, device=device, dtype=dtype)
        gain = torch.nn.init.ones_(gain)
        self.gain = torch.nn.Parameter(gain)
        self.eps = eps

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x_inv_std = 1/torch.sqrt(torch.mean((x * x), axis=-1) + self.eps)
        x_norm = torch.einsum("...i,...,i -> ...i", x, x_inv_std, self.gain)
        return x_norm.to(orig_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        def get_weights(d1, d2):
            w = torch.empty(d1, d2, device=device, dtype=dtype)
            std = np.sqrt(2 / (d1 + d2))
            w = torch.nn.init.trunc_normal_(w, std=std, a=-3*std, b=3*std)
            return torch.nn.Parameter(w)

        self.weights_pregate = get_weights(d_ff, d_model)
        self.weights_postgate = get_weights(d_ff, d_model)
        self.weights_postswiglu = get_weights(d_model, d_ff)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> torch.Tensor:
        x_pregate = torch.einsum("...i,ji->...j", x, self.weights_pregate)
        #silu = x_pregate * torch.sigmoid(x_pregate)
        silu = torch.nn.functional.silu(x_pregate)
        x_postgate = torch.einsum("...i,ji->...j", x, self.weights_postgate)
        swiglu = silu * x_postgate
        return torch.einsum("...i,ji -> ...j", swiglu, self.weights_postswiglu)
    
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        assert d_k % 2 == 0
        ks = (2*torch.arange(d_k // 2, device=device, dtype=dtype))/d_k
        denom = 1/(theta ** ks)
        num = torch.arange(max_seq_len, device=device)
        angles = torch.einsum("i,j->ij", num, denom)
        self.dtype = angles.dtype
        rotation = torch.exp(angles*1j)
        self.register_buffer("rotation", rotation, persistent=False)

    def forward(self, 
                x: Float[torch.Tensor, "... seq_len d_k"], 
                token_positions: Int[torch.Tensor, "... seq_len"]|int) -> torch.Tensor:
        x = rearrange(x, "... (half_d_k two) -> ... half_d_k two", two=2)
        x = torch.view_as_complex(x)
        if type(token_positions) == int:
            rotation = self.rotation[..., :token_positions, :]
        else:
            rotation = self.rotation[token_positions.flatten()].reshape(list(token_positions.shape) + [self.rotation.shape[-1]])
        rotated_x = torch.view_as_real(x*rotation)
        return rearrange(rotated_x, "... half_d_k two -> ... (half_d_k two)")

def softmax(tensor: torch.Tensor, dim_index: int) -> torch.Tensor:
    norm_exp = torch.exp(tensor - torch.max(tensor, axis=dim_index, keepdims=True).values)
    return norm_exp / torch.sum(norm_exp, axis=dim_index, keepdims=True)

def scaled_dot_product_attention(
        queries: Float[torch.Tensor, "... num_q d_k"],
        keys: Float[torch.Tensor, "... num_k d_k"],
        values: Float[torch.Tensor, "... num_k d_v"],
        mask: Bool[torch.Tensor, "num_q num_k"] | None = None) -> Float[torch.Tensor, "... num_q d_v"]:
    similarities = einsum(
            queries, keys, "... num_q d_k, ... num_k d_k -> ... num_q num_k")
    similarities /= math.sqrt(queries.shape[-1])
    if mask is not None:
        similarities = torch.where(mask, similarities, -torch.inf)
    return einsum(torch.softmax(similarities, -1), values, "... num_q num_k, ... num_k d_v -> ... num_q d_v")


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                 max_seq_len: int, theta: Optional[float], device=None, dtype=None):
        super().__init__()
        
        assert d_model % num_heads == 0
        d_k = d_model // num_heads

        if theta is not None:
            self.rope = RotaryPositionalEmbedding(
                    theta, d_k, max_seq_len, device=device, dtype=dtype)
        else:
            self.rope = None

        self.weights_qkv = get_weights((3*d_model, d_model), np.sqrt(2/(d_model + d_model)), device=device, dtype=dtype)
        self.weights_o = get_weights((d_model, d_model), np.sqrt(2/(d_model + d_model)), device=device, dtype=dtype)
        
        mask = torch.ones(max_seq_len, max_seq_len).tril() > 0
        self.register_buffer("mask", mask, persistent=False)

        self.dtype= self.weights_qkv.dtype
        self.num_heads = num_heads


    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"], 
                token_positions: Optional[Float[torch.Tensor, "... seq_len"]]=None):
        orig_dtype = x.dtype
        x = x.to(self.dtype)
        QKV = einsum(x, self.weights_qkv, 
        "... seq_len d_model_in, d_model_out_3x d_model_in -> ... seq_len d_model_out_3x")
        QKV = rearrange(QKV, "... seq_len (three num_heads d_k)-> three ... num_heads seq_len d_k", 
                        num_heads=self.num_heads, three=3)
        
        seq_len = x.shape[-2]
        Q, K, V = QKV[0], QKV[1], QKV[2]
        if self.rope is not None:
            if token_positions is not None:
                Q = self.rope.forward(Q, token_positions)
                K = self.rope.forward(K, token_positions)
            else:
                Q = self.rope.forward(Q, seq_len)
                K = self.rope.forward(K, seq_len)

        O = scaled_dot_product_attention(Q, K, V, self.mask[:seq_len,:seq_len])


        O = rearrange(O, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")
        result = einsum(O, self.weights_o, "... seq_len d_model_in, d_model_out d_model_in -> ... seq_len d_model_out")
        return result.to(orig_dtype)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 device=None, dtype=None, attention_dtype=None):
        super().__init__()
        self.mha_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=attention_dtype)
        self.ff_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"]):
        x_norm = self.mha_norm.forward(x)
        x_mha = self.mha.forward(x_norm)
        x2 = x + x_mha

        x2_norm = self.ff_norm.forward(x2)
        x2_ff = self.ff.forward(x2_norm)
        return x2 + x2_ff

class TransformerLM(torch.nn.Module):
    def __init__(self, *, vocab_size: int, num_layers: int, d_model: int, num_heads: int, 
                 d_ff: int, max_seq_len: int, theta: Optional[float], device=None, dtype=None, attention_dtype=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                    TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype, attention_dtype=attention_dtype))
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, indices: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        x = self.embedding.forward(indices)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.final_norm.forward(x)
        x = self.output_proj.forward(x)
        return x

         
        

        

