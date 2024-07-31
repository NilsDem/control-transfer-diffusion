# Attention adapted from https://github.com/Stability-AI/stable-audio-tools/tree/main/stable_audio_tools


import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from torch import Tensor, einsum
from packaging import version
from einops_exts import rearrange_many
from einops import rearrange, reduce, repeat
from torch.backends.cuda import sdp_kernel
from torch import Tensor, einsum, nn


# Taken from paper
def weight_normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = np.sqrt(n.numel() / x.numel())
    return x / torch.add(eps, n, alpha=alpha)


class WNConv2d(nn.Conv2d):

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(weight_normalize(self.weight))
        fan_in = self.weight[0].numel()
        weight = weight_normalize(self.weight) / np.sqrt(fan_in)
        return F.conv2d(x, weight, None, padding='same')


class WNLinear(nn.Linear):

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(weight_normalize(self.weight))
        fan_in = self.weight[0].numel()
        weight = weight_normalize(self.weight) / np.sqrt(fan_in)
        return F.linear(x, weight, None)


def pixel_norm(x: torch.FloatTensor, eps=1e-4, dim=1):
    return x / torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + eps)


def causal_mask(q: Tensor, k: Tensor) -> Tensor:
    b, i, j, device = q.shape[0], q.shape[-2], k.shape[-2], q.device
    mask = ~torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
    mask = repeat(mask, "n m -> b n m", b=b)
    return mask


def add_mask(sim: Tensor, mask: Tensor) -> Tensor:
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


def exists(val):
    return val is not None


class AttentionBase(nn.Module):

    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features**-0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads
        out_features = out_features  #default(out_features, features)

        self.to_out = nn.Linear(in_features=mid_features,
                                out_features=out_features)

        self.use_flash = torch.cuda.is_available() and version.parse(
            torch.__version__) >= version.parse('2.0.0')

        if not self.use_flash:
            return

        device_properties = torch.cuda.get_device_properties(
            torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # Use flash attention for A100 GPUs
            self.sdp_kernel_config = (True, False, False)
        else:
            # Don't use flash attention for other GPUs
            self.sdp_kernel_config = (False, True, True)

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v),
                                 "b n (h d) -> b h n d",
                                 h=self.num_heads)

        if not self.use_flash:
            if is_causal and not mask:
                # Mask out future tokens for causal attention
                mask = causal_mask(q, k)

            # Compute similarity matrix and add eventual mask
            sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale
            sim = add_mask(sim, mask) if mask is not None else sim

            # Get attention matrix with softmax
            attn = sim.softmax(dim=-1, dtype=torch.float32)

            # Compute values
            out = einsum("... n m, ... m d -> ... n d", attn, v)
        else:
            with sdp_kernel(*self.sdp_kernel_config):
                out = F.scaled_dot_product_attention(q,
                                                     k,
                                                     v,
                                                     attn_mask=mask,
                                                     is_causal=is_causal)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):

    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.context_features = context_features
        self.causal = causal
        mid_features = head_features * num_heads
        context_features = context_features if context_features is not None else features

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(in_features=features,
                              out_features=mid_features,
                              bias=False)
        self.to_kv = nn.Linear(in_features=context_features,
                               out_features=mid_features * 2,
                               bias=False)
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            out_features=out_features,
        )

    def forward(
        self,
        x: Tensor,  # [b, n, c]
        context: Optional[Tensor] = None,  # [b, m, d]
        context_mask: Optional[Tensor] = None,  # [b, m], false is masked,
        causal: Optional[bool] = False,
    ) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or context is not None, assert_message
        # Use context if provided
        context = context if context is not None else x
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)

        q, k, v = (self.to_q(x),
                   *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        if exists(context_mask):
            # Mask out cross-attention for padding tokens
            mask = repeat(context_mask, "b m -> b m d", d=v.shape[-1])
            k, v = k * mask, v * mask

        # Compute and return attention
        return self.attention(q, k, v, is_causal=self.causal or causal)


class SelfAttention1d(nn.Module):

    def __init__(self, c_in, n_head=1, dropout_rate=0.):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv1d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv1d(c_in, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        self.use_flash = False

        if not self.use_flash:
            return

        device_properties = torch.cuda.get_device_properties(
            torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # Use flash attention for A100 GPUs
            self.sdp_kernel_config = (True, False, False)
        else:
            # Don't use flash attention for other GPUs
            self.sdp_kernel_config = (False, True, True)

    def forward(self, input):
        n, c, s = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head,
                        s]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25

        if self.use_flash:
            with sdp_kernel(*self.sdp_kernel_config):
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=False).contiguous().view([n, c, s])
        else:
            att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
            y = (att @ v).transpose(2, 3).contiguous().view([n, c, s])

        return input + self.dropout(self.out_proj(y))


class CrossAttention1d(nn.Module):

    def __init__(self, c_in, context_in, n_head=1, dropout_rate=0.):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        #self.qkv_proj = nn.Conv1d(c_in, c_in * 3, 1)
        self.q_proj = nn.Conv1d(c_in, c_in * n_head, 1)
        self.kv_proj = nn.Conv1d(context_in, c_in * n_head * 2, 1)
        self.out_proj = nn.Conv1d(c_in * n_head, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        self.use_flash = False

        if not self.use_flash:
            return

        device_properties = torch.cuda.get_device_properties(
            torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # Use flash attention for A100 GPUs
            self.sdp_kernel_config = (True, False, False)
        else:
            # Don't use flash attention for other GPUs
            self.sdp_kernel_config = (False, True, True)

    def forward(self, input, context):
        n, c, s = input.shape
        s2 = context.shape[2]
        q = self.q_proj(self.norm(input))
        kv = self.kv_proj(context)
        q = q.view([n, self.n_head * 1, c, s]).transpose(2, 3)

        kv = kv.view([n, self.n_head * 2, c, s2]).transpose(2, 3)

        #q, k, v = qkv.chunk(3, dim=1)
        k, v = kv.chunk(2, dim=1)

        scale = k.shape[3]**-0.25

        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)

        y = (att @ v).transpose(2, 3).contiguous()

        y = rearrange(y, "b h n d -> b (h n) d")

        return input + self.dropout(self.out_proj(y))


class SelfAttention2d(nn.Module):

    def __init__(self, c_in, n_head=1, dropout_rate=0.):
        super().__init__()
        assert c_in % n_head == 0
        #self.norm = nn.GroupNorm(1, c_in)
        self.norm = nn.Identity()
        self.n_head = n_head
        self.k_proj = nn.Conv2d(c_in, c_in, 1)
        self.q_proj = nn.Conv2d(c_in, c_in, 1)
        self.v_proj = nn.Conv2d(c_in, c_in, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, h, w = input.shape
        # qkv = self.qkv_proj(self.norm(input))
        """qkv = qkv.view([n,  3, c // 8,
                        h * w]).transpose(2, 3)
        """
        q, k, v = self.q_proj(self.norm(input)), self.k_proj(
            self.norm(input)), self.v_proj(self.norm(input))

        q, k, v = q.reshape(n, -1,
                            h * w), k.reshape(n, -1,
                                              h * w), v.reshape(n, -1, h * w)

        #print(v.shape)
        scale = (h * w)**-0.25

        att = ((q * scale) @ (k.transpose(1, 2) * scale)).softmax(2)
        y = (att @ v).transpose(1, 2).contiguous()
        #.view([n, c, h * w])

        y = y.view([n, c, h, w])

        return input + self.dropout(self.out_proj(y))
