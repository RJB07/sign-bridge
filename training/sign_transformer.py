"""Small landmark transformer and a NumPy forward pass that matches eval mode."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff)
        self.ff2 = nn.Linear(ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attended)
        h = self.ln2(x)
        h = self.ff2(F.relu(self.ff1(h)))
        return x + self.drop(h)


class SignTransformer(nn.Module):
    def __init__(self, feature_dim: int, classes: int, d_model: int = 96, heads: int = 4, layers: int = 2, ff: int = 192, dropout: float = 0.1, seq_len: int = 32, stem: bool = False):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.ff = ff
        self.stem = stem
        self.proj = nn.Linear(feature_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        if stem:
            # Depthwise temporal filters stay cheap in the browser. Pointwise mixes channels.
            # Residual so the stem can start near the linear projection.
            self.dw1 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
            self.pw1 = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.dw2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
            self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.stem_ln = nn.LayerNorm(d_model)
            for conv in (self.dw1, self.pw1, self.dw2, self.pw2):
                nn.init.normal_(conv.weight, std=0.02)
                nn.init.zeros_(conv.bias)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, heads, ff, dropout) for _ in range(layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, classes)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        if self.stem:
            y = h.transpose(1, 2)
            y = F.relu(self.pw1(self.dw1(y)))
            y = F.relu(self.pw2(self.dw2(y)))
            h = h + self.stem_ln(y.transpose(1, 2))
        h = h + self.pos[:, : x.shape[1]]
        for block in self.blocks:
            h = block(h)
        h = self.final_ln(h).mean(dim=1)
        return self.fc(h)


def export_state(model: SignTransformer) -> dict[str, np.ndarray]:
    model.eval()
    weights: dict[str, np.ndarray] = {
        "proj.weight": model.proj.weight.detach().cpu().numpy(),
        "proj.bias": model.proj.bias.detach().cpu().numpy(),
        "pos": model.pos.detach().cpu().numpy()[0],
        "final_ln.weight": model.final_ln.weight.detach().cpu().numpy(),
        "final_ln.bias": model.final_ln.bias.detach().cpu().numpy(),
        "fc.weight": model.fc.weight.detach().cpu().numpy(),
        "fc.bias": model.fc.bias.detach().cpu().numpy(),
    }
    if model.stem:
        for name in ("dw1", "pw1", "dw2", "pw2"):
            conv = getattr(model, name)
            weights[f"{name}.weight"] = conv.weight.detach().cpu().numpy()
            weights[f"{name}.bias"] = conv.bias.detach().cpu().numpy()
        weights["stem_ln.weight"] = model.stem_ln.weight.detach().cpu().numpy()
        weights["stem_ln.bias"] = model.stem_ln.bias.detach().cpu().numpy()
    for index, block in enumerate(model.blocks):
        prefix = f"b{index}"
        weights[f"{prefix}.ln1.weight"] = block.ln1.weight.detach().cpu().numpy()
        weights[f"{prefix}.ln1.bias"] = block.ln1.bias.detach().cpu().numpy()
        weights[f"{prefix}.attn.in_weight"] = block.attn.in_proj_weight.detach().cpu().numpy()
        weights[f"{prefix}.attn.in_bias"] = block.attn.in_proj_bias.detach().cpu().numpy()
        weights[f"{prefix}.attn.out_weight"] = block.attn.out_proj.weight.detach().cpu().numpy()
        weights[f"{prefix}.attn.out_bias"] = block.attn.out_proj.bias.detach().cpu().numpy()
        weights[f"{prefix}.ln2.weight"] = block.ln2.weight.detach().cpu().numpy()
        weights[f"{prefix}.ln2.bias"] = block.ln2.bias.detach().cpu().numpy()
        weights[f"{prefix}.ff1.weight"] = block.ff1.weight.detach().cpu().numpy()
        weights[f"{prefix}.ff1.bias"] = block.ff1.bias.detach().cpu().numpy()
        weights[f"{prefix}.ff2.weight"] = block.ff2.weight.detach().cpu().numpy()
        weights[f"{prefix}.ff2.bias"] = block.ff2.bias.detach().cpu().numpy()
    return weights


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (x @ weight.T + bias).astype(np.float32)


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    return (y * weight + bias).astype(np.float32)


def _mha(x: np.ndarray, in_w: np.ndarray, in_b: np.ndarray, out_w: np.ndarray, out_b: np.ndarray, heads: int) -> np.ndarray:
    time, dim = x.shape
    qkv = _linear(x, in_w, in_b)
    q, k, v = np.split(qkv, 3, axis=-1)
    d_head = dim // heads
    def split(value: np.ndarray) -> np.ndarray:
        return value.reshape(time, heads, d_head).transpose(1, 0, 2)
    q, k, v = split(q), split(k), split(v)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d_head)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    mixed = np.matmul(weights, v).transpose(1, 0, 2).reshape(time, dim)
    return _linear(mixed.astype(np.float32), out_w, out_b)


def conv1d_same(seq: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """seq (T, C), weight (O, C, K), bias (O,) -> (T, O). Matches PyTorch Conv1d padding K//2."""
    time, _channels = seq.shape
    _out_ch, _in_ch, kernel = weight.shape
    pad = kernel // 2
    padded = np.pad(seq, ((pad, pad), (0, 0)))
    windows = np.stack([padded[index : index + kernel] for index in range(time)], axis=0)
    kernel_w = np.transpose(weight, (0, 2, 1))
    return (np.einsum("tkc,okc->to", windows, kernel_w) + bias).astype(np.float32)


def conv1d_depthwise(seq: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """seq (T, C), weight (C, 1, K), bias (C,) -> (T, C)."""
    time, _channels = seq.shape
    kernel = weight.shape[-1]
    pad = kernel // 2
    padded = np.pad(seq, ((pad, pad), (0, 0)))
    windows = np.stack([padded[index : index + kernel] for index in range(time)], axis=0)
    return (np.einsum("tkc,ck->tc", windows, weight[:, 0, :]) + bias).astype(np.float32)


def _apply_stem(h: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    if "dw1.weight" in weights:
        y = conv1d_depthwise(h, weights["dw1.weight"], weights["dw1.bias"])
        y = np.maximum(conv1d_same(y, weights["pw1.weight"], weights["pw1.bias"]), 0)
        y = conv1d_depthwise(y, weights["dw2.weight"], weights["dw2.bias"])
        y = np.maximum(conv1d_same(y, weights["pw2.weight"], weights["pw2.bias"]), 0)
        return h + _layer_norm(y, weights["stem_ln.weight"], weights["stem_ln.bias"])
    if "conv1.weight" in weights:
        y = np.maximum(conv1d_same(h, weights["conv1.weight"], weights["conv1.bias"]), 0)
        y = np.maximum(conv1d_same(y, weights["conv2.weight"], weights["conv2.bias"]), 0)
        return h + _layer_norm(y, weights["stem_ln.weight"], weights["stem_ln.bias"])
    return h


def forward_logits(seq: np.ndarray, weights: dict[str, np.ndarray], heads: int, layers: int) -> np.ndarray:
    h = _apply_stem(_linear(seq, weights["proj.weight"], weights["proj.bias"]), weights)
    h = h + weights["pos"][: seq.shape[0]]
    for index in range(layers):
        prefix = f"b{index}"
        normed = _layer_norm(h, weights[f"{prefix}.ln1.weight"], weights[f"{prefix}.ln1.bias"])
        h = h + _mha(
            normed,
            weights[f"{prefix}.attn.in_weight"],
            weights[f"{prefix}.attn.in_bias"],
            weights[f"{prefix}.attn.out_weight"],
            weights[f"{prefix}.attn.out_bias"],
            heads,
        )
        normed = _layer_norm(h, weights[f"{prefix}.ln2.weight"], weights[f"{prefix}.ln2.bias"])
        hidden = np.maximum(_linear(normed, weights[f"{prefix}.ff1.weight"], weights[f"{prefix}.ff1.bias"]), 0)
        h = h + _linear(hidden, weights[f"{prefix}.ff2.weight"], weights[f"{prefix}.ff2.bias"])
    h = _layer_norm(h, weights["final_ln.weight"], weights["final_ln.bias"]).mean(axis=0)
    return _linear(h[None, :], weights["fc.weight"], weights["fc.bias"])[0].astype(np.float32)


def parity_check(feature_dim: int = 273, classes: int = 5, stem: bool = False) -> float:
    torch.manual_seed(0)
    model = SignTransformer(feature_dim, classes, stem=stem)
    model.eval()
    sample = torch.randn(1, 32, feature_dim)
    with torch.no_grad():
        expected = model(sample).numpy()[0]
    actual = forward_logits(sample[0].numpy(), export_state(model), model.heads, len(model.blocks))
    return float(np.max(np.abs(expected - actual)))
