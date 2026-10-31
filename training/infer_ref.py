"""NumPy forward pass that mirrors the exported sequence model and the browser."""

from __future__ import annotations

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def conv1d_same(seq: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """seq (T, C), weight (O, C, 3), bias (O,) -> (T, O)."""
    t, channels = seq.shape
    out_ch = weight.shape[0]
    out = np.zeros((t, out_ch), dtype=np.float32)
    for time in range(t):
        for oc in range(out_ch):
            total = float(bias[oc])
            for ic in range(channels):
                for k in range(3):
                    src = time + k - 1
                    if src < 0 or src >= t:
                        continue
                    total += float(weight[oc, ic, k]) * float(seq[src, ic])
            out[time, oc] = total
    return out


def gru_mean(seq: np.ndarray, weight_ih: np.ndarray, weight_hh: np.ndarray, bias_ih: np.ndarray, bias_hh: np.ndarray) -> np.ndarray:
    """seq (T, H) -> mean hidden (H,). Gate order is reset, update, new."""
    hidden = weight_hh.shape[1]
    state = np.zeros(hidden, dtype=np.float32)
    outputs = []
    for time in range(seq.shape[0]):
        x = seq[time]
        gx = weight_ih @ x + bias_ih
        gh = weight_hh @ state + bias_hh
        r = _sigmoid(gx[:hidden] + gh[:hidden])
        z = _sigmoid(gx[hidden : 2 * hidden] + gh[hidden : 2 * hidden])
        n = np.tanh(gx[2 * hidden :] + r * gh[2 * hidden :])
        state = (1.0 - z) * n + z * state
        outputs.append(state.copy())
    return np.mean(np.stack(outputs), axis=0).astype(np.float32)


def forward_logits(seq: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    h = _relu(conv1d_same(seq, weights["conv1.weight"], weights["conv1.bias"]))
    h = _relu(conv1d_same(h, weights["conv2.weight"], weights["conv2.bias"]))
    pooled = gru_mean(
        h,
        weights["gru.weight_ih"],
        weights["gru.weight_hh"],
        weights["gru.bias_ih"],
        weights["gru.bias_hh"],
    )
    return (weights["fc.weight"] @ pooled + weights["fc.bias"]).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float32)
