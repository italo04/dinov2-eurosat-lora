"""diagnosticos.py — Norma de tokens y latencia fp16."""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def norma_tokens(m: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Calcula norma L2 por token del último bloque. Devuelve (B, N)."""
    m.eval()
    with torch.no_grad():
        out = m(x)
        # last_hidden_state: (B, N+1, D) — incluye CLS
        h = out.last_hidden_state  # (B, N+1, D)
        normas = h.norm(dim=-1)    # (B, N+1)
    return normas.cpu().numpy()


def medir_latencia(
    m: nn.Module,
    batch: int,
    img_size: int,
    device: str,
    reps: int = 20,
    warmup: int = 5,
    amp: bool = False,
) -> dict:
    """Mide latencia media de forward en ms."""
    m.eval()
    m.to(device)
    x = torch.randn(batch, 3, img_size, img_size, device=device)
    use_amp = amp and device == "cuda" and torch.cuda.is_available()

    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast("cuda"):
                    m(x)
            else:
                m(x)
        tiempos = []
        for _ in range(reps):
            t0 = time.perf_counter()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    m(x)
            else:
                m(x)
            if device == "cuda":
                torch.cuda.synchronize()
            tiempos.append((time.perf_counter() - t0) * 1000)

    return {
        "batch": batch,
        "img_size": img_size,
        "device": device,
        "amp": amp,
        "latencia_media_ms": float(np.mean(tiempos)),
        "latencia_std_ms": float(np.std(tiempos)),
    }


def remedir_fp16_b1(m: nn.Module, device: str) -> pd.DataFrame:
    """Re-mide latencia fp16 con batch 1 y distintas condiciones."""
    filas = []
    for warmup in [5, 50]:
        for amp in [False, True]:
            res = medir_latencia(m, batch=1, img_size=224, device=device,
                                 reps=10, warmup=warmup, amp=amp)
            res["warmup"] = warmup
            filas.append(res)
    return pd.DataFrame(filas)
