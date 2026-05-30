#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

EPS = 1e-12


# Configuration

@dataclass
class Config:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Pilot block.
    pilot_len: int = 32
    index_mode: str = "centered"          # "centered" (s1 = 0) or "uncentered"

    # CNN refiner.
    cnn_channels: Tuple[int, int] = (32, 32)
    cnn_kernel_size: int = 5              # odd, length-preserving padding
    hidden_dim: int = 32
    dropout: float = 0.0
    gate_max: float = 1.0                 # maximum gate value in [0, gate_max]

    # Bounded local update (trust region on the correction).
    trust_h: float = 0.25                 # bound on |delta| for Re/Im of h
    trust_phase: float = 0.05             # bound on |delta| for phi
    damping: float = 1e-3                 # Levenberg-style damping in the solve

    # Synthetic training ranges.
    h_abs_range: Tuple[float, float] = (0.05, 0.30)
    train_snr_db_range: Tuple[float, float] = (0.0, 30.0)
    train_phase_span_deg_range: Tuple[float, float] = (0.0, 150.0)

    # Optimization.
    batch_size: int = 256
    learning_rate: float = 5e-4
    train_steps: int = 800
    print_every: int = 100

    # Loss weights.
    w_phase: float = 1e-3
    w_recon: float = 5e-2

    # Evaluation.
    eval_trials: int = 5000
    snr_grid_db: Tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30)
    span_grid_deg: Tuple[int, ...] = field(default_factory=lambda: tuple(range(0, 160, 10)))
    eval_snr_for_span_db: float = 30.0
    eval_span_for_snr_deg_range: Tuple[float, float] = (0.0, 120.0)

    save_figs: bool = True


# Utilities

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def db10(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), EPS))


def cplx(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.complex(a, b)


# Pilot model

def pilot_symbols(cfg: Config) -> np.ndarray:
    """Antipodal pilot sequence with equal +1 / -1 occupancy."""
    bits = np.tile([1, 0], int(np.ceil(cfg.pilot_len / 2)))[: cfg.pilot_len]
    return (2.0 * bits - 1.0).astype(np.complex128)


def pilot_index(cfg: Config) -> np.ndarray:
    n = np.arange(cfg.pilot_len, dtype=np.float64)
    if cfg.index_mode == "centered":
        return n - n.mean()
    if cfg.index_mode == "uncentered":
        return n
    raise ValueError("index_mode must be 'centered' or 'uncentered'")


# Lifted phase-aware least squares (closed form) — initial estimate

def lifted_ls(y: torch.Tensor, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Two-parameter phase-aware LS.

    Returns eta = [Re(h), Im(h), phi] from the 2x2 normal equations built
    on the regressors {x[n], n * x[n]}.
    """
    e = torch.abs(x) ** 2
    s0 = torch.sum(e, dim=1)
    s1 = torch.sum(n * e, dim=1)
    s2 = torch.sum(n * n * e, dim=1)
    det = s0 * s2 - s1 * s1

    t0 = torch.sum(torch.conj(x) * y, dim=1)
    t1 = torch.sum(n * torch.conj(x) * y, dim=1)

    theta0 = (s2 * t0 - s1 * t1) / (det + EPS)      # h_hat
    theta1 = (s0 * t1 - s1 * t0) / (det + EPS)
    phi = torch.imag(theta1 / (theta0 + EPS))        # residual phase-slope diagnostic
    return torch.stack([torch.real(theta0), torch.imag(theta0), phi], dim=1)


def scalar_ls(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Conventional phase-stationary scalar LS (no phase term)."""
    num = torch.sum(torch.conj(x) * y, dim=1)
    den = torch.sum(torch.abs(x) ** 2, dim=1).clamp_min(EPS)
    h = num / den
    return torch.stack([torch.real(h), torch.imag(h), torch.zeros_like(torch.real(h))], dim=1)


def eta_to_yhat(eta: torch.Tensor, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    h = cplx(eta[:, 0], eta[:, 1])[:, None]
    phi = eta[:, 2][:, None]
    return h * torch.exp(1j * phi * n) * x


def local_update(y: torch.Tensor, x: torch.Tensor, n: torch.Tensor,
                 eta: torch.Tensor, damping: float) -> torch.Tensor:

    h = cplx(eta[:, 0], eta[:, 1])[:, None]
    phase = torch.exp(1j * eta[:, 2][:, None] * n)
    residual = y - h * phase * x

    basis = torch.stack([
        phase * x,                 # d/d Re(h)
        1j * phase * x,            # d/d Im(h)
        1j * h * n * phase * x,    # d/d phi
    ], dim=2)

    lhs = torch.real(torch.einsum("bni,bnj->bij", torch.conj(basis), basis))
    rhs = torch.real(torch.einsum("bni,bn->bi", torch.conj(basis), residual))
    eye = torch.eye(3, device=y.device)[None]
    system = lhs + damping * eye
    return torch.linalg.solve(system, rhs[:, :, None]).squeeze(-1)


# Synthetic data generator

def synth_batch(cfg: Config, B: int,
                snr_db: float | None = None,
                phase_span_range: Tuple[float, float] | None = None,
                phase_span_deg: float | None = None,
                noise_free: bool = False) -> Dict[str, torch.Tensor]:
    x0 = pilot_symbols(cfg)
    n0 = pilot_index(cfg)
    nrange = max(float(n0.max() - n0.min()), 1.0)

    X = np.tile(x0[None, :], (B, 1))
    N = np.tile(n0[None, :], (B, 1))

    h_abs = np.exp(np.random.uniform(*np.log(np.maximum(cfg.h_abs_range, 1e-8)), size=B))
    h = h_abs * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=B))

    if phase_span_deg is not None:
        spans = np.full(B, float(phase_span_deg))
    else:
        lo, hi = phase_span_range or cfg.train_phase_span_deg_range
        spans = np.random.uniform(lo, hi, size=B)
    signs = np.random.choice([-1.0, 1.0], size=B)
    phi = signs * np.deg2rad(spans) / nrange     # total span maps to per-sample slope

    clean = h[:, None] * np.exp(1j * phi[:, None] * N) * X

    if noise_free:
        y = clean.copy()
    else:
        snrs = np.full(B, snr_db) if snr_db is not None \
            else np.random.uniform(*cfg.train_snr_db_range, size=B)
        sigp = np.mean(np.abs(clean) ** 2, axis=1)
        nv = sigp / np.maximum(10.0 ** (snrs / 10.0), EPS)
        noise = np.sqrt(nv[:, None] / 2.0) * (
            np.random.randn(B, cfg.pilot_len) + 1j * np.random.randn(B, cfg.pilot_len))
        y = clean + noise

    eta = np.column_stack([h.real, h.imag, phi])
    dev = cfg.device
    return {
        "y": torch.tensor(y, dtype=torch.complex64, device=dev),
        "x": torch.tensor(X, dtype=torch.complex64, device=dev),
        "n": torch.tensor(N, dtype=torch.float32, device=dev),
        "clean": torch.tensor(clean, dtype=torch.complex64, device=dev),
        "eta": torch.tensor(eta, dtype=torch.float32, device=dev),
        "span_deg": torch.tensor(spans, dtype=torch.float32, device=dev),
    }


# Residual-gated CNN refiner

class ResidualGatedRefiner(nn.Module):
    """Lifted-LS initialization + CNN-gated bounded local update."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        c1, c2 = cfg.cnn_channels
        k = cfg.cnn_kernel_size
        if k % 2 == 0:
            raise ValueError("cnn_kernel_size must be odd")

        # Input channels: Re/Im of matched-filtered y, Re/Im of residual, index.
        self.encoder = nn.Sequential(
            nn.Conv1d(5, c1, k, padding=k // 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Conv1d(c1, c2, k, padding=k // 2), nn.GELU(),
        )
        # Pooled (mean + max) features feed a small head producing a scalar gate.
        self.gate_head = nn.Sequential(
            nn.LayerNorm(2 * c2),
            nn.Linear(2 * c2, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def _features(self, y, x, n, eta0):
        z = torch.conj(x) * y
        residual = torch.conj(x) * (y - eta_to_yhat(eta0, x, n))
        scale = torch.sqrt(torch.mean(torch.abs(z) ** 2, dim=1, keepdim=True)).clamp_min(EPS)
        ns = torch.max(torch.abs(n), dim=1, keepdim=True).values.clamp_min(1.0)
        return torch.stack([
            torch.real(z) / scale, torch.imag(z) / scale,
            torch.real(residual) / scale, torch.imag(residual) / scale,
            n / ns,
        ], dim=1).float()

    def forward(self, y, x, n) -> Dict[str, torch.Tensor]:
        eta0 = lifted_ls(y, x, n)

        delta = local_update(y, x, n, eta0, self.cfg.damping)
        delta = torch.stack([
            torch.clamp(delta[:, 0], -self.cfg.trust_h, self.cfg.trust_h),
            torch.clamp(delta[:, 1], -self.cfg.trust_h, self.cfg.trust_h),
            torch.clamp(delta[:, 2], -self.cfg.trust_phase, self.cfg.trust_phase),
        ], dim=1)

        enc = self.encoder(self._features(y, x, n, eta0))
        pooled = torch.cat([enc.mean(dim=2), enc.max(dim=2).values], dim=1)
        gate = self.cfg.gate_max * torch.sigmoid(self.gate_head(pooled))

        eta = eta0 + gate * delta
        return {"eta": eta, "eta0": eta0, "gate": gate, "yhat": eta_to_yhat(eta, x, n)}


# Training

def h_nmse(eta_hat: torch.Tensor, eta_true: torch.Tensor) -> torch.Tensor:
    num = torch.sum((eta_hat[:, :2] - eta_true[:, :2]) ** 2, dim=1)
    den = torch.sum(eta_true[:, :2] ** 2, dim=1).clamp_min(1e-10)
    return num / den


def loss_fn(out, batch, cfg: Config) -> torch.Tensor:
    h_loss = torch.mean(h_nmse(out["eta"], batch["eta"]))
    phase_loss = torch.mean((out["eta"][:, 2] - batch["eta"][:, 2]) ** 2)
    recon = torch.mean(torch.mean(torch.abs(out["yhat"] - batch["clean"]) ** 2, dim=1)
                       / torch.mean(torch.abs(batch["clean"]) ** 2, dim=1).clamp_min(EPS))
    return h_loss + cfg.w_phase * phase_loss + cfg.w_recon * recon


def train(cfg: Config) -> ResidualGatedRefiner:
    model = ResidualGatedRefiner(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=2e-5)
    print(f"Training: steps={cfg.train_steps}, batch={cfg.batch_size}, device={cfg.device}")
    model.train()
    for step in range(1, cfg.train_steps + 1):
        batch = synth_batch(cfg, cfg.batch_size)
        out = model(batch["y"], batch["x"], batch["n"])
        loss = loss_fn(out, batch, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % cfg.print_every == 0 or step == cfg.train_steps:
            print(f"  step {step:5d}/{cfg.train_steps} | loss {loss.item():.4e}")
    return model.eval()


ESTIMATORS = ("Scalar LS", "Lifted LS", "CNN refinement")


@torch.no_grad()
def _estimates(model, batch) -> Dict[str, torch.Tensor]:
    return {
        "Scalar LS": scalar_ls(batch["y"], batch["x"]),
        "Lifted LS": lifted_ls(batch["y"], batch["x"], batch["n"]),
        "CNN refinement": model(batch["y"], batch["x"], batch["n"])["eta"],
    }


@torch.no_grad()
def eval_vs_snr(cfg: Config, model) -> Dict[str, np.ndarray]:
    out = {name: [] for name in ESTIMATORS}
    for snr in cfg.snr_grid_db:
        batch = synth_batch(cfg, cfg.eval_trials, snr_db=float(snr),
                            phase_span_range=cfg.eval_span_for_snr_deg_range)
        for name, est in _estimates(model, batch).items():
            out[name].append(float(torch.mean(h_nmse(est, batch["eta"]))))
    return {name: db10(np.array(v)) for name, v in out.items()}


@torch.no_grad()
def eval_vs_span(cfg: Config, model) -> Dict[str, np.ndarray]:
    out = {name: [] for name in ESTIMATORS}
    for span in cfg.span_grid_deg:
        batch = synth_batch(cfg, cfg.eval_trials, snr_db=cfg.eval_snr_for_span_db,
                            phase_span_deg=float(span))
        for name, est in _estimates(model, batch).items():
            out[name].append(float(torch.mean(h_nmse(est, batch["eta"]))))
    return {name: db10(np.array(v)) for name, v in out.items()}


def plot(cfg: Config, snr_res, span_res) -> None:
    if not _HAVE_MPL:
        print("matplotlib not available; skipping plots.")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    for name in ESTIMATORS:
        ax[0].plot(cfg.snr_grid_db, snr_res[name], marker="o", label=name)
    ax[0].set_xlabel("SNR [dB]"); ax[0].set_ylabel("h-NMSE [dB]")
    ax[0].set_title(f"NMSE vs SNR (phase span "
                    f"{cfg.eval_span_for_snr_deg_range[0]:.0f}-"
                    f"{cfg.eval_span_for_snr_deg_range[1]:.0f} deg)")
    ax[0].grid(True, alpha=0.3); ax[0].legend()

    for name in ESTIMATORS:
        ax[1].plot(cfg.span_grid_deg, span_res[name], marker="s", label=name)
    ax[1].set_xlabel("Pilot phase span [deg]"); ax[1].set_ylabel("h-NMSE [dB]")
    ax[1].set_title(f"NMSE vs phase span (SNR {cfg.eval_snr_for_span_db:.0f} dB)")
    ax[1].grid(True, alpha=0.3); ax[1].legend()

    fig.tight_layout()
    if cfg.save_figs:
        fig.savefig("phase_aware_cnn_refinement.png", dpi=200, bbox_inches="tight")
        print("saved figure: phase_aware_cnn_refinement.png")
    plt.show()


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    model = train(cfg)

    print("\nEvaluating ...")
    snr_res = eval_vs_snr(cfg, model)
    span_res = eval_vs_span(cfg, model)

    print("\nh-NMSE [dB] vs SNR")
    header = "  SNR | " + " | ".join(f"{n:>15s}" for n in ESTIMATORS)
    print(header)
    for i, snr in enumerate(cfg.snr_grid_db):
        row = " | ".join(f"{snr_res[n][i]:+15.2f}" for n in ESTIMATORS)
        print(f"  {snr:3d} | {row}")

    plot(cfg, snr_res, span_res)


if __name__ == "__main__":
    main()
