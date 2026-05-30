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
    pilot_len: int = 30                   # M
    index_mode: str = "uncentered"        

    # CNN gate network.
    cnn_channels: Tuple[int, int, int] = (32, 32, 48)
    cnn_kernel_size: int = 5              # odd, length-preserving padding
    cnn_dilation: int = 2
    hidden_dim: int = 40
    dropout: float = 0.0
    gate_max: float = 0.85                # beta_max, gate in [0, gate_max]

    # Bounded local update (trust region on the correction)
    trust_h: float = 0.25                 # kappa_h  : bound on |delta| for Re/Im h
    trust_phase: float = 0.08             # kappa_phi: bound on |delta| for phi
    damping: float = 1e-3                 # zeta in the damped Gauss-Newton solve

    # Synthetic training ranges
    h_abs_range: Tuple[float, float] = (0.02, 0.30)
    train_snr_db_range: Tuple[float, float] = (0.0, 30.0)
    train_phase_span_deg_range: Tuple[float, float] = (0.0, 160.0)

    # Optimization
    batch_size: int = 256
    learning_rate: float = 5e-4
    train_steps: int = 1400
    print_every: int = 200

    # Loss weights 
    w_phase: float = 1e-3
    w_recon: float = 5e-2
    w_delta: float = 8e-3

    # Evaluation
    eval_trials: int = 5000
    snr_grid_db: Tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30)
    span_grid_deg: Tuple[int, ...] = field(default_factory=lambda: tuple(range(5, 165, 10)))
    eval_span_for_snr_deg_range: Tuple[float, float] = (0.0, 120.0)
    eval_snr_for_span_db: float = 30.0
    failure_nmse_db: float = -20.0        # failure: NMSE > -20 dB


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
    """Antipodal BPSK pilot with equal +1 / -1 occupancy."""
    bits = np.tile([1, 0], int(np.ceil(cfg.pilot_len / 2)))[: cfg.pilot_len]
    return (2.0 * bits - 1.0).astype(np.complex128)


def pilot_index(cfg: Config) -> np.ndarray:
    n = np.arange(cfg.pilot_len, dtype=np.float64)
    if cfg.index_mode == "uncentered":
        return n
    if cfg.index_mode == "centered":
        return n - n.mean()
    raise ValueError("index_mode must be 'uncentered' or 'centered'")


# Model-based estimators

def lifted_ls(r: torch.Tensor, c: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Closed-form phase-aware (lifted) LS.

    Returns psi = [Re(h), Im(h), phi] from the 2x2 normal equations on the
    regressors {c[m], m * c[m]}.  h_hat = u0_hat,  phi_hat = Im(u1_hat / u0_hat).
    """
    e = torch.abs(c) ** 2
    s0 = torch.sum(e, dim=1)
    s1 = torch.sum(m * e, dim=1)
    s2 = torch.sum(m * m * e, dim=1)
    det = s0 * s2 - s1 * s1

    t0 = torch.sum(torch.conj(c) * r, dim=1)
    t1 = torch.sum(m * torch.conj(c) * r, dim=1)

    u0 = (s2 * t0 - s1 * t1) / (det + EPS)       # h_hat
    u1 = (s0 * t1 - s1 * t0) / (det + EPS)
    phi = torch.imag(u1 / (u0 + EPS))
    return torch.stack([torch.real(u0), torch.imag(u0), phi], dim=1)


def scalar_ls(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Conventional phase-stationary scalar LS (no phase term)."""
    num = torch.sum(torch.conj(c) * r, dim=1)
    den = torch.sum(torch.abs(c) ** 2, dim=1).clamp_min(EPS)
    h = num / den
    return torch.stack([torch.real(h), torch.imag(h), torch.zeros_like(torch.real(h))], dim=1)


def psi_to_rhat(psi: torch.Tensor, c: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    h = cplx(psi[:, 0], psi[:, 1])[:, None]
    phi = psi[:, 2][:, None]
    return h * torch.exp(1j * phi * m) * c


def gauss_newton_step(r: torch.Tensor, c: torch.Tensor, m: torch.Tensor,
                      psi: torch.Tensor, damping: float) -> torch.Tensor:
    """One damped Gauss-Newton step on the exact model (paper Eq.(8)-(9)).

    Jacobian columns: d r_hat / d (Re h, Im h, phi).
    """
    h = cplx(psi[:, 0], psi[:, 1])[:, None]
    phase = torch.exp(1j * psi[:, 2][:, None] * m)
    residual = r - h * phase * c

    J = torch.stack([
        phase * c,                 # d / d Re(h)
        1j * phase * c,            # d / d Im(h)
        1j * h * m * phase * c,    # d / d phi
    ], dim=2)

    JhJ = torch.real(torch.einsum("bmi,bmj->bij", torch.conj(J), J))
    JhR = torch.real(torch.einsum("bmi,bm->bi", torch.conj(J), residual))
    eye = torch.eye(3, device=r.device)[None]
    return torch.linalg.solve(JhJ + damping * eye, JhR[:, :, None]).squeeze(-1)


# Synthetic data generator

def synth_batch(cfg: Config, B: int,
                snr_db: float | None = None,
                phase_span_range: Tuple[float, float] | None = None,
                phase_span_deg: float | None = None,
                noise_free: bool = False) -> Dict[str, torch.Tensor]:
    c0 = pilot_symbols(cfg)
    m0 = pilot_index(cfg)
    mspan = max(float(m0.max() - m0.min()), 1.0)     # = M - 1

    C = np.tile(c0[None, :], (B, 1))
    Mm = np.tile(m0[None, :], (B, 1))

    log_lo, log_hi = np.log(max(cfg.h_abs_range[0], 1e-8)), np.log(cfg.h_abs_range[1])
    h_abs = np.exp(np.random.uniform(log_lo, log_hi, size=B))
    h = h_abs * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=B))

    if phase_span_deg is not None:
        spans = np.full(B, float(phase_span_deg))
    else:
        lo, hi = phase_span_range or cfg.train_phase_span_deg_range
        spans = np.random.uniform(lo, hi, size=B)
    signs = np.random.choice([-1.0, 1.0], size=B)
    phi = signs * np.deg2rad(spans) / mspan          # Phi = |phi|(M-1)

    clean = h[:, None] * np.exp(1j * phi[:, None] * Mm) * C

    if noise_free:
        r = clean.copy()
    else:
        snrs = np.full(B, snr_db) if snr_db is not None \
            else np.random.uniform(*cfg.train_snr_db_range, size=B)
        sigp = np.mean(np.abs(clean) ** 2, axis=1)
        nv = sigp / np.maximum(10.0 ** (snrs / 10.0), EPS)
        noise = np.sqrt(nv[:, None] / 2.0) * (
            np.random.randn(B, cfg.pilot_len) + 1j * np.random.randn(B, cfg.pilot_len))
        r = clean + noise

    psi = np.column_stack([h.real, h.imag, phi])
    dev = cfg.device
    return {
        "r": torch.tensor(r, dtype=torch.complex64, device=dev),
        "c": torch.tensor(C, dtype=torch.complex64, device=dev),
        "m": torch.tensor(Mm, dtype=torch.float32, device=dev),
        "clean": torch.tensor(clean, dtype=torch.complex64, device=dev),
        "psi": torch.tensor(psi, dtype=torch.float32, device=dev),
        "span_deg": torch.tensor(spans, dtype=torch.float32, device=dev),
    }


# Residual-gated CNN (RG-CNN)

class RGCNN(nn.Module):
    """Lifted-LS anchor + CNN-gated bounded Gauss-Newton correction."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        c1, c2, c3 = cfg.cnn_channels
        k, d = cfg.cnn_kernel_size, cfg.cnn_dilation
        if k % 2 == 0:
            raise ValueError("cnn_kernel_size must be odd")

        # Per-index features: Re/Im matched-filtered obs, Re/Im residual,
        # normalized index, active mask, local phase coordinate -> 7 channels.
        self.frame_dim = 6
        self.encoder = nn.Sequential(
            nn.Conv1d(7, c1, k, padding=k // 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Conv1d(c1, c2, k, padding=d * (k // 2), dilation=d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Conv1d(c2, c3, k, padding=d * (k // 2), dilation=d), nn.GELU(),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(2 * c3 + self.frame_dim),
            nn.Linear(2 * c3 + self.frame_dim, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        # Initialize the gate near zero, so optimization starts from the anchor.
        nn.init.zeros_(self.gate_head[-1].weight)
        gmax = max(cfg.gate_max, 1e-6)
        g0 = float(np.clip(0.35 * gmax, 1e-4, gmax - 1e-4))
        with torch.no_grad():
            self.gate_head[-1].bias.fill_(math.log(g0 / max(gmax - g0, 1e-6)))

    def _seq_features(self, r, c, m, psi0):
        z = torch.conj(c) * r
        residual = torch.conj(c) * (r - psi_to_rhat(psi0, c, m))
        scale = torch.sqrt(torch.mean(torch.abs(z) ** 2, dim=1, keepdim=True)).clamp_min(EPS)
        ms = torch.max(torch.abs(m), dim=1, keepdim=True).values.clamp_min(1.0)
        phase_coord = torch.clamp((psi0[:, 2:3] * m) / math.pi, -2.0, 2.0)
        active = (torch.abs(c) > 1e-12).float()
        return torch.stack([
            torch.real(z) / scale, torch.imag(z) / scale,
            torch.real(residual) / scale, torch.imag(residual) / scale,
            m / ms, active, phase_coord,
        ], dim=1).float()

    def _frame_features(self, r, c, m, psi0):
        r0 = psi_to_rhat(psi0, c, m)
        M = r.shape[1]
        py = (torch.mean(torch.abs(r) ** 2, dim=1)).clamp_min(EPS)
        pr = (torch.mean(torch.abs(r - r0) ** 2, dim=1)).clamp_min(EPS)
        h_abs = torch.sqrt(torch.sum(psi0[:, :2] ** 2, dim=1)).clamp_min(EPS)
        ms = torch.max(torch.abs(m), dim=1).values.clamp_min(1.0)
        span = torch.abs(psi0[:, 2]) * ms
        return torch.stack([
            psi0[:, 0] / h_abs, psi0[:, 1] / h_abs,
            torch.log10(h_abs + EPS),
            span / math.pi,
            torch.log10(pr / py + EPS),
            torch.log10(py / pr + EPS),
        ], dim=1).float()

    def forward(self, r, c, m) -> Dict[str, torch.Tensor]:
        psi0 = lifted_ls(r, c, m)

        delta = gauss_newton_step(r, c, m, psi0, self.cfg.damping)
        delta = torch.stack([
            torch.clamp(delta[:, 0], -self.cfg.trust_h, self.cfg.trust_h),
            torch.clamp(delta[:, 1], -self.cfg.trust_h, self.cfg.trust_h),
            torch.clamp(delta[:, 2], -self.cfg.trust_phase, self.cfg.trust_phase),
        ], dim=1)

        enc = self.encoder(self._seq_features(r, c, m, psi0))
        pooled = torch.cat([enc.mean(dim=2), enc.max(dim=2).values,
                            self._frame_features(r, c, m, psi0)], dim=1)
        gate = self.cfg.gate_max * torch.sigmoid(self.gate_head(pooled))

        psi = psi0 + gate * delta
        return {"psi": psi, "psi0": psi0, "gate": gate, "delta": delta,
                "rhat": psi_to_rhat(psi, c, m)}


# Training

def h_nmse(psi_hat: torch.Tensor, psi_true: torch.Tensor) -> torch.Tensor:
    num = torch.sum((psi_hat[:, :2] - psi_true[:, :2]) ** 2, dim=1)
    den = torch.sum(psi_true[:, :2] ** 2, dim=1).clamp_min(1e-10)
    return num / den


def loss_fn(out, batch, cfg: Config) -> torch.Tensor:
    h_loss = torch.mean(h_nmse(out["psi"], batch["psi"]))
    phase_loss = torch.mean((out["psi"][:, 2] - batch["psi"][:, 2]) ** 2)
    recon = torch.mean(torch.mean(torch.abs(out["rhat"] - batch["clean"]) ** 2, dim=1)
                       / torch.mean(torch.abs(batch["clean"]) ** 2, dim=1).clamp_min(EPS))
    # discourage unnecessary corrections (trust-normalized)
    dn = torch.stack([
        out["delta"][:, 0] / max(cfg.trust_h, EPS),
        out["delta"][:, 1] / max(cfg.trust_h, EPS),
        out["delta"][:, 2] / max(cfg.trust_phase, EPS),
    ], dim=1)
    delta_reg = torch.mean((out["gate"].squeeze(-1)[:, None] * dn) ** 2)
    return h_loss + cfg.w_phase * phase_loss + cfg.w_recon * recon + cfg.w_delta * delta_reg


def train(cfg: Config) -> RGCNN:
    model = RGCNN(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=2e-5)
    print(f"Training: steps={cfg.train_steps}, batch={cfg.batch_size}, device={cfg.device}")
    model.train()
    for step in range(1, cfg.train_steps + 1):
        batch = synth_batch(cfg, cfg.batch_size)
        out = model(batch["r"], batch["c"], batch["m"])
        loss = loss_fn(out, batch, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % cfg.print_every == 0 or step == cfg.train_steps:
            print(f"  step {step:5d}/{cfg.train_steps} | loss {loss.item():.4e}")
    return model.eval()


# Evaluation

ESTIMATORS = ("Conventional CE", "Lifted CE", "RG-CNN")


@torch.no_grad()
def _estimates(model, batch) -> Dict[str, torch.Tensor]:
    return {
        "Conventional CE": scalar_ls(batch["r"], batch["c"]),
        "Lifted CE": lifted_ls(batch["r"], batch["c"], batch["m"]),
        "RG-CNN": model(batch["r"], batch["c"], batch["m"])["psi"],
    }


@torch.no_grad()
def eval_vs_snr(cfg: Config, model) -> Dict[str, np.ndarray]:
    out = {name: [] for name in ESTIMATORS}
    for snr in cfg.snr_grid_db:
        batch = synth_batch(cfg, cfg.eval_trials, snr_db=float(snr),
                            phase_span_range=cfg.eval_span_for_snr_deg_range)
        for name, est in _estimates(model, batch).items():
            out[name].append(float(torch.mean(h_nmse(est, batch["psi"]))))
    return {name: db10(np.array(v)) for name, v in out.items()}


@torch.no_grad()
def eval_vs_span(cfg: Config, model, noise_free: bool = True) -> Dict[str, np.ndarray]:
    out = {name: [] for name in ESTIMATORS}
    for span in cfg.span_grid_deg:
        batch = synth_batch(cfg, cfg.eval_trials,
                            snr_db=None if noise_free else cfg.eval_snr_for_span_db,
                            phase_span_deg=float(span), noise_free=noise_free)
        for name, est in _estimates(model, batch).items():
            out[name].append(float(torch.mean(h_nmse(est, batch["psi"]))))
    return {name: db10(np.array(v)) for name, v in out.items()}


@torch.no_grad()
def eval_failure(cfg: Config, model) -> Dict[str, float]:
    """P_fail = Pr[NMSE > -20 dB] under large-drift stress."""
    batch = synth_batch(cfg, cfg.eval_trials, snr_db=cfg.eval_snr_for_span_db,
                        phase_span_range=(65.0, 160.0))
    thr = 10.0 ** (cfg.failure_nmse_db / 10.0)
    out = {}
    for name, est in _estimates(model, batch).items():
        out[name] = float(torch.mean((h_nmse(est, batch["psi"]) > thr).float()))
    return out


def plot(cfg: Config, snr_res, span_res, fail_res) -> None:
    if not _HAVE_MPL:
        print("matplotlib not available; skipping plots.")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    for name in ESTIMATORS:
        ax[0].plot(cfg.snr_grid_db, snr_res[name], marker="o", label=name)
    ax[0].set_xlabel("SNR [dB]"); ax[0].set_ylabel("CE NMSE [dB]")
    ax[0].set_title("NMSE vs SNR under residual phase drift")
    ax[0].grid(True, alpha=0.3); ax[0].legend()

    for name in ESTIMATORS:
        ax[1].plot(cfg.span_grid_deg, span_res[name], marker="s", label=name)
    ax[1].set_xlabel("Pilot phase span [deg]"); ax[1].set_ylabel("Error floor NMSE [dB]")
    ax[1].set_title("Deterministic NMSE floor vs phase span")
    ax[1].grid(True, alpha=0.3); ax[1].legend()

    names = list(ESTIMATORS)
    ax[2].barh(names, [fail_res[n] for n in names])
    ax[2].set_xlabel(r"$P_{\mathrm{fail}} = \Pr[\mathrm{NMSE} > -20\,\mathrm{dB}]$")
    ax[2].set_xlim(0, 1)
    ax[2].set_title("Tail-failure under large drift")
    for i, n in enumerate(names):
        ax[2].text(fail_res[n] + 0.01, i, f"{fail_res[n]:.2f}", va="center")

    fig.tight_layout()
    plt.show()


# Main

def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    model = train(cfg)

    print("\nEvaluating ...")
    snr_res = eval_vs_snr(cfg, model)
    span_res = eval_vs_span(cfg, model, noise_free=True)
    fail_res = eval_failure(cfg, model)

    print("\nCE NMSE [dB] vs SNR")
    print("  SNR | " + " | ".join(f"{n:>16s}" for n in ESTIMATORS))
    for i, snr in enumerate(cfg.snr_grid_db):
        print(f"  {snr:3d} | " + " | ".join(f"{snr_res[n][i]:+16.2f}" for n in ESTIMATORS))

    print("\nP_fail (NMSE > -20 dB) under large drift")
    for n in ESTIMATORS:
        print(f"  {n:18s}: {fail_res[n]:.3f}")

    plot(cfg, snr_res, span_res, fail_res)


if __name__ == "__main__":
    main()
