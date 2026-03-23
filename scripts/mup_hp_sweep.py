"""
muP Hyperparameter Sweep for nanochat

Systematic HP sweep script for muP models, informed by:
- Yang et al., "Tensor Programs V" (arXiv:2203.03466) — swept 6 HPs for GPT-3
  using random search on a 256-width proxy
- Muon optimizer literature: only LR and WD matter (momentum, ns_steps insensitive)
- nanochat's existing mup_coord_check.py and mup_transfer_check.py patterns

Runs on a single GPU. Default width 256 (base width) with production-matching depth=24,
bf16, torch.compile, and full LR/WD/momentum schedules from base_train.py.

Usage:
    # Primary use case: 2D LR × WD grid sweep (~15 min)
    python -m scripts.mup_hp_sweep --sweep-2d --save-dir /tmp/sweep

    # 1D sweeps (~3 min each)
    python -m scripts.mup_hp_sweep --sweep-lr
    python -m scripts.mup_hp_sweep --sweep-wd
    python -m scripts.mup_hp_sweep --sweep-init-scale
    python -m scripts.mup_hp_sweep --sweep-attn-temp
    python -m scripts.mup_hp_sweep --sweep-emb-mult
    python -m scripts.mup_hp_sweep --sweep-output-temp
    python -m scripts.mup_hp_sweep --sweep-beta2

    # Compare SP vs muP side-by-side
    python -m scripts.mup_hp_sweep --sweep-2d --compare
    python -m scripts.mup_hp_sweep --sweep-init-scale --compare

    # Custom widths and training budget
    python -m scripts.mup_hp_sweep --sweep-lr --widths 128,256,512 --steps 200
"""

import argparse
import json
import math
import os
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nanochat.gpt import GPT, GPTConfig


# ─── Default sweep ranges ─────────────────────────────────────────────────────

DEFAULT_LR_VALUES = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
DEFAULT_WD_VALUES = [0.0, 0.03, 0.06, 0.1, 0.2, 0.4]
DEFAULT_INIT_SCALE_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
DEFAULT_ATTN_TEMP_VALUES = [0.5, 0.75, 1.0, 1.44, 2.0, 3.0, 4.0]
DEFAULT_EMB_MULT_VALUES = [0.5, 1.0, 2.0, 4.0, 7.0, 10.0]
DEFAULT_OUTPUT_TEMP_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_BETA2_VALUES = [0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]

DEFAULT_WIDTHS = [256, 512, 1024, 2048, 4096]


# ─── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    widths: List[int] = field(default_factory=lambda: list(DEFAULT_WIDTHS))
    steps: int = 300
    batch_size: int = 32
    seq_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 24
    head_dim: int = 128
    aspect_ratio: int = 64
    window_pattern: str = "SSSL"
    seed: int = 42
    base_width: int = 256
    use_mup: bool = True
    # Base HPs (tuned at base_width=256, matching nanochat production defaults)
    matrix_lr: float = 0.02
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    scalar_lr: float = 0.5
    weight_decay: float = 0.28
    muon_lr_exponent: float = 0.0
    # Scaling (matching base_train.py auto-compute logic)
    target_param_data_ratio: float = 10.5
    # LR schedule (matching base_train.py)
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    # Gradient accumulation
    grad_accum_steps: int = 1


# ─── Data loading ──────────────────────────────────────────────────────────────

_data_loader_initialized = False

def create_data_loader(batch_size: int, seq_len: int, device: torch.device):
    """Create a streaming data loader from the nanochat training pipeline.
    Returns (loader_iterator, vocab_size).
    Falls back to infinite random data if the tokenizer/dataset isn't available."""
    global _data_loader_initialized
    try:
        from nanochat.tokenizer import get_tokenizer
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, batch_size, seq_len, split="train", device=device,
        )
        if not _data_loader_initialized:
            print(f"Streaming real training data "
                  f"(vocab_size={vocab_size}, seq_len={seq_len}, batch_size={batch_size}, "
                  f"{batch_size * seq_len:,} tokens/batch)")
            _data_loader_initialized = True
        return loader, vocab_size
    except Exception as e:
        if not _data_loader_initialized:
            print(f"Could not load training data ({e}), using random tokens")
            _data_loader_initialized = True
        vocab_size = 32768
        def random_loader():
            i = 0
            while True:
                rng = torch.Generator(device=device)
                rng.manual_seed(42 + i)
                x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=rng)
                y = torch.roll(x, -1, dims=1)
                y[:, -1] = -1
                yield x, y
                i += 1
        return random_loader(), vocab_size


# ─── Batch/depth scaling (matching base_train.py lines 250-306) ───────────────

def compute_scaling(config: SweepConfig):
    """Compute batch_lr_scale and weight_decay_scaled matching base_train.py.
    Returns (batch_lr_scale, weight_decay_scaled, total_batch_size_tokens, num_iterations).
    """
    # Build a reference model on meta device to get scaling params
    def _build_meta(depth, width=None):
        if width is None:
            base_dim = depth * config.aspect_ratio
        else:
            base_dim = width
        model_dim = ((base_dim + config.head_dim - 1) // config.head_dim) * config.head_dim
        n_head = model_dim // config.head_dim
        gpt_config = GPTConfig(
            sequence_len=config.seq_len, vocab_size=config.vocab_size,
            n_layer=depth, n_head=n_head, n_kv_head=n_head, n_embd=model_dim,
            window_pattern=config.window_pattern,
        )
        with torch.device('meta'):
            m = GPT(gpt_config)
        return m

    def _get_scaling_params(m):
        pc = m.num_scaling_params()
        return pc['transformer_matrices'] + pc['lm_head']

    # Our target model: depth=n_layer, width=base_width (the sweep proxy width)
    target_model = _build_meta(config.n_layer, width=config.base_width)
    num_scaling_params = _get_scaling_params(target_model)
    target_tokens = int(config.target_param_data_ratio * num_scaling_params)

    # d12 reference (where HPs are tuned in nanochat)
    d12_ref = _build_meta(12)
    D_REF = config.target_param_data_ratio * _get_scaling_params(d12_ref)
    B_REF = 2**19  # 524288

    # Auto-compute optimal batch size (Power Lines: B_opt ∝ D^0.383)
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print(f"Scaling: {num_scaling_params:,} scaling params, {target_tokens:,} target tokens")
    print(f"Auto-computed optimal total batch size: {total_batch_size:,} tokens")

    # Batch LR scaling: η ∝ √(B/B_ref)
    batch_lr_scale = (total_batch_size / B_REF) ** 0.5
    if batch_lr_scale != 1.0:
        print(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (ref: {B_REF:,})")

    # WD scaling: T_epoch framework — λ = λ_ref · √(B/B_ref) · (D_ref/D)
    weight_decay_scaled = config.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    if weight_decay_scaled != config.weight_decay:
        print(f"Scaling weight decay from {config.weight_decay:.6f} to {weight_decay_scaled:.6f}")

    # Number of iterations
    num_iterations = target_tokens // total_batch_size
    print(f"Auto-computed num_iterations: {num_iterations:,}")

    # Auto-compute grad_accum_steps to match total_batch_size
    device_batch_tokens = config.batch_size * config.seq_len
    grad_accum_steps = max(1, total_batch_size // device_batch_tokens)
    actual_total_batch = grad_accum_steps * device_batch_tokens
    if actual_total_batch != total_batch_size:
        print(f"Note: actual total_batch_size={actual_total_batch:,} (grad_accum={grad_accum_steps})")
    else:
        print(f"grad_accum_steps={grad_accum_steps} to match total_batch_size={total_batch_size:,}")

    return batch_lr_scale, weight_decay_scaled, total_batch_size, num_iterations, grad_accum_steps


# ─── LR / WD / Momentum schedules (matching base_train.py) ────────────────────

def get_lr_multiplier(step, num_steps, warmup_steps, warmdown_ratio, final_lr_frac):
    """Linear warmup, constant, linear warmdown."""
    warmdown_iters = round(warmdown_ratio * num_steps)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step <= num_steps - warmdown_iters:
        return 1.0
    else:
        progress = (num_steps - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


def get_muon_momentum(step, num_steps, warmdown_ratio):
    """Muon momentum schedule: warmup 0.85→0.97 (400 steps), warmdown 0.97→0.90."""
    warmdown_iters = round(warmdown_ratio * num_steps)
    warmdown_start = num_steps - warmdown_iters
    if step < 400:
        frac = step / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif step >= warmdown_start:
        progress = (step - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


def get_weight_decay_mult(step, num_steps):
    """Cosine decay to zero over training."""
    return 0.5 * (1 + math.cos(math.pi * step / num_steps))


# ─── Model creation & training ─────────────────────────────────────────────────

def create_model(width: int, config: SweepConfig, device: torch.device,
                 mup_base_width: int = 0, init_scale: float = 1.0,
                 attn_temp: float = 1.44, emb_mult: float = 1.0,
                 output_temp: float = 1.0):
    """Create a model with the specified width and optional HP overrides."""
    head_dim = config.head_dim
    n_head = max(1, width // head_dim)
    actual_width = n_head * head_dim

    gpt_config = GPTConfig(
        sequence_len=config.seq_len,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=actual_width,
        window_pattern=config.window_pattern,
        mup_base_width=mup_base_width,
        attn_temp=attn_temp,
        emb_mult=emb_mult,
        output_temp=output_temp,
    )

    with torch.device('meta'):
        model = GPT(gpt_config)
    model.to_empty(device=device)
    model.init_weights()

    # Apply init_scale: multiply all parameter inits by scalar
    if init_scale != 1.0:
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(init_scale)

    return model, gpt_config


def train_model(width: int, config: SweepConfig, device: torch.device,
                data_loader, batch_lr_scale: float = 1.0,
                weight_decay_scaled: float = 0.28,
                lr_mult: float = 1.0, wd_override: Optional[float] = None,
                init_scale: float = 1.0, attn_temp: float = 1.44,
                emb_mult: float = 1.0, output_temp: float = 1.0,
                beta2_override: Optional[float] = None) -> Tuple[List[float], int]:
    """Train a model at given width with HP overrides, return (losses, actual_width).
    Returns float('inf') as final loss if NaN is encountered.

    LRs are: base_lr * batch_lr_scale * lr_mult
    WD is: wd_override (if given) else weight_decay_scaled (already batch/depth scaled)
    """
    torch.manual_seed(config.seed)

    mup_base_width = config.base_width if config.use_mup else 0
    model, gpt_config = create_model(
        width, config, device, mup_base_width=mup_base_width,
        init_scale=init_scale, attn_temp=attn_temp,
        emb_mult=emb_mult, output_temp=output_temp,
    )
    actual_width = gpt_config.n_embd

    # Note: we skip torch.compile here — it doesn't affect numerics, only speed,
    # and the ~2 min compilation overhead per run dominates the ~30s training time.
    # Production uses torch.compile but for short HP sweeps it's counterproductive.

    # WD: if sweeping WD, use wd_override directly; otherwise use pre-scaled value
    wd = wd_override if wd_override is not None else weight_decay_scaled

    # LRs: base * batch_lr_scale * lr_mult (batch_lr_scale from scaling laws, lr_mult from sweep)
    effective_lr_scale = batch_lr_scale * lr_mult
    optimizer = model.setup_optimizer(
        unembedding_lr=config.unembedding_lr * effective_lr_scale,
        embedding_lr=config.embedding_lr * effective_lr_scale,
        scalar_lr=config.scalar_lr * effective_lr_scale,
        matrix_lr=config.matrix_lr * effective_lr_scale,
        weight_decay=wd,
        use_mup=config.use_mup,
        base_width=config.base_width,
        muon_lr_exponent=config.muon_lr_exponent,
    )

    # Patch beta2 for Muon groups if requested
    if beta2_override is not None:
        for group in optimizer.param_groups:
            if group.get('kind') == 'muon':
                group['beta2'] = beta2_override

    model.train()
    losses = []
    grad_accum = config.grad_accum_steps

    for step in range(config.steps):
        # Gradient accumulation loop
        accumulated_loss = 0.0
        for micro_step in range(grad_accum):
            x, y = next(data_loader)
            loss = model(x, y)
            loss_val = loss.detach().item()
            if not np.isfinite(loss_val):
                losses.append(float('inf'))
                del model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return losses, actual_width
            accumulated_loss += loss_val
            (loss / grad_accum).backward()

        losses.append(accumulated_loss / grad_accum)

        # Apply LR / momentum / WD schedules (matching base_train.py lines 525-532)
        lrm = get_lr_multiplier(step, config.steps, config.warmup_steps,
                                config.warmdown_ratio, config.final_lr_frac)
        mom = get_muon_momentum(step, config.steps, config.warmdown_ratio)
        wd_mult = get_weight_decay_mult(step, config.steps)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group.get('kind') == 'muon':
                group["momentum"] = mom
                group["weight_decay"] = wd * wd_mult

        optimizer.step()
        model.zero_grad(set_to_none=True)

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch._dynamo.reset()

    return losses, actual_width


# ─── Sweep runners ──────────────────────────────────────────────────────────────

def run_2d_sweep(config: SweepConfig, device: torch.device,
                 batch_lr_scale: float, weight_decay_scaled: float,
                 lr_values: List[float], wd_values: List[float]) -> Dict:
    """Run LR × WD grid at all widths. Returns structured results."""
    results = {
        'widths': [],
        'lr_values': lr_values,
        'wd_values': wd_values,
        'final_losses': {},    # final_losses[(width, lr, wd)] = float
        'all_losses': {},      # all_losses[(width, lr, wd)] = [loss_step0, ...]
    }

    total = len(config.widths) * len(lr_values) * len(wd_values)
    done = 0
    t0 = time.time()

    for width in config.widths:
        actual_width = None
        for lr in lr_values:
            for wd in wd_values:
                done += 1
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(f"  [{done}/{total}] width={width}, lr={lr}, wd={wd}...",
                      end=" ", flush=True)

                data_loader, _ = create_data_loader(config.batch_size, config.seq_len, device)
                losses, actual_width = train_model(
                    width, config, device, data_loader,
                    batch_lr_scale=batch_lr_scale,
                    weight_decay_scaled=weight_decay_scaled,
                    lr_mult=lr / config.matrix_lr,  # normalize: sweep value IS the absolute Muon LR
                    wd_override=wd,
                )
                final = losses[-1]
                results['final_losses'][(actual_width, lr, wd)] = final
                results['all_losses'][(actual_width, lr, wd)] = losses
                print(f"final_loss={final:.4f} (eta: {eta/60:.1f}m)")

        if actual_width is not None and actual_width not in results['widths']:
            results['widths'].append(actual_width)

    return results


def run_1d_sweep(config: SweepConfig, device: torch.device,
                 batch_lr_scale: float, weight_decay_scaled: float,
                 hp_name: str, hp_values: List[float]) -> Dict:
    """Run a sweep over a single HP at all widths."""
    results = {
        'widths': [],
        'hp_name': hp_name,
        'hp_values': hp_values,
        'final_losses': defaultdict(dict),
        'all_losses': {},
    }

    total = len(config.widths) * len(hp_values)
    done = 0
    t0 = time.time()

    for width in config.widths:
        actual_width = None
        for hp_val in hp_values:
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            print(f"  [{done}/{total}] width={width}, {hp_name}={hp_val}...",
                  end=" ", flush=True)

            # Build kwargs based on which HP we're sweeping
            kwargs = {}
            if hp_name == 'lr':
                kwargs['lr_mult'] = hp_val / config.matrix_lr
            elif hp_name == 'wd':
                kwargs['wd_override'] = hp_val
            elif hp_name == 'init_scale':
                kwargs['init_scale'] = hp_val
            elif hp_name == 'attn_temp':
                kwargs['attn_temp'] = hp_val
            elif hp_name == 'emb_mult':
                kwargs['emb_mult'] = hp_val
            elif hp_name == 'output_temp':
                kwargs['output_temp'] = hp_val
            elif hp_name == 'beta2':
                kwargs['beta2_override'] = hp_val

            data_loader, _ = create_data_loader(config.batch_size, config.seq_len, device)
            losses, actual_width = train_model(
                width, config, device, data_loader,
                batch_lr_scale=batch_lr_scale,
                weight_decay_scaled=weight_decay_scaled,
                **kwargs)
            final = losses[-1]
            results['final_losses'][actual_width][hp_val] = final
            results['all_losses'][(actual_width, hp_val)] = losses
            print(f"final_loss={final:.4f} (eta: {eta/60:.1f}m)")

        if actual_width is not None and actual_width not in results['widths']:
            results['widths'].append(actual_width)

    return results


# ─── Plotting helpers ────────────────────────────────────────────────────────

def _width_colors(widths):
    return plt.cm.viridis(np.linspace(0, 0.85, len(widths)))


def _find_optimal(losses_dict):
    """Find key with minimum loss from a dict."""
    return min(losses_dict, key=losses_dict.get)


# ─── 2D sweep plots ─────────────────────────────────────────────────────────

def plot_2d_heatmap(results: Dict, title_prefix: str = "muP",
                    save_path: Optional[str] = None):
    """Plot heatmap of final loss for each width in the LR × WD grid."""
    widths = results['widths']
    lr_values = results['lr_values']
    wd_values = results['wd_values']

    n_widths = len(widths)
    n_cols = min(3, n_widths)
    n_rows = (n_widths + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_widths == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    # Gather all finite losses for shared colorbar
    all_losses = [results['final_losses'].get((w, lr, wd), float('inf'))
                  for w in widths for lr in lr_values for wd in wd_values]
    finite_losses = [l for l in all_losses if np.isfinite(l)]
    if not finite_losses:
        print("All losses are inf/NaN, cannot plot heatmap")
        return
    vmin, vmax = min(finite_losses), max(finite_losses)

    for idx, w in enumerate(widths):
        ax = axes[idx]
        grid = np.full((len(wd_values), len(lr_values)), np.nan)
        for i, wd in enumerate(wd_values):
            for j, lr in enumerate(lr_values):
                val = results['final_losses'].get((w, lr, wd), float('inf'))
                grid[i, j] = val if np.isfinite(val) else np.nan

        im = ax.pcolormesh(
            np.arange(len(lr_values) + 1) - 0.5,
            np.arange(len(wd_values) + 1) - 0.5,
            grid, vmin=vmin, vmax=vmax, cmap='viridis_r')

        # Annotate cells
        for i in range(len(wd_values)):
            for j in range(len(lr_values)):
                val = grid[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6,
                            color='white' if val > (vmin + vmax) / 2 else 'black')

        # Mark optimum
        best_key = None
        best_loss = float('inf')
        for lr in lr_values:
            for wd in wd_values:
                loss = results['final_losses'].get((w, lr, wd), float('inf'))
                if loss < best_loss:
                    best_loss = loss
                    best_key = (lr, wd)
        if best_key is not None:
            j = lr_values.index(best_key[0])
            i = wd_values.index(best_key[1])
            ax.plot(j, i, '*', color='red', markersize=15, zorder=5)

        ax.set_xticks(range(len(lr_values)))
        ax.set_xticklabels([f'{v:.4g}' for v in lr_values], rotation=45, fontsize=7)
        ax.set_yticks(range(len(wd_values)))
        ax.set_yticklabels([f'{v:.3g}' for v in wd_values], fontsize=7)
        ax.set_xlabel('LR')
        ax.set_ylabel('WD')
        ax.set_title(f'width={w}', fontsize=10)

    for idx in range(n_widths, len(axes)):
        axes[idx].set_visible(False)

    fig.colorbar(im, ax=axes[:n_widths].tolist(), shrink=0.8, label='Final Loss')
    fig.suptitle(f'{title_prefix}: LR × WD Sweep', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    plt.show()


def plot_2d_optimal_transfer(results: Dict, title_prefix: str = "muP",
                             save_path: Optional[str] = None):
    """Plot optimal LR and optimal WD vs width from 2D sweep."""
    widths = results['widths']
    lr_values = results['lr_values']
    wd_values = results['wd_values']

    # Find optimal (LR, WD) per width
    opt_lrs, opt_wds, opt_losses = [], [], []
    for w in widths:
        best_lr, best_wd, best_loss = None, None, float('inf')
        for lr in lr_values:
            for wd in wd_values:
                loss = results['final_losses'].get((w, lr, wd), float('inf'))
                if loss < best_loss:
                    best_loss = loss
                    best_lr, best_wd = lr, wd
        opt_lrs.append(best_lr)
        opt_wds.append(best_wd)
        opt_losses.append(best_loss)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Optimal LR vs width
    ax = axes[0]
    ax.plot(widths, opt_lrs, 'o-', linewidth=2, markersize=8, color='tab:blue')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Optimal LR')
    ax.set_title('Optimal LR vs Width')
    ax.grid(True, alpha=0.3)

    # Optimal WD vs width
    ax = axes[1]
    ax.plot(widths, opt_wds, 'o-', linewidth=2, markersize=8, color='tab:orange')
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Optimal WD')
    ax.set_title('Optimal WD vs Width')
    ax.grid(True, alpha=0.3)

    # Best loss vs width
    ax = axes[2]
    ax.plot(widths, opt_losses, 'o-', linewidth=2, markersize=8, color='tab:green')
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Best Final Loss')
    ax.set_title('Best Achievable Loss vs Width')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{title_prefix}: Optimal HP Transfer from 2D Sweep', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved transfer plot to {save_path}")
    plt.show()


def plot_2d_loss_curves(results: Dict, title_prefix: str = "muP",
                        save_path: Optional[str] = None):
    """Plot loss curves at the optimal (LR, WD) per width from 2D sweep."""
    widths = results['widths']
    lr_values = results['lr_values']
    wd_values = results['wd_values']
    colors = _width_colors(widths)

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, w in enumerate(widths):
        # Find optimal config
        best_key, best_loss = None, float('inf')
        for lr in lr_values:
            for wd in wd_values:
                loss = results['final_losses'].get((w, lr, wd), float('inf'))
                if loss < best_loss:
                    best_loss = loss
                    best_key = (w, lr, wd)
        if best_key is not None:
            losses = results['all_losses'].get(best_key, [])
            if losses:
                ax.plot(losses, color=colors[i], linewidth=2,
                        label=f'w={w} (lr={best_key[1]:.4g}, wd={best_key[2]:.3g})')

    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'{title_prefix}: Loss Curves at Optimal (LR, WD)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")
    plt.show()


def plot_2d_comparison(results_sp: Dict, results_mup: Dict,
                       save_path: Optional[str] = None):
    """Plot SP vs muP heatmaps side by side (one row per width, 2 columns)."""
    widths = results_sp['widths']
    lr_values = results_sp['lr_values']
    wd_values = results_sp['wd_values']

    n_widths = len(widths)
    fig, axes = plt.subplots(n_widths, 2, figsize=(10, 4 * n_widths))
    if n_widths == 1:
        axes = axes.reshape(1, 2)

    # Shared color range
    all_losses = []
    for res in [results_sp, results_mup]:
        for w in widths:
            for lr in lr_values:
                for wd in wd_values:
                    val = res['final_losses'].get((w, lr, wd), float('inf'))
                    if np.isfinite(val):
                        all_losses.append(val)
    if not all_losses:
        print("All losses are inf/NaN, cannot plot comparison")
        return
    vmin, vmax = min(all_losses), max(all_losses)

    for col, (results, label) in enumerate([(results_sp, 'SP'), (results_mup, 'muP')]):
        for row, w in enumerate(widths):
            ax = axes[row, col]
            grid = np.full((len(wd_values), len(lr_values)), np.nan)
            for i, wd in enumerate(wd_values):
                for j, lr in enumerate(lr_values):
                    val = results['final_losses'].get((w, lr, wd), float('inf'))
                    grid[i, j] = val if np.isfinite(val) else np.nan

            im = ax.pcolormesh(
                np.arange(len(lr_values) + 1) - 0.5,
                np.arange(len(wd_values) + 1) - 0.5,
                grid, vmin=vmin, vmax=vmax, cmap='viridis_r')

            # Annotate
            for i in range(len(wd_values)):
                for j in range(len(lr_values)):
                    val = grid[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5,
                                color='white' if val > (vmin + vmax) / 2 else 'black')

            # Mark optimum
            best_loss = float('inf')
            best_ij = (0, 0)
            for i in range(len(wd_values)):
                for j in range(len(lr_values)):
                    if np.isfinite(grid[i, j]) and grid[i, j] < best_loss:
                        best_loss = grid[i, j]
                        best_ij = (j, i)
            ax.plot(best_ij[0], best_ij[1], '*', color='red', markersize=15, zorder=5)

            ax.set_xticks(range(len(lr_values)))
            ax.set_xticklabels([f'{v:.4g}' for v in lr_values], rotation=45, fontsize=6)
            ax.set_yticks(range(len(wd_values)))
            ax.set_yticklabels([f'{v:.3g}' for v in wd_values], fontsize=6)
            ax.set_xlabel('LR')
            ax.set_ylabel('WD')
            ax.set_title(f'{label}: width={w}', fontsize=9)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Final Loss')
    fig.suptitle('2D Sweep: SP vs muP', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    plt.show()


# ─── 1D sweep plots ─────────────────────────────────────────────────────────

def plot_1d_sweep(results: Dict, title_prefix: str = "muP",
                  save_path: Optional[str] = None):
    """Plot 1D sweep: final loss vs HP value for each width + optimal HP vs width."""
    widths = results['widths']
    hp_values = results['hp_values']
    hp_name = results['hp_name']
    final_losses = results['final_losses']
    colors = _width_colors(widths)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: final loss vs HP value
    ax = axes[0]
    for i, w in enumerate(widths):
        losses = [final_losses[w].get(v, float('inf')) for v in hp_values]
        ax.plot(hp_values, losses, 'o-', color=colors[i], linewidth=2, label=f'w={w}')
        opt_v = _find_optimal(final_losses[w])
        ax.plot(opt_v, final_losses[w][opt_v], '*', color=colors[i], markersize=15, zorder=5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(hp_name)
    ax.set_ylabel('Final Loss')
    ax.set_title(f'{title_prefix}: {hp_name} Sweep')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: optimal HP vs width
    ax = axes[1]
    opt_vals = [_find_optimal(final_losses[w]) for w in widths]
    ax.plot(widths, opt_vals, 'o-', linewidth=2, markersize=8, color='tab:blue')
    ax.axhline(y=hp_values[len(hp_values)//2], color='gray', linestyle='--', alpha=0.5,
               label=f'median ({hp_values[len(hp_values)//2]:.3g})')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel('Width')
    ax.set_ylabel(f'Optimal {hp_name}')
    ax.set_title(f'Optimal {hp_name} vs Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_1d_loss_curves(results: Dict, title_prefix: str = "muP",
                        save_path: Optional[str] = None):
    """Plot loss curves at the optimal HP value for each width."""
    widths = results['widths']
    hp_name = results['hp_name']
    final_losses = results['final_losses']
    colors = _width_colors(widths)

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, w in enumerate(widths):
        opt_v = _find_optimal(final_losses[w])
        losses = results['all_losses'].get((w, opt_v), [])
        if losses:
            ax.plot(losses, color=colors[i], linewidth=2,
                    label=f'w={w} ({hp_name}={opt_v:.3g})')

    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'{title_prefix}: Loss Curves at Optimal {hp_name}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")
    plt.show()


def plot_1d_comparison(results_sp: Dict, results_mup: Dict,
                       save_path: Optional[str] = None):
    """Plot SP vs muP comparison for a 1D sweep."""
    hp_name = results_sp['hp_name']
    hp_values = results_sp['hp_values']
    widths = results_sp['widths']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = _width_colors(widths)

    # Top row: final loss vs HP
    for col, (results, label) in enumerate([(results_sp, 'SP'), (results_mup, 'muP')]):
        ax = axes[0, col]
        for i, w in enumerate(widths):
            losses = [results['final_losses'][w].get(v, float('inf')) for v in hp_values]
            ax.plot(hp_values, losses, 'o-', color=colors[i], linewidth=2, label=f'w={w}')
            opt_v = _find_optimal(results['final_losses'][w])
            ax.plot(opt_v, results['final_losses'][w][opt_v], '*',
                    color=colors[i], markersize=15, zorder=5)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel(hp_name)
        ax.set_ylabel('Final Loss')
        ax.set_title(f'{label}: {hp_name} Sweep')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Shared y-axis for top row
    all_losses = []
    for res in [results_sp, results_mup]:
        for w in widths:
            for v in hp_values:
                val = res['final_losses'][w].get(v, float('inf'))
                if np.isfinite(val):
                    all_losses.append(val)
    if all_losses:
        y_min, y_max = min(all_losses) * 0.9, max(all_losses) * 1.1
        axes[0, 0].set_ylim(y_min, y_max)
        axes[0, 1].set_ylim(y_min, y_max)

    # Bottom left: optimal HP vs width
    ax = axes[1, 0]
    for results, label, color in [(results_sp, 'SP', 'tab:red'), (results_mup, 'muP', 'tab:blue')]:
        opt_vals = [_find_optimal(results['final_losses'][w]) for w in widths]
        ax.plot(widths, opt_vals, 'o-', linewidth=2, markersize=8, color=color, label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel('Width')
    ax.set_ylabel(f'Optimal {hp_name}')
    ax.set_title(f'Optimal {hp_name} vs Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: best loss vs width
    ax = axes[1, 1]
    for results, label, color in [(results_sp, 'SP', 'tab:red'), (results_mup, 'muP', 'tab:blue')]:
        opt_losses = [results['final_losses'][w][_find_optimal(results['final_losses'][w])] for w in widths]
        ax.plot(widths, opt_losses, 'o-', linewidth=2, markersize=8, color=color, label=label)
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Best Final Loss')
    ax.set_title('Best Achievable Loss vs Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{hp_name} Sweep: SP vs muP', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    plt.show()


# ─── Summary printing ──────────────────────────────────────────────────────────

def print_2d_summary(results: Dict, label: str):
    """Print summary table for 2D sweep."""
    widths = results['widths']
    lr_values = results['lr_values']
    wd_values = results['wd_values']

    print(f"\n{'='*70}")
    print(f"  {label}: 2D LR × WD Sweep Results")
    print(f"{'='*70}")

    for w in widths:
        print(f"\n  Width={w}:")
        # Find optimal
        best_lr, best_wd, best_loss = None, None, float('inf')
        for lr in lr_values:
            for wd in wd_values:
                loss = results['final_losses'].get((w, lr, wd), float('inf'))
                if loss < best_loss:
                    best_loss = loss
                    best_lr, best_wd = lr, wd
        print(f"    Best: lr={best_lr:.4g}, wd={best_wd:.3g}, loss={best_loss:.4f}")

    # HP transfer metric: spread of optimal LR and WD across widths
    opt_lrs, opt_wds = [], []
    for w in widths:
        best_lr, best_wd, best_loss = None, None, float('inf')
        for lr in lr_values:
            for wd in wd_values:
                loss = results['final_losses'].get((w, lr, wd), float('inf'))
                if loss < best_loss:
                    best_loss = loss
                    best_lr, best_wd = lr, wd
        opt_lrs.append(best_lr)
        opt_wds.append(best_wd)

    # Filter valid values for spread computation
    valid_lrs = [lr for lr in opt_lrs if lr is not None and lr > 0]
    if len(valid_lrs) >= 2:
        lr_spread = np.log2(max(valid_lrs)) - np.log2(min(valid_lrs))
        print(f"\n  Optimal LR spread (log2): {lr_spread:.3f}")
        print(f"    (0 = perfect transfer, >1 = poor transfer)")

    # WD spread (handle zeros)
    valid_wds = [wd for wd in opt_wds if wd is not None]
    if len(valid_wds) >= 2:
        unique_wds = len(set(valid_wds))
        print(f"  Optimal WD values: {valid_wds} (unique: {unique_wds}/{len(valid_wds)})")


def print_1d_summary(results: Dict, label: str):
    """Print summary table for 1D sweep."""
    widths = results['widths']
    hp_name = results['hp_name']
    hp_values = results['hp_values']
    final_losses = results['final_losses']

    print(f"\n{'='*70}")
    print(f"  {label}: {hp_name} Sweep Results")
    print(f"{'='*70}")

    # Header
    header = f"{'Width':>8}"
    for v in hp_values:
        header += f" | {v:>7.3g}"
    header += f" | {'Best':>7} | {'Opt':>7}"
    print(header)
    print("-" * len(header))

    opt_vals = []
    for w in widths:
        row = f"{w:>8}"
        for v in hp_values:
            loss = final_losses[w].get(v, float('inf'))
            if np.isfinite(loss):
                row += f" | {loss:>7.4f}"
            else:
                row += f" | {'inf':>7}"
        opt_v = _find_optimal(final_losses[w])
        opt_vals.append(opt_v)
        opt_loss = final_losses[w][opt_v]
        row += f" | {opt_loss:>7.4f} | {opt_v:>7.3g}"
        print(row)

    # Spread metric
    valid_opts = [v for v in opt_vals if v > 0]
    if len(valid_opts) >= 2:
        log_opts = np.log2(np.array(valid_opts))
        spread = log_opts.max() - log_opts.min()
        print(f"\nOptimal {hp_name} spread (log2): {spread:.3f}")
        print(f"  (0 = perfect transfer, >1 = poor transfer)")


# ─── Save results ───────────────────────────────────────────────────────────────

def save_results(results: Dict, mode: str, save_dir: str):
    """Save raw results as JSON."""
    # Convert tuple keys to string keys for JSON serialization
    serializable = {}
    for k, v in results.items():
        if k in ('final_losses', 'all_losses') and isinstance(v, dict):
            new_dict = {}
            for key, val in v.items():
                str_key = str(key)
                if isinstance(val, dict):
                    new_dict[str_key] = {str(kk): vv for kk, vv in val.items()}
                else:
                    new_dict[str_key] = val
            serializable[k] = new_dict
        elif isinstance(v, defaultdict):
            serializable[k] = {str(kk): dict(vv) if isinstance(vv, dict) else vv
                                for kk, vv in v.items()}
        else:
            serializable[k] = v

    path = os.path.join(save_dir, 'sweep_results.json')
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Saved results to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='muP Hyperparameter Sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode flags
    mode_group = parser.add_argument_group('Sweep modes (default: --sweep-2d)')
    mode_group.add_argument('--sweep-2d', action='store_true', default=False,
                            help='LR × WD grid (primary use case)')
    mode_group.add_argument('--sweep-lr', action='store_true', default=False,
                            help='1D: learning rate only')
    mode_group.add_argument('--sweep-wd', action='store_true', default=False,
                            help='1D: weight decay only')
    mode_group.add_argument('--sweep-init-scale', action='store_true', default=False,
                            help='1D: initialization scale')
    mode_group.add_argument('--sweep-attn-temp', action='store_true', default=False,
                            help='1D: attention temperature')
    mode_group.add_argument('--sweep-emb-mult', action='store_true', default=False,
                            help='1D: embedding multiplier')
    mode_group.add_argument('--sweep-output-temp', action='store_true', default=False,
                            help='1D: output temperature')
    mode_group.add_argument('--sweep-beta2', action='store_true', default=False,
                            help='1D: Muon beta2')

    # Model/Training
    parser.add_argument('--widths', type=str, default=','.join(str(w) for w in DEFAULT_WIDTHS),
                        help='Comma-separated widths')
    parser.add_argument('--steps', type=int, default=300,
                        help='Optimizer steps per run (default: 300)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Per-device batch size (default: 32, matches production)')
    parser.add_argument('--seq-len', type=int, default=2048,
                        help='Sequence length (default: 2048)')
    parser.add_argument('--n-layer', type=int, default=24,
                        help='Number of transformer layers (default: 24, matches production d24)')
    parser.add_argument('--aspect-ratio', type=int, default=64,
                        help='model_dim = depth * aspect_ratio (default: 64, matches production)')
    parser.add_argument('--head-dim', type=int, default=128,
                        help='Head dimension (default: 128, matches production)')
    parser.add_argument('--window-pattern', type=str, default='SSSL',
                        help='Sliding window pattern (default: SSSL, matches production)')
    parser.add_argument('--base-width', type=int, default=256,
                        help='Base width for muP (default: 256)')
    parser.add_argument('--compare', action='store_true', default=False,
                        help='Run both SP and muP')
    parser.add_argument('--lr-values', type=str, default=None,
                        help='Comma-separated LR values (overrides default grid)')
    parser.add_argument('--wd-values', type=str, default=None,
                        help='Comma-separated WD values (overrides default grid)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--target-param-data-ratio', type=float, default=10.5,
                        help='Data:param ratio for scaling laws (default: 10.5, matches production)')
    # LR schedule
    parser.add_argument('--warmup-steps', type=int, default=40,
                        help='LR warmup steps (default: 40, matches production)')
    parser.add_argument('--warmdown-ratio', type=float, default=0.65,
                        help='Fraction of steps for LR warmdown (default: 0.65)')
    parser.add_argument('--final-lr-frac', type=float, default=0.05,
                        help='Final LR as fraction of peak (default: 0.05)')
    parser.add_argument('--grad-accum-steps', type=int, default=0,
                        help='Gradient accumulation steps (0 = auto-compute from scaling laws)')

    # Output
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots + JSON results')
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help='Skip plt.show()')

    args = parser.parse_args()

    # Default to --sweep-2d if no mode specified
    any_mode = (args.sweep_2d or args.sweep_lr or args.sweep_wd or
                args.sweep_init_scale or args.sweep_attn_temp or
                args.sweep_emb_mult or args.sweep_output_temp or args.sweep_beta2)
    if not any_mode:
        args.sweep_2d = True

    if args.no_plot:
        import matplotlib
        matplotlib.use('Agg')

    # Setup
    widths = [int(w) for w in args.widths.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get vocab_size from tokenizer
    try:
        from nanochat.tokenizer import get_tokenizer
        vocab_size = get_tokenizer().get_vocab_size()
    except Exception:
        vocab_size = 32768

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    base_config = SweepConfig(
        widths=widths,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        window_pattern=args.window_pattern,
        seed=args.seed,
        base_width=args.base_width,
        target_param_data_ratio=args.target_param_data_ratio,
        warmup_steps=args.warmup_steps,
        warmdown_ratio=args.warmdown_ratio,
        final_lr_frac=args.final_lr_frac,
        grad_accum_steps=args.grad_accum_steps,
    )

    # Compute batch/depth scaling (matching base_train.py scaling laws)
    batch_lr_scale, weight_decay_scaled, total_batch_size, num_iterations, grad_accum = compute_scaling(base_config)
    # Respect explicit CLI override; otherwise use auto-computed value
    if args.grad_accum_steps > 0:
        base_config.grad_accum_steps = args.grad_accum_steps
        print(f"Using CLI grad_accum_steps={args.grad_accum_steps} (auto-computed was {grad_accum})")
        # Recompute batch_lr_scale and weight_decay_scaled for the actual batch size
        actual_total_batch = args.grad_accum_steps * base_config.batch_size * base_config.seq_len
        B_REF = 2**19  # 524288, matching base_train.py
        batch_lr_scale = (actual_total_batch / B_REF) ** 0.5
        # Recompute WD scaling with actual batch
        d12_ref = None
        with torch.device('meta'):
            base_dim = 12 * base_config.aspect_ratio
            model_dim = ((base_dim + base_config.head_dim - 1) // base_config.head_dim) * base_config.head_dim
            n_head = model_dim // base_config.head_dim
            d12_ref = GPT(GPTConfig(
                sequence_len=base_config.seq_len, vocab_size=base_config.vocab_size,
                n_layer=12, n_head=n_head, n_kv_head=n_head, n_embd=model_dim,
                window_pattern=base_config.window_pattern,
            ))
        pc = d12_ref.num_scaling_params()
        D_REF = base_config.target_param_data_ratio * (pc['transformer_matrices'] + pc['lm_head'])
        # Target tokens for our proxy model
        proxy_ref = None
        with torch.device('meta'):
            base_dim = base_config.base_width
            model_dim = ((base_dim + base_config.head_dim - 1) // base_config.head_dim) * base_config.head_dim
            n_head_p = model_dim // base_config.head_dim
            proxy_ref = GPT(GPTConfig(
                sequence_len=base_config.seq_len, vocab_size=base_config.vocab_size,
                n_layer=base_config.n_layer, n_head=n_head_p, n_kv_head=n_head_p, n_embd=model_dim,
                window_pattern=base_config.window_pattern,
            ))
        pc2 = proxy_ref.num_scaling_params()
        target_tokens = int(base_config.target_param_data_ratio * (pc2['transformer_matrices'] + pc2['lm_head']))
        weight_decay_scaled = base_config.weight_decay * math.sqrt(actual_total_batch / B_REF) * (D_REF / target_tokens)
        total_batch_size = actual_total_batch
        print(f"Recomputed for actual batch: batch_lr_scale={batch_lr_scale:.4f}, "
              f"weight_decay_scaled={weight_decay_scaled:.6f}, total_batch={actual_total_batch:,}")
    else:
        base_config.grad_accum_steps = grad_accum
    print(f"\nProduction-matching setup:")
    print(f"  depth={base_config.n_layer}, base_width={base_config.base_width}, "
          f"seq_len={base_config.seq_len}, batch_size={base_config.batch_size}")
    print(f"  total_batch_size={total_batch_size:,} tokens, grad_accum={base_config.grad_accum_steps}, "
          f"num_iterations={num_iterations:,}")
    print(f"  batch_lr_scale={batch_lr_scale:.4f}, weight_decay_scaled={weight_decay_scaled:.6f}")
    print(f"  steps_per_sweep_run={base_config.steps} (truncated training for HP search)")

    # ── 2D sweep ───────────────────────────────────────────────────────────

    # Parse custom sweep values if provided
    custom_lr = [float(x) for x in args.lr_values.split(',')] if args.lr_values else None
    custom_wd = [float(x) for x in args.wd_values.split(',')] if args.wd_values else None

    if args.sweep_2d:
        lr_values = custom_lr or DEFAULT_LR_VALUES
        wd_values = custom_wd or DEFAULT_WD_VALUES

        if args.compare:
            # SP
            print("\n" + "=" * 60)
            print("2D Sweep: Standard Parameterization (SP)")
            print("=" * 60)
            base_config.use_mup = False
            results_sp = run_2d_sweep(base_config, device, batch_lr_scale, weight_decay_scaled,
                                      lr_values, wd_values)
            print_2d_summary(results_sp, "SP")

            # muP
            print("\n" + "=" * 60)
            print("2D Sweep: muP")
            print("=" * 60)
            base_config.use_mup = True
            results_mup = run_2d_sweep(base_config, device, batch_lr_scale, weight_decay_scaled,
                                       lr_values, wd_values)
            print_2d_summary(results_mup, "muP")

            # Plots
            save = os.path.join(args.save_dir, 'sweep_2d_comparison.png') if args.save_dir else None
            plot_2d_comparison(results_sp, results_mup, save_path=save)

            if args.save_dir:
                save_results(results_mup, '2d_compare', args.save_dir)
        else:
            print("\n" + "=" * 60)
            print("2D Sweep: muP")
            print("=" * 60)
            base_config.use_mup = True
            results = run_2d_sweep(base_config, device, batch_lr_scale, weight_decay_scaled,
                                   lr_values, wd_values)
            print_2d_summary(results, "muP")

            save_hm = os.path.join(args.save_dir, 'sweep_2d_heatmap.png') if args.save_dir else None
            plot_2d_heatmap(results, save_path=save_hm)

            save_tr = os.path.join(args.save_dir, 'sweep_2d_transfer.png') if args.save_dir else None
            plot_2d_optimal_transfer(results, save_path=save_tr)

            save_lc = os.path.join(args.save_dir, 'sweep_2d_loss_curves.png') if args.save_dir else None
            plot_2d_loss_curves(results, save_path=save_lc)

            if args.save_dir:
                save_results(results, '2d', args.save_dir)

    # ── 1D sweeps ──────────────────────────────────────────────────────────

    sweep_1d_configs = [
        ('sweep_lr',          'lr',          custom_lr or DEFAULT_LR_VALUES),
        ('sweep_wd',          'wd',          custom_wd or DEFAULT_WD_VALUES),
        ('sweep_init_scale',  'init_scale',  DEFAULT_INIT_SCALE_VALUES),
        ('sweep_attn_temp',   'attn_temp',   DEFAULT_ATTN_TEMP_VALUES),
        ('sweep_emb_mult',    'emb_mult',    DEFAULT_EMB_MULT_VALUES),
        ('sweep_output_temp', 'output_temp', DEFAULT_OUTPUT_TEMP_VALUES),
        ('sweep_beta2',       'beta2',       DEFAULT_BETA2_VALUES),
    ]

    for flag_name, hp_name, hp_values in sweep_1d_configs:
        if not getattr(args, flag_name):
            continue

        if args.compare:
            # SP
            print("\n" + "=" * 60)
            print(f"{hp_name} Sweep: Standard Parameterization (SP)")
            print("=" * 60)
            base_config.use_mup = False
            results_sp = run_1d_sweep(base_config, device, batch_lr_scale, weight_decay_scaled,
                                      hp_name, hp_values)
            print_1d_summary(results_sp, "SP")

            # muP
            print("\n" + "=" * 60)
            print(f"{hp_name} Sweep: muP")
            print("=" * 60)
            base_config.use_mup = True
            results_mup = run_1d_sweep(base_config, device, batch_lr_scale, weight_decay_scaled,
                                       hp_name, hp_values)
            print_1d_summary(results_mup, "muP")

            save = os.path.join(args.save_dir, f'sweep_{hp_name}_comparison.png') if args.save_dir else None
            plot_1d_comparison(results_sp, results_mup, save_path=save)

            if args.save_dir:
                save_results(results_mup, f'{hp_name}_compare', args.save_dir)
        else:
            print("\n" + "=" * 60)
            print(f"{hp_name} Sweep: muP")
            print("=" * 60)
            base_config.use_mup = True
            results = run_1d_sweep(base_config, device, batch_lr_scale, weight_decay_scaled,
                                   hp_name, hp_values)
            print_1d_summary(results, "muP")

            save_sweep = os.path.join(args.save_dir, f'sweep_{hp_name}.png') if args.save_dir else None
            plot_1d_sweep(results, save_path=save_sweep)

            save_lc = os.path.join(args.save_dir, f'sweep_{hp_name}_loss_curves.png') if args.save_dir else None
            plot_1d_loss_curves(results, save_path=save_lc)

            if args.save_dir:
                save_results(results, hp_name, args.save_dir)


if __name__ == '__main__':
    main()
