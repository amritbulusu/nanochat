"""
CompleteP Depth Transfer Check for nanochat

Validates that optimal learning rates transfer across model depths under CompleteP.
For each depth, sweeps over LR multipliers and records final loss. Under correct
CompleteP, the optimal LR multiplier should be ~1.0 at all depths (i.e., the same LR
works everywhere). Without depth scaling, the optimal LR typically shifts with depth.

Reference: Dey et al., "Don't be lazy: CompleteP" (arXiv:2505.01618)
Reference: Mlodozeniec et al., "CompletedP" (arXiv:2512.22382)

Usage:
    # Quick check
    python -m scripts.depth_transfer_check

    # Compare no scaling vs alpha=0.5 vs alpha=1.0
    python -m scripts.depth_transfer_check --compare

    # With muP also enabled (joint width+depth)
    python -m scripts.depth_transfer_check --compare --use-mup

    # Save plots
    python -m scripts.depth_transfer_check --compare --save-dir temp/depth_transfer
"""

import argparse
import os
os.environ["NANOCHAT_DTYPE"] = "float32"
import torch
import torch._dynamo
torch._dynamo.config.disable = True
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from nanochat.gpt import GPT, GPTConfig


@dataclass
class DepthTransferConfig:
    depths: List[int]
    lr_multipliers: List[float]
    model_dim: int = 256          # fixed width (decoupled from depth)
    num_steps: int = 200
    batch_size: int = 8
    seq_len: int = 128
    vocab_size: int = 32768
    seed: int = 42
    use_mup: bool = False
    base_width: int = 128
    base_depth: int = 2
    alpha: float = 1.0
    # Base learning rates
    matrix_lr: float = 0.12
    embedding_lr: float = 6.0
    unembedding_lr: float = 0.12
    # Multi-seed averaging
    num_seeds: int = 1
    base_seed: int = 42
    # LR decay
    lr_decay: bool = False
    # Data diversity
    num_batches: int = 1


def load_batches(num_batches: int, batch_size: int, seq_len: int, device: torch.device):
    """Load multiple batches. Falls back to random data if dataset unavailable."""
    try:
        from nanochat.tokenizer import get_tokenizer
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, batch_size, seq_len, split="train", device=device,
        )
        batches = []
        for i, (x, y) in enumerate(loader):
            batches.append((x, y))
            if len(batches) >= num_batches:
                break
        print(f"Loaded {len(batches)} real training batch(es) (vocab_size={vocab_size})")
        return batches, vocab_size
    except Exception as e:
        print(f"Could not load training data ({e}), using random tokens")
        vocab_size = 32768
        batches = []
        for i in range(num_batches):
            rng = torch.Generator(device=device)
            rng.manual_seed(42 + i)
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=rng)
            y = torch.roll(x, -1, dims=1)
            y[:, -1] = -1
            batches.append((x, y))
        return batches, vocab_size


def create_model(depth: int, config: DepthTransferConfig, device: torch.device,
                 completep_base_depth: int = 0, alpha: float = 1.0):
    """Create a model with specified depth at fixed width."""
    head_dim = 64
    width = config.model_dim
    n_head = max(1, width // head_dim)
    actual_width = n_head * head_dim

    mup_base_width = config.base_width if config.use_mup else 0

    gpt_config = GPTConfig(
        sequence_len=config.seq_len,
        vocab_size=config.vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=actual_width,
        window_pattern="L",
        mup_base_width=mup_base_width,
        completep_base_depth=completep_base_depth,
        depth_branch_alpha=alpha,
    )

    with torch.device('meta'):
        model = GPT(gpt_config)
    model.to_empty(device=device)
    model.init_weights()

    return model, gpt_config


def train_model(depth: int, lr_mult: float, config: DepthTransferConfig,
                device: torch.device, batches: List,
                completep_base_depth: int = 0, alpha: float = 1.0):
    """Train a model at given depth and LR multiplier, return loss history."""
    torch.manual_seed(config.seed)

    model, gpt_config = create_model(depth, config, device,
                                      completep_base_depth=completep_base_depth,
                                      alpha=alpha)

    optimizer = model.setup_optimizer(
        unembedding_lr=config.unembedding_lr * lr_mult,
        embedding_lr=config.embedding_lr * lr_mult,
        matrix_lr=config.matrix_lr * lr_mult,
        weight_decay=0.0,
        use_mup=config.use_mup,
        base_width=config.base_width,
    )

    # Cosine LR decay schedule
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    min_lr_ratio = 0.1

    model.train()
    losses = []
    num_batches = len(batches)

    for step in range(config.num_steps):
        if config.lr_decay:
            progress = step / max(config.num_steps - 1, 1)
            decay = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = base_lr * decay

        x, y = batches[step % num_batches]
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
            loss = model(x, y)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return losses


def ewm_final(losses: List[float], alpha: float = 0.9) -> float:
    """Compute exponentially weighted moving average and return final value."""
    ewm = losses[0]
    for loss in losses[1:]:
        ewm = alpha * loss + (1 - alpha) * ewm
    return ewm


def find_optimal_lr(final_losses: Dict[float, float]) -> float:
    """Find the LR multiplier with the lowest final loss."""
    return min(final_losses, key=final_losses.get)


def run_depth_transfer_check(config: DepthTransferConfig, device: torch.device,
                              batches: List,
                              completep_base_depth: int = 0, alpha: float = 1.0) -> Dict:
    """Run LR sweep across all depths, averaging over seeds."""
    seeds = list(range(config.base_seed, config.base_seed + config.num_seeds))
    results = {
        'depths': [],
        'lr_multipliers': config.lr_multipliers,
        'losses': {},
        'final_losses': defaultdict(dict),
        'final_losses_stderr': defaultdict(dict),
    }

    for depth in config.depths:
        seed_label = f" (seeds {seeds[0]}-{seeds[-1]})" if len(seeds) > 1 else ""
        for lr_mult in config.lr_multipliers:
            print(f"  depth={depth}, lr_mult={lr_mult:.4f}...{seed_label}", end=" ", flush=True)

            seed_ewm_losses = []
            seed_loss_curves = []
            for seed in seeds:
                config_copy_seed = config.seed
                config.seed = seed
                losses = train_model(depth, lr_mult, config, device, batches,
                                     completep_base_depth=completep_base_depth,
                                     alpha=alpha)
                config.seed = config_copy_seed
                seed_loss_curves.append(losses)
                seed_ewm_losses.append(ewm_final(losses))

            mean_ewm = np.mean(seed_ewm_losses)
            stderr_ewm = np.std(seed_ewm_losses, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else 0.0
            mean_curve = np.mean(seed_loss_curves, axis=0).tolist()

            results['losses'][(depth, lr_mult)] = mean_curve
            results['final_losses'][depth][lr_mult] = mean_ewm
            results['final_losses_stderr'][depth][lr_mult] = stderr_ewm
            print(f"final_loss={mean_ewm:.4f}" + (f" ± {stderr_ewm:.4f}" if len(seeds) > 1 else ""))

        if depth not in results['depths']:
            results['depths'].append(depth)

    return results


def plot_comparison(all_results: List[Dict], labels: List[str],
                    config: DepthTransferConfig, save_path: Optional[str] = None):
    """Plot depth transfer check: compare different parameterizations."""
    n_modes = len(all_results)
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_modes, figsize=(5 * n_modes, 4 * n_rows))
    if n_modes == 1:
        axes = axes.reshape(-1, 1)

    for col, (results, label) in enumerate(zip(all_results, labels)):
        depths = results['depths']
        lr_mults = results['lr_multipliers']
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(depths)))

        # Top: LR sweep curves
        ax = axes[0, col]
        for i, d in enumerate(depths):
            losses = np.array([results['final_losses'][d][m] for m in lr_mults])
            ax.plot(lr_mults, losses, 'o-', color=colors[i], linewidth=2, label=f'L={d}')
            if 'final_losses_stderr' in results and d in results['final_losses_stderr']:
                stderrs = np.array([results['final_losses_stderr'][d].get(m, 0) for m in lr_mults])
                if stderrs.any():
                    ax.fill_between(lr_mults, losses - stderrs, losses + stderrs, color=colors[i], alpha=0.2)
            opt_mult = find_optimal_lr(results['final_losses'][d])
            opt_loss = results['final_losses'][d][opt_mult]
            ax.plot(opt_mult, opt_loss, '*', color=colors[i], markersize=15, zorder=5)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('LR Multiplier')
        ax.set_ylabel('Final Loss')
        ax.set_title(f'{label}: Final Loss vs LR')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom: optimal LR vs depth
        ax = axes[1, col]
        opt_mults = [find_optimal_lr(results['final_losses'][d]) for d in depths]
        ax.plot(depths, opt_mults, 'o-', linewidth=2, markersize=8, color='tab:blue')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='target (1.0)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.set_xticks(depths)
        ax.set_xticklabels(depths)
        ax.set_xlabel('Depth')
        ax.set_ylabel('Optimal LR Multiplier')
        ax.set_title(f'{label}: Optimal LR vs Depth')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Shared y-axis for top row
    all_losses_flat = []
    for results in all_results:
        for d in results['depths']:
            for m in results['lr_multipliers']:
                all_losses_flat.append(results['final_losses'][d][m])
    y_min = min(all_losses_flat) * 0.9
    y_max = min(all_losses_flat) * 3.0
    for col in range(n_modes):
        axes[0, col].set_ylim(y_min, y_max)

    mup_str = " + muP" if config.use_mup else ""
    fig.suptitle(f'Depth Transfer Check{mup_str} (width={config.model_dim})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    plt.show()


def print_summary(results: Dict, label: str):
    """Print a summary table of the LR sweep results."""
    depths = results['depths']
    lr_mults = results['lr_multipliers']
    final_losses = results['final_losses']

    print(f"\n{'='*70}")
    print(f"  {label}: Depth LR Sweep Results")
    print(f"{'='*70}")

    header = f"{'Depth':>8}"
    for m in lr_mults:
        header += f" | {m:>7.3f}"
    header += f" | {'Best':>7} | {'Opt LR':>7}"
    print(header)
    print("-" * len(header))

    opt_mults = []
    for d in depths:
        row = f"{d:>8}"
        for m in lr_mults:
            loss = final_losses[d][m]
            row += f" | {loss:>7.4f}"
        opt_m = find_optimal_lr(final_losses[d])
        opt_mults.append(opt_m)
        opt_loss = final_losses[d][opt_m]
        row += f" | {opt_loss:>7.4f} | {opt_m:>7.3f}"
        print(row)

    opt_mults_arr = np.array(opt_mults)
    log_opt = np.log2(opt_mults_arr)
    spread = log_opt.max() - log_opt.min()
    print(f"\nOptimal LR spread (log2): {spread:.3f}")
    print(f"  (0 = perfect transfer, >1 = poor transfer)")


def main():
    parser = argparse.ArgumentParser(description='CompleteP Depth Transfer Check')
    parser.add_argument('--depths', type=str, default='2,4,8,16',
                        help='Comma-separated list of depths to test')
    parser.add_argument('--model-dim', type=int, default=256,
                        help='Fixed model width (decoupled from depth)')
    parser.add_argument('--lr-mults', type=str,
                        default='0.03125,0.044,0.0625,0.088,0.125,0.177,0.25,0.354,0.5,0.707,1.0,1.414,2.0,2.828,4.0,5.657,8.0',
                        help='Comma-separated LR multipliers to sweep')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of training steps per run')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--base-depth', type=int, default=2,
                        help='Base depth for CompleteP scaling')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Branch scaling exponent (1.0=CompleteP, 0.5=Depth-muP)')
    parser.add_argument('--use-mup', action='store_true',
                        help='Also enable muP width scaling')
    parser.add_argument('--base-width', type=int, default=128,
                        help='Base width for muP scaling')
    parser.add_argument('--compare', action='store_true',
                        help='Run no scaling, alpha=0.5, and alpha=1.0 side by side')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds to average over')
    parser.add_argument('--lr-decay', action='store_true',
                        help='Enable cosine LR decay to lr/10')
    parser.add_argument('--num-batches', type=int, default=1,
                        help='Number of data batches to cycle through')

    args = parser.parse_args()
    depths = [int(d) for d in args.depths.split(',')]
    lr_mults = sorted(float(m) for m in args.lr_mults.split(','))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batches, vocab_size = load_batches(args.num_batches, args.batch_size, args.seq_len, device)

    config = DepthTransferConfig(
        depths=depths,
        lr_multipliers=lr_mults,
        model_dim=args.model_dim,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        seed=args.seed,
        use_mup=args.use_mup,
        base_width=args.base_width,
        base_depth=args.base_depth,
        alpha=args.alpha,
        num_seeds=args.seeds,
        base_seed=args.seed,
        lr_decay=args.lr_decay,
        num_batches=args.num_batches,
    )

    if args.compare:
        configs_to_run = [
            (0, 1.0, "No depth scaling"),
            (args.base_depth, 0.5, "α=0.5 (Depth-μP)"),
            (args.base_depth, 1.0, "α=1.0 (CompleteP)"),
        ]

        all_results = []
        all_labels = []

        for base_depth, alpha, label in configs_to_run:
            print(f"\n{'='*60}")
            print(f"Running: {label}")
            print(f"{'='*60}")
            results = run_depth_transfer_check(config, device, batches,
                                                completep_base_depth=base_depth,
                                                alpha=alpha)
            all_results.append(results)
            all_labels.append(label)
            print_summary(results, label)

        # Compare
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for results, label in zip(all_results, all_labels):
            depths = results['depths']
            opt_mults = [find_optimal_lr(results['final_losses'][d]) for d in depths]
            spread = np.log2(max(opt_mults)) - np.log2(min(opt_mults))
            print(f"{label:>25}: optimal LR spread (log2) = {spread:.3f}")

        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'depth_transfer_comparison.png')
        plot_comparison(all_results, all_labels, config, save_path)

    else:
        completep_base_depth = args.base_depth if args.alpha > 0 else 0
        label = f"α={args.alpha}" if completep_base_depth > 0 else "No depth scaling"
        print(f"\n{'='*60}")
        print(f"Running Depth Transfer Check ({label})")
        print(f"{'='*60}")
        print(f"Depths: {depths}, Width: {config.model_dim}")

        results = run_depth_transfer_check(config, device, batches,
                                            completep_base_depth=completep_base_depth,
                                            alpha=args.alpha)
        print_summary(results, label)

        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'depth_transfer.png')
        plot_comparison([results], [label], config, save_path)


if __name__ == '__main__':
    main()
