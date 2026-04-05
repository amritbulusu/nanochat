"""
CompleteP Depth Coordinate Check for nanochat

Validates that activation magnitudes are independent of model depth under CompleteP.
Analogous to mup_coord_check.py (which checks width independence under muP).

Reference: Dey et al., "Don't be lazy: CompleteP" (arXiv:2505.01618)
Reference: Mlodozeniec et al., "CompletedP" (arXiv:2512.22382)

Usage:
    # Quick check
    python -m scripts.depth_coord_check --depths 2,4,8,16 --steps 10

    # Compare no scaling vs alpha=0.5 vs alpha=1
    python -m scripts.depth_coord_check --compare

    # With muP enabled (joint width+depth check)
    python -m scripts.depth_coord_check --compare --use-mup
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

from nanochat.gpt import GPT, GPTConfig


def load_batch(batch_size: int, seq_len: int, device: torch.device):
    """Load a single batch. Falls back to random data if dataset unavailable."""
    try:
        from nanochat.tokenizer import get_tokenizer
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, batch_size, seq_len, split="train", device=device,
        )
        x, y = next(loader)
        print(f"Loaded real training data (vocab_size={vocab_size})")
        return x, y, vocab_size
    except Exception as e:
        print(f"Could not load training data ({e}), using random tokens")
        vocab_size = 32768
        rng = torch.Generator(device=device)
        rng.manual_seed(42)
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=rng)
        y = torch.roll(x, -1, dims=1)
        y[:, -1] = -1
        return x, y, vocab_size


@dataclass
class DepthCoordConfig:
    depths: List[int]
    model_dim: int = 256          # fixed width (decoupled from depth)
    num_steps: int = 10
    batch_size: int = 4
    seq_len: int = 128
    vocab_size: int = 32768
    seed: int = 42
    use_mup: bool = False
    base_width: int = 128
    base_depth: int = 2
    alpha: float = 1.0            # branch scaling exponent
    matrix_lr: float = 0.12
    embedding_lr: float = 6.0
    unembedding_lr: float = 0.12
    num_seeds: int = 1
    base_seed: int = 42


class DepthActivationRecorder:
    """Records activation statistics for depth coord check."""

    def __init__(self):
        self.stats: Dict[str, List[float]] = defaultdict(list)
        self.hooks = []

    def _get_stat(self, tensor: torch.Tensor) -> float:
        return tensor.float().abs().mean().item()

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if output is not None and isinstance(output, torch.Tensor):
                self.stats[name].append(self._get_stat(output))
        return hook

    def register_hooks(self, model: GPT) -> None:
        """Register hooks on embedding, first/last block outputs, and logits."""
        # Embedding
        self.hooks.append(
            model.transformer.wte.register_forward_hook(self._make_hook('word embedding')))

        # First and last block attention + FFN outputs
        n_layer = model.config.n_layer
        for i in [0, n_layer - 1]:
            block = model.transformer.h[i]
            self.hooks.append(
                block.attn.c_proj.register_forward_hook(self._make_hook(f'attn output.{i}')))
            self.hooks.append(
                block.mlp.c_proj.register_forward_hook(self._make_hook(f'FFN output.{i}')))

        # Middle block (to see depth-interior behavior)
        mid = n_layer // 2
        if mid != 0 and mid != n_layer - 1:
            block = model.transformer.h[mid]
            self.hooks.append(
                block.attn.c_proj.register_forward_hook(self._make_hook(f'attn output.mid({mid})')))
            self.hooks.append(
                block.mlp.c_proj.register_forward_hook(self._make_hook(f'FFN output.mid({mid})')))

        # Output logits (with muP scaling applied)
        mup_base = model.config.mup_base_width
        n_embd = model.config.n_embd
        def logit_hook(module, input, output):
            if output is not None and isinstance(output, torch.Tensor):
                scaled = output
                if mup_base > 0:
                    scaled = output * (mup_base / n_embd)
                self.stats['output logits'].append(self._get_stat(scaled))
        self.hooks.append(model.lm_head.register_forward_hook(logit_hook))

        # Residual stream after final block (pre-norm, pre-backout)
        # We hook the last block itself to get its output = residual stream state
        def resid_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.stats['residual stream (final)'].append(self._get_stat(output))
        self.hooks.append(model.transformer.h[-1].register_forward_hook(resid_hook))

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_step_stats(self) -> Dict[str, float]:
        step_stats = {}
        for name, values in self.stats.items():
            if values:
                step_stats[name] = np.mean(values)
        self.stats = defaultdict(list)
        return step_stats


def create_model(depth: int, config: DepthCoordConfig, device: torch.device,
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


def run_depth_coord_check(config: DepthCoordConfig, device: torch.device,
                          x: torch.Tensor, y: torch.Tensor,
                          completep_base_depth: int = 0, alpha: float = 1.0) -> Dict:
    """Run coordinate check across all depths, averaging over seeds."""
    seeds = list(range(config.base_seed, config.base_seed + config.num_seeds))
    results = {
        'depths': [],
        'steps': list(range(config.num_steps)),
        'stats': defaultdict(lambda: defaultdict(list)),
        'stats_stderr': defaultdict(lambda: defaultdict(list)),
        'losses': defaultdict(list),
    }

    for depth in config.depths:
        seed_label = f" (seeds {seeds[0]}-{seeds[-1]})" if len(seeds) > 1 else ""
        mode = "no scaling" if completep_base_depth == 0 else f"α={alpha}"
        print(f"\n  depth={depth}, {mode}...{seed_label}")

        seed_stats = defaultdict(list)
        seed_losses = []

        for seed_idx, seed in enumerate(seeds):
            torch.manual_seed(seed)

            model, gpt_config = create_model(depth, config, device,
                                              completep_base_depth=completep_base_depth,
                                              alpha=alpha)

            optimizer = model.setup_optimizer(
                unembedding_lr=config.unembedding_lr,
                embedding_lr=config.embedding_lr,
                matrix_lr=config.matrix_lr,
                weight_decay=0.0,
                use_mup=config.use_mup,
                base_width=config.base_width,
            )

            recorder = DepthActivationRecorder()
            recorder.register_hooks(model)
            model.train()

            step_values = defaultdict(list)
            losses = []

            for step in range(config.num_steps):
                with torch.amp.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
                    loss = model(x, y)

                losses.append(loss.item())
                step_stats = recorder.get_step_stats()
                for layer, value in step_stats.items():
                    step_values[layer].append(value)

                if step == 0 and seed_idx == 0:
                    print(f"    Step {step}: loss={loss.item():.4f}")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            seed_losses.append(losses)
            for layer, vals in step_values.items():
                seed_stats[layer].append(vals)

            recorder.remove_hooks()
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results['depths'].append(depth)
        results['losses'][depth] = np.mean(seed_losses, axis=0).tolist()

        for layer, all_seed_vals in seed_stats.items():
            arr = np.array(all_seed_vals)
            means = arr.mean(axis=0).tolist()
            stderrs = (arr.std(axis=0, ddof=1) / np.sqrt(len(seeds))).tolist() if len(seeds) > 1 else [0.0] * len(means)
            results['stats'][depth][layer] = means
            results['stats_stderr'][depth][layer] = stderrs

        final_losses = [sl[-1] for sl in seed_losses]
        print(f"    Final loss: {np.mean(final_losses):.4f}" +
              (f" ± {np.std(final_losses, ddof=1)/np.sqrt(len(seeds)):.4f}" if len(seeds) > 1 else ""))

    return results


def compute_depth_dependence(results: Dict) -> Dict[str, float]:
    """Compute slope of log(activation) vs log(depth). ~0 = depth-independent."""
    depths = np.array(results['depths'])
    log_depths = np.log2(depths)
    final_step = len(results['steps']) - 1

    # Only use layers that exist at all depths (first/last block indices differ)
    # Use a common subset: layers present in the smallest AND largest depth
    common_layers = set(results['stats'][depths[0]].keys())
    for d in depths:
        common_layers &= set(results['stats'][d].keys())

    slopes = {}
    for layer in sorted(common_layers):
        values = [results['stats'][d][layer][final_step] for d in depths]
        log_values = np.log2(np.array(values) + 1e-10)
        if len(log_depths) >= 2:
            slope, _ = np.polyfit(log_depths, log_values, 1)
            slopes[layer] = slope

    return slopes


def plot_depth_comparison(results_list: List[Dict], labels: List[str],
                          config: DepthCoordConfig, save_path: Optional[str] = None):
    """Plot depth coord check: columns = different alpha values, rows = layers."""
    n_cols = len(results_list)
    # Find common layers across all results
    common_layers = None
    for results in results_list:
        depths = results['depths']
        layers = set(results['stats'][depths[0]].keys())
        for d in depths:
            layers &= set(results['stats'][d].keys())
        common_layers = layers if common_layers is None else common_layers & layers

    layer_names = sorted(common_layers)
    n_rows = len(layer_names) + 1  # +1 for loss curves

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    steps = results_list[0]['steps']
    step_colors = plt.cm.plasma(np.linspace(0, 1, len(steps)))

    for col, (results, label) in enumerate(zip(results_list, labels)):
        depths = results['depths']

        for row, layer in enumerate(layer_names):
            ax = axes[row, col]
            for s, step in enumerate(steps):
                values = [results['stats'][d].get(layer, [0]*(s+1))[s] for d in depths]
                ax.plot(depths, values, 'o-', color=step_colors[s], linewidth=1.5,
                        label=f'step {step}' if (row == 0 and col == 0) else None)
            ax.set_xscale('log', base=2)
            ax.set_xticks(depths)
            ax.set_xticklabels(depths, fontsize=7)
            ax.set_title(f'{label}: {layer}', fontsize=9)
            ax.set_xlabel('Depth')
            ax.set_ylabel('Mean |activation|')
            ax.grid(True, alpha=0.3)

        # Loss curves row
        ax = axes[n_rows - 1, col]
        depth_colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
        for j, d in enumerate(depths):
            ax.plot(steps, results['losses'][d], label=f'L={d}', color=depth_colors[j], linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'{label}: Loss Curves')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=7, loc='best')

    mup_str = " + muP" if config.use_mup else ""
    fig.suptitle(f'Depth Coordinate Check{mup_str} (width={config.model_dim})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='CompleteP Depth Coordinate Check')
    parser.add_argument('--depths', type=str, default='2,4,8,16,32',
                        help='Comma-separated list of depths to test')
    parser.add_argument('--model-dim', type=int, default=256,
                        help='Fixed model width (decoupled from depth)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=4)
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

    args = parser.parse_args()
    depths = [int(d) for d in args.depths.split(',')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    x, y, vocab_size = load_batch(args.batch_size, args.seq_len, device)

    config = DepthCoordConfig(
        depths=depths,
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
            results = run_depth_coord_check(config, device, x, y,
                                             completep_base_depth=base_depth,
                                             alpha=alpha)
            all_results.append(results)
            all_labels.append(label)

            slopes = compute_depth_dependence(results)
            print(f"\n  Depth dependence slopes:")
            for layer, slope in slopes.items():
                status = "OK" if abs(slope) < 0.15 else "WARN"
                print(f"    {layer}: {slope:+.4f} [{status}]")

        # Summary table
        print(f"\n{'='*60}")
        print("SUMMARY: Depth Dependence Slopes")
        print(f"{'='*60}")
        all_slopes = [compute_depth_dependence(r) for r in all_results]
        common_layers = set(all_slopes[0].keys())
        for s in all_slopes:
            common_layers &= set(s.keys())
        header = f"{'Layer':<30}" + "".join(f" {l:>15}" for l in all_labels)
        print(header)
        print("-" * len(header))
        for layer in sorted(common_layers):
            row = f"{layer:<30}"
            for slopes in all_slopes:
                row += f" {slopes.get(layer, float('nan')):>15.4f}"
            print(row)

        # Plot
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'depth_coord_check_comparison.png')
        plot_depth_comparison(all_results, all_labels, config, save_path)

    else:
        label = f"α={config.alpha}" if config.alpha != 1.0 else "CompleteP"
        base_depth = args.base_depth if args.alpha > 0 else 0
        print(f"\n{'='*60}")
        print(f"Running Depth Coordinate Check ({label})")
        print(f"{'='*60}")
        print(f"Depths: {depths}, Width: {config.model_dim}")

        results = run_depth_coord_check(config, device, x, y,
                                         completep_base_depth=base_depth,
                                         alpha=config.alpha)

        slopes = compute_depth_dependence(results)
        print(f"\n{'='*60}")
        print("Depth Dependence (slope on log-log plot)")
        print("Expected: ~0 for depth-independent")
        print(f"{'='*60}")
        for layer, slope in slopes.items():
            status = "OK" if abs(slope) < 0.15 else "WARN"
            print(f"  {layer}: {slope:+.4f} [{status}]")

        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'depth_coord_check.png')
        plot_depth_comparison([results], [label], config, save_path)


if __name__ == '__main__':
    main()
