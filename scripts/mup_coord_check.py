"""
muP Coordinate Check for nanochat

This script validates muP implementation by checking that activation magnitudes
are independent of model width. Based on EleutherAI's nanoGPT-mup and Microsoft's
mup library.

Reference: https://blog.eleuther.ai/mutransfer/
Reference: Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
           Hyperparameter Transfer" (arXiv:2203.03466), Sections B.1 and F.

Usage:
    python -m scripts.mup_coord_check --widths 128,256,512,1024 --steps 10
    python -m scripts.mup_coord_check --use-mup --widths 128,256,512,1024
    python -m scripts.mup_coord_check --compare --detailed
    python -m scripts.mup_coord_check --compare --muon-lr-exponent 0.5
"""

import argparse
import os
os.environ["NANOCHAT_DTYPE"] = "float32"
import torch
import torch._dynamo
torch._dynamo.config.disable = True
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

from nanochat.gpt import GPT, GPTConfig


def load_batch(batch_size: int, seq_len: int, device: torch.device):
    """Load a single batch from the nanochat training pipeline.
    Falls back to random data if the tokenizer/dataset isn't available."""
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
class CoordCheckConfig:
    widths: List[int]
    num_steps: int = 10
    batch_size: int = 4
    seq_len: int = 128
    vocab_size: int = 32768
    n_layer: int = 2
    seed: int = 42
    use_mup: bool = False
    base_width: int = 128
    # Learning rates (tuned at base_width=128)
    matrix_lr: float = 0.12
    embedding_lr: float = 6.0
    unembedding_lr: float = 0.12
    # Detailed diagnostics
    detailed: bool = False
    # Muon LR exponent: 1.0 = base/width (standard muP), 0.5 = sqrt(base/width)
    # Paper Section C.1: Frobenius-normalizing optimizers may need exponent 0.5
    muon_lr_exponent: float = 0.0
    num_seeds: int = 1
    base_seed: int = 42


class ActivationRecorder:
    """Records activation statistics during forward pass using hooks."""

    def __init__(self, detailed: bool = False):
        self.stats: Dict[str, List[float]] = defaultdict(list)
        self.hooks = []
        self.detailed = detailed
        self._last_q = None
        self._ve_cache: Dict[int, torch.Tensor] = {}

    def _get_stat(self, tensor: torch.Tensor) -> float:
        """Compute mean absolute value (l1 norm per element)."""
        if tensor is None:
            return 0.0
        if tensor.dtype == torch.bool:
            return tensor.float().abs().mean().item()
        return tensor.float().abs().mean().item()

    def _make_hook(self, name: str):
        """Create a forward hook that records output statistics."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if output is not None and isinstance(output, torch.Tensor):
                self.stats[name].append(self._get_stat(output))
        return hook

    def _make_attn_logit_hook(self, name: str, n_head: int, n_kv_head: int, head_dim: int):
        """Create a hook on c_k that computes pre-softmax attention logit magnitudes.

        We hook onto c_k's forward, then use the most recent c_q output to compute
        q @ k^T / sqrt(d) for a single batch element to measure attention logit scale.
        """
        def q_hook(module, input, output):
            self._last_q = output.detach()

        def k_hook(module, input, output):
            if self._last_q is None:
                return
            q = self._last_q
            k = output.detach()
            B, T, _ = q.shape
            q = q[0:1].view(1, T, n_head, head_dim)
            k = k[0:1].view(1, T, n_kv_head, head_dim)
            # Apply QK norm (same as model)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            # Expand k for GQA
            if n_head != n_kv_head:
                k = k.repeat_interleave(n_head // n_kv_head, dim=2)
            # Compute logits: q @ k^T / sqrt(d) — just for first few positions
            T_sub = min(T, 32)
            q_sub = q[:, :T_sub].transpose(1, 2)  # (1, H, T_sub, D)
            k_sub = k[:, :T_sub].transpose(1, 2)  # (1, H, T_sub, D)
            logits = torch.matmul(q_sub, k_sub.transpose(-2, -1)) / (head_dim ** 0.5)
            self.stats[name].append(logits.float().abs().mean().item())
            self._last_q = None

        return q_hook, k_hook

    def register_hooks(self, model: GPT) -> None:
        """Register forward hooks on key layers."""
        # Embedding
        h = model.transformer.wte.register_forward_hook(self._make_hook('word embedding'))
        self.hooks.append(h)

        # Each transformer block
        for i, block in enumerate(model.transformer.h):
            if str(i) in model.value_embeds:
                n_kv_head = block.attn.n_kv_head
                head_dim = block.attn.head_dim

                def ve_hook(module, input, output, layer=i):
                    if output is not None and isinstance(output, torch.Tensor):
                        self.stats[f'value embed.{layer}'].append(self._get_stat(output))
                        self._ve_cache[layer] = output.detach()

                def c_v_hook(module, input, output, layer=i):
                    if output is not None and isinstance(output, torch.Tensor):
                        self.stats[f'value current.{layer}'].append(self._get_stat(output))

                def gate_hook(module, input, output, layer=i, n_kv_head=n_kv_head, head_dim=head_dim):
                    if output is None or not isinstance(output, torch.Tensor):
                        return
                    gate = 3 * torch.sigmoid(output.detach())
                    self.stats[f'value gate.{layer}'].append(self._get_stat(gate))
                    ve = self._ve_cache.pop(layer, None)
                    if ve is None:
                        return
                    ve = ve.view(ve.shape[0], ve.shape[1], n_kv_head, head_dim)
                    branch = gate.unsqueeze(-1) * ve
                    self.stats[f'value branch.{layer}'].append(self._get_stat(branch))

                self.hooks.append(model.value_embeds[str(i)].register_forward_hook(ve_hook))
                self.hooks.append(block.attn.c_v.register_forward_hook(c_v_hook))
                self.hooks.append(block.attn.ve_gate.register_forward_hook(gate_hook))

            # Attention output
            h = block.attn.c_proj.register_forward_hook(self._make_hook(f'attn output.{i}'))
            self.hooks.append(h)
            # MLP output
            h = block.mlp.c_proj.register_forward_hook(self._make_hook(f'FFN output.{i}'))
            self.hooks.append(h)

            # Detailed: attention logit magnitudes
            if self.detailed:
                n_head = block.attn.n_head
                n_kv_head = block.attn.n_kv_head
                head_dim = block.attn.head_dim
                q_hook, k_hook = self._make_attn_logit_hook(
                    f'attn logits.{i}', n_head, n_kv_head, head_dim)
                h1 = block.attn.c_q.register_forward_hook(q_hook)
                h2 = block.attn.c_k.register_forward_hook(k_hook)
                self.hooks.extend([h1, h2])

        # Output logits: hook on lm_head, but apply muP scaling to match what forward() does
        mup_base = model.config.mup_base_width
        n_embd = model.config.n_embd
        def logit_hook(module, input, output):
            if output is not None and isinstance(output, torch.Tensor):
                scaled = output
                if mup_base > 0:
                    scaled = output * (mup_base / n_embd)
                self.stats['output logits'].append(self._get_stat(scaled))
        h = model.lm_head.register_forward_hook(logit_hook)
        self.hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self._ve_cache = {}
        self._last_q = None

    def get_step_stats(self) -> Dict[str, float]:
        """Get mean stats for the current step and reset."""
        step_stats = {}
        for name, values in self.stats.items():
            if values:
                step_stats[name] = np.mean(values)
        self.stats = defaultdict(list)
        self._ve_cache = {}
        self._last_q = None
        return step_stats


def create_model(width: int, config: CoordCheckConfig, device: torch.device, mup_base_width: int = 0) -> Tuple[GPT, GPTConfig]:
    """Create a model with the specified width."""
    head_dim = 64
    n_head = max(1, width // head_dim)
    actual_width = n_head * head_dim

    gpt_config = GPTConfig(
        sequence_len=config.seq_len,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=actual_width,
        window_pattern="L",
        mup_base_width=mup_base_width,
    )

    with torch.device('meta'):
        model = GPT(gpt_config)
    model.to_empty(device=device)
    model.init_weights()

    return model, gpt_config


def setup_optimizer_mup(model: GPT, config: CoordCheckConfig, width: int):
    """Set up optimizer with muP scaling using the native use_mup flag."""
    optimizer = model.setup_optimizer(
        unembedding_lr=config.unembedding_lr,
        embedding_lr=config.embedding_lr,
        matrix_lr=config.matrix_lr,
        weight_decay=0.0,
        use_mup=True,
        base_width=config.base_width,
        muon_lr_exponent=config.muon_lr_exponent,
    )
    return optimizer


def setup_optimizer_sp(model: GPT, config: CoordCheckConfig, width: int):
    """Set up optimizer with standard parameterization (current nanochat)."""
    optimizer = model.setup_optimizer(
        unembedding_lr=config.unembedding_lr,
        embedding_lr=config.embedding_lr,
        matrix_lr=config.matrix_lr,
        weight_decay=0.0,
        use_mup=False,
    )
    return optimizer


def record_detailed_stats(model: GPT, results: Dict, width: int, step: int):
    """Record weight update norms, gradient norms, and spectral norms per parameter group."""
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        # Simplify name for display
        short_name = name.replace('transformer.', '').replace('.weight', '')
        # Gradient Frobenius norm
        grad_norm = p.grad.float().norm().item()
        results['detailed_stats'][width][f'grad norm: {short_name}'].append(grad_norm)
        # Gradient spectral norm (top singular value) — only for 2D weight matrices
        if p.grad.ndim == 2:
            try:
                # svd_lowrank is faster than full SVD when we only need the top singular value
                U, S, V = torch.svd_lowrank(p.grad.float(), q=1)
                results['detailed_stats'][width][f'grad spectral: {short_name}'].append(S[0].item())
            except Exception:
                pass


def record_weight_update_norms(model: GPT, params_before: Dict[str, torch.Tensor],
                                results: Dict, width: int):
    """Record ||delta_W|| (Frobenius) and spectral norm of delta_W for each parameter after optimizer step."""
    for name, p in model.named_parameters():
        if name not in params_before:
            continue
        short_name = name.replace('transformer.', '').replace('.weight', '')
        delta = p.data.float() - params_before[name]
        # Frobenius norm of update
        results['detailed_stats'][width][f'update norm: {short_name}'].append(delta.norm().item())
        # Spectral norm of update — only for 2D weight matrices
        if delta.ndim == 2:
            try:
                U, S, V = torch.svd_lowrank(delta, q=1)
                results['detailed_stats'][width][f'update spectral: {short_name}'].append(S[0].item())
            except Exception:
                pass


def run_coord_check(config: CoordCheckConfig, device: torch.device,
                    x: torch.Tensor, y: torch.Tensor) -> Dict:
    """Run coordinate check across all widths, averaging over multiple seeds."""
    seeds = list(range(config.base_seed, config.base_seed + config.num_seeds))
    results = {
        'widths': [],
        'steps': list(range(config.num_steps)),
        'stats': defaultdict(lambda: defaultdict(list)),       # mean across seeds
        'stats_stderr': defaultdict(lambda: defaultdict(list)), # stderr across seeds
        'losses': defaultdict(list),
        'detailed_stats': defaultdict(lambda: defaultdict(list)),
    }

    for width in config.widths:
        seed_label = f" (seeds {seeds[0]}-{seeds[-1]})" if len(seeds) > 1 else ""
        print(f"\nTraining width={width}...{seed_label}")

        mup_base_width = config.base_width if config.use_mup else 0

        # Collect stats across seeds: {layer: [[step0, step1, ...], ...per seed]}
        seed_stats = defaultdict(list)
        seed_losses = []

        for seed_idx, seed in enumerate(seeds):
            torch.manual_seed(seed)

            model, gpt_config = create_model(width, config, device, mup_base_width=mup_base_width)
            actual_width = gpt_config.n_embd

            if config.use_mup:
                optimizer = setup_optimizer_mup(model, config, actual_width)
            else:
                optimizer = setup_optimizer_sp(model, config, actual_width)

            recorder = ActivationRecorder(detailed=config.detailed)
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
                    print(f"  Step {step}: loss={loss.item():.4f}, layers={list(step_stats.keys())}")

                loss.backward()

                if config.detailed:
                    record_detailed_stats(model, results, actual_width, step)
                    params_before = {name: p.data.float().clone()
                                     for name, p in model.named_parameters()
                                     if p.grad is not None}

                optimizer.step()

                if config.detailed:
                    record_weight_update_norms(model, params_before, results, actual_width)

                optimizer.zero_grad(set_to_none=True)

            seed_losses.append(losses)
            for layer, vals in step_values.items():
                seed_stats[layer].append(vals)

            recorder.remove_hooks()
            del model, optimizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        results['widths'].append(actual_width)

        # Average across seeds
        mean_losses = np.mean(seed_losses, axis=0).tolist()
        results['losses'][actual_width] = mean_losses

        for layer, all_seed_vals in seed_stats.items():
            arr = np.array(all_seed_vals)  # (num_seeds, num_steps)
            means = arr.mean(axis=0).tolist()
            stderrs = (arr.std(axis=0, ddof=1) / np.sqrt(len(seeds))).tolist() if len(seeds) > 1 else [0.0] * len(means)
            results['stats'][actual_width][layer] = means
            results['stats_stderr'][actual_width][layer] = stderrs

        final_losses = [sl[-1] for sl in seed_losses]
        print(f"  Final loss: {np.mean(final_losses):.4f}" + (f" ± {np.std(final_losses, ddof=1)/np.sqrt(len(seeds)):.4f}" if len(seeds) > 1 else ""))

    return results


def plot_coord_check(results: Dict, config: CoordCheckConfig, save_path: Optional[str] = None):
    """Plot coordinate check: one subplot per layer, x=width (log2), y=mean |activation|, lines=steps."""
    widths = results['widths']
    steps = results['steps']
    stats = results['stats']

    layer_names = list(stats[widths[0]].keys())
    n_layers = len(layer_names)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols

    param_type = "muP" if config.use_mup else "SP"
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    step_colors = plt.cm.plasma(np.linspace(0, 1, len(steps)))

    for i, layer in enumerate(layer_names):
        ax = axes[i]
        for s, step in enumerate(steps):
            values = [stats[w][layer][s] for w in widths]
            ax.plot(widths, values, 'o-', color=step_colors[s], linewidth=1.5,
                    label=f'step {step}' if i == 0 else None)
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels(widths, fontsize=7)
        ax.set_title(layer, fontsize=9)
        ax.set_xlabel('Width')
        ax.set_ylabel('Mean |activation|')
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc='best')

    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Coordinate Check ({param_type}): Activation Magnitude vs Width', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_loss_curves(results: Dict, config: CoordCheckConfig, title: str = "", save_path: Optional[str] = None):
    """Plot loss curves across widths to verify HP transfer."""
    widths = results['widths']
    steps = results['steps']
    losses = results['losses']

    fig, ax = plt.subplots(figsize=(5 * 2, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))

    for i, w in enumerate(widths):
        ax.plot(steps, losses[w], label=f'width={w}', color=colors[i], linewidth=2)

    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Curves Across Widths{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation for final loss spread
    final_losses = [losses[w][-1] for w in widths]
    spread = max(final_losses) - min(final_losses)
    ax.annotate(f'Final loss spread: {spread:.4f}', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")

    plt.show()


def plot_comparison(results_sp: Dict, results_mup: Dict, config: CoordCheckConfig, save_path: Optional[str] = None):
    """Plot SP vs muP: one subplot per layer (left=SP, right=muP), x=width (log2), y=mean |activation|, lines=steps."""
    widths = results_sp['widths']
    steps = results_sp['steps']

    layer_names = list(results_sp['stats'][widths[0]].keys())
    n_layers = len(layer_names)

    # n_layers activation rows + 1 loss row, 2 cols (SP | muP)
    n_rows, n_cols = n_layers + 1, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

    step_colors = plt.cm.plasma(np.linspace(0, 1, len(steps)))
    width_colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))

    for row, layer in enumerate(layer_names):
        # Shared y-axis range across SP and muP for this layer
        all_vals = [results_sp['stats'][w][layer][s] for w in widths for s in range(len(steps))] + \
                   [results_mup['stats'][w][layer][s] for w in widths for s in range(len(steps))]
        y_min, y_max = min(all_vals) * 0.9, max(all_vals) * 1.1

        for col, (results, label) in enumerate([(results_sp, 'SP'), (results_mup, 'muP')]):
            ax = axes[row, col]
            for s, step in enumerate(steps):
                values = [results['stats'][w][layer][s] for w in widths]
                ax.plot(widths, values, 'o-', color=step_colors[s], linewidth=1.5,
                        label=f'step {step}' if (row == 0 and col == 0) else None)
            ax.set_xscale('log', base=2)
            ax.set_xticks(widths)
            ax.set_xticklabels(widths, fontsize=7)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'{label}: {layer}', fontsize=9)
            ax.set_xlabel('Width')
            ax.set_ylabel('Mean |activation|')
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=7, loc='best')

    # Loss curves row (log scale so low-loss detail is visible)
    all_losses = [v for r in (results_sp, results_mup) for w in widths for v in r['losses'][w]]
    loss_min, loss_max = min(all_losses) * 0.9, max(all_losses) * 1.1

    for col, (results, label) in enumerate([(results_sp, 'SP'), (results_mup, 'muP')]):
        ax = axes[n_layers, col]
        for j, w in enumerate(widths):
            ax.plot(steps, results['losses'][w], label=f'w={w}', color=width_colors[j], linewidth=2)
        ax.set_yscale('log')
        ax.set_ylim(loss_min, loss_max)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'{label}: Loss Curves')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        final_losses = [results['losses'][w][-1] for w in widths]
        spread = max(final_losses) - min(final_losses)
        ax.annotate(f'Spread: {spread:.4f}', xy=(0.65, 0.95), xycoords='axes fraction', fontsize=9)

    fig.suptitle('Coordinate Check: SP vs muP', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def plot_detailed(results: Dict, config: CoordCheckConfig, save_path: Optional[str] = None):
    """Plot detailed diagnostics: gradient norms, weight update norms, attention logits."""
    widths = results['widths']
    detailed = results['detailed_stats']
    if not detailed or not detailed[widths[0]]:
        print("No detailed stats recorded. Use --detailed flag.")
        return

    # Collect all detailed metric names
    metric_names = sorted(detailed[widths[0]].keys())

    # Group by category
    categories = defaultdict(list)
    for name in metric_names:
        if name.startswith('grad spectral:'):
            categories['Gradient Spectral Norms'].append(name)
        elif name.startswith('grad norm:'):
            categories['Gradient Norms'].append(name)
        elif name.startswith('update spectral:'):
            categories['Update Spectral Norms'].append(name)
        elif name.startswith('update norm:'):
            categories['Weight Update Norms'].append(name)
        elif name.startswith('attn logits'):
            categories['Attention Logit Magnitudes'].append(name)

    for cat_name, names in categories.items():
        n = len(names)
        n_cols = min(4, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        steps = results['steps']
        width_colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))

        for i, name in enumerate(names):
            ax = axes[i]
            for j, w in enumerate(widths):
                values = detailed[w].get(name, [])
                if values:
                    ax.plot(range(len(values)), values, color=width_colors[j],
                            linewidth=1.5, label=f'w={w}' if i == 0 else None)
            ax.set_title(name.split(': ', 1)[-1] if ': ' in name else name, fontsize=8)
            ax.set_xlabel('Step')
            ax.set_ylabel('Norm')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        axes[0].legend(fontsize=7, loc='best')
        param_type = "muP" if config.use_mup else "SP"
        fig.suptitle(f'{cat_name} ({param_type})', fontsize=14)
        plt.tight_layout()

        if save_path:
            cat_slug = cat_name.lower().replace(' ', '_')
            path = save_path.replace('.png', f'_{cat_slug}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved {cat_name} plot to {path}")

        plt.show()


def plot_spectral_vs_width(results: Dict, config: CoordCheckConfig, save_path: Optional[str] = None):
    """Plot spectral norms vs width (like coord check plots) to verify width-independence.

    For each 2D weight matrix, plots the spectral norm of its gradient and update
    at the final training step as a function of model width. Under correct muP/CompleteP
    scaling, these should be flat (width-independent), indicating feature learning
    in all layers. Growing spectral norms indicate lazy/NTK-like behavior.
    """
    widths = results['widths']
    detailed = results['detailed_stats']
    if not detailed or not detailed[widths[0]]:
        print("No detailed stats recorded. Use --detailed flag.")
        return

    # Collect spectral metrics
    spectral_categories = {
        'Gradient Spectral Norms vs Width': [n for n in sorted(detailed[widths[0]].keys()) if n.startswith('grad spectral:')],
        'Update Spectral Norms vs Width': [n for n in sorted(detailed[widths[0]].keys()) if n.startswith('update spectral:')],
    }

    for cat_name, names in spectral_categories.items():
        if not names:
            continue
        n = len(names)
        n_cols = min(4, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        steps = results['steps']
        step_colors = plt.cm.plasma(np.linspace(0, 1, len(steps)))

        for i, name in enumerate(names):
            ax = axes[i]
            for step_idx, step in enumerate(steps):
                values = []
                for w in widths:
                    step_vals = detailed[w].get(name, [])
                    if step_idx < len(step_vals):
                        values.append(step_vals[step_idx])
                    else:
                        values.append(np.nan)
                ax.plot(np.log2(widths), values, 'o-', color=step_colors[step_idx],
                        linewidth=1.2, markersize=4,
                        label=f'step {step}' if i == 0 else None)
            short_name = name.split(': ', 1)[-1] if ': ' in name else name
            ax.set_title(short_name, fontsize=8)
            ax.set_xlabel('log2(width)')
            ax.set_ylabel('Spectral Norm')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        axes[0].legend(fontsize=6, loc='best', ncol=2)
        param_type = "muP" if config.use_mup else "SP"
        fig.suptitle(f'{cat_name} ({param_type}) — flat = width-independent (good)', fontsize=12)
        plt.tight_layout()

        if save_path:
            cat_slug = cat_name.lower().replace(' ', '_').replace(' ', '_')
            path = save_path.replace('.png', f'_{cat_slug}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved {cat_name} plot to {path}")

        plt.show()


def compute_width_dependence(results: Dict) -> Dict[str, float]:
    """Compute how much activations scale with width (slope on log-log plot)."""
    widths = np.array(results['widths'])
    log_widths = np.log2(widths)
    final_step = len(results['steps']) - 1

    slopes = {}
    for layer in results['stats'][widths[0]].keys():
        values = [results['stats'][w][layer][final_step] for w in widths]
        log_values = np.log2(np.array(values) + 1e-10)
        slope, _ = np.polyfit(log_widths, log_values, 1)
        slopes[layer] = slope

    return slopes


def compute_spectral_width_dependence(results: Dict) -> Dict[str, float]:
    """Compute how spectral norms scale with width (slope on log-log plot).

    For correct muP/CompleteP, spectral norms of gradients and updates should
    be width-independent (slope ~0), indicating feature learning in all layers.
    Positive slopes suggest lazy/NTK behavior where features don't evolve.
    """
    widths = np.array(results['widths'])
    log_widths = np.log2(widths)
    detailed = results.get('detailed_stats', {})
    if not detailed or not detailed[widths[0]]:
        return {}

    spectral_names = [n for n in sorted(detailed[widths[0]].keys())
                      if 'spectral' in n]

    slopes = {}
    for name in spectral_names:
        values = []
        for w in widths:
            step_vals = detailed[w].get(name, [])
            # Use the final step value
            values.append(step_vals[-1] if step_vals else np.nan)
        values = np.array(values)
        if np.any(np.isnan(values)) or np.any(values <= 0):
            continue
        log_values = np.log2(values)
        slope, _ = np.polyfit(log_widths, log_values, 1)
        slopes[name] = slope

    return slopes


def main():
    parser = argparse.ArgumentParser(description='muP Coordinate Check')
    parser.add_argument('--widths', type=str, default='128,256,512,1024',
                        help='Comma-separated list of widths to test')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--n-layer', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--use-mup', action='store_true',
                        help='Use muP learning rate scaling')
    parser.add_argument('--base-width', type=int, default=128,
                        help='Base width for muP scaling')
    parser.add_argument('--compare', action='store_true',
                        help='Run both SP and muP and compare')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds to average over (EleutherAI uses 5)')
    parser.add_argument('--detailed', action='store_true',
                        help='Record detailed diagnostics: gradient norms, weight update norms, '
                             'attention logit magnitudes')
    parser.add_argument('--muon-lr-exponent', type=float, default=0.0,
                        help='Muon LR exponent for muP: 1.0 = (base/width)^1 (standard muP), '
                             '0.5 = (base/width)^0.5 (for Frobenius-normalizing optimizers, '
                             'see Yang et al. Section C.1)')

    args = parser.parse_args()

    # Parse widths
    widths = [int(w) for w in args.widths.split(',')]

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load a single batch of real training data (reused every step)
    x, y, vocab_size = load_batch(args.batch_size, args.seq_len, device)

    # Create config
    config = CoordCheckConfig(
        widths=widths,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        seed=args.seed,
        use_mup=args.use_mup,
        base_width=args.base_width,
        detailed=args.detailed,
        muon_lr_exponent=args.muon_lr_exponent,
        num_seeds=args.seeds,
        base_seed=args.seed,
    )

    if args.compare:
        # Run both SP and muP
        print("\n" + "="*60)
        print("Running Standard Parameterization (SP)")
        print("="*60)
        config.use_mup = False
        results_sp = run_coord_check(config, device, x, y)

        print("\n" + "="*60)
        print("Running muP")
        if config.muon_lr_exponent != 1.0:
            print(f"  (Muon LR exponent: {config.muon_lr_exponent})")
        print("="*60)
        config.use_mup = True
        results_mup = run_coord_check(config, device, x, y)

        # Compute slopes
        print("\n" + "="*60)
        print("Width Dependence (slope on log-log plot)")
        print("Expected: ~0 for width-independent, positive = grows with width")
        print("="*60)

        slopes_sp = compute_width_dependence(results_sp)
        slopes_mup = compute_width_dependence(results_mup)

        print(f"\n{'Layer':<20} {'SP Slope':>12} {'muP Slope':>12}")
        print("-"*46)
        for layer in slopes_sp:
            print(f"{layer:<20} {slopes_sp[layer]:>12.4f} {slopes_mup[layer]:>12.4f}")

        # Spectral norm width dependence (if --detailed)
        if config.detailed:
            spectral_slopes_sp = compute_spectral_width_dependence(results_sp)
            spectral_slopes_mup = compute_spectral_width_dependence(results_mup)
            if spectral_slopes_sp:
                print(f"\n{'='*60}")
                print("Spectral Norm Width Dependence (slope on log-log plot)")
                print("Expected: ~0 for feature learning, positive = lazy/NTK regime")
                print("="*60)
                print(f"\n{'Parameter':<30} {'SP Slope':>12} {'muP Slope':>12}")
                print("-"*56)
                for name in spectral_slopes_sp:
                    short = name.split(': ', 1)[-1] if ': ' in name else name
                    sp_val = spectral_slopes_sp.get(name, float('nan'))
                    mup_val = spectral_slopes_mup.get(name, float('nan'))
                    print(f"{short:<30} {sp_val:>12.4f} {mup_val:>12.4f}")

        # Plot comparison
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'coord_check_comparison.png')
        plot_comparison(results_sp, results_mup, config, save_path)

        # Plot detailed diagnostics if requested
        if config.detailed:
            for results, label in [(results_sp, 'SP'), (results_mup, 'muP')]:
                config.use_mup = (label == 'muP')
                detail_save = None
                spectral_save = None
                if args.save_dir:
                    detail_save = os.path.join(args.save_dir, f'detailed_{label.lower()}.png')
                    spectral_save = os.path.join(args.save_dir, f'spectral_{label.lower()}.png')
                plot_detailed(results, config, detail_save)
                plot_spectral_vs_width(results, config, spectral_save)

    else:
        # Run single mode
        param_type = "muP" if config.use_mup else "SP"
        print(f"\n{'='*60}")
        print(f"Running Coordinate Check ({param_type})")
        print(f"{'='*60}")
        print(f"Widths: {widths}")
        print(f"Steps: {config.num_steps}")
        print(f"Base width: {config.base_width}")
        if config.use_mup and config.muon_lr_exponent != 1.0:
            print(f"Muon LR exponent: {config.muon_lr_exponent}")

        results = run_coord_check(config, device, x, y)

        # Compute slopes
        slopes = compute_width_dependence(results)
        print("\n" + "="*60)
        print("Width Dependence (slope on log-log plot)")
        print("Expected for muP: ~0 (width-independent)")
        print("="*60)
        for layer, slope in slopes.items():
            status = "OK" if abs(slope) < 0.1 else "WARN"
            print(f"  {layer}: {slope:+.4f} [{status}]")

        # Spectral norm width dependence (if --detailed)
        if config.detailed:
            spectral_slopes = compute_spectral_width_dependence(results)
            if spectral_slopes:
                print("\n" + "="*60)
                print("Spectral Norm Width Dependence (slope on log-log plot)")
                print("Expected: ~0 for feature learning, positive = lazy/NTK regime")
                print("="*60)
                for name, slope in spectral_slopes.items():
                    short = name.split(': ', 1)[-1] if ': ' in name else name
                    status = "OK" if abs(slope) < 0.1 else "WARN"
                    print(f"  {short}: {slope:+.4f} [{status}]")

        # Loss curve analysis
        final_losses = [results['losses'][w][-1] for w in results['widths']]
        loss_spread = max(final_losses) - min(final_losses)
        print(f"\nFinal loss spread across widths: {loss_spread:.4f}")
        print(f"Expected for muP: low spread (similar losses across widths)")

        # Plot activations
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'coord_check_{param_type.lower()}.png')
        plot_coord_check(results, config, save_path)

        # Plot loss curves
        loss_save_path = None
        if args.save_dir:
            loss_save_path = os.path.join(args.save_dir, f'loss_curves_{param_type.lower()}.png')
        plot_loss_curves(results, config, title=param_type, save_path=loss_save_path)

        # Plot detailed diagnostics if requested
        if config.detailed:
            detail_save = None
            spectral_save = None
            if args.save_dir:
                detail_save = os.path.join(args.save_dir, f'detailed_{param_type.lower()}.png')
                spectral_save = os.path.join(args.save_dir, f'spectral_{param_type.lower()}.png')
            plot_detailed(results, config, detail_save)
            plot_spectral_vs_width(results, config, spectral_save)


if __name__ == '__main__':
    main()
