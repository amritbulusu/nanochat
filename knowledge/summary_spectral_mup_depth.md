# Spectral Condition for μP under Width-Depth Scaling

**Paper:** arXiv:2603.00541 (Feb 2026)
**Authors:** Zheng, Wang, Zhang, Li (Renmin University / ByteDance Seed)

## Core Contribution

A unified spectral framework for μP under **joint width-depth scaling**. Introduces a single "spectral μP condition" that characterizes how weight norms and per-step update norms should scale with width (n) and depth (L). Recovers existing μP results (SGD, AdamW, CompleteP) as special cases, and extends to Muon, Muon-Kimi, Shampoo, SOAP, SSO, Lion, Sophia.

## The Spectral μP Condition

For a residual network with L blocks, each containing weight matrices W_l:

1. **Initial condition**: block multiplier × weight norm products = Θ(1/L) per hidden block
2. **First-order update condition**: α_l × ||ΔW_l|| × ||W_l|| = Θ(1/L) — each block's per-step contribution is O(1/L)
3. **Second-order update condition**: automatically satisfied when (1) and (2) hold (for block depth ≥ 2)

Key result: For all optimizers with 2-layer residual blocks, the block multiplier must be α_l = Θ(1/L).

## Muon-Kimi μP (Table 1 — main result)

Base model: (n_base=256, L_base=4). Scaling ratios: r_n = n/n_base, r_L = L/L_base.

| | Input weights | Hidden weights | Output weights |
|---|---|---|---|
| **Block Multiplier** | α_base | **α_base / r_L** | **α_base / r_n** |
| **Initial Variance** | σ²_base/d₀ or σ²_base | σ²_base / r_n | **σ²_base** (not /r_n) |
| **Learning Rate** | η_base | **η_base / √r_n** | η_base |

Key: hidden LR scales with **width only** (1/√r_n), **NO depth scaling**. Block multiplier scales as **1/r_L** (depth scaling needed in forward pass).

## Original Muon μP (Table in Appendix B.2)

| | Input weights | Hidden weights | Output weights |
|---|---|---|---|
| **Block Multiplier** | α_base | **α_base / r_L** | **α_base / r_n** |
| **Initial Variance** | σ²_base/d₀ or σ²_base | σ²_base / r_n | **σ²_base** |
| **Learning Rate** | **η_base × √r_n** | **η_base** | **η_base × √r_n** |
| **Weight Decay** | λ_base/√r_n | λ_base | λ_base/√r_n |

Key difference from Muon-Kimi: hidden LR = η_base (**NO width OR depth scaling**). Input/output LR scale as √r_n.

The reason: Muon's SVD-projected update U@V^T has ||A_l||_rms = √(n_in/n_out) = Θ(1) for hidden layers. The RMS norm is already O(1), so no LR compensation needed. Muon-Kimi's RMS normalization prefactor (0.2√max(n_in,n_out)) changes the update scale, requiring 1/√r_n LR correction.

## Critical Finding: Normalization Layers Mask Depth Scaling

The paper's own experimental data reveals:

**With LayerNorm** (Table D.5):
- SP optimal LR = log₂(η_base) = -7 for ALL depths 4→256 (transfers perfectly!)
- μP optimal LR = log₂(η_base) = -7 for ALL depths 4→256 (also perfect!)
- Conclusion: **Both SP and μP transfer LR across depths when LayerNorm is present**

**Without LayerNorm** (Table D.7):
- SP: training becomes unstable (NaN), optimal LR shifts at large depths
- μP: remains stable, optimal LR transfers for L ≥ 32

The paper explicitly acknowledges this (Section 5.1): *"One may notice that under SP the optimal learning rate appears to transfer reasonably well along the depths in our experiments. We attribute this to two factors: (1) the tested depths are still moderate; (2) modern architectural components such as LayerNorm and QKNorm substantially enhance training stability, partially masking the underlying scaling pathology of SP at practical depths."*

## Implications for nanochat

### Architecture mapping
- nanochat uses Polar Express (Newton-Schulz) for Muon → functionally equivalent to **original Muon** (not Muon-Kimi)
- nanochat has **RMSNorm** (learnable=False) + **QK normalization** → equivalent to LayerNorm+QKNorm in the paper
- nanochat operates at d12-d24 (depths 12-24)

### Why "no depth scaling looks better" at depths 2-16
The paper's data directly explains this: **with normalization layers, SP already transfers LR perfectly across depths 4-256**. At nanochat's operating depths (12-24) with RMSNorm and QKNorm, branch scaling provides essentially zero benefit. The pathology only manifests either:
1. At extreme depths without normalization, OR
2. With normalization removed

### What depth scaling actually provides
1. **Theoretical correctness**: Ensures the spectral condition holds regardless of depth
2. **Stability insurance**: Would prevent divergence if normalization were ever removed/weakened
3. **Marginal loss improvement at extreme depths**: At L=256 with LayerNorm, μP achieves 3.678 vs SP's 3.688 — a 0.01 improvement
4. **Feature norm stability**: Even with normalization, feature norms grow more with SP than μP at large depths

### Practical recommendation
For nanochat at d12-d24 with RMSNorm + QKNorm:
- **Branch scaling 1/L is theoretically correct** but provides negligible empirical benefit at these depths
- **No LR depth scaling needed** — for original Muon, hidden LR doesn't scale with depth OR width
- The existing width muP + RMSNorm + QKNorm already handles depth stability at practical scales
- CompleteP becomes essential only at extreme depths (L > 64) or without normalization
- If pursuing CompleteP, α=1 is correct: hidden LR factor = m_L^{α-1} = 1 (no LR change needed)

### Muon's variance reduction step
nanochat's Muon has a per-parameter "variance reduction" second-moment normalization step after Polar Express. This is not present in the paper's analysis of original Muon. It may provide additional update scale control that further diminishes the need for explicit depth scaling. This is an open question.

## Key Equations

For original Muon, hidden layer update: ΔW_l = -η_l × U_l × V_l^T (SVD projection)
- ||A_l||_rms = ||U_l V_l^T||_rms = √(n_in/n_out) = Θ(1) for hidden layers
- First-order condition: α_l × η_l × ||A_l||_rms × ||W_l||_rms = Θ(1/L)
- With α_l = Θ(1/L), ||A_l||_rms = Θ(1), ||W_l||_rms = Θ(1): need η_l = Θ(1)

For Muon-Kimi, prefactor changes ||A_l||_rms = Θ(√n), so η_l = Θ(1/√n).

## Reference
- GitHub: https://github.com/ML-GSAI/Width-Depth-muP
- Based on experiments with GPT-2 style models, Muon-Kimi + AdamW, OpenWebText, base model (256, 4)
