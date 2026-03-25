# muP Changes & Assumptions

## Changes Made (on `mup_hp_sweep` branch)

### 1. Output logit scaling (`gpt.py` forward)
```python
logits *= self.config.mup_base_width / self.config.n_embd  # = 1/m_d
```
**Assumption:** Standard muP rule. 1/m_d (not 1/sqrt(m_d)) is required for all training steps, not just init.

### 2. lm_head init std = 0.02 under muP (vs 0.001 for SP)
**Assumption:** Larger init gives stronger initial logit signal. The logit scaling in forward handles width-independence, so a fixed larger init is safe. Taken from EleutherAI reference.

### 3. c_proj / mlp.c_proj non-zero init under muP
SP uses zero init (transformer blocks start "off"). muP uses `uniform(-s, s)` with `s = sqrt(3)/sqrt(n_embd)`.
**Assumption:** "Zero init causes attn/FFN outputs to vanish as width increases with muP LR scaling." **This assumption was written when muon_lr_exponent was expected to be non-zero. With exponent=0, the rationale may no longer hold.** Needs testing: does zero init work fine with muon_lr_exponent=0?

### 4. No width-dependent LR scaling for Muon (muon_lr_exponent=0)
**Assumption:** Muon's Newton-Schulz orthogonalization normalizes ||update||_F ≈ 1 regardless of width, making updates already O(1). "Empirically confirmed: exponent 0 and 1 give identical transfer behavior." **That empirical test was only at small widths (256→512). Exponent 0.5 is theoretically motivated but untested.**

### 5. No width-dependent LR scaling for AdamW embedding/output groups
- `emb_lr_scale = 1.0` (input embeddings)
- `output_lr_scale = 1.0` (lm_head — logit scaling handles gradients)
**Assumption:** Standard muP. Input embeddings don't need scaling. Output handled by forward-pass logit scaling + Adam normalization.

### 6. Value embedding LR scaling: NO width scaling (ve_lr_scale = 1.0)
Tested `ve_lr_scale = width_ratio` (= 1/m_d) on 2026-03-24. **Reverted after ablation.**
- Coord check: VE fix introduced -0.54 slope at attn_output.1 (vs 0.19 without fix)
- Transfer check: VE fix worsened muP spread from 1.0 to ~1.3 (log2 scale)
- Best achievable loss unchanged (0.0162 either way)
**Conclusion:** VEs behave like input embeddings for LR scaling purposes, not like hidden layers. Keep `ve_lr_scale = 1.0`.

### 7. Attention scaling unchanged (QK-norm replaces standard muP attention rule)
Standard muP says use Q^TK/d_head (not 1/sqrt(d_head)). nanochat uses QK-norm which makes Q,K unit vectors, so attention scores are bounded in [-1,1] regardless of dimension.
**Assumption:** QK-norm achieves width-independent attention without needing the standard muP scaling rule. head_dim=128 is constant across widths.

### 8. Hidden layer init variance = 1/n_embd
Variance = 1/(base_width * m_d) = σ²_base/m_d where σ²_base = 1/base_width.
**Assumption:** This matches the standard muP rule (init variance ∝ 1/m_d). No change needed.

## Unvalidated / Untested

- **No valid d14 production run exists.** All previous runs were data-starved (2 ClimbMix shards, epoch=13). Results are meaningless.
- **muon_lr_exponent=0 vs 0.5:** Theory suggests 0.5 for Frobenius-norm optimizers. Only tested 0 vs 1 at small scale.
- **c_proj non-zero init:** May not be needed with exponent=0. Untested ablation.
- **~~VE fix~~:** Tested and reverted. VE LR scaling (1/m_d) hurts both coord check and transfer.
- **Depth scaling (CompleteP):** Not implemented. muP only handles width scaling; depth scaling may also be needed for transfer across model sizes.
- **Sweep-tuned HPs (LR=0.05, WD=0.2):** Found optimal at width=256 with production batch size. Never validly tested at production width due to data starvation.
