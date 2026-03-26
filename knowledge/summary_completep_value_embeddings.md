# CompleteP, muP, and Value Embeddings

## Purpose

This note summarizes three things:

1. How value embeddings / value residuals work in ResFormer.
2. What CompleteP changes relative to width-only muP.
3. A draft prescription for how nanochat should think about `value_embeds` and `ve_gate`.

I separate **paper-backed conclusions** from **my proposed prescription** because the literature does not directly cover the exact value-embedding mechanism used in this repo.

## TL;DR

- In attention, `Q` and `K` decide **where to read**, while `V` decides **what content is moved**.
- ResFormer improves deep information flow by injecting an earlier value representation into later attention layers, so later layers can route earlier token-level information with the current attention map.
- CompleteP is about **depth scaling of residual branches**. Its key practical rule is to scale each residual branch contribution to the residual stream by `1/L` as depth `L` grows.
- The cleanest way to apply CompleteP to value embeddings is: **treat the value-embedding path as part of the attention branch**, not as a separate residual shortcut that gets its own unmanaged depth budget.
- For nanochat specifically, I do **not** think the current `sqrt(base_width / n_embd)` VE gate scaling is theoretically grounded. I also do **not** think initializing `value_embeds` “like `c_v`” is a clean μP argument.

## 1. How Value Residuals Work

In a standard transformer attention block, the output is roughly:

```text
A = softmax(QK^T)
U = A V
```

`Q` and `K` choose which tokens attend to which other tokens. `V` carries the content that gets aggregated and sent forward.

The ResFormer paper argues that deep transformers often preserve hidden-state residuals but still lose useful early token-level information in the **value** pathway. Their fix is a **value residual**: later layers mix in the first layer's value representation before applying attention. The result is that later layers can still route early token-level content using their own current attention patterns.

This matters because the value path is the content path inside attention. If it drifts too far from early token information, attention may still be structurally expressive but less effective at preserving or reusing token-local content.

Paper-backed summary:

- ResFormer adds residual structure in the value channel, not only in the hidden-state channel.
- Their analysis suggests later layers effectively learn a `Delta V` correction on top of the reused earlier value state.
- The benefit is better propagation of token-level information through depth.

Source: [Value Residual Learning (ACL 2025)](https://aclanthology.org/2025.acl-long.1375/)

## 2. How nanochat Differs from ResFormer

nanochat is not implementing the exact ResFormer construction.

In [nanochat/gpt.py](/home/amrit/nanochat/nanochat/gpt.py#L100), the value path is:

```text
v = c_v(x)
v = v + gate(x) * ve(token_id)
```

where:

- `c_v(x)` is the learned value projection from the current hidden state.
- `ve(token_id)` is a per-layer token embedding table.
- `gate(x)` is a small learned gate over KV heads.

So the repo is doing **token-conditioned value injection** rather than **reuse of first-layer value states**. Functionally, this still modifies the content path inside attention, but it is a distinct architecture.

Implication: any scaling rule for nanochat's `value_embeds` is an extrapolation from the ResFormer idea, not a direct theorem from the paper.

Relevant local code:

- [nanochat/gpt.py](/home/amrit/nanochat/nanochat/gpt.py#L100)
- [nanochat/gpt.py](/home/amrit/nanochat/nanochat/gpt.py#L261)
- [nanochat/gpt.py](/home/amrit/nanochat/nanochat/gpt.py#L505)

## 3. muP: What It Covers

Standard μP is primarily about **width scaling**.

Its core goal is to preserve:

- stable hidden activation scales,
- stable output scales,
- and maximal feature-learning updates

as width grows.

For transformers, the standard μP playbook includes:

- treating the readout specially,
- scaling readout logits by `1 / width_multiplier`,
- and using width-aware init / optimizer rules for hidden matrices.

The official μP repo also recommends using `1 / d_head` attention scaling rather than `1 / sqrt(d_head)` in standard transformer settings.

Sources:

- [Tensor Programs V / μP](https://arxiv.org/abs/2203.03466)
- [microsoft/mup README](https://github.com/microsoft/mup/blob/main/README.md)

## 4. CompleteP: What It Adds

CompleteP extends the scaling story from width to **depth** in residual networks and transformers.

The core issue is that width-stable networks can still become **lazy** as depth grows. "Lazy" here means deeper layers increasingly behave like their linearization, so the network is technically stable but does not fully exploit nonlinear feature learning.

CompleteP's argument is:

- stability alone is not enough,
- maximal update alone is not enough,
- the parameterization should also preserve **complete feature learning** in every layer as depth increases.

Their practical conclusion is that for deep pre-LN transformers, the residual branch contribution should scale like `1 / L`, not merely `1 / sqrt(L)`, if we want both depth-wise HP transfer and non-lazy behavior.

Useful paper-backed points:

- Only CompleteP achieves reliable depth-wise HP transfer in their experiments.
- Their summary rule is: scale each residual block output by `1 / L` before adding it to the residual stream.
- They also need matching scaling adjustments for LayerNorm, bias learning rates, AdamW epsilon, and weight decay.

Sources:

- [CompleteP paper on arXiv](https://arxiv.org/abs/2505.01618)
- [OpenReview page](https://openreview.net/forum?id=lMU2kaMANl)
- [Cerebras summary](https://www.cerebras.ai/blog/cerebras-at-neurips-2025-nine-papers-from-pretraining-to-inference)

## 5. What CompleteP Implies for Value Embeddings

This is the key conceptual bridge.

CompleteP is fundamentally about the **amount of branch output that reaches the residual stream per layer**.

The value-embedding term does **not** directly update the residual stream on its own. It changes the attention branch content:

```text
U_attn = A (V_current + V_extra)
```

and then the attention branch output is projected and added into the residual stream.

That means the clean interpretation is:

- `value_embeds` belong to the **attention branch internals**,
- but their effect on the model should still obey the **same depth budget as the rest of the attention branch**.

This suggests a practical rule:

> Do not give the VE path its own independent depth shortcut. Make the full attention branch, including the VE contribution, obey the same CompleteP residual scaling.

This is the most important prescription in this note.

## 6. Prescription Draft for nanochat

This section is my recommendation, not a published theorem.

### 6.1 Forward Pass

Recommended structure:

1. Build the attention content as:

```text
v_total = c_v(x) + gate(x) * ve(idx)
```

2. Compute attention normally:

```text
y_attn = A v_total
```

3. Apply output projection.
4. Apply the **depth scaling on the full attention branch output**, not only on `gate * ve`.

In other words, if nanochat adopts CompleteP, the scaling target should be the whole branch contribution to `x`, not a VE-only special case.

### 6.2 Width Scaling

Treat `value_embeds` as more **embedding-like** than **hidden-matrix-like** for μP width scaling.

Why:

- `c_v` is a fan-in projection from hidden coordinates.
- `ve(idx)` is a lookup table.
- They enter the same tensor slot, but they are not born from the same scaling geometry.

Therefore:

- do **not** justify `value_embeds` init by saying they should match `c_v` because they share shape;
- instead check whether `gate * ve` has width-stable coordinate magnitude and width-stable update size.

### 6.3 Initialization

Current code initializes `value_embeds` with the same `1 / sqrt(n_embd)`-style scale as `c_v`. I do not think that is a strong argument.

Draft recommendation:

- initialize `value_embeds` with an **embedding-style** rule, then empirically calibrate so `gate * ve` is on the same order as `c_v(x)` at the base width;
- initialize `ve_gate` small enough that the VE path starts as a correction, not a dominant route.

Practical target:

- at init, `||gate * ve||` should be a modest fraction of `||c_v(x)||`, not far larger and not vanishingly smaller.

### 6.4 Optimizer Grouping

Recommended default:

- keep `value_embeds` in an AdamW-style embedding/non-matrix group;
- keep `ve_gate` with other learned projection-like parameters unless there is strong evidence it should be separated.

I would **not** assume `value_embeds` should use hidden-matrix μP LR scaling.

I would also avoid ad hoc VE-specific width multipliers unless they are justified by coord checks.

### 6.5 What I Would Remove First

If I had to simplify the current implementation before re-deriving it carefully, I would first remove the VE-specific forward scaling:

```text
gate *= sqrt(base_width / n_embd)
```

in [nanochat/gpt.py](/home/amrit/nanochat/nanochat/gpt.py#L105).

Reason: I found no paper-backed justification for that special μP factor on the VE path.

## 7. Validation Plan

To make this prescription credible, nanochat needs VE-specific diagnostics.

Minimum checks:

1. Coord check over width for:
   - `c_v(x)`
   - `ve(idx)`
   - `gate(x)`
   - `gate * ve`
   - `A c_v(x)`
   - `A (gate * ve)`
   - final attention branch output
2. Depth check over `L` for the same quantities if CompleteP-style scaling is added.
3. Update-size tracking for `value_embeds` and `ve_gate`.
4. Ablations:
   - current init vs embedding-style init,
   - VE gate scaling on vs off,
   - branch-level CompleteP scaling only vs VE-specific scaling.

Success criterion:

- the VE path should remain nontrivial, width-stable, and depth-stable,
- without bypassing the intended residual-branch budget.

## 8. 5090 Experiment Results

I implemented a first-pass VE-path prescription on branch `completep-ve-experiment`:

- removed the VE-specific `sqrt(base_width / n_embd)` gate shrink,
- changed `value_embeds` to an embedding-style constant-std init (`ve_init_std = 0.10`),
- aligned `value_embeds` optimizer treatment with the main embedding table,
- and extended the coord check to record `value embed`, `value current`, `value gate`, and `value branch`.

### Coord Check Compare

Command:

```bash
MPLBACKEND=Agg ./.venv/bin/python -m scripts.mup_coord_check \
  --compare --widths 128,256,512,1024,2048 \
  --save-dir temp/completep_ve_gpu_coord
```

Key slope results:

| Metric | SP slope | muP slope | Read |
|--------|---------:|----------:|------|
| `value embed.1` | -0.4647 | -0.0025 | big improvement |
| `value current.1` | -0.0479 | -0.0205 | fine |
| `output logits` | 0.2309 | 0.0437 | improved |
| `value gate.1` | 0.1978 | 0.2114 | still bad |
| `value branch.1` | -0.2709 | 0.2095 | still bad |
| `attn output.1` | -0.3338 | 0.2494 | still bad |

Interpretation:

- The embedding/init side of the VE path is much better under the new prescription.
- The remaining width drift is now concentrated in the **gated VE contribution after it enters the attention branch**.
- This strongly suggests the next problem is not the `value_embeds` table itself, but the scaling of the **gate and branch composition inside attention**.

### Transfer Check Compare

Command:

```bash
MPLBACKEND=Agg ./.venv/bin/python -m scripts.mup_transfer_check \
  --compare --widths 128,256,512,1024,2048 \
  --save-dir temp/completep_ve_gpu_transfer
```

Optimal LR spread across `128 -> 2048`:

- SP: `1.000` log2
- muP: `0.500` log2

So with the new VE treatment, muP shows a **2.0x improvement in LR transfer** over SP on the 5090 run.

Interpretation:

- The VE-path changes do **not** break μP transfer.
- They make the width-transfer story better than SP up to width 2048.
- But they do **not** fully solve VE coordination inside the attention block.

### What This Means

The current branch validates two points:

1. Treating `value_embeds` as embedding-like was the right move.
2. The remaining problem has shifted to the **gate-controlled branch composition**, which still grows with width.

So the next experiment should target:

- gate parameterization,
- gate init,
- and whether the VE term should be normalized or budgeted at the attention-branch level more explicitly.

## 9. Bottom Line

The cleanest synthesis is:

- **μP handles width**.
- **CompleteP handles depth**.
- **Value embeddings should be treated as attention-branch content modifiers whose total contribution is governed at the branch level**.

So the safest design principle for nanochat is:

> Treat `value_embeds` as embedding-like for width parametrization, but treat their effect as part of the attention residual branch for depth parametrization.

That is the prescription I would start from.

## Sources

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [microsoft/mup README](https://github.com/microsoft/mup/blob/main/README.md)
- [Don't be lazy: CompleteP enables compute-efficient deep transformers](https://arxiv.org/abs/2505.01618)
- [CompleteP on OpenReview](https://openreview.net/forum?id=lMU2kaMANl)
- [Cerebras summary of CompleteP](https://www.cerebras.ai/blog/cerebras-at-neurips-2025-nine-papers-from-pretraining-to-inference)
- [Value Residual Learning (ResFormer), ACL 2025](https://aclanthology.org/2025.acl-long.1375/)
- Local implementation references:
  - [nanochat/gpt.py](/home/amrit/nanochat/nanochat/gpt.py#L100)
  - [nanochat/gpt.py](/home/amrit/nanochat/nanochat/gpt.py#L261)
  - [nanochat/gpt.py](/home/amrit/nanochat/nanochat/gpt.py#L388)
