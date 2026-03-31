---
title: "What Can We Actually Modify in Attention? A Taxonomy of Every Variant"
description: "Attention has five independent axes of variation: number of heads, KV representation, attention pattern, storage and caching, and layer-level architecture. Every variant in the literature is a modification of exactly one or two of these axes."
date: 2026-03-31
tags: [machine-learning, attention, transformers, architecture]
order: 1
---

Every attention variant you will ever encounter — Multi-Query Attention, GQA, FlashAttention, Longformer, Linear Attention, MLA, Mamba — is a modification of vanilla multi-head attention. But the papers rarely frame it that way. They present the new thing from scratch, and you are left wondering: what actually changed?

This blog answers that question by breaking the vanilla attention formula into its constituent choices. There are exactly five axes on which you can make a different decision. Every variant in the literature sits at a specific point in this five-dimensional space. Once you see the space, every subsequent paper becomes: "which axis did this modify, and why?"

This is the most important blog in the series. The twelve blogs that follow are each one axis, one variant, one paper.

---

## The Running Model

We need a concrete model to anchor the numbers throughout. Fix:

- $d_\text{model} = 512$
- $h = 8$ heads
- $d_k = d_v = 64$ (so $h \cdot d_k = 512 = d_\text{model}$)
- $L = 12$ layers
- $N = 512$ tokens (training), $n$ tokens generated so far (inference)
- fp16 throughout (2 bytes per element)

This is roughly BERT-base. The numbers will scale cleanly and are easy to verify.

---

## Vanilla Attention, Written with Every Choice Made Explicit

The standard multi-head attention formula is:

$$
\text{MHA}(X) = W_O \cdot \text{concat}(o^1, \ldots, o^h)
$$

where for each head $i \in \{1, \ldots, h\}$:

$$
Q^i = X W_Q^i, \quad K^i = X W_K^i, \quad V^i = X W_V^i
$$

$$
A^i = \text{softmax}\!\left(\frac{Q^i {K^i}^T + M}{\sqrt{d_k}}\right), \quad o^i = A^i V^i
$$

Every symbol here is a choice:

| Symbol | Current choice | What could change |
|---|---|---|
| h | 8 separate heads | Fewer, shared, or grouped heads |
| $W_K^i, W_V^i$ | One per head, full rank | Shared across heads, or low-rank |
| M | Causal mask (−∞ for future positions) | Local window, global tokens, cross |
| KV persistence | Nothing saved across steps | Cache K, V, or a compressed latent |
| Layer placement | Sequential: Attn → AddNorm → FFN → AddNorm | Parallel with FFN, different norm |

These five rows are the five axes. We go through each one.

---

## Axis 1: Number of Heads

### What h controls

h heads means h separate projections of the input into (Q, K, V) space, h separate attention computations, and concatenation at the end.

**Parameter cost** for the projection matrices: 4 matrices per head ($W_Q, W_K, W_V$ each $d_\text{model} \times d_k$, plus $W_O$ at $d_\text{model} \times d_\text{model}$):

$$
\text{params per layer} = h \cdot 3 \cdot d_{\text{model}} \cdot d_k + d_{\text{model}}^2
$$

With our model: $8 \cdot 3 \cdot 512 \cdot 64 + 512^2 = 786{,}432 + 262{,}144 = 1{,}048{,}576 \approx 1\text{M}$ per layer.

**What multiple heads buy.** Each head attends with its own learned projection. Head 1 might learn to track syntactic dependencies; head 5 might track coreference. The concatenation aggregates all signals. This is the same mechanism as having h separate embedding spaces that each answer a different question about the input.

**The tradeoff.** More heads → more capacity, but each head has dimension $d_k = d_\text{model}/h$. Halve the heads, each head gets twice the width. At extreme cases: $h=1$ is single-head attention (full-width representations, one signal); $h=512$ is degenerate (each head operates on a scalar).

**Variants on this axis:** none of the major variants actually change h. What they change is how many unique K/V projections exist for those h heads — that is Axis 2.

---

## Axis 2: KV Representation

### The default: one K and V projection per head

Vanilla MHA has $h$ query heads, $h$ key heads, and $h$ value heads. The query for head $i$ uses $W_Q^i$; the key uses $W_K^i$; the value uses $W_V^i$. All three projection matrices are unique per head.

**KV parameter cost** in our model (K and V only):

$$
\text{params}_{KV} = h \cdot 2 \cdot d_{\text{model}} \cdot d_k = 8 \cdot 2 \cdot 512 \cdot 64 = 524{,}288
$$

**KV cache cost per token** during inference (more on this in Axis 4):

$$
\text{bytes per token} = L \cdot h \cdot 2 \cdot d_k \cdot 2 = 12 \cdot 8 \cdot 2 \cdot 64 \cdot 2 = 24{,}576 \text{ bytes} \approx 24\text{ KB}
$$

At 4096 tokens: 98 MB. At 128K tokens: 3 GB. The h factor in this formula is where the KV cache pressure comes from.

### What you can change

**Multi-Query Attention (MQA)**: all h query heads share one K head and one V head. Instead of $W_K^1, \ldots, W_K^h$, there is a single $W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$.

$$
Q^i = X W_Q^i \quad \forall i, \qquad K = X W_K, \qquad V = X W_V
$$

Head i attends: $A^i = \text{softmax}(Q^i K^T / \sqrt{d_k})$, $o^i = A^i V$.

KV cache per token: $L \cdot 1 \cdot 2 \cdot d_k \cdot 2 = 12 \cdot 2 \cdot 64 \cdot 2 = 3{,}072 \text{ bytes}$ — exactly h = 8 times smaller.

**Grouped-Query Attention (GQA)**: h query heads, g KV heads, h/g queries per KV group. Intermediate point between MHA and MQA. With g=2 (4 queries per KV group), KV cache is h/g = 4× smaller.

**Multi-head Latent Attention (MLA, DeepSeek)**: instead of caching K and V directly, project the input to a low-rank latent $c_t \in \mathbb{R}^{d_c}$ where $d_c \ll h \cdot d_k$. K and V are recovered from $c_t$ via learned up-projections at inference time. The cache stores only $c_t$ — $d_c$ elements per token instead of $2 \cdot h \cdot d_k$.

With $d_c = 512$ vs. $2 \cdot h \cdot d_k = 2 \cdot 8 \cdot 64 = 1024$: cache is 2× smaller. At larger $h$, the compression is proportionally larger.

**Summary of KV variants:**

| Variant | KV heads | Cache per token | Quality |
|---|---|---|---|
| MHA | h = 8 | 24 KB | Best |
| GQA (g=2) | g = 2 | 6 KB | Near-MHA |
| MQA | 1 | 3 KB | Slight degradation |
| MLA | — (latent d_c) | 2d_c bytes | Competitive |

---

## Axis 3: Attention Pattern

### The default: full (dense) attention with causal mask

The mask $M$ in the attention formula is $-\infty$ for disallowed positions and $0$ for allowed ones. The causal mask (used in decoder self-attention) sets $M[i,j] = -\infty$ for all $j > i$. This means every position can attend to all positions at or before it.

**Full attention pattern**: every query attends to every allowed key. The attention matrix $A$ has $O(N^2)$ non-zero entries. This costs $\Theta(N^2 d)$ FLOPs and $\Theta(N^2)$ memory — the quadratic wall from the previous blog.

### What you can change

**Local (sliding window) attention**: position $i$ only attends to positions $[i-w, i+w]$ for window size $w$. Non-zero entries: $O(Nw)$. FLOPs: $O(Nwd)$ — linear in $N$. Longformer uses this as the default pattern for most tokens.

**Global + local**: a few special tokens (e.g., [CLS], task tokens) attend to all positions ($O(N)$ non-zeros for that token), while all other tokens use local windows. BigBird, Longformer for classification.

**Cross-attention**: queries come from one sequence (decoder), keys and values from another (encoder output). The pattern is fully dense but $N_q$ and $N_k$ may differ. Shape of $A$: $N_q \times N_k$.

**Bidirectional vs. causal**: encoder self-attention uses M = 0 everywhere (no mask) — every position attends to every other. Decoder self-attention uses causal mask — only past and present.

**Block-sparse**: a learned or fixed pattern of $B \times B$ blocks that are computed; all others are set to $-\infty$. FlashAttention directly supports block-sparse patterns by skipping zero blocks in the tiling loop.

**Name**: the attention pattern is specified by the *attention bias* or *attention mask* $M$. Any function of positions $(i, j) \to \{0, -\infty\}$ defines a valid pattern. The pattern determines the sparsity structure of $A$.

**Key insight**: changing the pattern changes FLOPs and memory (some patterns reduce both to $O(N)$), but it also changes what information each position can access. Local windows are memory-efficient but cannot propagate information more than $w$ steps per layer. Global tokens recover long-range connectivity at $O(N)$ extra cost.

---

## Axis 4: Storage and Caching

### The default: stateless (no cache)

During training, attention is computed over the full context for all positions simultaneously. Nothing is saved between forward passes except the learned weights. Memory is dominated by activations: $O(N^2)$ per layer for the attention matrices, $O(Nd)$ for everything else.

### What you can change

**KV cache (autoregressive inference)**: when generating token $t+1$, all previous K and V tensors are cached. The new query attends over the full $[1, t]$ context using the cache. Per-token memory: $h \cdot 2 \cdot d_k \cdot L$ (as computed in Axis 2). Growth: linear in sequence length, constant per step.

**No KV cache (recurrent models)**: instead of caching the full K and V history, maintain a fixed-size state (typically $d_\text{state} \times d_\text{model}$ or similar). New tokens update the state. Memory: $O(1)$ per step. The tradeoff: lossy — the model cannot retrieve arbitrary past tokens.

**Sliding-window cache (Mistral, Gemma)**: only cache the last $w$ tokens. Memory: $O(w)$ regardless of sequence length. Positions outside the window cannot be attended to directly (but information propagates via multiple layers).

**Paged attention (vLLM)**: physical memory for KV caches is managed in fixed-size pages, allocated on demand. This is an implementation detail (not a model change), but it drastically improves memory utilization when serving many requests with different context lengths simultaneously.

**Quantized KV cache**: store K and V in int8 or int4 instead of fp16. Halves or quarters the cache size at the cost of some precision. In our running model: 24 KB/token → 12 KB/token in int8.

**Why this axis is separate from Axis 2.** Axis 2 modifies *what is computed* (fewer or compressed K/V projections). Axis 4 modifies *what is stored* across steps. MQA (Axis 2) reduces cache size by reducing h. Sliding-window cache (Axis 4) reduces cache size by bounding the stored history. Both help, and they compose.

---

## Axis 5: Layer-Level Architecture

### The default: Post-norm, sequential Attn → FFN

The vanilla transformer block (from "Attention Is All You Need") is:

$$
x' = \text{LayerNorm}(x + \text{Attention}(x))
$$

$$
x'' = \text{LayerNorm}(x' + \text{FFN}(x'))
$$

This is the *Post-norm* arrangement: normalization happens after the residual addition. It is numerically unstable at initialization and typically requires careful learning rate warmup.

### What you can change

**Pre-norm (most modern models)**: normalize the input *before* the sublayer, not after:

$$
x' = x + \text{Attention}(\text{LayerNorm}(x))
$$

$$
x'' = x' + \text{FFN}(\text{LayerNorm}(x'))
$$

Gradients flow through the residual path without passing through LayerNorm. Training is more stable; warmup requirements are reduced. GPT-2, LLaMA, and essentially every model trained after 2020 uses pre-norm.

**Parallel attention + FFN (PaLM, Falcon)**: compute attention and FFN on the same input simultaneously and add both to the residual:

$$
x' = x + \text{Attention}(\text{LayerNorm}(x)) + \text{FFN}(\text{LayerNorm}(x))
$$

Saves one LayerNorm pass. More importantly, attention and FFN projections can be fused into a single matrix multiply on hardware. ~15% throughput improvement.

**RMSNorm instead of LayerNorm**: LayerNorm subtracts the mean and divides by standard deviation, then applies learned scale and shift. RMSNorm skips the mean subtraction — divides only by root mean square, applies only scale:

$$
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}} \cdot \gamma_i
$$

Fewer operations, no shift parameter, comparable quality. LLaMA, Mistral, Gemma all use RMSNorm.

**No attention at certain layers**: some architectures (Jamba, Zamba) alternate between attention layers and SSM (state space model) layers. The motivation: SSM layers handle long-range dependencies with O(N) compute; attention layers are retained for precision on specific tasks. This is the most aggressive modification on this axis — replacing attention with a different mechanism entirely.

**Why this axis is rarely the punchline.** Layer-level changes (pre-norm, RMSNorm, parallel sublayers) are important engineering decisions, but they do not change what attention *does*. They change how the signal flows and how efficiently the computation runs. The papers that introduce these changes (GPT-2, LLaMA, PaLM) mention them as implementation details. They are axis 5, but the story is elsewhere.

---

## The Variant Map

With five axes defined, every major attention variant maps to a specific location:

| Variant | Axis 1 (Heads) | Axis 2 (KV rep) | Axis 3 (Pattern) | Axis 4 (Cache) | Axis 5 (Architecture) |
|---|---|---|---|---|---|
| Vanilla MHA | h heads | 1 KV per head | Full | No cache | Post-norm, sequential |
| MQA | h heads | 1 shared KV | Full | KV cache | — |
| GQA | h heads | g < h KV heads | Full | KV cache | — |
| MLA | h heads | Low-rank latent | Full | Latent cache | — |
| Longformer | h heads | 1 KV per head | Local + global | KV cache | — |
| Linear Attn | h heads | 1 KV per head | Kernel approx | Running state | — |
| Mamba | No heads | No KV | No explicit pattern | Recurrent state | Replace Attn |
| LLaMA | h heads | 1 KV per head | Full | KV cache | Pre-norm, RMSNorm |

**Reading the table**: LLaMA is vanilla MHA on axes 1–4, with axis 5 changed to pre-norm + RMSNorm. Mamba abandons attention entirely. Note that FlashAttention is absent — it is not a variant, it is a hardware-aware implementation of the same full attention. It does not appear in this table for the same reason CUDA kernels do not: it changes how attention runs, not what it computes.

---

## Why This Framing Matters

Most discussions of attention variants present each technique as a standalone invention. "Here is MQA. Here is GQA. Here is linear attention." After reading three such papers, you have three disconnected techniques.

The taxonomy shows the underlying structure: each technique answers one of five questions about the vanilla attention formula. Once you understand which question is being answered and why, you can:

1. **Predict the tradeoff** before reading the evaluation. GQA modifies Axis 2 → smaller KV cache, possibly slightly reduced quality. Longformer modifies Axis 3 → O(N) compute, but each position has limited direct access to far-away positions.

2. **Combine techniques correctly**. GQA and sliding-window cache are axes 2 and 4 — they are independent, so they compose. GQA + pre-norm is axes 2 and 5 — also independent. MQA + linear attention would modify axes 2 and 3 — may interact (the key sharing affects the linear factorization).

3. **Understand what a new paper actually claims**. When a paper says "our method reduces memory by 8×," ask: which axis? If axis 2 (KV representation), it is a cache reduction. If axis 3 (pattern), it may also reduce FLOPs. If axis 4 (caching strategy), it may be a serving infrastructure improvement with no quality change at all.

---

## Summary

Vanilla attention has five independent axes of variation:

**Axis 1 — Heads**: how many parallel Q/K/V projections. Almost always fixed; the interesting variation is in what the heads share (Axis 2).

**Axis 2 — KV representation**: how many unique K and V projections exist across heads. Modifying this directly controls KV cache size. Variants: MQA, GQA, MLA.

**Axis 3 — Attention pattern**: which positions attend to which. Full attention is O(N²). Sparse patterns (local, global, block) reduce to O(N) or O(Nw). Variants: Longformer, BigBird, block-sparse FlashAttention.

**Axis 4 — Storage and caching**: what persists across inference steps. The standard KV cache is one choice; sliding windows, recurrent states, and quantized caches are others.

**Axis 5 — Layer-level architecture**: pre/post-norm, norm type, sequential/parallel sublayers. These affect training stability and throughput, not what attention computes.

Every paper in Phase 3 of this series modifies exactly one or two of these axes. The next blog begins Phase 3 with the first axis: KV representation.

---

*Previous: [Why Vanilla Attention Breaks at Scale](/blog/attention-why-vanilla-breaks)*  
*Next: Multi-Query and Grouped-Query Attention — Axis 2*
