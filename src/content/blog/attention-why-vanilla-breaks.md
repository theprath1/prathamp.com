---
title: "Why Vanilla Attention Breaks at Scale: The O(n²) Wall"
description: "A precise accounting of why standard attention becomes infeasible at long sequences: exact FLOP counts, memory costs, KV-cache growth, and the HBM bandwidth bottleneck that makes memory worse than compute."
date: 2026-03-31
tags: [machine-learning, attention, transformers, efficiency, flashattention]
order: 2
---

The first two blogs built attention from scratch: the alignment model, the Q/K/V abstraction, scaled dot-product, multi-head. Everything fits on a whiteboard. The formula is clean.

Now double the sequence length. Double it again. Watch the cost explode.

This blog is a precise accounting of *why* vanilla attention fails at scale. Not vague gestures at "quadratic complexity" — exact FLOP counts, byte counts, and a measurement of which bottleneck actually kills you first.

---

## The Running Example, Scaled Up

We've been working with three vectors. That was enough to derive the mechanism. It is not enough to understand the scaling problem.

Fix a concrete model: **d_model = 512, d_k = d_v = 64, h = 8 heads, L = 12 layers**. This is roughly BERT-base. We'll vary sequence length n from 128 to 16 384.

For each bottleneck, we'll state a general formula, then plug in numbers.

---

## Bottleneck 1: Compute

### Counting FLOPs in QK^T

The attention score matrix S = QK^T / √d_k. Focus on QK^T first.

Q has shape (n, d_k). K has shape (n, d_k). The product QK^T has shape (n, n).

**Entry-by-entry cost.** Each entry S[i,j] = q_i · k_j is a dot product of two d_k-dimensional vectors. That costs d_k multiplications and d_k − 1 additions ≈ 2d_k floating-point operations.

There are n² entries total, so:

$$\text{FLOPs}(QK^T) = 2 d_k \cdot n^2$$

**The remaining steps.** Scaling by 1/√d_k: n² FLOPs (elementwise). Softmax over each row: 3 passes of n operations per row (max, exp, divide), times n rows ≈ 3n² FLOPs. AV: same structure as QK^T, this time (n×n)·(n×d_v), costing 2d_v·n² FLOPs.

Total per head:

$$\text{FLOPs}_{\text{head}} = 2d_k n^2 + n^2 + 3n^2 + 2d_v n^2 = n^2(2d_k + 2d_v + 4)$$

With d_k = d_v = 64:

$$\text{FLOPs}_{\text{head}} = n^2 \cdot 132 \approx n^2 \cdot 128$$

Multiply by h = 8 heads:

$$\boxed{\text{FLOPs}_{\text{attention}} \approx 1024 \cdot n^2}$$

**Compared to the FFN.** The feed-forward sublayer in each transformer block is two linear projections: d_model → 4·d_model → d_model. Cost per token: 2·d_model·4·d_model + 2·4·d_model·d_model = 16·d_model² multiplied by n tokens = 16·d_model²·n. With d_model = 512:

$$\text{FLOPs}_{\text{FFN}} = 16 \cdot 512^2 \cdot n = 4{,}194{,}304 \cdot n$$

### The Crossover Table

Attention is O(n²) and FFN is O(n). At short sequences, FFN dominates. At some crossover n*, attention takes over.

Set FLOPs_attention = FLOPs_FFN:

$$1024 \cdot n^2 = 4{,}194{,}304 \cdot n \implies n^* = 4{,}096$$

| n | FLOPs (attn, billion) | FLOPs (FFN, billion) | Dominant |
|---|---|---|---|
| 128 | 0.017 | 0.537 | FFN |
| 512 | 0.268 | 2.147 | FFN |
| 1 024 | 1.074 | 4.295 | FFN |
| 2 048 | 4.295 | 8.590 | FFN |
| **4 096** | **17.180** | **17.180** | **tie** |
| 8 192 | 68.719 | 34.360 | **Attention** |
| 16 384 | 274.878 | 68.719 | **Attention** |

Beyond 4K tokens, attention dominates compute. At 16K tokens, attention costs 4× the FFN. Modern long-context models at 128K tokens spend essentially all their compute budget in attention.

---

## Bottleneck 2: Memory

The compute bottleneck is painful but manageable with faster hardware. The memory bottleneck is structurally different: you cannot parallelize your way out of it.

### The N×N Matrices

Standard attention materializes two full n×n matrices:

- **S** = QK^T / √d_k, the raw score matrix (n × n floats)
- **P** = softmax(S), the attention weight matrix (n × n floats)

In fp16 (2 bytes per float), each matrix costs:

$$\text{bytes}(S) = \text{bytes}(P) = 2 \cdot n^2 \text{ bytes}$$

For n = 4 096, one head, one layer:

$$2 \times 2 \times 4096^2 = 67{,}108{,}864 \text{ bytes} = 64 \text{ MB}$$

Scale to h = 8 heads and L = 12 layers:

$$64 \text{ MB} \times 8 \times 12 = 6{,}144 \text{ MB} \approx 6 \text{ GB}$$

Just for the attention weight matrices. No activations, no parameters. At n = 8 192 this quadruples to 24 GB. At n = 16 384 it is 96 GB — already exceeding the VRAM of most GPUs.

### Why HBM Bandwidth Matters More Than VRAM Size

You might think: buy a GPU with more VRAM and the problem goes away. It does not. The real bottleneck is *bandwidth* to high-bandwidth memory (HBM).

Modern GPUs have two tiers of memory:

| Memory | Capacity | Bandwidth |
|---|---|---|
| SRAM (on-chip) | 20 MB (A100) | ~19 TB/s |
| HBM (off-chip) | 40–80 GB | ~1.5–2 TB/s |

SRAM is 10× faster than HBM. But it is also 2000× smaller. Matrices larger than ~20 MB must live in HBM.

**Standard attention's HBM access pattern.** Count the reads and writes:

1. Load Q, K from HBM → compute QK^T → write S to HBM: Θ(Nd) reads + Θ(N²) writes
2. Load S from HBM → compute softmax(S) → write P to HBM: Θ(N²) reads + Θ(N²) writes
3. Load P, V from HBM → compute PV → write O to HBM: Θ(N²) reads + Θ(Nd) writes

Total HBM accesses: **Θ(Nd + N²)**

At n = 4 096, d = 64 (per head):

$$\text{HBM reads/writes} = 4 \cdot 4096 \cdot 64 + 3 \cdot 4096^2 = 1{,}048{,}576 + 50{,}331{,}648 \approx 51 \text{ M elements}$$

The N² term is 48× larger than the Nd term. The attention computation is *memory-bandwidth-bound*, not compute-bound. The GPU's arithmetic units sit idle, waiting for data from HBM.

This is the core insight behind FlashAttention: attention is not limited by how fast your GPU can multiply — it is limited by how fast data can flow from HBM to SRAM. Reducing HBM accesses matters far more than reducing FLOPs.

---

## Bottleneck 3: KV Cache During Autoregressive Inference

The O(n²) compute and memory costs are training-time problems. During inference, there is a third, distinct problem: **KV cache growth**.

### What the KV Cache Is

Autoregressive generation produces tokens one at a time. At step t, the model attends over all previous t tokens. Recomputing keys and values from scratch at each step would cost O(t²) total — the same quadratic we saw at training.

The fix: cache the K and V tensors for all previous tokens. When generating token t+1, compute only one new row of Q, K, V; concatenate K_{new} and V_{new} to the cache; compute attention with the full cached K and V.

This turns per-step compute from O(t²) to O(t) at the cost of storing the cache.

### Cost per Token

At each layer, for each head, the cache holds one K vector and one V vector per token. Each vector has d_k or d_v dimensions. In fp16:

$$\text{cache bytes per token} = 2 \cdot L \cdot h \cdot (d_k + d_v) \cdot 2 \text{ bytes}$$

With L = 32 layers, h = 32 heads, d_k = d_v = 128 (GPT-3 scale):

$$\text{bytes per token} = 2 \cdot 32 \cdot 32 \cdot 128 \cdot 2 = 524{,}288 = 0.5 \text{ MB/token}$$

This is 0.5 MB per token in the context window. Not per batch — per *token*.

### KV Cache Growth Table

| Context length | KV cache size | GPU VRAM needed (A100 40GB) |
|---|---|---|
| 1 024 | 0.5 GB | 1.25% |
| 4 096 | 2 GB | 5% |
| 16 384 | 8 GB | 20% |
| 32 768 | 16 GB | 40% |
| 65 536 | 32 GB | 80% |
| **128 000** | **64 GB** | **>100% — OOM** |

A 128K-token context uses 64 GB for KV cache alone on a GPT-3-scale model. This leaves nothing for parameters (~350 GB for GPT-3, distributed across multiple GPUs) or activations. Longer contexts require either fewer layers, narrower heads, or a fundamentally different caching strategy.

---

## Three Problems, Three Lineages of Solutions

Each bottleneck independently motivated a research direction. Understanding which bottleneck each addresses is the key to understanding why these techniques look so different.

### The O(n²) Memory Bottleneck → FlashAttention

**Problem**: Materializing S and P requires O(n²) HBM reads and writes, making standard attention memory-bandwidth-bound.

**Solution**: Never write S or P to HBM. Fuse all three attention steps into a single kernel. Use tiling and the *online softmax trick* to compute the correct result from blocks of Q, K, V that fit in SRAM.

**Result**: HBM accesses drop from Θ(Nd + N²) to Θ(N²d²/M) where M is SRAM size. For typical M >> d², this is a large constant-factor reduction. The computation is now compute-bound, not memory-bound.

FlashAttention solves this by tiling Q, K, V into blocks that fit in SRAM and using the online softmax trick to accumulate the result without ever writing S or P to HBM. It is an implementation optimization, not an architectural change — it computes exactly the same attention, just without the N×N materialization.

### The KV Cache Bottleneck → GQA, MQA, MLA

**Problem**: Each layer, each head needs its own K and V cache. Memory grows as L × h × d per token.

**Solutions**:
- **Multi-Query Attention (MQA)**: all query heads share one K and one V head. Cache shrinks by factor h.
- **Grouped-Query Attention (GQA)**: query heads are grouped; each group shares one K/V pair. Intermediate tradeoff.
- **Multi-head Latent Attention (MLA, DeepSeek)**: compress K and V into a low-rank latent vector; only cache the latent. Cache shrinks by factor d_kv/d_latent.

### The O(n²) Compute Bottleneck → Sparse and Linear Attention

**Problem**: Even with memory solved, FLOPs grow as n² for long sequences.

**Solutions**:
- **Sparse attention** (Longformer, BigBird): only attend to a local window plus a few global tokens. FLOPs drop to O(n).
- **Linear attention**: replace the softmax with a kernel function that can be factored. The order of operations changes: instead of (QK^T)V, compute Q(K^TV). This is O(nd²) per step — linear in n.
- **State-space models** (Mamba, S4): replace attention entirely with a recurrent update. O(n) compute and O(1) inference state.

---

## Summary

| Bottleneck | Cause | Scaling | Typical threshold | Primary fix |
|---|---|---|---|---|
| Compute | QK^T and AV matrix multiplies | O(n²d) | n > 4 096 | Sparse attention, linear attention |
| Memory | Materializing S and P | O(n²) bytes | n > 2 048 | FlashAttention |
| HBM bandwidth | Reading/writing S and P | Θ(Nd + N²) accesses | n > 1 024 | FlashAttention |
| KV cache | Caching K, V for autoregression | O(L · h · d) per token | > 16K tokens | MQA, GQA, MLA |

None of these is a quirk that will be engineered away. They are structural consequences of how matrix multiplication scales. The only way out is to change what computation is done (sparse/linear), how it is scheduled on hardware (FlashAttention), or what gets stored (MQA/GQA/MLA).

The next blog maps all five axes you can modify in attention — and shows how every variant in the literature is a specific point in that space.

---

*Previous: [From Soft Alignment to Queries, Keys, and Values](/blog/attention-q-k-v-from-scratch)*  
*Next: [What Can We Actually Modify in Attention?](/blog/attention-what-can-we-modify)*
