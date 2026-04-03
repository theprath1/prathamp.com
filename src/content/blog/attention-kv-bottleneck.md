---
title: "The KV Bottleneck Explained Deeply: Why Inference Is Memory-Bound"
description: "Building from exact byte counts to the fundamental insight: autoregressive inference is bottlenecked not by arithmetic but by memory bandwidth from loading keys and values. Every KV cache optimization in the literature is a response to this single bottleneck — derived step by step with concrete numbers."
date: 2026-04-02
tags: [machine-learning, attention, transformers, kv-cache, inference, memory-bandwidth]
order: 2
---

The previous blogs established three bottlenecks of vanilla attention: quadratic compute, quadratic memory, and linear KV cache growth. We gave the numbers. Now we go deeper into the third one — the KV cache — because it is the bottleneck that dominates modern inference and the one that the next several blogs will systematically attack.

This blog is not about solutions. It is about understanding the problem with enough precision that every solution in the series becomes obvious.

---

## The Running Model

We continue with the same model from the previous blogs:

- $d_\text{model} = 512$
- $h = 8$ heads
- $d_k = d_v = 64$ (so $h \cdot d_k = 512 = d_\text{model}$)
- $L = 12$ layers
- fp16 throughout (2 bytes per element)

We established in the previous blog that this model costs 24 KB per token in KV cache. We will now trace exactly where those bytes come from, why they matter more than FLOPs, and what happens when you scale up.

---

## The GPU Memory Hierarchy

Before we can understand *why* inference is memory-bound, we need to understand *where* data lives on a GPU. Modern GPUs have a multi-level memory hierarchy, and the speed difference between levels is enormous.

### Three tiers

A GPU has three main tiers of memory. We will use the NVIDIA A100 as our concrete reference:

| Memory tier | Capacity | Bandwidth | Latency |
|---|---|---|---|
| **Registers + SRAM** (on-chip) | ~20 MB | ~19 TB/s | ~1 ns |
| **HBM** (high-bandwidth memory, off-chip) | 40–80 GB | ~2 TB/s | ~100 ns |
| **CPU DRAM** (main memory) | >1 TB | ~50 GB/s | ~500 ns |

The key numbers: SRAM is roughly $10\times$ faster than HBM. HBM is roughly $40\times$ faster than CPU DRAM. But SRAM is roughly $2{,}000\times$ to $4{,}000\times$ smaller than HBM.

### What fits where

In our running model, the KV cache for one head, one layer, at context length $t = 4{,}096$ is:

$$
(d_k + d_v) \cdot t \cdot 2 = 128 \cdot 4{,}096 \cdot 2 = 1{,}048{,}576 \text{ bytes} = 1 \text{ MB per head per layer}
$$

Across all 8 heads and 12 layers: $8 \times 12 \times 1$ MB $= 96$ MB.

This is $4.8\times$ larger than the A100's entire SRAM. The KV cache *cannot* live in SRAM. It must live in HBM. Every time the model needs to read the cached keys and values for an attention step, those bytes must travel from HBM to SRAM at HBM bandwidth — roughly $2$ TB/s.

**Numerical check.** At $t = 512$ tokens: KV cache $= 8 \times 12 \times 128 \times 512 \times 2 = 12{,}582{,}912$ bytes $\approx 12$ MB. Still smaller than SRAM's 20 MB — in principle, the KV cache for a very short context *could* fit. But the model's weight matrices also need SRAM space during computation, so in practice even 12 MB is too much. For any meaningful context length, the KV cache lives in HBM.

### Why this hierarchy matters for inference

During training, the input is a full batch of sequences processed in parallel. The attention computation involves large matrix-matrix multiplies ($Q K^\top$, $A V$), which have much higher arithmetic intensity and are often compute-bound in practice. The GPU's arithmetic units matter much more here than in autoregressive decoding.

During autoregressive inference, we generate one token at a time. The attention computation involves matrix-*vector* multiplies ($K_\text{cache}^\top q$, $V_\text{cache}^\top a$). The arithmetic intensity of a matrix-vector multiply is fundamentally different from a matrix-matrix multiply. We will derive this precisely in the next section.

---

## What Happens During One Autoregressive Step

We are generating token $t+1$. All previous tokens $1, 2, \ldots, t$ have already been processed. Their keys and values sit in the KV cache in HBM.

The computation for one step has three phases. We trace each one for a single head, then scale to all heads and layers.

### Phase 1: Compute the New Query, Key, and Value

The embedding of the new token $x_{t+1} \in \mathbb{R}^{d_\text{model}}$ is projected through three weight matrices:

$$
q_{t+1} = x_{t+1} W_Q^i, \quad k_{t+1} = x_{t+1} W_K^i, \quad v_{t+1} = x_{t+1} W_V^i
$$

Each projection is a matrix-vector multiply. Let us count exactly.

**FLOPs for one projection.** The output $q_{t+1} \in \mathbb{R}^{d_k}$ has $d_k = 64$ elements. Each element is a dot product of $x_{t+1} \in \mathbb{R}^{512}$ with one column of $W_Q^i \in \mathbb{R}^{512 \times 64}$. A dot product of two $d_\text{model}$-dimensional vectors requires $d_\text{model}$ multiplications and $d_\text{model} - 1$ additions. Using the standard convention of counting each multiply-add as 2 FLOPs:

$$
\text{FLOPs per element} = 2 \cdot d_\text{model} = 2 \times 512 = 1{,}024
$$

There are $d_k = 64$ output elements, so:

$$
\text{FLOPs per projection} = 2 \cdot d_\text{model} \cdot d_k = 2 \times 512 \times 64 = 65{,}536
$$

Three projections (Q, K, V) per head, $h = 8$ heads:

$$
\text{FLOPs}_\text{proj} = h \cdot 3 \cdot 2 \cdot d_\text{model} \cdot d_k = 8 \times 3 \times 65{,}536 = 1{,}572{,}864
$$

**Bytes loaded from HBM.** To compute one projection, we must load the weight matrix $W_Q^i \in \mathbb{R}^{d_\text{model} \times d_k}$ from HBM. It has $d_\text{model} \times d_k = 512 \times 64 = 32{,}768$ elements. In fp16 (2 bytes per element):

$$
\text{bytes per weight matrix} = 32{,}768 \times 2 = 65{,}536 \text{ bytes} = 64 \text{ KB}
$$

We also load the input vector $x_{t+1}$ (512 elements, 1 KB) — this is tiny and shared across all projections, so we can ignore it. Three matrices per head, $h = 8$ heads:

$$
\text{bytes}_\text{proj} = h \cdot 3 \cdot d_\text{model} \cdot d_k \cdot 2 = 8 \times 3 \times 65{,}536 = 1{,}572{,}864 \text{ bytes} \approx 1.5 \text{ MB}
$$

### The Roofline Model: Compute-bound vs. Memory-bound

We now have both the FLOPs and the bytes for Phase 1. The ratio of these two quantities has a name.

**Arithmetic intensity** is the number of floating-point operations performed per byte of data transferred from memory. It is measured in FLOP/byte.

$$
\text{arithmetic intensity} = \frac{\text{FLOPs}}{\text{bytes loaded from HBM}}
$$

This concept comes from the **Roofline model** (Williams, Waterman, and Patterson, 2009). The Roofline model says: every hardware platform has a **ridge point** — the arithmetic intensity at which the platform transitions from being memory-bandwidth-limited to compute-limited. The ridge point is:

$$
\text{ridge point} = \frac{\text{peak compute (FLOP/s)}}{\text{peak memory bandwidth (bytes/s)}}
$$

For the A100 GPU:

$$
\text{ridge point} = \frac{312 \times 10^{12} \text{ FLOP/s}}{2 \times 10^{12} \text{ bytes/s}} = 156 \text{ FLOP/byte}
$$

If an operation's arithmetic intensity is *below* 156 FLOP/byte, it is **memory-bandwidth-bound**: the GPU finishes its arithmetic before the next chunk of data arrives from HBM. It sits idle, waiting for data.

If an operation's arithmetic intensity is *above* 156 FLOP/byte, it is **compute-bound**: data arrives faster than the GPU can process it. The arithmetic units are the bottleneck.

**Phase 1's arithmetic intensity:**

$$
\text{arithmetic intensity}_\text{proj} = \frac{1{,}572{,}864 \text{ FLOPs}}{1{,}572{,}864 \text{ bytes}} = 1.0 \text{ FLOP/byte}
$$

This is $156\times$ below the ridge point. The projection phase is catastrophically **memory-bandwidth-bound**. For every FLOP the GPU performs, it must load one byte from HBM. But the GPU *could* perform 156 FLOPs per byte loaded if the data were available. The arithmetic units are idle more than 99% of the time.

**Why is the intensity exactly 1.0?** This is not a coincidence. A matrix-vector multiply of shape $(m \times n) \cdot (n \times 1)$ performs $2mn$ FLOPs and loads $mn \cdot 2$ bytes (the matrix in fp16) plus $n \cdot 2$ bytes (the vector, negligible for large $m$). The arithmetic intensity is:

$$
\frac{2mn}{2mn} = 1.0 \text{ FLOP/byte}
$$

Under this simplified fp16 accounting, the arithmetic intensity is approximately 1.0 FLOP/byte for large matrix-vector multiplies regardless of the matrix dimensions. That is why autoregressive inference — where nearly every operation is a matrix-vector multiply — tends to be memory-bandwidth-bound.

**Contrast with training.** During training, the projection is a matrix-matrix multiply: $Q = X W_Q$ where $X \in \mathbb{R}^{N \times d_\text{model}}$ and $W_Q \in \mathbb{R}^{d_\text{model} \times d_k}$. The FLOPs are $2 \cdot N \cdot d_\text{model} \cdot d_k$. The bytes loaded are the weight matrix ($d_\text{model} \cdot d_k \cdot 2$ bytes) plus the input ($N \cdot d_\text{model} \cdot 2$ bytes). For large $N$, the arithmetic intensity approaches:

$$
\frac{2 \cdot N \cdot d_\text{model} \cdot d_k}{d_\text{model} \cdot d_k \cdot 2 + N \cdot d_\text{model} \cdot 2} \approx \frac{2 N d_k}{2N} = d_k = 64 \text{ FLOP/byte}
$$

At 64 FLOP/byte, training is still below the ridge point but much closer — especially with larger batch sizes that push the effective $N$ higher. The key difference: training's arithmetic intensity scales with $N$ (batch $\times$ sequence length), while inference's arithmetic intensity is stuck at 1.0 regardless of $t$.

### Phase 2: Append to the KV Cache

The new $k_{t+1}$ and $v_{t+1}$ vectors are appended to the cache in HBM.

Each vector has $d_k = 64$ elements $= 128$ bytes in fp16. Two vectors (K and V) per head, $h = 8$ heads:

$$
\text{bytes written} = h \cdot 2 \cdot d_k \cdot 2 = 8 \times 2 \times 64 \times 2 = 2{,}048 \text{ bytes} = 2 \text{ KB}
$$

This is negligible compared to the reads we are about to do in Phase 3. The write is a one-time cost per step that does not scale with context length. We include it for completeness but will not track it further.

### Phase 3: Compute Attention Over the Full Context

This is where the KV cache bottleneck lives. The new query $q_{t+1}$ must attend to all $t+1$ cached keys, produce a softmax distribution, and use it to weight all $t+1$ cached values.

We trace each sub-step for one head in full detail.

**Step 3a: Score computation.**

$$
s = K_\text{cache}^\top q_{t+1} \in \mathbb{R}^{t+1}
$$

This is the matrix-vector product of $K_\text{cache} \in \mathbb{R}^{(t+1) \times d_k}$ transposed with $q_{t+1} \in \mathbb{R}^{d_k}$.

Equivalently, each score $s_j = k_j^\top q_{t+1}$ is a dot product of two $d_k$-dimensional vectors. There are $t+1$ such dot products.

*FLOPs:* Each dot product costs $2 d_k$ FLOPs. There are $t+1$ of them:

$$
\text{FLOPs}_{3a} = 2 \cdot d_k \cdot (t+1) = 2 \times 64 \times (t+1) = 128(t+1)
$$

*Bytes loaded:* The entire $K_\text{cache}$ must be read from HBM — $(t+1) \times d_k$ elements, each 2 bytes:

$$
\text{bytes}_{3a} = (t+1) \cdot d_k \cdot 2 = (t+1) \times 64 \times 2 = 128(t+1) \text{ bytes}
$$

Plus the query vector $q_{t+1}$: $d_k \times 2 = 128$ bytes. This is negligible for large $t$.

*Arithmetic intensity:*

$$
\frac{128(t+1)}{128(t+1)} = 1.0 \text{ FLOP/byte}
$$

Again, exactly 1.0. This is a matrix-vector multiply, and as we showed, all matrix-vector multiplies in fp16 have arithmetic intensity 1.0.

**Step 3b: Scaling and softmax.**

First, divide each score by $\sqrt{d_k} = \sqrt{64} = 8$: that is $t+1$ divisions. Then apply the softmax, which requires three passes over the $t+1$ scores:

1. Find the maximum: $t$ comparisons
2. Compute $e^{s_j - s_\text{max}}$ for each $j$: $t+1$ exponentiations
3. Normalize (divide by sum): $t+1$ additions to compute the sum, then $t+1$ divisions

Total: roughly $4(t+1)$ FLOPs. The bytes involved are the score vector $s$ which was just computed — it likely still resides in SRAM from Step 3a. If we must read it from HBM: $2(t+1)$ bytes.

This step's cost is dominated by Steps 3a and 3c. We include the count but it does not change the analysis.

**Step 3c: Weighted sum of values.**

$$
o_{t+1} = V_\text{cache}^\top \cdot a_{t+1} \in \mathbb{R}^{d_v}
$$

where $a_{t+1} = \text{softmax}(s / \sqrt{d_k}) \in \mathbb{R}^{t+1}$ is the attention weight vector.

This is another matrix-vector multiply: $V_\text{cache} \in \mathbb{R}^{(t+1) \times d_v}$ transposed with $a_{t+1} \in \mathbb{R}^{t+1}$.

Equivalently, this is a weighted sum of the $t+1$ value vectors: $o_{t+1} = \sum_{j=1}^{t+1} a_j \cdot v_j$. Each $a_j \cdot v_j$ costs $d_v$ multiplications, and summing costs $t \cdot d_v$ additions.

*FLOPs:*

$$
\text{FLOPs}_{3c} = 2 \cdot d_v \cdot (t+1) = 128(t+1)
$$

*Bytes loaded:* The full $V_\text{cache}$: $(t+1) \cdot d_v \cdot 2 = 128(t+1)$ bytes.

*Arithmetic intensity:* $128(t+1) / 128(t+1) = 1.0$ FLOP/byte.

### Total for Phase 3, One Head

Summing Steps 3a, 3b, and 3c:

$$
\text{FLOPs}_\text{attn} = 128(t+1) + 4(t+1) + 128(t+1) = 260(t+1)
$$

$$
\text{bytes}_\text{attn} = 128(t+1) + 128(t+1) = 256(t+1)
$$

(The softmax bytes are negligible since the data is likely already in SRAM.)

$$
\text{arithmetic intensity}_\text{attn} = \frac{260(t+1)}{256(t+1)} \approx 1.0 \text{ FLOP/byte}
$$

The arithmetic intensity stays near 1.0 as $t$ grows. It does not improve with longer context. In this roofline analysis, the attention computation during inference remains memory-bandwidth-bound across sequence lengths rather than suddenly becoming compute-bound at larger $t$.

### Scaling to All Heads and All Layers

Each of the $h = 8$ heads performs the same computation with different weight matrices. Each of the $L = 12$ layers performs the same computation with different parameters. Since each head has its own KV cache, the total bytes read are:

$$
\text{bytes}_\text{attn, total} = L \cdot h \cdot 256(t+1) = 12 \times 8 \times 256 \times (t+1) = 24{,}576 \cdot (t+1)
$$

**Numerical check.** At $t+1 = 4{,}096$:

$$
24{,}576 \times 4{,}096 = 100{,}663{,}296 \text{ bytes} \approx 96 \text{ MB}
$$

This is 96 MB of HBM reads per generation step — just for loading the KV cache. At A100 HBM bandwidth of 2 TB/s, this takes:

$$
\frac{96 \times 10^6}{2 \times 10^{12}} = 48 \times 10^{-6} \text{ s} = 48 \text{ microseconds}
$$

Each generation step spends at least 48 microseconds just loading KV cache data from HBM, regardless of how fast the arithmetic is.

**Numerical check at $t+1 = 128{,}000$:**

$$
24{,}576 \times 128{,}000 = 3{,}145{,}728{,}000 \text{ bytes} \approx 3 \text{ GB}
$$

At 2 TB/s: $3 \times 10^9 / (2 \times 10^{12}) = 1.5$ ms per step. That is 1.5 ms per token — for the KV cache reads alone.

---

## The Fundamental Equation of Inference Throughput

We can now write down the time for one autoregressive step. Since the computation is memory-bandwidth-bound, time is determined by bytes loaded, not FLOPs.

### Deriving the total bytes per step

The total bytes loaded per step consist of two categories:

**Category 1: Model weight bytes** (loaded once per step, independent of $t$).

A transformer layer has:
- Attention projections: $W_Q, W_K, W_V \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$ (each is $h$ head matrices concatenated) and $W_O \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$. Total: $4 \cdot d_\text{model}^2$ parameters.
- FFN: two matrices, $W_1 \in \mathbb{R}^{d_\text{model} \times 4d_\text{model}}$ and $W_2 \in \mathbb{R}^{4d_\text{model} \times d_\text{model}}$. Total: $2 \times 4 \times d_\text{model}^2 = 8 \cdot d_\text{model}^2$ parameters.
- LayerNorm parameters: negligible ($2 \times d_\text{model}$ per norm, two norms per layer).

Total per layer: $(4 + 8) \cdot d_\text{model}^2 = 12 \cdot d_\text{model}^2$ parameters. In fp16:

$$
\text{weight bytes per layer} = 12 \cdot d_\text{model}^2 \cdot 2 = 24 \cdot d_\text{model}^2
$$

**Numerical check**: $24 \times 512^2 = 24 \times 262{,}144 = 6{,}291{,}456$ bytes $\approx 6$ MB per layer.

Across $L = 12$ layers:

$$
\text{total weight bytes} = L \cdot 24 \cdot d_\text{model}^2 = 12 \times 6{,}291{,}456 = 75{,}497{,}472 \approx 72 \text{ MB}
$$

(The slight discrepancy from $12 \times 6 = 72$ is because we are rounding. The exact value is $12 \times 24 \times 512^2 = 75{,}497{,}472$ bytes $= 72$ MB.)

**Category 2: KV cache bytes** (grows linearly with $t$):

$$
\text{KV bytes} = L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot (t+1) = 24{,}576 \cdot (t+1)
$$

### The total and the crossover

$$
\boxed{B_\text{step}(t) = \underbrace{24 L \cdot d_\text{model}^2}_{\text{weights (fixed)}} + \underbrace{L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot t}_{\text{KV cache (grows with } t)}}
$$

Per-step time at HBM bandwidth $W$:

$$
T_\text{step}(t) = \frac{B_\text{step}(t)}{W}
$$

**The crossover.** KV cache reads exceed model weight reads when:

$$
L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot t > 24 L \cdot d_\text{model}^2
$$

The $L$ cancels from both sides, leaving:

$$
h \cdot (d_k + d_v) \cdot 2 \cdot t > 24 \cdot d_\text{model}^2
$$

Since $d_k = d_v$ and $h \cdot d_k = d_\text{model}$:

$$
h \cdot 2 d_k \cdot 2 \cdot t > 24 \cdot d_\text{model}^2
$$

$$
4 \cdot d_\text{model} \cdot t > 24 \cdot d_\text{model}^2
$$

$$
t > 6 \cdot d_\text{model}
$$

**Numerical check**: $6 \times 512 = 3{,}072$. So $t^* \approx 3{,}072$ tokens.

Let us verify: $24{,}576 \times 3{,}072 = 75{,}497{,}472$ bytes $= 72$ MB $\approx$ weight bytes. Correct.

This is a remarkably clean result: the crossover occurs at approximately $t^* = 6 \cdot d_\text{model}$, independent of $h$, $d_k$, or $L$.

**Interpretation.** For our small model ($d_\text{model} = 512$), the crossover is at ~3K tokens. For LLaMA-7B ($d_\text{model} = 4{,}096$), the crossover is at $6 \times 4{,}096 = 24{,}576$ tokens. For GPT-3 ($d_\text{model} = 12{,}288$), it is at ~74K tokens. Wider models have a later crossover because their weight matrices are proportionally larger.

But every model eventually crosses — and modern models routinely operate at 100K+ tokens.

| Context length $t$ | Weight bytes | KV cache bytes | Dominant load |
|---|---|---|---|
| 512 | 72 MB | 12 MB | Weights |
| 1,024 | 72 MB | 24 MB | Weights |
| 2,048 | 72 MB | 48 MB | Weights |
| **3,072** | **72 MB** | **72 MB** | **Tie** |
| 4,096 | 72 MB | 96 MB | **KV cache** |
| 8,192 | 72 MB | 192 MB | **KV cache** |
| 16,384 | 72 MB | 384 MB | **KV cache** |

Beyond the crossover, every doubling of context length doubles the per-step latency (since the dominant cost — loading the KV cache — doubles). The model weights are a fixed cost that does not grow with context.

### Deriving per-step latency

At A100 bandwidth $W = 2$ TB/s:

$$
T_\text{step}(t) = \frac{B_\text{step}(t)}{W} = \frac{72 \times 10^6 + 24{,}576 \cdot t}{2 \times 10^{12}}
$$

**Numerical check** at $t = 4{,}096$:

$$
T_\text{step} = \frac{72 \times 10^6 + 100{,}663{,}296}{2 \times 10^{12}} = \frac{172{,}663{,}296}{2 \times 10^{12}} = 86.3 \text{ microseconds}
$$

**At $t = 128{,}000$:**

$$
T_\text{step} = \frac{72 \times 10^6 + 3{,}145{,}728{,}000}{2 \times 10^{12}} = \frac{3{,}217{,}728{,}000}{2 \times 10^{12}} = 1.61 \text{ ms}
$$

At 128K context, a single token takes 1.6 ms. To generate 100 tokens, the model spends 160 ms just on memory transfers. And this is for our tiny 12-layer model.

---

## Why Batch Size Cannot Save You

A natural reaction: if each step is memory-bound because the arithmetic intensity is 1.0, batch more requests together to amortize the weight loading. This is correct — but it hits a wall.

### How batching helps

With batch size $B$, Phase 1 stays the same: the weight matrices are loaded once and applied to $B$ input vectors. This is now a matrix-matrix multiply — $W_Q \cdot X_\text{batch}^\top$ where $X_\text{batch} \in \mathbb{R}^{B \times d_\text{model}}$. The FLOPs scale by $B$, the weight bytes stay constant, so the arithmetic intensity becomes $B$ FLOP/byte. At $B = 156$, we reach the A100's ridge point and Phase 1 becomes compute-bound.

Phase 3 is different. Each request in the batch has its own context — its own KV cache. The $K_\text{cache}$ for request 1 is different from the $K_\text{cache}$ for request 2 (they are answering different prompts). So the bytes loaded in Phase 3 are:

$$
\text{bytes}_{3, \text{batched}} = B \cdot L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot t = B \cdot 24{,}576 \cdot t
$$

The FLOPs also scale by $B$. Each request's query attends to its own cache — there is no sharing. So the arithmetic intensity of Phase 3 remains:

$$
\frac{B \cdot 260(t+1)}{B \cdot 256(t+1)} \approx 1.0 \text{ FLOP/byte}
$$

Batching does *not* help Phase 3 at all. The KV cache for each request must be loaded separately, and the FLOPs and bytes both scale linearly with $B$.

### The memory capacity wall

Even if batching could help compute, it hits a hard wall: the KV caches must all fit in HBM simultaneously.

At $t = 4{,}096$, each request's KV cache is 96 MB. With $B = 156$:

$$
156 \times 96 \text{ MB} = 14{,}976 \text{ MB} \approx 15 \text{ GB}
$$

Add the model weights (72 MB for our small model — negligible here, but for a 7B model the weights are ~14 GB in fp16), activations, optimizer states (if finetuning), and framework overhead. On an A100 with 80 GB HBM, we might have ~60 GB available for KV caches, allowing $60{,}000 / 96 \approx 625$ concurrent requests at 4K context.

Now scale to $t = 128{,}000$: each cache is 3 GB. Maximum concurrent requests: $60{,}000 / 3{,}000 \approx 20$.

The situation for larger models is much worse. For LLaMA-7B at 128K context, each cache is 64 GB — it does not even fit on a single GPU. Batch size must be 1, and multi-GPU parallelism is required just to hold one request.

This is the fundamental tension:

$$
\boxed{\text{Batching cures the compute problem but amplifies the memory capacity problem.}}
$$

The KV cache is the binding constraint on both throughput (bandwidth) and concurrency (capacity). It is the single bottleneck that limits how many tokens per second a serving system can produce and how many users it can serve simultaneously.

---

## Deriving the Per-Token KV Cache Formula

Let us derive a clean, closed-form expression for the KV cache cost per token.

### Starting from first principles

At layer $\ell$, head $i$, for one token at position $t$, the KV cache stores:

- One key vector: $k_t^{(\ell, i)} \in \mathbb{R}^{d_k}$
- One value vector: $v_t^{(\ell, i)} \in \mathbb{R}^{d_v}$

In fp16, each element is 2 bytes. So the cache for one token, one head, one layer is:

$$
\text{bytes}_{1,1,1} = (d_k + d_v) \cdot 2
$$

With $d_k = d_v = 64$: $(64 + 64) \times 2 = 256$ bytes.

**Numerical check:** 256 bytes stores two 64-element fp16 vectors. $64 \times 2 = 128$ bytes per vector, two vectors: $128 + 128 = 256$. Correct.

### Summing over heads

There are $h$ heads per layer, each with its own K and V:

$$
\text{bytes}_{1,\text{all heads},1} = h \cdot (d_k + d_v) \cdot 2
$$

With $h = 8$: $8 \times 128 \times 2 = 2{,}048$ bytes $= 2$ KB per token per layer.

### Summing over layers

There are $L$ layers, each with its own attention:

$$
\text{bytes per token} = L \cdot h \cdot (d_k + d_v) \cdot 2
$$

With $L = 12$: $12 \times 2{,}048 = 24{,}576$ bytes $= 24$ KB per token.

### Simplification using $h \cdot d_k = d_\text{model}$

In all standard architectures, the head dimension $d_k$ is chosen so that $h \cdot d_k = d_\text{model}$. This is not a constraint — it is a design convention that keeps the total width constant. Using this identity:

$$
h \cdot (d_k + d_v) = h \cdot 2 d_k = 2 \cdot h \cdot d_k = 2 \cdot d_\text{model}
$$

Substituting:

$$
\text{bytes per token} = L \cdot 2 \cdot d_\text{model} \cdot 2 = 4 \cdot L \cdot d_\text{model}
$$

$$
\boxed{\text{KV cache bytes per token} = 4 \cdot L \cdot d_\text{model}}
$$

**Numerical check**: $4 \times 12 \times 512 = 24{,}576$ bytes $= 24$ KB. Matches.

This formula reveals that the KV cache cost per token depends on exactly two hyperparameters: model depth $L$ and model width $d_\text{model}$. It does *not* depend on $h$ or $d_k$ individually — only on their product $d_\text{model}$. Doubling the number of heads while halving $d_k$ (keeping $d_\text{model}$ constant) does not change the KV cache size at all.

### Scaling to real models

The total KV cache for a sequence of length $t$ is:

$$
\text{total KV bytes} = 4 \cdot L \cdot d_\text{model} \cdot t
$$

Let us compute this for production models at $t = 128{,}000$ tokens:

**GPT-2 Large** ($d_\text{model} = 1{,}280$, $L = 36$):

$$
4 \times 36 \times 1{,}280 \times 128{,}000 = 23{,}592{,}960{,}000 \approx 22 \text{ GB}
$$

**Numerical check of per-token cost**: $4 \times 36 \times 1{,}280 = 184{,}320$ bytes $\approx 180$ KB/token. $180 \times 128{,}000 = 23{,}040{,}000$ KB $\approx 22$ GB. Consistent.

**LLaMA-7B** ($d_\text{model} = 4{,}096$, $L = 32$):

$$
4 \times 32 \times 4{,}096 \times 128{,}000 = 67{,}108{,}864{,}000 \approx 64 \text{ GB}
$$

Per-token: $4 \times 32 \times 4{,}096 = 524{,}288$ bytes $= 512$ KB/token.

**LLaMA-65B** ($d_\text{model} = 8{,}192$, $L = 80$):

$$
4 \times 80 \times 8{,}192 \times 128{,}000 = 335{,}544{,}320{,}000 \approx 320 \text{ GB}
$$

Per-token: $4 \times 80 \times 8{,}192 = 2{,}621{,}440$ bytes $\approx 2.5$ MB/token.

**GPT-3 175B** ($d_\text{model} = 12{,}288$, $L = 96$):

$$
4 \times 96 \times 12{,}288 \times 128{,}000 = 603{,}979{,}776{,}000 \approx 576 \text{ GB}
$$

Per-token: $4 \times 96 \times 12{,}288 = 4{,}718{,}592$ bytes $\approx 4.5$ MB/token.

| Model | $d_\text{model}$ | $L$ | bytes/token | KV cache at 128K |
|---|---|---|---|---|
| Our running model | 512 | 12 | 24 KB | 3 GB |
| GPT-2 Large | 1,280 | 36 | 180 KB | 22 GB |
| LLaMA-7B | 4,096 | 32 | 512 KB | 64 GB |
| LLaMA-65B | 8,192 | 80 | 2.5 MB | 320 GB |
| GPT-3 175B | 12,288 | 96 | 4.5 MB | 576 GB |

The model weights of LLaMA-65B are ~130 GB in fp16 (65 billion parameters $\times$ 2 bytes). The KV cache at 128K tokens is $320 / 130 \approx 2.5\times$ larger than the model itself. The cache has become the dominant consumer of GPU memory.

---

## The KV Cache vs. Model Weight Ratio

This is worth formalizing. The ratio of KV cache bytes to model weight bytes tells us how much of the GPU's memory and bandwidth is consumed by the cache versus the model.

### Deriving the ratio

Model weight bytes (as we derived):

$$
\text{weight bytes} = 24 \cdot L \cdot d_\text{model}^2
$$

(This is $12 \cdot d_\text{model}^2$ parameters per layer $\times$ $L$ layers $\times$ 2 bytes.)

KV cache bytes at context length $t$:

$$
\text{KV bytes} = 4 \cdot L \cdot d_\text{model} \cdot t
$$

The ratio:

$$
\frac{\text{KV bytes}}{\text{weight bytes}} = \frac{4 \cdot L \cdot d_\text{model} \cdot t}{24 \cdot L \cdot d_\text{model}^2}
$$

The $L$ cancels. One power of $d_\text{model}$ cancels. What remains is:

$$
\boxed{\frac{\text{KV bytes}}{\text{weight bytes}} = \frac{t}{6 \cdot d_\text{model}}}
$$

**Numerical check** with our running model at $t = 4{,}096$:

$$
\frac{4{,}096}{6 \times 512} = \frac{4{,}096}{3{,}072} = 1.33
$$

So the KV cache is $1.33\times$ the model weights. Verifying: $96 \text{ MB} / 72 \text{ MB} = 1.33$. Correct.

**At $t = 128{,}000$ for LLaMA-65B** ($d_\text{model} = 8{,}192$):

$$
\frac{128{,}000}{6 \times 8{,}192} = \frac{128{,}000}{49{,}152} = 2.60
$$

The KV cache is $2.6\times$ the model weights. We computed 320 GB cache vs. ~130 GB weights earlier — $320/130 \approx 2.46$. The small discrepancy is because the actual model includes embedding layers and LM head not counted in our $12 d_\text{model}^2$ per-layer estimate, but the formula captures the correct order of magnitude.

**Interpretation.** The ratio $t / (6 \cdot d_\text{model})$ tells us:
- At the crossover ($t = 6 \cdot d_\text{model}$), the ratio is exactly 1. KV cache = weights.
- The ratio grows *linearly* with $t$. Every additional token adds a fixed cost.
- The ratio shrinks *inversely* with $d_\text{model}$. Wider models have proportionally more weights, so the cache takes longer to overtake them. But it always does eventually.

---

## The Three Levers for Reducing KV Cache

The formula $\text{bytes/token} = L \cdot h \cdot (d_k + d_v) \cdot 2$ tells us where to push. There are three major levers we will focus on in this series, and many KV cache optimizations pull one or more of them.

### Lever 1: Reduce the number of KV heads

If instead of $h$ unique KV heads, we use $g < h$ KV heads (sharing each across $h/g$ query heads), the per-token cache becomes:

$$
\text{bytes/token} = L \cdot g \cdot (d_k + d_v) \cdot 2
$$

The reduction factor — by dividing the original by the new — is:

$$
\frac{L \cdot h \cdot (d_k + d_v) \cdot 2}{L \cdot g \cdot (d_k + d_v) \cdot 2} = \frac{h}{g}
$$

The $L$, $(d_k + d_v)$, and the 2 all cancel. The reduction is exactly $h/g$.

With $g = 1$ (all queries share one KV head): factor $= h = 8\times$. This is **Multi-Query Attention** (MQA, Shazeer 2019).

With $g = 2$: factor $= 4\times$. This is **Grouped-Query Attention** (GQA, Ainslie et al. 2023).

**Numerical check.** MQA on our running model:

$$
12 \times 1 \times 128 \times 2 = 3{,}072 \text{ bytes/token} = 3 \text{ KB}
$$

Reduction: $24{,}576 / 3{,}072 = 8 = h$. Correct.

GQA with $g = 2$:

$$
12 \times 2 \times 128 \times 2 = 6{,}144 \text{ bytes/token} = 6 \text{ KB}
$$

Reduction: $24{,}576 / 6{,}144 = 4 = h/g = 8/2$. Correct.

**Tradeoff.** Fewer KV heads means all query heads in a group must use the same key-value representation. This limits the model's ability to attend to different features in different heads. The quality cost depends on how redundant the original heads were.

### Lever 2: Compress the KV representation

Instead of caching $k_t \in \mathbb{R}^{d_k}$ and $v_t \in \mathbb{R}^{d_v}$ per head, project the input to a shared low-dimensional latent $c_t \in \mathbb{R}^{d_c}$ where $d_c \ll h \cdot (d_k + d_v)$. Cache only $c_t$; reconstruct per-head K and V from $c_t$ using learned up-projection matrices during attention.

Per-token cache: $L \cdot d_c \cdot 2$ bytes.

With $d_c = 512$ (matching $d_\text{model}$), the cache per token per layer is $512 \times 2 = 1{,}024$ bytes. Compare to MHA: $h \cdot (d_k + d_v) \cdot 2 = 8 \times 128 \times 2 = 2{,}048$ bytes. Reduction: $2\times$.

For larger models with more heads, the compression ratio improves proportionally. This is **Multi-head Latent Attention** (MLA, DeepSeek-V2).

**Tradeoff.** The up-projection matrices (from $c_t$ to per-head K and V) must be applied during every attention step, adding compute. This trades memory for FLOPs — the opposite direction from what FlashAttention does.

### Lever 3: Cache fewer tokens

Instead of caching all $t$ tokens, cache only the most recent $w$ tokens. The total cache is bounded:

$$
\text{cache bytes} = L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot w
$$

This is constant regardless of $t$.

With $w = 512$ on our running model:

$$
12 \times 8 \times 128 \times 2 \times 512 = 12{,}582{,}912 \text{ bytes} = 12 \text{ MB}
$$

Compare to full cache at $t = 128{,}000$: $3$ GB. A $250\times$ reduction.

**Tradeoff.** Positions more than $w$ tokens ago are invisible to the current layer's attention. Information can propagate further than $w$ through multi-layer composition — if token $A$ at position $i$ attends to token $B$ at position $i - w/2$, and token $B$ attends to token $C$ at position $i - w$, then $A$ indirectly accesses $C$ through two layers. But direct access is limited to the window.

### Composing the levers

These three levers compose cleanly in the simple accounting used here. GQA ($g = 2$) + sliding window ($w = 512$):

$$
12 \times 2 \times 128 \times 2 \times 512 = 3{,}145{,}728 \text{ bytes} = 3 \text{ MB (fixed)}
$$

Compare to vanilla MHA at $t = 128{,}000$: $\approx 3$ GB. Combined reduction: $1{,}000\times$.

**Numerical check**: the two individual reductions are $h/g = 4\times$ (from GQA) and $128{,}000 / 512 = 250\times$ (from sliding window). Product: $4 \times 250 = 1{,}000$. And $3{,}000 / 3 = 1{,}000$. Correct — the reductions compose multiplicatively as claimed.

---

## Why Memory Bandwidth Is the Right Metric

We have used "bytes loaded from HBM" as the primary cost metric throughout this blog. Let us justify this choice with a direct comparison.

### Compute time vs. memory time

Consider generating one token at context length $t = 4{,}096$ with our running model.

**Total FLOPs** (across all heads and layers, Phase 3 only):

$$
\text{FLOPs}_\text{total} = L \cdot h \cdot 260(t+1) = 12 \times 8 \times 260 \times 4{,}096 = 102{,}236{,}160 \approx 100 \text{ million}
$$

**Time if compute-bound** (limited by arithmetic throughput, A100 at 312 TFLOPS fp16):

$$
T_\text{compute} = \frac{100 \times 10^6}{312 \times 10^{12}} = 0.32 \text{ microseconds}
$$

**Time if memory-bound** (limited by HBM bandwidth, A100 at 2 TB/s):

$$
T_\text{memory} = \frac{96 \times 10^6}{2 \times 10^{12}} = 48 \text{ microseconds}
$$

The ratio:

$$
\frac{T_\text{memory}}{T_\text{compute}} = \frac{48}{0.32} = 150
$$

The memory transfer takes $150\times$ longer than the computation. This is almost exactly the ridge-point ratio of 156 — not a coincidence, since our arithmetic intensity is $\approx 1.0$ and the ridge point is 156.

**The GPU sits idle 99.3% of the time**, waiting for KV cache data to arrive from HBM. The actual per-step time is $T_\text{step} = \max(T_\text{compute}, T_\text{memory}) = 48$ microseconds (the Roofline model takes the maximum, since the bottleneck determines the runtime).

### Does the ratio improve with context?

As $t$ increases, both FLOPs and bytes grow proportionally:

$$
\text{FLOPs} \propto t, \quad \text{bytes} \propto t
$$

The ratio $T_\text{memory} / T_\text{compute}$ stays constant at $\approx 150$. The operation never becomes compute-bound, no matter how long the context.

This is the mathematical inevitability of matrix-vector multiplies: the arithmetic intensity is always 1.0 FLOP/byte in fp16, the hardware's ridge point is always 156 FLOP/byte, and the gap never closes.

**The implication:** in the KV-bandwidth-dominant regime, reducing KV cache size can translate into roughly proportional inference speedups. Halve the KV bytes loaded and you can often cut a large part of the per-step latency, though fixed costs such as weight loading still remain.

---

## Connecting to the Papers

Two papers in our collection directly address the bottlenecks we have derived.

### FlashAttention (Dao et al., 2022)

FlashAttention solves the $O(N^2)$ memory materialization problem during *training*. Recall from the "Why Vanilla Attention Breaks" blog that standard attention materializes the $N \times N$ score matrix $S = Q K^\top$ and the attention weight matrix $P = \text{softmax}(S)$ to HBM. These matrices have $O(N^2)$ elements, requiring $O(N^2)$ HBM reads and writes.

FlashAttention avoids this by tiling $Q$, $K$, $V$ into blocks that fit in SRAM and using the **online softmax algorithm** — a technique for computing softmax incrementally, block by block, without needing all scores simultaneously. The key insight: you can compute $\text{softmax}(Q K^\top) V$ one block of $K, V$ at a time, keeping only a running maximum and running sum in SRAM. No $N \times N$ matrix is ever written to HBM.

The IO complexity drops from $\Theta(Nd + N^2)$ (standard) to a lower tiled bound that depends on SRAM size $M$ (FlashAttention), substantially reducing HBM traffic. The exact constant-factor gain depends on $N$, $d$, $M$, and the implementation, so it is better not to compress it to a single universal number.

**What FlashAttention does not do.** FlashAttention does not reduce the KV cache during autoregressive inference. During generation, the bottleneck is not materializing $S$ and $P$ (which are $1 \times t$ vectors for a single query, not $N \times N$ matrices). The bottleneck is loading the KV cache itself. FlashAttention solves a training bottleneck (Axis 4: storage), not an inference bottleneck (Axis 2: KV representation).

### GQA (Ainslie et al., 2023)

GQA directly attacks the inference KV bottleneck by pulling Lever 1: reducing the number of KV heads from $h$ to $g$. The per-token cache drops by $h/g$, and inference speed improves proportionally at long contexts.

The paper's central finding: with $g = 8$ KV groups on T5-XXL (which has $h = 64$ query heads), quality remains within 0.1 points of full MHA while inference is $5.4\times$ faster. The quality cost of reducing KV heads is far smaller than the bandwidth saving — exactly because many heads are redundant.

The next blog derives GQA in full detail.

### Complementarity

These two papers are complementary:

| | FlashAttention | GQA |
|---|---|---|
| **Bottleneck addressed** | Training memory | Inference bandwidth |
| **Axis modified** | Axis 4 (storage) | Axis 2 (KV representation) |
| **What changes** | How attention is computed | What K/V tensors exist |
| **Cache reduction** | None | $h/g$ factor |
| **Speed improvement** | Training 1.5–3× | Inference up to $h/g$ |

They compose cleanly: use FlashAttention during training for memory efficiency, and GQA during inference for bandwidth efficiency. Modern systems (LLaMA 2, Mistral) use both.

---

## Summary

Autoregressive inference is memory-bandwidth-bound, not compute-bound. We derived this from first principles using the Roofline model: in our simplified accounting, every attention operation during token generation behaves like a matrix-vector multiply with arithmetic intensity near 1.0 FLOP/byte, which is far below the A100's ridge point. In this regime, the GPU spends most of its time waiting for data from HBM rather than doing arithmetic.

The dominant memory load is the KV cache. Its cost per token is $4 \cdot d_\text{model} \cdot L$ bytes in fp16 — a formula that depends only on model width and depth. The KV cache overtakes the model weight load at context length $t^* = 6 \cdot d_\text{model}$, and beyond this point, per-step latency grows linearly with context. At production scale (LLaMA-65B, 128K tokens), the KV cache reaches 320 GB — 2.5× the model's own weight memory.

Batching helps compute efficiency but amplifies the memory capacity problem, creating a fundamental tension. Three major levers for reducing the KV cache are: reduce KV heads (MQA/GQA, Lever 1), compress the KV representation (MLA, Lever 2), or cache fewer tokens (sliding window, Lever 3). These levers compose cleanly in the simple accounting used here, and many KV cache papers pull one or more of them.

---

*Previous: [What Can We Actually Modify in Attention?](/blog/attention-what-can-we-modify)*
*Next: [Grouped-Query Attention: Fewer KV Heads, Same Quality](/blog/attention-gqa)*
