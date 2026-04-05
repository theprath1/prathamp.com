---
title: "Why Vanilla Attention Breaks at Scale: The O(n²) Wall"
description: "A precise accounting of why standard attention becomes infeasible at long sequences: exact FLOP counts, memory costs, KV-cache growth, and the HBM bandwidth bottleneck that makes memory worse than compute."
date: 2026-03-31
tags: [machine-learning, attention, transformers, efficiency, flashattention]
order: 2
---

The first two blogs built attention from scratch: the alignment model, the Q/K/V abstraction, scaled dot-product, multi-head. Everything fits on a whiteboard. The formula is clean.

Now double the sequence length. Double it again. Watch the cost explode.

This blog is a precise accounting of why vanilla attention fails at scale. Not vague gestures at "quadratic complexity" but exact FLOP counts, byte counts, and a measurement of which bottleneck actually kills you first.

---

## The Running Model, Scaled Up

We've been working with three vectors. That was enough to derive the mechanism. It is not enough to understand the scaling problem.

Fix a concrete model: $d_\text{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$, $L = 12$ layers, and fp16 throughout (2 bytes per element). We will vary sequence length $n$ during training and generated context length $t$ during autoregressive inference.

Two identities will be used repeatedly: $h \cdot d_k = 8 \cdot 64 = 512 = d_\text{model}$ and $d_k = d_v = 64$. These look innocent. They are not. Almost every scaling law in this post comes from expanding these two equalities inside the attention formulas.

---

## 1. What Actually Scales with Sequence Length?

Before counting anything, let us write the per-head attention computation once:

$$
S = \frac{QK^\top}{\sqrt{d_k}}, \qquad P = \text{softmax}(S), \qquad O = PV
$$

For one head, the tensors have shapes $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$, $S \in \mathbb{R}^{n \times n}$, $P \in \mathbb{R}^{n \times n}$, and $O \in \mathbb{R}^{n \times d_v}$.

The source of the trouble is now visible. $Q$, $K$, $V$, and $O$ all scale like $n \cdot d$, but $S$ and $P$ scale like $n^2$. This is the whole story in one line — the trouble starts the moment we materialize pairwise interactions between all positions. The quadratic term is not a side effect. It is built into the object we are computing.

### Numerical check

At $n = 512$:

$$
n^2 = 512^2 = 262{,}144
$$

At $n = 4{,}096$:

$$
n^2 = 4{,}096^2 = 16{,}777{,}216
$$

The sequence length grew by a factor of $\frac{4{,}096}{512} = 8$, but the number of pairwise interactions grew by $\frac{16{,}777{,}216}{262{,}144} = 64 = 8^2$. This is the **quadratic growth law**: doubling sequence length multiplies pairwise interactions by 4, and multiplying length by 8 multiplies interactions by 64.

---

## 2. Bottleneck 1: Compute

Start with the arithmetic. This is the bottleneck people usually mention first, and for good reason: the FLOP count really does blow up.

### 2.1 FLOPs in $QK^\top$

The score matrix before scaling is $QK^\top$.

$Q$ has shape $(n, d_k)$, $K^\top$ has shape $(d_k, n)$, and their product has shape $(n, n)$. By the **definition of matrix multiplication**, each entry is a dot product:

$$
[QK^\top]_{ij} = \sum_{r=1}^{d_k} Q_{ir} K_{jr}
$$

One dot product of length $d_k$ costs $d_k$ multiplications and $d_k - 1$ additions. Using the standard FLOP convention that a multiply and an add each count as one floating-point operation, this is approximately $2d_k$ FLOPs per entry. There are $n^2$ entries, so:

$$
\text{FLOPs}(QK^\top) = 2 d_k n^2
$$

For our running model, $d_k = 64$:

$$
\text{FLOPs}(QK^\top) = 2 \cdot 64 \cdot n^2 = 128 n^2
$$

### 2.2 FLOPs in Scaling, Softmax, and $PV$

Now add the rest of the attention computation. The dot-product score matrix is not the whole story.

**Scaling.** Dividing each entry of $QK^\top$ by $\sqrt{d_k}$ is one elementwise operation per entry:

$$
\text{FLOPs}(\text{scale}) = n^2
$$

**Softmax.** For each of the $n$ rows, we need one pass to find the row maximum, one pass to exponentiate shifted scores, one pass to sum exponentials, and one pass to divide by the sum. That is approximately $4n$ scalar operations per row, hence

$$
\text{FLOPs}(\text{softmax}) \approx 4n^2
$$

**Value mixing.** The output matrix is $PV$ with shape $(n, n) \cdot (n, d_v) \to (n, d_v)$. By the same dot-product counting as above:

$$
\text{FLOPs}(PV) = 2 d_v n^2
$$

Since $d_v = 64$:

$$
\text{FLOPs}(PV) = 128 n^2
$$

### 2.3 Total FLOPs Per Head

Summing the four terms:

$$
\text{FLOPs}_\text{head}
= 2d_k n^2 + n^2 + 4n^2 + 2d_v n^2
$$

Factor out $n^2$ by the **distributive law of multiplication over addition**:

$$
\text{FLOPs}_\text{head}
= n^2(2d_k + 2d_v + 5)
$$

Substitute $d_k = d_v = 64$:

$$
\text{FLOPs}_\text{head}
= n^2(128 + 128 + 5)
= 261 n^2
$$

For all $h = 8$ heads:

$$
\text{FLOPs}_\text{attn}
= 8 \cdot 261 n^2
= 2{,}088 n^2
$$

To keep the scaling law readable, we approximate this as

$$
\boxed{\text{FLOPs}_\text{attn} \approx 2{,}048\, n^2 = 4 d_\text{model} n^2}
$$

The approximation comes from ignoring the small softmax/scaling constant and using $h \cdot 2d_k + h \cdot 2d_v = 4d_\text{model}$.

### 2.4 Compare to the FFN

The raw number $2{,}088 n^2$ does not mean much by itself. The right comparison inside a transformer block is the feed-forward network.

The standard FFN maps $d_\text{model} \to 4d_\text{model} \to d_\text{model}$. For one token, the first linear layer costs $2 \cdot d_\text{model} \cdot 4d_\text{model} = 8 d_\text{model}^2$ FLOPs, and the second linear layer costs the same $8 d_\text{model}^2$, so the FFN cost per token is $16 d_\text{model}^2$. Across $n$ tokens:

$$
\text{FLOPs}_\text{FFN} = 16 d_\text{model}^2 n
$$

With $d_\text{model} = 512$:

$$
\text{FLOPs}_\text{FFN} = 16 \cdot 512^2 \cdot n
= 4{,}194{,}304\, n
$$

### 2.5 The Compute Crossover

Attention is quadratic in $n$. The FFN is linear in $n$. So there must be a sequence length at which attention stops being the secondary cost and becomes the dominant one.

Using the clean approximation $\text{FLOPs}_\text{attn} \approx 2{,}048 n^2$:

$$
2{,}048 n^2 = 4{,}194{,}304 n
$$

Cancel one factor of $n$ from both sides:

$$
2{,}048 n = 4{,}194{,}304
$$

Divide both sides by $2{,}048$:

$$
n = 2{,}048
$$

So the approximate compute crossover is

$$
\boxed{n^* \approx 4 d_\text{model} = 2{,}048}
$$

If we keep the exact $2{,}088 n^2$ coefficient, the crossover is slightly lower:

$$
n^* = \frac{4{,}194{,}304}{2{,}088} \approx 2{,}009
$$

### 2.6 Numerical table

| $n$ | Attn FLOPs / layer | FFN FLOPs / layer | Dominant |
|---|---|---|---|
| 128 | 0.034 B | 0.537 B | FFN |
| 512 | 0.545 B | 2.147 B | FFN |
| 1,024 | 2.190 B | 4.295 B | FFN |
| 2,048 | 8.760 B | 8.590 B | Near tie |
| 4,096 | 35.041 B | 17.180 B | Attention |
| 8,192 | 140.164 B | 34.360 B | Attention |

At short sequences, the FFN dominates compute. Around 2K tokens, attention catches up. Beyond that, attention becomes the arithmetic bottleneck. This is the first way vanilla attention breaks.

---

## 3. Bottleneck 2: Training Memory

The compute bottleneck hurts. The memory bottleneck usually hurts earlier.

### 3.1 What gets materialized

This is the crucial implementation detail that the compact formula hides. Standard attention does not merely imply an $n \times n$ interaction pattern — it usually materializes two dense $n \times n$ matrices per head. The first is the score matrix $S = \frac{QK^\top}{\sqrt{d_k}}$ and the second is the attention weight matrix $P = \text{softmax}(S)$. Each has exactly $n^2$ elements. In fp16, each element is 2 bytes, so one matrix costs $2 n^2$ bytes and the pair costs $4 n^2$ bytes per head, per layer.

### 3.2 Numerical check

At $n = 4{,}096$:

$$
n^2 = 4{,}096^2 = 16{,}777{,}216
$$

So one $n \times n$ fp16 matrix costs

$$
2 \cdot 16{,}777{,}216 = 33{,}554{,}432 \text{ bytes}
$$

which is

$$
32 \text{ MB}
$$

to a good binary-unit approximation.

Two matrices per head means

$$
64 \text{ MB/head/layer}
$$

Multiply by $h = 8$ heads:

$$
64 \text{ MB} \times 8 = 512 \text{ MB/layer}
$$

Multiply by $L = 12$ layers:

$$
512 \text{ MB} \times 12 = 6{,}144 \text{ MB} \approx 6 \text{ GB}
$$

This is only for the two attention matrices — it does not include Q, K, V activations, FFN activations, residual streams, optimizer state, or gradients.

### 3.3 Memory table

| $n$ | One matrix $S$ or $P$ | Two matrices / head | All heads / layer | All heads / 12 layers |
|---|---|---|---|---|
| 512 | 0.5 MB | 1 MB | 8 MB | 96 MB |
| 1,024 | 2 MB | 4 MB | 32 MB | 384 MB |
| 2,048 | 8 MB | 16 MB | 128 MB | 1.5 GB |
| 4,096 | 32 MB | 64 MB | 512 MB | 6 GB |
| 8,192 | 128 MB | 256 MB | 2 GB | 24 GB |
| 16,384 | 512 MB | 1 GB | 8 GB | 96 GB |

By 8K tokens, the attention matrices alone already consume tens of gigabytes across layers. This is the second way vanilla attention breaks.

### 3.4 Why the $n \times n$ matrices dominate everything else

It is worth making the comparison explicit, because otherwise "quadratic memory" can still sound abstract.

One Q, K, or V tensor has shape $n \times d_k$ (or $n \times d_v$). At $n = 4{,}096$ and $d_k = d_v = 64$, one such tensor costs $4{,}096 \cdot 64 \cdot 2 = 524{,}288$ bytes, which is only $0.5$ MB. So all three linear activations together cost about $1.5$ MB per head. Compare that to the two dense matrices at $64$ MB per head — the ratio is $\frac{64}{1.5} \approx 42.7$. At 4K tokens, the $n \times n$ attention matrices are already more than forty times larger than the combined Q, K, and V activations. This is the practical meaning of "quadratic memory dominates linear activations." The $n \times n$ objects are not just asymptotically larger — they are already dominating by large constants at practical sequence lengths.

---

## 4. Why Memory Gets Worse Than Compute: HBM Traffic

Compute tells us how many arithmetic operations happen. It does not tell us how fast the hardware can feed data to those operations.

That is where the real training bottleneck appears. Long-context attention is often limited less by multiplication than by movement.

### 4.1 SRAM vs HBM

Modern GPUs have fast on-chip SRAM and registers with very high bandwidth, and much larger off-chip HBM with much lower bandwidth. The exact numbers vary by device, but the pattern is stable: SRAM is extremely fast and extremely small, while HBM is much larger and much slower. The problem is that $S$ and $P$ stop fitting on-chip surprisingly early, and once they spill to HBM, every pass over them becomes a bandwidth problem.

### 4.2 Exact forward-pass traffic

Let us count the dominant tensor traffic for one head in a standard forward pass. This is the bookkeeping FlashAttention is designed around.

**Step 1.** Read $Q$ and $K$ to compute $QK^\top$:

$$
2nd_k
$$

elements read.

Write the score matrix $S$:

$$
n^2
$$

elements written.

**Step 2.** Read $S$ to apply softmax:

$$
n^2
$$

elements read.

Write $P$:

$$
n^2
$$

elements written.

**Step 3.** Read $P$ and $V$ to compute $PV$:

$$
n^2 + nd_v
$$

elements read.

Write $O$:

$$
nd_v
$$

elements written.

Add everything:

$$
2nd_k + n^2 + n^2 + n^2 + nd_v + nd_v
$$

Since $d_k = d_v = d$, this becomes

$$
4nd + 3n^2
$$

elements moved per head in the forward pass.

### 4.3 Numerical check

For our running model, $d = 64$ and $n = 4{,}096$.

First compute the linear term:

$$
4nd = 4 \cdot 4{,}096 \cdot 64 = 1{,}048{,}576
$$

Now the quadratic term:

$$
3n^2 = 3 \cdot 4{,}096^2 = 3 \cdot 16{,}777{,}216 = 50{,}331{,}648
$$

Total:

$$
4nd + 3n^2 = 1{,}048{,}576 + 50{,}331{,}648 = 51{,}380{,}224
$$

elements moved.

The linear term is only about 1 million elements, while the quadratic term is over 50 million. The ratio is $\frac{50{,}331{,}648}{1{,}048{,}576} = 48$, so at 4K tokens, the $n^2$ traffic is already 48 times larger than the $nd$ traffic. This is the core reason FlashAttention exists — the bottleneck is not only arithmetic, it is moving those $n^2$ matrices to and from HBM.

---

## 5. Bottleneck 3: KV Cache During Autoregressive Inference

The first two bottlenecks are primarily training-time problems. Inference introduces a different one, and in modern long-context generation it is often the decisive one.

### 5.1 Why the cache exists at all

During autoregressive generation, token $t+1$ attends over all previous tokens $1, \ldots, t$.

If we recomputed all keys and values from scratch at every generation step, total work over the full generated sequence would become quadratic again. So practical decoders cache the keys and values from previous steps. That cache is what makes autoregressive decoding feasible in the first place.

### 5.2 Cache bytes per token

For one token, one head, one layer, we store one key vector of length $d_k$ and one value vector of length $d_v$, giving $d_k + d_v$ elements. In fp16, each element is 2 bytes, so the cost is $(d_k + d_v) \cdot 2$ bytes. Across $h$ heads and $L$ layers:

$$
\boxed{\text{KV bytes/token} = L \cdot h \cdot (d_k + d_v) \cdot 2}
$$

### 5.3 Numerical check

Substitute the running model:

$$
L \cdot h \cdot (d_k + d_v) \cdot 2
= 12 \cdot 8 \cdot (64 + 64) \cdot 2
$$

First add the key and value widths:

$$
64 + 64 = 128
$$

Then multiply:

$$
12 \cdot 8 \cdot 128 \cdot 2
= 12 \cdot 8 \cdot 256
= 12 \cdot 2{,}048
= 24{,}576 \text{ bytes}
$$

So the cache cost is

$$
\boxed{24{,}576 \text{ bytes/token} \approx 24 \text{ KB/token}}
$$

### 5.4 Growth table

| Context length $t$ | KV cache size |
|---|---|
| 1,024 | 24 MB |
| 4,096 | 96 MB |
| 16,384 | 384 MB |
| 32,768 | 768 MB |
| 65,536 | 1.5 GB |
| 128,000 | 3.0 GB |

This is for our small 12-layer model. The important point is not just the number but the scaling law: $\text{KV cache} \propto t$. The cache grows linearly with generated context length and never shrinks unless we explicitly evict or compress it.

### 5.5 A cleaner closed form

Because $d_k = d_v$ and $h d_k = d_\text{model}$, the cache formula simplifies in a surprisingly clean way. Start from $\text{KV bytes/token} = L \cdot h \cdot (d_k + d_v) \cdot 2$. Since $d_k = d_v$, we have $d_k + d_v = 2d_k$, so $\text{KV bytes/token} = L \cdot h \cdot 2d_k \cdot 2 = 4L(h d_k)$. Now use $h d_k = d_\text{model}$:

$$
\boxed{\text{KV bytes/token} = 4L d_\text{model}}
$$

### 5.6 Numerical check of the closed form

Substitute $L = 12$ and $d_\text{model} = 512$:

$$
4 \cdot 12 \cdot 512 = 24{,}576
$$

bytes per token, exactly matching the earlier derivation.

This form is worth remembering because it shows that, for standard MHA, the per-token KV cost depends only on model depth and width. The head count disappears once we use the conventional relation $h d_k = d_\text{model}$.

### 5.7 Why this is different from the $n^2$ wall

This is the part that confuses almost everyone. The KV cache is **not** another version of the training-time $n^2$ wall. It is different in two ways. First, the stored memory grows linearly with context length, $O(t)$. Second, every new token must read the whole cache accumulated so far, so the per-step bandwidth cost also grows linearly with $t$. That is why long-context inference feels slow even when we generate only one token at a time — each step drags a longer and longer KV history through memory. The next blog will derive this bandwidth bottleneck in much more detail. For now, the key fact is simple: training breaks on $n^2$, while inference breaks on the KV cache.

---

## 6. Three Problems, Three Lineages of Solutions

Different attention papers look different because they are attacking different bottlenecks. Once we separate the bottlenecks, the literature becomes much easier to parse.

### 6.1 Compute bottleneck

The first bottleneck is that $QK^\top$ and $PV$ scale quadratically in sequence length. Typical fixes include sparse attention, local-window attention, linear attention, and state-space replacements — all of which change what pairwise interactions are computed.

### 6.2 Memory / HBM bottleneck

The second bottleneck is that materializing $S$ and $P$ forces large $n^2$ tensors through HBM. The canonical fix here is FlashAttention. It does not change the attention formula — it changes the schedule, tiling the computation so the large matrices are never written to HBM in the first place.

### 6.3 KV-cache bottleneck

The third bottleneck is that inference stores and rereads one K and one V vector per layer, per head, per token. Typical fixes include Multi-Query Attention (MQA), Grouped-Query Attention (GQA), Multi-head Latent Attention (MLA), sliding-window caches, and KV quantization, all of which change what gets stored across decoding steps.

### 6.4 Why one fix does not solve the others

This is the unifying insight of the whole post. FlashAttention solves the training memory traffic problem, but it does **not** shrink the inference KV cache. GQA shrinks the inference KV cache, but it does **not** remove the quadratic training-time $n \times n$ interaction pattern. Sparse or linear attention reduce quadratic arithmetic, but they may or may not help the KV cache, depending on whether they also change what is stored. So when two papers claim to make attention "efficient," they may not be addressing the same bottleneck at all.

---

## Summary

Vanilla attention breaks in three distinct ways. First, its arithmetic cost grows as $O(n^2)$ because every token interacts with every other token. Second, its training-time memory and HBM traffic are dominated by the dense $n \times n$ score and probability matrices. Third, its autoregressive inference path accumulates a KV cache whose size grows linearly with context and whose bandwidth cost grows with every generated token.

These three bottlenecks are why the literature branches: sparse and linear methods attack quadratic compute, FlashAttention attacks training-time memory traffic, and MQA/GQA/MLA attack the KV cache. The next blog zooms in on that third bottleneck and derives the KV-cache story much more deeply.

---

*Previous: [From Soft Alignment to Queries, Keys, and Values](/blog/attention-q-k-v-from-scratch)*  
*Next: [What Can We Actually Modify in Attention?](/blog/attention-what-can-we-modify)*
