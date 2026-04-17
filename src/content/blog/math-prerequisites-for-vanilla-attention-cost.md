---
title: "Mathematical Prerequisites for the Cost of Vanilla Attention"
description: "Building the cost-accounting foundations for vanilla attention — the FLOP count of a matrix multiplication, the byte count of a tensor, the bandwidth of a memory hierarchy, arithmetic intensity as FLOPs per byte, and the roofline model with its ridge point — all derived step by step with one consistent two-token attention example."
date: 2026-03-31
tags: [machine-learning, attention, transformers, mathematics, performance, roofline, memory, efficiency]
order: 3
---

The next two posts in this series explain *why* vanilla attention becomes infeasible at long sequences. [Why Vanilla Attention Breaks at Scale](/blog/attention-why-vanilla-breaks) traces the $O(n^2)$ wall in compute and memory during training. [The KV Bottleneck](/blog/attention-kv-bottleneck) shows that during autoregressive inference the bottleneck flips: it is no longer compute but memory bandwidth that decides how fast the model runs.

Both posts speak the same language — **FLOPs, bytes, bandwidth, arithmetic intensity, the roofline model** — and neither stops to build that language from scratch. This post does.

By the end, you will have five tools. You will be able to count the FLOPs in any matrix multiplication, count the bytes of any tensor, divide the two to get an **arithmetic intensity**, compare that intensity against a GPU's **ridge point** to decide whether the operation is compute-bound or memory-bound, and reason about why the same operation can be compute-bound in training but memory-bound in single-token inference. Every boxed result in the two follow-up posts will then read as an application of a formula we have already derived here.

---

## The Running Example

A single attention head on a sequence of 2 tokens, with head dimension $d_k = d_v = 4$, stored in fp16 (2 bytes per element). The queries, keys, and values are 2×4 matrices:

$$
Q = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix}, \qquad
K = \begin{pmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{pmatrix}, \qquad
V = \begin{pmatrix} 2 & 0 & 0 & 0 \\ 0 & 2 & 0 & 0 \end{pmatrix}
$$

We will use these matrices to count FLOPs in $QK^\top$, count bytes of the resulting attention score matrix, compute an arithmetic intensity, and compare it to a hypothetical GPU's ridge point. In the two follow-up posts, the same formulas run with $n$ in the thousands and $d_k = 64$, but the structure is identical.

---

## 1. FLOPs: Counting the Work in a Matrix Multiplication

### 1.1 Motivation

Before we can say anything about attention's cost, we need to count one thing precisely: how many floating-point operations does a matrix multiplication cost? The attention score matrix $S = QK^\top$, the projection $PV$ after softmax, and every linear layer in the network are all matrix multiplications. If we can count FLOPs for a generic matmul, we can count them for attention.

### 1.2 FLOPs in a single dot product

A **floating-point operation** (FLOP) is one scalar multiplication or one scalar addition between floating-point numbers. The dot product of two vectors of length $d$

$$
a^\top b = a_1 b_1 + a_2 b_2 + \cdots + a_d b_d
$$

consists of $d$ multiplications and $d - 1$ additions. That is $2d - 1$ FLOPs. We follow the standard convention in the performance literature and round this up to

$$
\boxed{\text{FLOPs(dot product of length } d) = 2d}
$$

The "−1" disappears because for large $d$ it is negligible and because hardware typically fuses a multiply and an add into one unit (the **FMA**, fused multiply-add). Counting each FMA as 2 FLOPs is the convention every GPU spec sheet uses.

### 1.3 FLOPs in a matrix multiplication

Consider the product $C = AB$ where $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$, so $C \in \mathbb{R}^{m \times n}$. Each entry $C_{ij}$ is a dot product between row $i$ of $A$ and column $j$ of $B$, both of length $k$:

$$
C_{ij} = \sum_{\ell=1}^{k} A_{i\ell} B_{\ell j}
$$

By Section 1.2, each such dot product costs $2k$ FLOPs. There are $mn$ entries in $C$, so:

$$
\boxed{\text{FLOPs}(AB) = 2mkn \quad \text{for } A \in \mathbb{R}^{m \times k},\; B \in \mathbb{R}^{k \times n}}
$$

This is the single most important formula in this post. Every attention-cost calculation we will do is an application of it.

### 1.4 Numerical check with the running example

For $Q K^\top$, we have $Q \in \mathbb{R}^{2 \times 4}$ and $K^\top \in \mathbb{R}^{4 \times 2}$, so $m = 2$, $k = 4$, $n = 2$. By the boxed formula:

$$
\text{FLOPs}(QK^\top) = 2 \cdot 2 \cdot 4 \cdot 2 = 32
$$

Let us confirm by direct counting. The product $S = QK^\top$ is:

$$
S = \begin{pmatrix} Q_1 \cdot K_1 & Q_1 \cdot K_2 \\ Q_2 \cdot K_1 & Q_2 \cdot K_2 \end{pmatrix}
= \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}
$$

Each of the 4 entries is a dot product of length 4. Each dot product costs $2 \cdot 4 = 8$ FLOPs (4 multiplies + 3 adds ≈ 8 under FMA convention). Total: $4 \cdot 8 = 32$ FLOPs. $\checkmark$

### 1.5 Interpretation

Every matrix multiply's FLOP count is **the product of its three shape dimensions, times two**. The attention score matrix $S = QK^\top$ at scale $n$ tokens and head dimension $d_k$ is a matmul of shapes $n \times d_k$ times $d_k \times n$, which by our formula costs $2 \cdot n \cdot d_k \cdot n = 2 d_k n^2$ FLOPs. That $n^2$ is the $O(n^2)$ wall that [Why Vanilla Attention Breaks](/blog/attention-why-vanilla-breaks) opens with — and it comes directly from this formula.

---

## 2. Bytes: The Size of a Tensor in Memory

### 2.1 Motivation

FLOPs measure work. But the GPU also has to *move* the inputs and outputs of that work between memory and compute units. To reason about which side dominates, we need to count bytes. The $n \times n$ attention matrix that looks innocent on paper becomes a memory problem because we have to store it somewhere.

### 2.2 Bytes of a scalar

Modern transformers usually run in **fp16** (half precision) or **bf16** (bfloat16) during training and inference. Both use 16 bits — that is, 2 bytes — per scalar. fp32 uses 4 bytes per scalar, fp8 uses 1 byte. We denote the number of bytes per element as $b$; for the rest of this post and the follow-ups, $b = 2$ unless stated.

### 2.3 Bytes of a tensor

A tensor with shape $(d_1, d_2, \ldots, d_r)$ stored densely in memory occupies

$$
\boxed{\text{bytes}(T) = b \cdot \prod_{i=1}^{r} d_i}
$$

bytes, where the product is the total number of scalars (the number of entries in the tensor).

### 2.4 Numerical check with the running example

The attention score matrix $S = QK^\top \in \mathbb{R}^{2 \times 2}$ has 4 entries. In fp16:

$$
\text{bytes}(S) = 2 \cdot 2 \cdot 2 = 8 \text{ bytes}
$$

We can verify by counting: each of the 4 entries of $S$ is one fp16 number of size 2 bytes, giving $4 \cdot 2 = 8$. $\checkmark$

The queries matrix $Q \in \mathbb{R}^{2 \times 4}$ has 8 entries, so $\text{bytes}(Q) = 2 \cdot 8 = 16$. Same for $K$ and $V$. Total input to the attention head: $3 \cdot 16 = 48$ bytes.

### 2.5 Interpretation

The attention score matrix at scale is $n \times n$, so its size is $2 n^2$ bytes in fp16. This becomes the second $O(n^2)$ scaling in vanilla attention — not FLOPs but memory. For $n = 8192$ that is already $2 \cdot 8192^2 \approx 134$ MB *per head per layer*, and we will see in [Why Vanilla Attention Breaks](/blog/attention-why-vanilla-breaks) that this number multiplied across heads and layers is what blows up training memory.

---

## 3. Memory Hierarchy: Bandwidth and the HBM/SRAM Split

### 3.1 Motivation

Counting bytes is not the same as counting *time*. A byte sitting in on-chip SRAM is accessible in roughly a nanosecond; the same byte sitting in off-chip HBM takes tens of times longer to reach the compute units. To predict whether attention will be slow because of compute or slow because of data movement, we need one more number: bandwidth.

### 3.2 The two-level memory model

A modern GPU has two levels of memory that matter for our analysis:

- **HBM** (high-bandwidth memory): the large, off-chip memory where weights, activations, and the KV cache live. On an A100, the capacity is roughly 40–80 GB and the bandwidth is roughly 1.5–2 TB/s.
- **SRAM** (on-chip memory): registers and shared memory inside each streaming multiprocessor. Orders of magnitude faster (bandwidth in the 10s of TB/s) but orders of magnitude smaller (hundreds of KB per SM, tens of MB total).

The exact numbers vary by device. What matters for every derivation in this series is the *pattern*:

$$
\text{bandwidth}(\text{SRAM}) \gg \text{bandwidth}(\text{HBM}),
\qquad
\text{capacity}(\text{SRAM}) \ll \text{capacity}(\text{HBM})
$$

### 3.3 Bandwidth as a rate

**Bandwidth** is the maximum number of bytes per second the memory can deliver to the compute units. If a tensor has $B$ bytes and lives in memory with bandwidth $\beta$ (bytes per second), reading the whole tensor takes at least

$$
\boxed{t_\text{read} = \frac{B}{\beta}}
$$

seconds. This is a lower bound: the bandwidth number on the spec sheet assumes perfect utilization.

### 3.4 Numerical check with the running example

Suppose our (hypothetical) GPU has HBM bandwidth $\beta = 2 \times 10^{12}$ bytes/s = 2 TB/s. Reading the 8-byte attention matrix $S$ from HBM takes at least:

$$
t_\text{read}(S) = \frac{8}{2 \times 10^{12}} = 4 \times 10^{-12} \text{ s} = 4 \text{ ps}
$$

Reading the 16-byte query matrix $Q$ takes 8 ps. These numbers are absurdly small because our example is tiny. The point of the formula is not the ps — it is that doubling the matrix size doubles the read time, linearly, forever.

### 3.5 Interpretation

Bandwidth is a *rate*, not a total. The 2 TB/s number is not a budget; it is a ceiling. If we need to read a 40 GB KV cache once per generated token and HBM runs at 2 TB/s, we have a lower bound of $40 / 2000 = 0.02$ s per token — 50 tokens/sec — regardless of how fast the compute units can run. This is the core mechanism behind the KV bottleneck.

---

## 4. Arithmetic Intensity: FLOPs per Byte

### 4.1 Motivation

An operation has two costs: a compute cost (FLOPs) and a data-movement cost (bytes). Whichever finishes *last* determines the actual runtime. To compare the two, we need a single number that says how much compute the operation does *per byte of data it moves*. That number is arithmetic intensity.

### 4.2 Definition

The **arithmetic intensity** $I$ of an operation is the ratio of its FLOPs to its bytes-moved:

$$
\boxed{I = \frac{\text{FLOPs}}{\text{bytes moved}}}
$$

Units: FLOPs/byte. "Bytes moved" means bytes that must travel between HBM and the compute units — reading inputs, writing outputs, re-reading spilled intermediates.

The numerator was derived in Section 1, the denominator in Section 2. Arithmetic intensity is a derived quantity, not a new one.

### 4.3 Derivation of intensity for matrix multiplication

Consider the matmul $C = AB$ with $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$, $C \in \mathbb{R}^{m \times n}$, all in fp16 (so $b = 2$).

- FLOPs (by Section 1.3): $2mkn$.
- Bytes moved, assuming we read $A$ once, read $B$ once, and write $C$ once: $2(mk + kn + mn)$.

The intensity is:

$$
I_\text{matmul} = \frac{2mkn}{2(mk + kn + mn)} = \frac{mkn}{mk + kn + mn}
$$

Two special cases are worth naming.

**Square matmul** ($m = n = k$, so an $n \times n$ times $n \times n$ product):

$$
I = \frac{n \cdot n \cdot n}{n^2 + n^2 + n^2} = \frac{n^3}{3n^2} = \frac{n}{3}
$$

Intensity grows linearly with $n$ — big square matmuls are compute-heavy.

**Matrix–vector product** ($n = 1$, so $A \in \mathbb{R}^{m \times k}$ times a vector in $\mathbb{R}^{k}$):

$$
I = \frac{m \cdot k \cdot 1}{mk + k + m} \approx \frac{mk}{mk} = 1
$$

Intensity is roughly 1 FLOP per byte — bandwidth-dominated. This is the core reason single-token autoregressive decoding is memory-bound: every linear layer becomes a matrix–vector product.

### 4.4 Numerical check with the running example

For $S = QK^\top$ with $m = n = 2$ and $k = 4$:

$$
I = \frac{2 \cdot 4 \cdot 2}{2 \cdot 4 + 4 \cdot 2 + 2 \cdot 2} = \frac{16}{8 + 8 + 4} = \frac{16}{20} = 0.8 \text{ FLOPs/byte}
$$

Let us double-check by going back to counted totals: 32 FLOPs (Section 1.4) and 40 bytes moved ($16 + 16 + 8 = 40$, by Section 2.4). Ratio: $32 / 40 = 0.8$. $\checkmark$

### 4.5 Interpretation

Arithmetic intensity is not a property of hardware — it is a property of the *operation*. The square matmul and the matrix–vector product have different intensities because one reuses inputs far more than the other. The same matrix read from HBM once feeds thousands of dot products in a big matmul, amortizing the read; a matvec uses each byte of the matrix exactly once.

---

## 5. The Roofline Model

### 5.1 Motivation

We have a per-operation number (intensity, FLOPs/byte) and two per-hardware numbers (peak compute in FLOPs/s, peak bandwidth in bytes/s). We want one question answered: given these three numbers, how fast can the operation run? The **roofline model** (Williams, Waterman, and Patterson, 2009) is the clean answer.

### 5.2 The two upper bounds on throughput

For an operation with intensity $I$ running on hardware with peak compute $\pi$ (FLOPs/s) and peak bandwidth $\beta$ (bytes/s), the achievable FLOP throughput is bounded by two things:

1. **Compute ceiling.** The hardware cannot exceed its peak compute rate: $\text{throughput} \leq \pi$.
2. **Bandwidth ceiling.** The operation moves one byte every $1/\beta$ seconds, and every byte moved fuels $I$ FLOPs. So the FLOP rate cannot exceed $I \cdot \beta$.

Taking the smaller of the two:

$$
\boxed{\text{throughput} = \min(\pi,\; I \cdot \beta)}
$$

This is the **roofline bound**. Plotted on log axes with $I$ on the x-axis and throughput on the y-axis, it looks like a slanted line ($I \cdot \beta$) meeting a horizontal line ($\pi$) — a "roof" shape, hence the name.

### 5.3 Derivation of the ridge point

The two ceilings meet where $\pi = I \cdot \beta$. Solving for $I$ gives the **ridge point** $I^*$:

$$
\boxed{I^* = \frac{\pi}{\beta}}
$$

The ridge point is a hardware-only number — it depends on the GPU, not on the operation. An operation with $I < I^*$ sits on the bandwidth-limited slope and is called **memory-bound**. An operation with $I > I^*$ sits under the compute ceiling and is called **compute-bound**.

### 5.4 Numerical check with the running example

Take a hypothetical GPU with $\pi = 300 \times 10^{12}$ FLOPs/s (300 TFLOPs/s in fp16) and $\beta = 2 \times 10^{12}$ bytes/s (2 TB/s). The ridge point is:

$$
I^* = \frac{300 \times 10^{12}}{2 \times 10^{12}} = 150 \text{ FLOPs/byte}
$$

Our $QK^\top$ operation has $I = 0.8$ (Section 4.4). Since $0.8 \ll 150$, the operation is *memory-bound* — by a factor of nearly 200. Its predicted throughput is:

$$
\text{throughput}(QK^\top) = \min(300 \times 10^{12},\; 0.8 \cdot 2 \times 10^{12}) = 1.6 \times 10^{12} \text{ FLOPs/s}
$$

Only about 0.5% of peak compute is achievable — because the operation is so small it cannot even come close to the ridge. A full-scale $QK^\top$ at $n = 4096$, $d_k = 64$ would have intensity $I = \frac{n \cdot d_k \cdot n}{n d_k + d_k n + n^2} = \frac{n d_k}{2 d_k + n} \approx 64$ for large $n$, still below the ridge — so even at scale, $QK^\top$ is memory-bound.

### 5.5 Compute-bound vs memory-bound: the same formula, two regimes

The roofline model unifies two regimes that feel different but are the same formula:

**Compute-bound** ($I > I^*$): throughput is $\pi$, and making memory faster does nothing. This is where large square matmuls live. To go faster, buy more FLOPs.

**Memory-bound** ($I < I^*$): throughput is $I \cdot \beta$, and making the processor faster does nothing. This is where single-token decoding and the attention score matrix live. To go faster, either raise $I$ (by doing more work per byte, e.g. fusing kernels as in FlashAttention) or raise $\beta$ (better memory).

This split is the single most useful lens in the two follow-up posts. [Why Vanilla Attention Breaks](/blog/attention-why-vanilla-breaks) notes that the $QK^\top$ and softmax operations are memory-bound because their intensity is low — every pass over the $n \times n$ score matrix touches HBM. [The KV Bottleneck](/blog/attention-kv-bottleneck) shows that at inference time the KV cache is read once per generated token with intensity $\approx 1$, placing the entire decoding loop deep in memory-bound territory.

---

## Summary

We built five tools, each derived from the one before it. **FLOPs** count work: the dot product costs $2d$ and a matmul $A B$ with $A$ of shape $m \times k$ and $B$ of shape $k \times n$ costs $\boxed{2mkn}$. **Bytes** count tensor size: shape product times bytes-per-element, typically 2 in fp16. The **memory hierarchy** splits GPU memory into fast small SRAM and slow large HBM, and reading a tensor takes at least $B/\beta$ seconds. The **arithmetic intensity** $I = \text{FLOPs} / \text{bytes}$ is a property of the operation, not the hardware — and it is high for square matmuls (grows like $n/3$) but low for matrix–vector products ($\approx 1$). The **roofline model** combines all of these: achievable throughput is $\min(\pi, I \beta)$, and the ridge point $I^* = \pi / \beta$ decides whether an operation is compute-bound or memory-bound.

With these tools in hand, we are ready for [Why Vanilla Attention Breaks at Scale](/blog/attention-why-vanilla-breaks) and [The KV Bottleneck Explained](/blog/attention-kv-bottleneck), where every cost table, every "this is memory-bound" claim, and every ridge-point comparison is a direct application of the five formulas above.
