---
title: "Grouped-Query Attention: Fewer KV Heads, Same Quality"
description: "Building GQA from the ground up — from multi-head attention to multi-query attention to grouped-query attention — showing exactly how sharing KV heads across query groups reduces the KV cache by a factor of h/g while preserving nearly all of MHA's quality. Every formula derived, every number verified."
date: 2026-04-02
tags: [machine-learning, attention, transformers, gqa, mqa, kv-cache, inference]
order: 1
---

The previous blog established that autoregressive inference is memory-bandwidth-bound, and that the KV cache is the dominant memory load at long contexts. The formula was exact: $\text{bytes/token} = L \cdot h \cdot (d_k + d_v) \cdot 2$. The factor $h$ — the number of KV heads — appears linearly. Reduce $h_\text{KV}$ and the cache shrinks proportionally.

This blog derives the two techniques that pull this lever: **Multi-Query Attention** (MQA, Shazeer 2019), which collapses all KV heads to one, and **Grouped-Query Attention** (GQA, Ainslie et al. 2023), which finds the sweet spot in between. We re-derive both from the vanilla formula, verify every number with our running model, and trace through the GQA paper's key experimental results.

---

## The Running Model

Same as every blog in this series:

- $d_\text{model} = 512$
- $h = 8$ query heads
- $d_k = d_v = 64$
- $L = 12$ layers
- fp16 (2 bytes per element)

All KV cache numbers are per-token unless stated otherwise.

---

## Vanilla MHA: The Baseline to Beat

In standard multi-head attention, each head $i \in \{1, \ldots, h\}$ has its own query, key, and value projection:

$$
Q^i = X W_Q^i, \quad K^i = X W_K^i, \quad V^i = X W_V^i
$$

where $W_Q^i, W_K^i \in \mathbb{R}^{d_\text{model} \times d_k}$ and $W_V^i \in \mathbb{R}^{d_\text{model} \times d_v}$. Each head computes attention independently:

$$
A^i = \text{softmax}\!\left(\frac{Q^i {K^i}^\top}{\sqrt{d_k}}\right), \quad o^i = A^i V^i
$$

The outputs are concatenated and projected:

$$
\text{MHA}(X) = \text{concat}(o^1, \ldots, o^h) \, W_O
$$

### Counting every parameter and byte

Let us be extremely precise about what MHA costs. We count three things: parameters, cache, and bandwidth.

**Total attention parameters per layer.** There are four categories of weight matrices:

1. Query projections: $h$ matrices $W_Q^i \in \mathbb{R}^{d_\text{model} \times d_k}$, each with $d_\text{model} \cdot d_k = 512 \times 64 = 32{,}768$ parameters. Total: $h \cdot d_\text{model} \cdot d_k = 8 \times 32{,}768 = 262{,}144$.
2. Key projections: same count. Total: $262{,}144$.
3. Value projections: same count ($d_v = d_k$). Total: $262{,}144$.
4. Output projection: $W_O \in \mathbb{R}^{(h \cdot d_v) \times d_\text{model}} = \mathbb{R}^{512 \times 512}$. Total: $d_\text{model}^2 = 262{,}144$.

Grand total per layer: $4 \times 262{,}144 = 1{,}048{,}576 \approx 1\text{M}$ parameters.

**Numerical check**: This is $4 \cdot d_\text{model}^2 = 4 \times 512^2 = 4 \times 262{,}144 = 1{,}048{,}576$. Correct — MHA's attention block has exactly $4 d_\text{model}^2$ parameters, independent of how those parameters are divided among heads.

**KV parameters specifically.** The KV projections are categories 2 and 3 above:

$$
\text{KV params per layer} = h \cdot d_\text{model} \cdot (d_k + d_v) = 8 \times 512 \times (64 + 64) = 524{,}288
$$

As a fraction of total attention parameters: $524{,}288 / 1{,}048{,}576 = 0.5$. Exactly half of the attention parameters are in KV projections. This makes sense: Q and O each contribute $d_\text{model}^2$ parameters, and K+V together also contribute $h \cdot d_\text{model} \cdot (d_k + d_v) = 2 d_\text{model}^2$ in our running model where $d_k = d_v$ and $h d_k = d_\text{model}$.

**KV cache per token per layer.** At inference, for each processed token, we store one key vector and one value vector per head:

$$
\text{KV cache per token per layer} = h \cdot (d_k + d_v) \cdot 2 = 8 \times (64 + 64) \times 2 = 2{,}048 \text{ bytes}
$$

Let us break this down further — element by element — to make sure we understand exactly what is stored:

- Head 1: $k_t^1 \in \mathbb{R}^{64}$ (128 bytes) + $v_t^1 \in \mathbb{R}^{64}$ (128 bytes) = 256 bytes
- Head 2: $k_t^2 \in \mathbb{R}^{64}$ (128 bytes) + $v_t^2 \in \mathbb{R}^{64}$ (128 bytes) = 256 bytes
- ...
- Head 8: $k_t^8 \in \mathbb{R}^{64}$ (128 bytes) + $v_t^8 \in \mathbb{R}^{64}$ (128 bytes) = 256 bytes

Total: $8 \times 256 = 2{,}048$ bytes per token per layer. Across $L = 12$ layers:

$$
12 \times 2{,}048 = 24{,}576 \text{ bytes} = 24 \text{ KB per token}
$$

**KV memory bandwidth per step.** At context length $t$, generating the next token requires loading all $t$ cached K and V vectors from HBM (one full read of the entire cache per step):

$$
\text{bandwidth}_\text{KV} = L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot t = 24{,}576 \cdot t \text{ bytes}
$$

At $t = 4{,}096$: $24{,}576 \times 4{,}096 = 100{,}663{,}296$ bytes $\approx 96$ MB per step.

At $t = 128{,}000$: $24{,}576 \times 128{,}000 = 3{,}145{,}728{,}000$ bytes $\approx 3$ GB per step.

These are the numbers to beat. Every variant in this blog reduces one or more of these costs.

---

## Multi-Query Attention: The Extreme Case

**Multi-Query Attention** (MQA) was introduced by Shazeer (2019). The idea is the simplest possible attack on the KV cache: keep $h$ separate query heads, but use a *single* shared key head and a *single* shared value head.

### Deriving MQA from MHA

Start from the MHA formula and make one change: instead of $h$ separate key matrices $W_K^1, \ldots, W_K^h$ and $h$ separate value matrices $W_V^1, \ldots, W_V^h$, use a single shared $W_K$ and a single shared $W_V$.

There is now one $W_K \in \mathbb{R}^{d_\text{model} \times d_k}$ and one $W_V \in \mathbb{R}^{d_\text{model} \times d_v}$. Each query head still has its own $W_Q^i$:

$$
Q^i = X W_Q^i \quad \forall\, i \in \{1, \ldots, h\}, \qquad K = X W_K, \qquad V = X W_V
$$

Head $i$ computes attention using its own query against the shared key and value:

$$
A^i = \text{softmax}\!\left(\frac{Q^i K^\top}{\sqrt{d_k}}\right), \quad o^i = A^i V
$$

Every head uses the same $K$ and $V$. The concatenation and output projection are unchanged:

$$
\text{MQA}(X) = \text{concat}(o^1, \ldots, o^h) \, W_O
$$

### Deriving the savings step by step

**KV parameters per layer.** We now have one $W_K$ and one $W_V$ instead of $h$ of each:

$$
\text{KV params}_\text{MQA} = d_\text{model} \cdot (d_k + d_v) = 512 \times (64 + 64) = 65{,}536
$$

Compare to MHA: $524{,}288$. The ratio:

$$
\frac{\text{KV params}_\text{MHA}}{\text{KV params}_\text{MQA}} = \frac{524{,}288}{65{,}536} = 8 = h
$$

MQA uses $h\times$ fewer KV parameters. The factor is exactly the number of heads, because we went from $h$ copies to 1 copy.

**Total attention parameters per layer.** Q projections are unchanged: $h \cdot d_\text{model} \cdot d_k = 262{,}144$. KV projections: $65{,}536$ (down from $524{,}288$). Output projection: $262{,}144$ (unchanged).

Total: $262{,}144 + 65{,}536 + 262{,}144 = 589{,}824$.

Compare to MHA: $1{,}048{,}576$. Ratio: $1{,}048{,}576 / 589{,}824 = 1.78$. MQA has $44\%$ fewer attention parameters — a significant reduction, but less than the $8\times$ cache reduction because the Q and O matrices are unchanged.

**KV cache per token per layer.** Only one K vector and one V vector are cached (shared across all heads):

$$
\text{KV cache}_\text{MQA} = 1 \cdot (d_k + d_v) \cdot 2 = (64 + 64) \times 2 = 256 \text{ bytes}
$$

Let us spell out exactly what is stored per token per layer in MQA:

- Shared K: $k_t \in \mathbb{R}^{64}$ (128 bytes)
- Shared V: $v_t \in \mathbb{R}^{64}$ (128 bytes)
- Total: 256 bytes

Compare to MHA's 2,048 bytes per token per layer. Ratio: $2{,}048 / 256 = 8 = h$. Exactly $h\times$ smaller.

Across $L = 12$ layers: $12 \times 256 = 3{,}072$ bytes $= 3$ KB/token.

**Numerical check**: MHA was 24 KB/token. $24 / 3 = 8 = h$. Correct.

**Bandwidth per step at $t = 4{,}096$:**

$$
3{,}072 \times 4{,}096 = 12{,}582{,}912 \text{ bytes} \approx 12 \text{ MB}
$$

Compare to MHA's 96 MB — an $8\times$ reduction. At A100 bandwidth of 2 TB/s:

$$
T_\text{KV}^\text{MQA} = \frac{12 \times 10^6}{2 \times 10^{12}} = 6 \text{ microseconds}
$$

Compare to MHA's 48 microseconds. The KV cache loading time dropped from 48 $\mu$s to 6 $\mu$s. This is the primary source of MQA's inference speedup.

**Bandwidth per step at $t = 128{,}000$:**

$$
3{,}072 \times 128{,}000 = 393{,}216{,}000 \text{ bytes} \approx 375 \text{ MB}
$$

Compare to MHA's 3 GB. Still $8\times$ smaller.

### What MQA changes about the attention computation

This is the part that confuses almost everyone on first encounter. In MHA, each head's attention scores are:

$$
s_j^i = \frac{(q^i)^\top k_j^i}{\sqrt{d_k}}
$$

Head $i$'s query $q^i$ is dotted with head $i$'s keys $k_j^i$. Each head operates in its own subspace.

In MQA:

$$
s_j^i = \frac{(q^i)^\top k_j}{\sqrt{d_k}}
$$

Head $i$'s query $q^i$ — which lives in a head-specific subspace defined by $W_Q^i$ — is dotted with the *shared* key $k_j$ — which lives in a single subspace defined by the shared $W_K$.

The attention *scores* are different per head (because each $q^i$ is different), so each head produces a different attention distribution $a^i$. But the *values* being weighted are the same $V$ for all heads. So the output of each head is:

$$
o^i = \sum_j a_j^i \cdot v_j
$$

Different heads produce different weighted combinations of the same value vectors. In MHA, different heads produce different weighted combinations of different value vectors.

**Concrete example.** Suppose at position $t$, head 1 assigns attention weight 0.8 to token 5 and 0.2 to token 12, while head 3 assigns weight 0.5 to each. In MHA:

$$
o^1 = 0.8 \cdot v_5^1 + 0.2 \cdot v_{12}^1, \quad o^3 = 0.5 \cdot v_5^3 + 0.5 \cdot v_{12}^3
$$

The two outputs differ in both the *weights* and the *values*. In MQA:

$$
o^1 = 0.8 \cdot v_5 + 0.2 \cdot v_{12}, \quad o^3 = 0.5 \cdot v_5 + 0.5 \cdot v_{12}
$$

The outputs differ only in the weights. Both heads are selecting from the same "library" of value representations. This is a meaningful capacity reduction — the model can no longer represent different information per head in the value space.

### The cost of MQA: empirical evidence

The GQA paper (Ainslie et al., 2023, Table 1) quantifies the quality cost on T5-XXL:

| Model | Avg. quality | Inference time (s/sample) |
|---|---|---|
| MHA-XXL | 47.2 | 1.51 |
| MQA-XXL (uptrained) | 46.6 | 0.24 |

The quality drops by 0.6 points. The inference time drops by $6.3\times$.

Is a 0.6-point quality drop acceptable? It depends on the application. For some tasks, 0.6 points is noise. For others — especially when the model is already near the state of the art — 0.6 points is the difference between best-in-class and second-tier.

The question becomes: can we find a middle ground that recovers most of the quality while keeping most of the speed?

---

## Grouped-Query Attention: The Interpolation

**Grouped-Query Attention** (GQA) is the answer. It was introduced by Ainslie et al. (2023) as a generalization that places MHA and MQA at opposite ends of a single spectrum.

### The idea

Divide the $h$ query heads into $g$ groups of equal size. Each group of $h/g$ query heads shares one key head and one value head. There are $g$ KV heads total.

Three boundary cases:

- $g = h$: every group has exactly one query head → every query head has its own KV head → this is MHA
- $g = 1$: all $h$ query heads are in one group → all share one KV head → this is MQA
- $1 < g < h$: the intermediate case → this is GQA

### Writing the formula explicitly

Let $g$ denote the number of KV groups. The number of query heads per group is $h/g$ (we require $g$ divides $h$). The heads are partitioned into consecutive groups:

- Group 1: query heads $\{1, 2, \ldots, h/g\}$
- Group 2: query heads $\{h/g + 1, \ldots, 2h/g\}$
- ...
- Group $g$: query heads $\{h - h/g + 1, \ldots, h\}$

For our running model with $h = 8$ and $g = 2$: group 1 has query heads $\{1, 2, 3, 4\}$, group 2 has query heads $\{5, 6, 7, 8\}$. Each group has $h/g = 4$ query heads sharing one KV head.

For KV group $j \in \{1, \ldots, g\}$, compute the shared key and value:

$$
K^j = X W_K^j, \quad V^j = X W_V^j
$$

For query head $i$ belonging to group $j = \lceil i \cdot g / h \rceil$, compute:

$$
Q^i = X W_Q^i
$$

$$
A^i = \text{softmax}\!\left(\frac{Q^i {K^j}^\top}{\sqrt{d_k}}\right)
$$

$$
o^i = A^i V^j
$$

The output is the same concatenation and projection as always:

$$
\text{GQA}(X) = \text{concat}(o^1, \ldots, o^h) \, W_O
$$

### Deriving the savings for general $g$

**KV parameters per layer.** There are $g$ key matrices of size $d_\text{model} \times d_k$ and $g$ value matrices of size $d_\text{model} \times d_v$:

$$
\text{KV params}_\text{GQA} = g \cdot d_\text{model} \cdot (d_k + d_v)
$$

With $g = 2$:

$$
2 \times 512 \times (64 + 64) = 131{,}072
$$

With $g = 4$:

$$
4 \times 512 \times (64 + 64) = 262{,}144
$$

**Numerical check** of the reduction factor for $g = 2$:

$$
\frac{\text{KV params}_\text{MHA}}{\text{KV params}_\text{GQA}} = \frac{524{,}288}{131{,}072} = 4 = \frac{h}{g} = \frac{8}{2}
$$

For $g = 4$: $524{,}288 / 262{,}144 = 2 = 8/4$. Correct in both cases.

**KV cache per token per layer.** Each of the $g$ KV groups caches one K vector and one V vector:

$$
\text{KV cache}_\text{GQA} = g \cdot (d_k + d_v) \cdot 2
$$

Let us compute this for every possible $g$ in our $h = 8$ model. Since $g$ must divide $h$, the options are $g \in \{1, 2, 4, 8\}$:

$g = 1$ (MQA): $1 \times 128 \times 2 = 256$ bytes per layer

$g = 2$: $2 \times 128 \times 2 = 512$ bytes per layer

$g = 4$: $4 \times 128 \times 2 = 1{,}024$ bytes per layer

$g = 8$ (MHA): $8 \times 128 \times 2 = 2{,}048$ bytes per layer

Across $L = 12$ layers:

| $g$ | bytes/token/layer | bytes/token (all layers) | KB/token |
|---|---|---|---|
| 1 (MQA) | 256 | 3,072 | 3 |
| 2 | 512 | 6,144 | 6 |
| 4 | 1,024 | 12,288 | 12 |
| 8 (MHA) | 2,048 | 24,576 | 24 |

**Numerical check**: each row is exactly $g/8$ of the MHA row. $3 / 24 = 1/8$, $6 / 24 = 1/4$, $12 / 24 = 1/2$. Correct.

### The general reduction formula

Dividing the MHA cache by the GQA cache:

$$
\frac{\text{KV cache}_\text{MHA}}{\text{KV cache}_\text{GQA}} = \frac{h \cdot (d_k + d_v) \cdot 2}{g \cdot (d_k + d_v) \cdot 2}
$$

The $(d_k + d_v)$ factor appears in both numerator and denominator — it cancels. The factor of 2 (bytes per element) also cancels. What remains is:

$$
\frac{h}{g}
$$

$$
\boxed{\text{KV cache reduction factor} = \frac{h}{g}}
$$

This is the central equation. The reduction depends only on the ratio of query heads to KV groups. Not on $d_k$, not on $d_v$, not on $d_\text{model}$, not on $L$, not on the sequence length. Just $h/g$.

**Interpretation.** This result says something powerful: regardless of model size, head dimension, or sequence length, GQA with $g$ groups reduces the KV cache by exactly $h/g$. A model with 128 heads and 16 groups gets the same $8\times$ reduction as a model with 8 heads and 1 group (MQA). The formula is universal.

### Bandwidth savings at every context length

The bandwidth per step for GQA at context length $t$:

$$
\text{BW}_\text{GQA}(t) = L \cdot g \cdot (d_k + d_v) \cdot 2 \cdot t
$$

For our model with $g = 2$, at several context lengths:

| Context $t$ | MHA bandwidth | GQA-2 bandwidth | MQA bandwidth |
|---|---|---|---|
| 512 | 12 MB | 3 MB | 1.5 MB |
| 1,024 | 24 MB | 6 MB | 3 MB |
| 4,096 | 96 MB | 24 MB | 12 MB |
| 16,384 | 384 MB | 96 MB | 48 MB |
| 128,000 | 3,000 MB | 750 MB | 375 MB |

**Numerical check** for GQA-2 at $t = 4{,}096$: $6{,}144 \times 4{,}096 = 25{,}165{,}824$ bytes $\approx 24$ MB. $96 / 24 = 4 = h/g$. Correct.

At $t = 128{,}000$: $6{,}144 \times 128{,}000 = 786{,}432{,}000 \approx 750$ MB. $3{,}000 / 750 = 4$. Correct.

---

## Converting an MHA Checkpoint to GQA

A key practical contribution of the GQA paper is a recipe for converting existing MHA models to GQA *without training from scratch*. Pre-training from scratch costs millions of dollars. If we can convert an existing checkpoint, we save almost all of that cost.

### The conversion problem

We have a trained MHA model with $h$ key projection matrices $W_K^1, \ldots, W_K^h \in \mathbb{R}^{d_\text{model} \times d_k}$ and $h$ value projection matrices $W_V^1, \ldots, W_V^h \in \mathbb{R}^{d_\text{model} \times d_v}$. We want to produce $g < h$ key matrices and $g$ value matrices.

The query projections $W_Q^1, \ldots, W_Q^h$ and the output projection $W_O$ remain unchanged. Only the KV projections change.

### The mean-pooling conversion

The paper's approach: for each group, average the original KV heads assigned to that group.

**Step 1.** Choose $g$ (must divide $h$). Partition heads into $g$ groups of size $h/g$.

**Step 2.** For group $j \in \{1, \ldots, g\}$, let $G_j = \{(j-1) \cdot h/g + 1, \ldots, j \cdot h/g\}$ be the set of original head indices in this group. Compute:

$$
W_K^{(j)} = \frac{1}{|G_j|} \sum_{i \in G_j} W_K^i = \frac{g}{h} \sum_{i \in G_j} W_K^i
$$

$$
W_V^{(j)} = \frac{1}{|G_j|} \sum_{i \in G_j} W_V^i = \frac{g}{h} \sum_{i \in G_j} W_V^i
$$

This is **mean pooling** — each element of the new weight matrix is the arithmetic mean of the corresponding elements from the original heads in the group. Matrix addition is elementwise: $(A + B)_{ij} = A_{ij} + B_{ij}$.

### Tracing the conversion for our running model

In our model with $h = 8$ and $g = 2$:

- Group 1 ($G_1$): original heads $\{1, 2, 3, 4\}$
- Group 2 ($G_2$): original heads $\{5, 6, 7, 8\}$

Each original $W_K^i \in \mathbb{R}^{512 \times 64}$ has 32,768 parameters. The mean-pooled group key:

$$
W_K^{(1)} = \frac{1}{4}\left(W_K^1 + W_K^2 + W_K^3 + W_K^4\right) \in \mathbb{R}^{512 \times 64}
$$

To see what happens at the element level, take element $(r, c)$ for row $r \in \{1, \ldots, 512\}$ and column $c \in \{1, \ldots, 64\}$:

$$
[W_K^{(1)}]_{rc} = \frac{1}{4}\left([W_K^1]_{rc} + [W_K^2]_{rc} + [W_K^3]_{rc} + [W_K^4]_{rc}\right)
$$

Suppose the four original values at position $(1, 1)$ are $0.12, -0.05, 0.08, 0.03$. Then:

$$
[W_K^{(1)}]_{1,1} = \frac{0.12 + (-0.05) + 0.08 + 0.03}{4} = \frac{0.18}{4} = 0.045
$$

This is repeated for all $512 \times 64 = 32{,}768$ elements.

**Counting the resulting parameters.** Before conversion: $8 \times 2 \times 32{,}768 = 524{,}288$ KV parameters. After conversion: $2 \times 2 \times 32{,}768 = 131{,}072$ KV parameters. Reduction: $524{,}288 / 131{,}072 = 4 = h/g$. Correct.

### Why mean pooling works best

The paper (Ainslie et al., 2023, Figure 4) compares three conversion methods for T5-Large uptrained to MQA with $\alpha = 0.05$:

| Method | Performance score |
|---|---|
| Mean pooling | $\approx 55.6$ |
| First head selection | $\approx 55.2$ |
| Random initialization | $\approx 54.4$ |

**Mean pooling** averages all $h/g$ original heads in each group. It preserves the componentwise average signal across those heads, so it tends to retain the shared structure that multiple heads learned in common.

**First head selection** picks $W_K^1$ and discards $W_K^2, \ldots, W_K^{h/g}$. It preserves one head's learned features perfectly but completely loses the other $h/g - 1$ heads' contributions. In our model with $g = 2$, selecting the first head discards 3 out of 4 heads' worth of learned representations.

**Random initialization** creates $W_K^{(j)}$ with random weights drawn from an initialization distribution. It discards *all* learned information and relies entirely on uptraining to learn new representations from scratch.

The ranking makes intuitive sense: more information preserved → better starting point → less work for uptraining → higher final quality.

**Numerical argument for why mean beats first.** Consider a toy case where each head's key projection adds a different "feature direction" to the representation. Head 1 projects onto direction $u_1$, head 2 onto $u_2$, etc. The mean pooling result $\frac{1}{4}(u_1 + u_2 + u_3 + u_4)$ retains a component along all four directions — attenuated by $1/4$, but present. First-head selection keeps only $u_1$ and has zero component along $u_2, u_3, u_4$. In this toy setup, the mean is a more balanced initialization because it preserves information from all four original directions rather than discarding three of them outright.

### Uptraining: adapting to the new structure

After conversion, the model is pre-trained for an additional $\alpha$ fraction of the original pre-training steps. The paper uses $\alpha = 0.05$ — just 5% of the original compute.

**Why uptraining is necessary.** After mean-pooling, the KV projections have changed. Head 1's query $W_Q^1$ was trained to work with head 1's key $W_K^1$ — they learned complementary representations. Now head 1's query must work with the group's mean key $W_K^{(1)} = \frac{1}{4}(W_K^1 + W_K^2 + W_K^3 + W_K^4)$. The dot products $q_t^1 \cdot k_t^{(1)}$ will produce different attention patterns than $q_t^1 \cdot k_t^1$.

Let us trace this quantitatively. Before conversion, the attention score for head 1 between query position $t$ and key position $s$ is:

$$
\text{score}_\text{before} = (x_t W_Q^1)^\top (x_s W_K^1) = x_t^\top (W_Q^1 {W_K^1}^\top) x_s
$$

After conversion (before uptraining):

$$
\text{score}_\text{after} = (x_t W_Q^1)^\top (x_s W_K^{(1)}) = x_t^\top \left(W_Q^1 \cdot \frac{1}{4}\sum_{i=1}^{4} {W_K^i}^\top\right) x_s
$$

By the **distributive law of matrix multiplication**:

$$
= \frac{1}{4} \sum_{i=1}^{4} x_t^\top (W_Q^1 {W_K^i}^\top) x_s = \frac{1}{4}\left(\text{score}_{11} + \text{score}_{12} + \text{score}_{13} + \text{score}_{14}\right)
$$

where $\text{score}_{1i} = x_t^\top (W_Q^1 {W_K^i}^\top) x_s$ is what head 1's query would score against head $i$'s key.

The post-conversion score is the average of what head 1's query would have scored against all four original keys. This is a reasonable starting point — but it is not what the model was optimized for. The $\text{score}_{12}, \text{score}_{13}, \text{score}_{14}$ terms are "cross-head" interactions that the original model never trained on.

Uptraining allows $W_Q^1$ to adapt. It learns to produce queries that give useful attention patterns when scored against the mean key projection. With 5% uptraining ($\alpha = 0.05$), this adaptation is sufficient to recover nearly all quality.

**Uptraining cost.** The paper reports approximately 600 TPUv3 chip-days for uptraining T5-XXL at $\alpha = 0.05$. The original T5-XXL pre-training cost was roughly 12,000 chip-days. So $0.05 \times 12{,}000 = 600$ chip-days — consistent with the reported number. This is a one-time investment that converts an MHA model to a faster inference model.

### How quality changes with uptraining proportion

The paper (Figure 5) shows performance as a function of $\alpha$ for T5-XXL with MQA and GQA-8:

| $\alpha$ | MQA quality | GQA-8 quality | MHA baseline |
|---|---|---|---|
| 0 (no uptraining) | ~53.5 | ~55.5 | 56.2 |
| 0.02 | ~55.5 | ~56.5 | 56.2 |
| 0.05 | ~56.5 | ~57.0 | 56.2 |
| 0.10 | ~57.0 | ~57.0 | 56.2 |

Two observations:

1. **GQA starts higher than MQA** even at $\alpha = 0$. The mean-pooled GQA checkpoint is a better starting point because it has $g = 8$ distinct KV heads rather than just 1, preserving more of the original model's representational capacity.

2. **Both reach diminishing returns around $\alpha = 0.05$**. Doubling the uptraining to $\alpha = 0.10$ gives marginal improvement. The 5% threshold appears to be sufficient for the query projections to adapt.

---

## The Quality-Speed Tradeoff: Tracing the Paper's Results

The GQA paper's central result (Figure 3, Table 1) shows the Pareto frontier of quality vs. inference speed. We trace the key data points in detail.

### Experimental setup

The paper uses T5 models (Raffel et al., 2020) implemented in JAX with Flax:

- **T5-Large** with standard MHA (baseline, small but fast)
- **T5-XXL** with MHA (baseline, large and slow)
- **T5-XXL** uptrained to MQA ($\alpha = 0.05$)
- **T5-XXL** uptrained to GQA-8 ($\alpha = 0.05$, i.e., 8 KV groups out of 64 query heads)

Evaluation on summarization tasks (CNN/Daily Mail, arXiv, PubMed, MediaSum, MultiNews), translation (WMT), and question answering (TriviaQA).

### Main results

| Model | Attention | $T_\text{infer}$ (s) | Average | CNN | arXiv | PubMed | MediaSum | MultiNews | WMT | TriviaQA |
|---|---|---|---|---|---|---|---|---|---|---|
| MHA-Large | MHA | 0.37 | 46.0 | 42.9 | 44.6 | 46.2 | 35.5 | 46.6 | 27.7 | 78.2 |
| MHA-XXL | MHA | 1.51 | 47.2 | 43.8 | 45.6 | 47.5 | 36.4 | 46.9 | 28.4 | 81.9 |
| MQA-XXL | MQA | 0.24 | 46.6 | 43.0 | 45.0 | 46.9 | 36.1 | 46.5 | 28.5 | 81.3 |
| GQA-8-XXL | GQA-8 | 0.28 | 47.1 | 43.5 | 45.4 | 47.7 | 36.3 | 47.2 | 28.4 | 81.6 |

### Interpreting the numbers

**GQA-8-XXL vs. MHA-XXL (the key comparison):**
- Quality: 47.1 vs. 47.2, a drop of just 0.1 points
- Speed: 0.28s vs. 1.51s, a $5.4\times$ speedup

The quality difference is small on the reported tasks. On PubMed, GQA-8 actually *outperforms* MHA (47.7 vs. 47.5). On arXiv, GQA-8 loses 0.2 points (45.4 vs. 45.6). These per-task fluctuations are consistent with the average gap being small, though the paper does not report a formal significance test here.

**MQA-XXL vs. MHA-XXL:**
- Quality: 46.6 vs. 47.2, a drop of 0.6 points
- Speed: 0.24s vs. 1.51s, a $6.3\times$ speedup

MQA is faster than GQA-8 (0.24s vs. 0.28s) but loses more quality (0.6 vs. 0.1 points).

**GQA-8-XXL vs. MHA-Large (the "free lunch" comparison):**
- Quality: 47.1 vs. 46.0, GQA-8-XXL is **1.1 points better**
- Speed: 0.28s vs. 0.37s, GQA-8-XXL is also **faster**

This comparison reveals GQA's real power: a large model with GQA can be both higher quality *and* faster than a smaller model with full MHA. You get the quality of a large model at an inference cost below a smaller model. This helps explain why GQA became a common choice in production deployments of large models.

### Why the speedup is $5.4\times$ and not $8\times$

T5-XXL has $h = 64$ query heads. GQA-8 uses $g = 8$ KV groups, so the KV cache is $64/8 = 8\times$ smaller. But the end-to-end speedup is only $5.4\times$.

We derived the formula in the previous blog:

$$
\text{speedup} = \frac{B_\text{weights} + B_\text{KV}^\text{MHA}}{B_\text{weights} + B_\text{KV}^\text{GQA}}
$$

The KV cache is not the only contributor to per-step time. Model weight loading, FFN computation, LayerNorm, and the output projection all contribute fixed costs that are the same for MHA and GQA. The $B_\text{weights}$ term in both numerator and denominator pulls the speedup below $h/g$.

At infinite context length, $B_\text{weights}$ becomes negligible and the speedup approaches $h/g = 8$. The T5-XXL experiments use moderate context lengths (512–2048 tokens for most tasks), where the weight load is still significant relative to the KV cache.

### The effect of number of groups (Figure 6)

The paper varies $g$ from 1 (MQA) to 64 (MHA) for GQA-XXL and measures inference time per sample:

| GQA groups $g$ | Time per sample (s) | Relative to MHA |
|---|---|---|
| 1 (MQA) | ~0.24 | $6.3\times$ faster |
| 4 | ~0.26 | $5.8\times$ faster |
| 8 | ~0.28 | $5.4\times$ faster |
| 16 | ~0.4 | $3.8\times$ faster |
| 32 | ~0.7 | $2.2\times$ faster |
| 64 (MHA) | ~1.51 | $1.0\times$ |

The inference time increases roughly linearly with $g$ for small $g$ and then steeply as $g$ approaches $h$. Going from 1 to 8 groups adds only 0.04s. Going from 8 to 64 groups adds 1.23s.

This non-linearity arises because the KV cache load is $\propto g$, but the fixed costs (weights, FFN) create a floor below which the total time cannot drop regardless of $g$. At small $g$, the fixed costs dominate and changes in $g$ have little effect. At large $g$, the KV cache dominates and changes in $g$ are proportionally expensive.

---

## Why GQA Preserves Quality

This is the part that confuses almost everyone. Reducing KV heads from 64 to 8 is an $8\times$ reduction in the key-value capacity. Why doesn't quality collapse?

### Head redundancy

Research on attention head pruning has consistently found that many heads are redundant:

- Voita et al. (2019) showed that in a 6-layer, 8-head Transformer, only 2–3 heads per layer are "important" for translation quality. The rest can be pruned with minimal quality loss.
- Michel et al. (2019) demonstrated that for BERT, removing 20–40% of heads has negligible effect on downstream task performance.

The implication: in MHA with 64 heads, many heads learn overlapping or nearly identical key-value representations. When GQA groups these heads and replaces their individual K/V with a shared K/V, the information loss is small because the individual heads were not contributing unique information.

### What the query heads learn to do

In GQA, the $h/g$ query heads within each group share one K and V. But each query head still has its own $W_Q^i$. The query projections are free to learn different "questions" to ask of the shared key-value representation.

Think of it as a library analogy. In MHA, each head has its own library (K and V) and its own search query (Q). In GQA, groups of heads share a library but each head still has its own search query. If the individual libraries were mostly redundant copies of the same books, consolidating them into one shared library per group loses little — the search queries can still find different information by asking different questions.

The key insight: **the diversity in attention comes more from the queries than from the keys and values.** Different heads attend to different positions primarily because their query projections differ, not because their key projections differ. GQA preserves all query diversity while reducing key-value redundancy.

### Mathematical argument

Consider two query heads $i_1$ and $i_2$ in the same group, sharing key $K^j$. Their attention distributions are:

$$
a^{i_1} = \text{softmax}\left(\frac{Q^{i_1} {K^j}^\top}{\sqrt{d_k}}\right), \quad a^{i_2} = \text{softmax}\left(\frac{Q^{i_2} {K^j}^\top}{\sqrt{d_k}}\right)
$$

Even though $K^j$ is the same, the attention distributions differ because $Q^{i_1} \neq Q^{i_2}$. The query matrices project the input into different subspaces, producing different attention scores against the same keys.

The two heads will attend to different positions as long as $W_Q^{i_1}$ and $W_Q^{i_2}$ are sufficiently different. Since these query projections are not constrained to be similar (they are separate learned parameters), the model retains the ability to attend to multiple different features simultaneously — as many as there are query heads.

---

## Deriving the Inference Time Improvement

We can predict the speedup from GQA analytically.

### The speedup formula

Per-step inference time is dominated by memory bandwidth. The total bytes loaded per step:

$$
B_\text{step} = B_\text{weights} + B_\text{KV}(t)
$$

For MHA:

$$
B_\text{step}^\text{MHA} = B_\text{weights} + L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot t
$$

For GQA with $g$ groups:

$$
B_\text{step}^\text{GQA} = B_\text{weights} + L \cdot g \cdot (d_k + d_v) \cdot 2 \cdot t
$$

The speedup ratio:

$$
\text{speedup}(t) = \frac{B_\text{step}^\text{MHA}}{B_\text{step}^\text{GQA}} = \frac{B_\text{weights} + L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot t}{B_\text{weights} + L \cdot g \cdot (d_k + d_v) \cdot 2 \cdot t}
$$

### Limiting behavior

**Short context ($t \to 0$).** Both numerator and denominator approach $B_\text{weights}$:

$$
\text{speedup} \to \frac{B_\text{weights}}{B_\text{weights}} = 1
$$

No speedup. The model spends all its time loading weights, which are the same for MHA and GQA.

**Long context ($t \to \infty$).** The $B_\text{weights}$ term becomes negligible:

$$
\text{speedup} \to \frac{L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot t}{L \cdot g \cdot (d_k + d_v) \cdot 2 \cdot t}
$$

The $L$, $(d_k + d_v)$, 2, and $t$ all cancel:

$$
\text{speedup} \to \frac{h}{g}
$$

The maximum achievable speedup is exactly $h/g$ — the same as the cache reduction factor.

### Numerical verification at every context length

Using our running model ($B_\text{weights} = 72$ MB, $g = 2$, $h = 8$):

**At $t = 512$:**

- $B_\text{KV}^\text{MHA} = 24{,}576 \times 512 = 12{,}582{,}912 \approx 12$ MB
- $B_\text{KV}^\text{GQA} = 6{,}144 \times 512 = 3{,}145{,}728 \approx 3$ MB

$$
\text{speedup} = \frac{72 + 12}{72 + 3} = \frac{84}{75} = 1.12\times
$$

At short context, almost no speedup.

**At $t = 2{,}048$:**

- $B_\text{KV}^\text{MHA} = 48$ MB
- $B_\text{KV}^\text{GQA} = 12$ MB

$$
\text{speedup} = \frac{72 + 48}{72 + 12} = \frac{120}{84} = 1.43\times
$$

**At $t = 4{,}096$:**

- $B_\text{KV}^\text{MHA} = 96$ MB
- $B_\text{KV}^\text{GQA} = 24$ MB

$$
\text{speedup} = \frac{72 + 96}{72 + 24} = \frac{168}{96} = 1.75\times
$$

**At $t = 16{,}384$:**

- $B_\text{KV}^\text{MHA} = 384$ MB
- $B_\text{KV}^\text{GQA} = 96$ MB

$$
\text{speedup} = \frac{72 + 384}{72 + 96} = \frac{456}{168} = 2.71\times
$$

**At $t = 128{,}000$:**

- $B_\text{KV}^\text{MHA} = 3{,}000$ MB
- $B_\text{KV}^\text{GQA} = 750$ MB

$$
\text{speedup} = \frac{72 + 3{,}000}{72 + 750} = \frac{3{,}072}{822} = 3.74\times
$$

Approaching the theoretical maximum of $h/g = 4\times$.

| Context $t$ | Speedup | % of theoretical max ($4\times$) |
|---|---|---|
| 512 | $1.12\times$ | 28% |
| 2,048 | $1.43\times$ | 36% |
| 4,096 | $1.75\times$ | 44% |
| 16,384 | $2.71\times$ | 68% |
| 128,000 | $3.74\times$ | 93% |

The speedup grows monotonically with context length, approaching $h/g$ asymptotically. At 128K tokens, we achieve 93% of the theoretical maximum. GQA's value increases with context length — precisely when inference is most expensive.

---

## GQA and Related KV-Head Choices in Modern Architectures

Since the GQA paper, modern language models have explored different points on the MHA-GQA-MQA spectrum. We trace a few concrete choices made by major architectures.

**LLaMA 2 70B** (Meta, 2023): $h = 64$ query heads, $g = 8$ KV groups. Cache reduction: $64/8 = 8\times$. At $d_\text{model} = 8{,}192$ and $L = 80$, the MHA cache would be $4 \times 80 \times 8{,}192 = 2.5$ MB/token. With GQA-8: $2.5 / 8 = 312.5$ KB/token. At 128K tokens: $312.5 \times 128{,}000 = 40$ GB instead of 320 GB.

**Mistral 7B** (Mistral AI, 2023): $h = 32$ query heads, $g = 8$ KV groups. Cache reduction: $32/8 = 4\times$. Combined with sliding window attention ($w = 4{,}096$), the cache is bounded at $g \cdot (d_k + d_v) \cdot 2 \cdot L \cdot w$ bytes regardless of sequence length. This is Lever 1 (GQA) and Lever 3 (sliding window) composed.

**Gemma** (Google, 2024): uses different KV-head choices by model size rather than one uniform GQA setting. Gemma 2B uses MQA ($\text{num\_kv\_heads} = 1$), while Gemma 7B uses full MHA ($\text{num\_kv\_heads} = 16$). This is a useful reminder that modern models do not all choose the same point on the MHA-GQA-MQA spectrum; the right choice depends on size, quality targets, and serving constraints.

Across these examples, architects are clearly optimizing the same tradeoff between KV-cache size and quality, even when they land on different points of the spectrum. LLaMA 2 70B and Mistral 7B both use GQA, while Gemma spans MQA and MHA across sizes. The broader pattern is that KV-head sharing became a central deployment design choice after GQA made the tradeoff explicit.

---

## The Unified View: MHA, GQA, MQA as One Formula

All three variants — MHA, GQA, and MQA — are a single attention mechanism parameterized by $g$:

$$
\boxed{\text{Attention}_{g}(X) = \text{concat}(o^1, \ldots, o^h) \, W_O}
$$

where for query head $i$ with group index $j = \lceil i \cdot g / h \rceil$:

$$
Q^i = X W_Q^i, \quad K^j = X W_K^j, \quad V^j = X W_V^j
$$

$$
o^i = \text{softmax}\!\left(\frac{Q^i {K^j}^\top}{\sqrt{d_k}}\right) V^j
$$

The only thing that varies is $g$. Everything else — the query projections, the output projection, the softmax, the scaling — is identical.

| $g$ | Name | KV heads | KV cache per token | Query heads per KV head |
|---|---|---|---|---|
| $h$ | MHA | $h$ unique | $L \cdot h \cdot (d_k + d_v) \cdot 2$ | 1 |
| $1 < g < h$ | GQA-$g$ | $g$ shared | $L \cdot g \cdot (d_k + d_v) \cdot 2$ | $h/g$ |
| $1$ | MQA | 1 shared | $L \cdot (d_k + d_v) \cdot 2$ | $h$ |

**There is no structural difference between these three methods.** They are the same formula with different values of one integer parameter. GQA is the general family. MHA ($g = h$) and MQA ($g = 1$) are boundary cases.

This unified view makes clear that choosing $g$ is a single design decision with a clean tradeoff: smaller $g$ → smaller cache → faster inference → potentially lower quality. The engineering challenge is finding the $g$ that maximizes inference speed while keeping quality within tolerance. The GQA paper showed this $g$ exists and is easy to find.

---

## Summary

The KV cache during autoregressive inference costs $L \cdot h \cdot (d_k + d_v) \cdot 2$ bytes per token. Multi-Query Attention (MQA) collapses all $h$ KV heads to 1, reducing the cache by $h\times$ at the cost of a measurable quality drop (0.6 points on T5-XXL). Grouped-Query Attention (GQA) generalizes this to $g$ KV groups, each shared by $h/g$ query heads, reducing the cache by exactly $h/g$. The GQA paper showed that with $g = 8$ on T5-XXL, quality is within 0.1 points of full MHA while inference is $5.4\times$ faster. Existing MHA checkpoints can be converted to GQA by mean-pooling KV heads within each group and uptraining for just 5% of the original pre-training compute. The inference speedup approaches $h/g$ at long context lengths and is most impactful exactly when inference is most expensive. MHA, GQA, and MQA are not three separate inventions — they are a single formula parameterized by the number of KV groups $g$.

---

*Previous: [The KV Bottleneck Explained Deeply](/blog/attention-kv-bottleneck)*
*Next: Multi-Query Attention — The Extreme Compression (MQA)*
