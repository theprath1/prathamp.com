---
title: "DeepSeek-V4 Hybrid Attention: CSA and HCA from Scratch"
description: "Building the long-context attention of DeepSeek-V4 from the ground up — dual KV streams, overlapping softmax compression, the lightning indexer, top-$k$ sparse selection, and heavier compression — all derived step by step on a six-token example."
date: 2026-04-24
tags: [machine-learning, attention, transformers, kv-cache, long-context, deepseek, sparse-attention]
order: 1
---

DeepSeek-V4 ([DeepSeek-AI, 2026](https://huggingface.co/collections/deepseek-ai/deepseek-v4)) introduces two Mixture-of-Experts models — V4-Pro (1.6T parameters, 49B activated) and V4-Flash (284B parameters, 13B activated) — both supporting a context length of **one million tokens**. At 1M-context inference, V4-Pro spends only $27\%$ of the single-token FLOPs and $10\%$ of the KV cache of DeepSeek-V3.2. V4-Flash pushes that to $10\%$ of FLOPs and $7\%$ of KV cache.

The architectural change that drives those numbers is not the Mixture-of-Experts backbone, not the Muon optimizer, not the Manifold-Constrained Hyper-Connections. It is the **hybrid attention** that interleaves two new attention mechanisms across layers:

- **Compressed Sparse Attention (CSA)** — compress the KV cache by a factor of $m$, then attend sparsely over the compressed entries using a learned index.
- **Heavily Compressed Attention (HCA)** — compress the KV cache by a much larger factor $m' \gg m$, then attend densely over the (tiny) compressed cache.

We will derive both mechanisms from scratch on a six-token example, trace every softmax and weighted sum by hand, and verify the KV-cache ledger that yields the $10\%$ headline number.

The DeepSeek Sparse Attention (DSA) mechanism from V3.2, which CSA builds on top of, was derived in [DeepSeek Sparse Attention from scratch](/blog/attention-deepseek-sparse). We will restate what we need so the post is self-contained.

---

## The Running Example

We take a causal sequence of $n = 6$ tokens with hidden size $d = 2$. The hidden states are:

$$
H = \begin{bmatrix} \mathbf{h}_0 \\ \mathbf{h}_1 \\ \mathbf{h}_2 \\ \mathbf{h}_3 \\ \mathbf{h}_4 \\ \mathbf{h}_5 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 2 & 1 \\ 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix} \in \mathbb{R}^{6 \times 2}.
$$

The attention-specific parameters we fix for this post:

- **Compressed head dimension** $c = 2$ — the width of one compressed KV entry.
- **CSA compression ratio** $m = 2$ — how many tokens go into one CSA compressed entry.
- **HCA compression ratio** $m' = 3$ — how many tokens go into one HCA compressed entry.
- **Sparse selection budget** $k = 1$ — how many CSA compressed entries each query keeps.
- **Number of query heads** $n_h = 1$ — kept to one for readability. Production V4-Pro uses $n_h = 128$.
- **Indexer head dimension** $c^I = 2$ and **number of indexer heads** $n_h^I = 1$.
- **Query compression dimension** $d_c = 2$.

Production V4-Pro values are $c = 512$, $m = 4$, $m' = 128$, $k = 1024$, $n_h = 128$. We use smaller numbers so the arithmetic fits on one line. Every ratio we compute will scale up unchanged.

We focus on the **query token $t = 5$** — the last token in the sequence — because a causal model attends only to preceding tokens, and $t = 5$ has the most preceding context (tokens $0, 1, 2, 3, 4$) to attend over.

---

## 1. The Problem: Quadratic KV Cache

In standard Multi-Head Attention ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) with GQA ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) as the BF16 GQA8 baseline the DeepSeek-V4 paper uses, the KV cache stores one key and one value vector per token per KV head per layer. With head dimension $128$, $8$ KV groups, $L$ layers, and BF16 (2 bytes per element):

$$
\text{KV cache bytes per token per layer} = 2 \cdot (n_h^{KV} \cdot d_h) \cdot 2 = 2 \cdot (8 \cdot 128) \cdot 2 = 4096 \text{ bytes}.
$$

For a sequence of $n = 10^6$ tokens and $L = 61$ layers (V4-Pro), that is:

$$
n \cdot L \cdot 4096 = 10^6 \cdot 61 \cdot 4096 \approx 2.5 \cdot 10^{11} \text{ bytes} = 250 \text{ GB per request}.
$$

This is a **quadratic-in-sequence** read pattern during decode — each new token must read all preceding KV entries — and a **linear-in-sequence** storage pattern. Both blow up at $10^6$ tokens. CSA and HCA attack both numbers at their root: they reduce the number of stored KV entries by a factor $m$ or $m'$, and they reduce how many entries each query reads by the sparse selection $k$.

---

## 2. Strategy: Compress Groups of Tokens into One Entry

The common idea behind CSA and HCA is the **compression operation**: take a window of $m$ (or $m'$) contiguous hidden states, project each into a $c$-dimensional KV candidate, generate per-position weights, and combine them into a single compressed entry.

Symbolically, if $C_{1:m} \in \mathbb{R}^{m \times c}$ is a block of $m$ KV candidates and $S_{1:m} \in \mathbb{R}^{m \times c}$ is a matching block of weights (one weight vector per candidate), then the compressed entry is:

$$
C^{\text{Comp}} = \sum_{j=1}^{m} S_j \odot C_j \in \mathbb{R}^{c},
$$

where $\odot$ is the **Hadamard product** (elementwise multiplication). This is a learned, data-dependent pooling over the window — every output coordinate is a convex combination of the same coordinate across the $m$ positions, with a separate softmax per coordinate.

HCA stops here. CSA adds a twist — each compressed entry draws from $2m$ positions, with overlap between neighbours, so that block boundaries blur — and then performs **sparse selection** over the compressed entries.

We derive CSA first, then specialize to HCA.

---

## 3. CSA Step 1 — Dual KV Streams and Compression Weights

CSA maintains **two** series of raw KV entries and **two** series of compression weights:

$$
C^a = H \cdot W^{aKV}, \qquad C^b = H \cdot W^{bKV},
$$

$$
Z^a = H \cdot W^{aZ}, \qquad Z^b = H \cdot W^{bZ},
$$

where $W^{aKV}, W^{bKV}, W^{aZ}, W^{bZ} \in \mathbb{R}^{d \times c}$ are learnable. All four have shape $\mathbb{R}^{n \times c}$ after the multiplication.

To trace the computation we pick concrete weight matrices:

$$
W^{aKV} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W^{bKV} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix},
$$

$$
W^{aZ} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad W^{bZ} = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}.
$$

$W^{aKV}$ is the identity (stream $a$ preserves $H$); $W^{bKV}$ swaps coordinates (stream $b$ is a reshuffled view). $W^{aZ}$ keeps only column 0 of $H$; $W^{bZ}$ keeps only column 1 of $H$ and places it in column 0 of $Z^b$. These choices are intentional — they let us check each derivation by eye.

Computing $C^a = H \cdot W^{aKV}$ (using the **matrix multiplication rule** $(HW)_{ij} = \sum_k H_{ik} W_{kj}$):

$$
C^a = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 2 & 1 \\ 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}.
$$

Computing $C^b = H \cdot W^{bKV}$ (swap columns):

$$
C^b = \begin{bmatrix} 0 & 1 \\ 2 & 0 \\ 1 & 2 \\ 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}.
$$

Computing $Z^a = H \cdot W^{aZ}$ (column 0 of $H$ into column 0 of $Z^a$; column 1 zeroed):

$$
Z^a = \begin{bmatrix} 1 & 0 \\ 0 & 0 \\ 2 & 0 \\ 1 & 0 \\ 0 & 0 \\ 1 & 0 \end{bmatrix}.
$$

Computing $Z^b = H \cdot W^{bZ}$ (column 1 of $H$ into column 0 of $Z^b$; column 1 zeroed):

$$
Z^b = \begin{bmatrix} 0 & 0 \\ 2 & 0 \\ 1 & 0 \\ 1 & 0 \\ 1 & 0 \\ 0 & 0 \end{bmatrix}.
$$

**Why two streams and not one?** The very next step will softmax *over twice* as many positions as one block's width. If there were only one stream, the softmax for a block would only see $m$ elements. With two, it sees $2m$ — giving each compressed entry a view that extends across the block boundary. The paper calls this **overlapped compression**. We will see the overlap in Step 2.

---

## 4. CSA Step 2 — The Overlapping $2m$-Softmax

For each compressed index $i \in \{0, 1, \ldots, n/m - 1\}$, CSA constructs the $i$-th compressed entry from a window of **$2m$** raw positions:

- positions $mi, mi+1, \ldots, m(i+1)-1$ from stream $a$ (the "current block"),
- positions $m(i-1), m(i-1)+1, \ldots, mi-1$ from stream $b$ (the "previous block").

For $i = 0$ the $b$-positions would be $-m, \ldots, -1$, which do not exist. The paper pads: $Z^b_{-m:-1}$ is set to $-\infty$ and $C^b_{-m:-1}$ to zero.

Adding learnable positional biases $B^a, B^b \in \mathbb{R}^{m \times c}$ (we choose $B^a = B^b = 0$ for readability), the softmax is:

$$
\begin{bmatrix} S^a_{mi:m(i+1)-1} \\ S^b_{m(i-1):mi-1} \end{bmatrix} = \text{Softmax}_{\text{row}}\!\left( \begin{bmatrix} Z^a_{mi:m(i+1)-1} + B^a \\ Z^b_{m(i-1):mi-1} + B^b \end{bmatrix} \right),
$$

where $\text{Softmax}_{\text{row}}$ denotes a softmax applied **down the row dimension** — for each of the $c$ output columns independently, normalize across the $2m$ stacked rows. Each column therefore gets its own independent **softmax distribution** over $2m$ positions.

The softmax function itself is:

$$
\text{Softmax}(x)_j = \frac{\exp(x_j)}{\sum_{k=1}^{2m} \exp(x_k)}.
$$

We use $m = 2$, so $2m = 4$. Let us compute both CSA compressed entries that query $t = 5$ is allowed to see.

### Block $i = 0$

- $a$-positions: $0, 1$; $b$-positions: $-2, -1$ (padded with $-\infty$ in $Z^b$, zero in $C^b$).

Column $0$ of the stacked input:

$$
\left[Z^a_{0:1,0} \ \Big\Vert \ Z^b_{-2:-1,0}\right] = [\,1,\ 0,\ -\infty,\ -\infty\,].
$$

(Here $\Vert$ stacks vertically; we display horizontally to save space.) The **exponential of $-\infty$ is $0$**, so the denominator is $e^1 + e^0 = 2.718 + 1 = 3.718$. The softmax is:

$$
[S^a_{0,0},\ S^a_{1,0},\ S^b_{-2,0},\ S^b_{-1,0}] = \left[\tfrac{e}{e+1},\ \tfrac{1}{e+1},\ 0,\ 0\right] \approx [0.731,\ 0.269,\ 0,\ 0].
$$

Numerical check — the weights must sum to $1$ (a softmax is a probability distribution):

$$
0.731 + 0.269 + 0 + 0 = 1.000. \ \checkmark
$$

Column $1$: all entries of $Z^a[:,1]$ and $Z^b[:,1]$ are zero, so the input is $[0, 0, -\infty, -\infty]$. Softmax gives:

$$
[S^a_{0,1},\ S^a_{1,1},\ S^b_{-2,1},\ S^b_{-1,1}] = [0.5,\ 0.5,\ 0,\ 0].
$$

### Block $i = 1$

- $a$-positions: $2, 3$; $b$-positions: $0, 1$ (now real, no padding).

Column $0$ of the stacked input:

$$
[Z^a_{2,0},\ Z^a_{3,0},\ Z^b_{0,0},\ Z^b_{1,0}] = [2,\ 1,\ 0,\ 2].
$$

The denominator is $e^2 + e^1 + e^0 + e^2 = 7.389 + 2.718 + 1 + 7.389 = 18.496$. The softmax is:

$$
\left[\tfrac{7.389}{18.496},\ \tfrac{2.718}{18.496},\ \tfrac{1}{18.496},\ \tfrac{7.389}{18.496}\right] \approx [0.400,\ 0.147,\ 0.054,\ 0.400].
$$

Numerical check: $0.400 + 0.147 + 0.054 + 0.400 = 1.001 \approx 1$ (rounding). $\checkmark$

Column $1$: all zeros, so softmax $= [0.25, 0.25, 0.25, 0.25]$.

Notice the overlap already: block $i = 1$'s softmax **includes positions $0$ and $1$** through the $b$-stream, while block $i = 0$ also used positions $0$ and $1$ through the $a$-stream. This is the "overlapped" in overlapped compression — information from positions $0, 1$ is available to both compressed entries.

---

## 5. CSA Step 3 — The Weighted Sum

Given the softmax weights, the compressed entry is:

$$
\boxed{\ C_i^{\text{Comp}} \ = \ \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j \ + \ \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j.\ }
$$

Each term is a $c$-dimensional vector; $\odot$ denotes the **Hadamard product** (elementwise multiplication). The sum has exactly $2m$ terms — one per row of the softmax.

### Computing $C_0^{\text{Comp}}$

Column $0$:

$$
C_0^{\text{Comp}}[0] = 0.731 \cdot C^a_0[0] + 0.269 \cdot C^a_1[0] + 0 \cdot C^b_{-2}[0] + 0 \cdot C^b_{-1}[0].
$$

Substituting $C^a_0[0] = 1$, $C^a_1[0] = 0$, and the padded-zero $C^b$ entries:

$$
C_0^{\text{Comp}}[0] = 0.731 \cdot 1 + 0.269 \cdot 0 + 0 + 0 = 0.731.
$$

Column $1$:

$$
C_0^{\text{Comp}}[1] = 0.5 \cdot C^a_0[1] + 0.5 \cdot C^a_1[1] + 0 + 0 = 0.5 \cdot 0 + 0.5 \cdot 2 = 1.0.
$$

So

$$
C_0^{\text{Comp}} \approx [0.731,\ 1.000] \in \mathbb{R}^2.
$$

**Sanity check — range**: every entry is a convex combination (weights are non-negative and sum to one) of values in $\{0, 1, 2\}$, so $C_0^{\text{Comp}}$ must lie in $[0, 2]^2$. Both components $0.731$ and $1.000$ fall in $[0, 2]$. $\checkmark$

### Computing $C_1^{\text{Comp}}$

Column $0$:

$$
C_1^{\text{Comp}}[0] = 0.400 \cdot C^a_2[0] + 0.147 \cdot C^a_3[0] + 0.054 \cdot C^b_0[0] + 0.400 \cdot C^b_1[0].
$$

Substituting $C^a_2[0] = 2$, $C^a_3[0] = 1$, $C^b_0[0] = 0$, $C^b_1[0] = 2$:

$$
C_1^{\text{Comp}}[0] = 0.400 \cdot 2 + 0.147 \cdot 1 + 0.054 \cdot 0 + 0.400 \cdot 2 = 0.800 + 0.147 + 0 + 0.800 = 1.747.
$$

Column $1$:

$$
C_1^{\text{Comp}}[1] = 0.25 \cdot C^a_2[1] + 0.25 \cdot C^a_3[1] + 0.25 \cdot C^b_0[1] + 0.25 \cdot C^b_1[1].
$$

Substituting $1, 1, 1, 0$:

$$
C_1^{\text{Comp}}[1] = 0.25 \cdot 1 + 0.25 \cdot 1 + 0.25 \cdot 1 + 0.25 \cdot 0 = 0.75.
$$

So

$$
C_1^{\text{Comp}} \approx [1.747,\ 0.750] \in \mathbb{R}^2.
$$

Note the overlap concretely: $C^b_0 = [0, 1]$ (drawn from token $h_0$) and $C^b_1 = [2, 0]$ (drawn from token $h_1$) both appear in $C_1^{\text{Comp}}$ through the $b$-stream, even though tokens $0$ and $1$ are "in" block $0$. The block boundary is intentionally blurred.

**Interpretation.** The six-token sequence has been compressed from a $6 \times 2$ tensor of raw KV entries into a $3 \times 2$ tensor of compressed entries (we showed the first two; $C_2^{\text{Comp}}$ is analogous). The compression ratio is $m = 2$, exactly as advertised.

---

## 6. CSA Step 4 — The Lightning Indexer

Compression alone cuts the KV cache by $m$. Sparse selection cuts the per-query read by another factor. CSA's **lightning indexer** is the mechanism that scores which compressed entries the query should actually attend to.

For query token $t$, the indexer performs four operations.

### 6.1 Produce a compressed latent query

$$
\mathbf{c}_t^Q = \mathbf{h}_t \cdot W^{DQ}, \qquad W^{DQ} \in \mathbb{R}^{d \times d_c}.
$$

With our $d = 2$, $d_c = 2$, pick $W^{DQ} = I$ (identity). Then $\mathbf{c}_5^Q = \mathbf{h}_5 = [1, 0]$.

### 6.2 Up-project to indexer query heads

$$
[\mathbf{q}^I_{t,1};\ \ldots;\ \mathbf{q}^I_{t,n_h^I}] = \mathbf{c}_t^Q \cdot W^{IUQ}, \qquad W^{IUQ} \in \mathbb{R}^{d_c \times c^I n_h^I}.
$$

With $n_h^I = 1$ and $c^I = 2$, pick $W^{IUQ} = I$. Then $\mathbf{q}^I_{5,1} = [1, 0]$.

### 6.3 Produce per-head indexer weights

$$
[w^I_{t,1};\ \ldots;\ w^I_{t,n_h^I}] = \mathbf{h}_t \cdot W^w, \qquad W^w \in \mathbb{R}^{d \times n_h^I}.
$$

With $n_h^I = 1$ and $W^w = [1, 1]^T$, we get $w^I_{5,1} = \mathbf{h}_5 \cdot [1, 1]^T = 1 \cdot 1 + 0 \cdot 1 = 1$.

### 6.4 Score each compressed block

Given compressed indexer keys $K^{\text{IComp}}_s \in \mathbb{R}^{c^I}$ (produced by the same compression operation as $C^{\text{Comp}}$ but with a separate set of weight matrices — we assume they are given for this section and return to their construction below), the index score is:

$$
\boxed{\ I_{t,s} = \sum_{h=1}^{n_h^I} w^I_{t,h} \cdot \text{ReLU}\!\left(\mathbf{q}^I_{t,h} \cdot K^{\text{IComp}}_s\right).\ }
$$

**ReLU** (Rectified Linear Unit) is the activation $\text{ReLU}(x) = \max(x, 0)$. It clamps the per-head contribution to be non-negative — a block that is "negatively" scored by one head does not drag down the total.

### Why ReLU and not softmax?

This is the part that confuses almost everyone. In standard attention the scores are softmax-normalized, because we want a probability distribution over keys. Here we want a **ranking** — the top-$k$ largest scores — and a ranking is invariant to monotone transforms. ReLU is cheaper than softmax, does not require cross-block normalization (each $I_{t,s}$ is computed independently), and keeps the indexer's output bounded below.

Concretely, the ReLU lets us implement the whole indexer in FP4 without exploding logits — the paper notes that "attention computation within the lightning indexer is performed in FP4 precision," which is only viable because we never exponentiate. This is the **straight-through estimator** strategy ([Jacob et al., 2018](https://arxiv.org/abs/1712.05877)) applied to attention scoring.

### Numerical check

Suppose the compressed indexer keys come out to:

$$
K^{\text{IComp}}_0 = [0.5,\ 0.5], \qquad K^{\text{IComp}}_1 = [1.0,\ 1.0].
$$

Then:

$$
\mathbf{q}^I_{5,1} \cdot K^{\text{IComp}}_0 = 1 \cdot 0.5 + 0 \cdot 0.5 = 0.5,
$$

$$
\mathbf{q}^I_{5,1} \cdot K^{\text{IComp}}_1 = 1 \cdot 1.0 + 0 \cdot 1.0 = 1.0.
$$

Both are positive, so ReLU leaves them unchanged:

$$
I_{5,0} = 1 \cdot \text{ReLU}(0.5) = 0.5, \qquad I_{5,1} = 1 \cdot \text{ReLU}(1.0) = 1.0.
$$

$I_{5,1} > I_{5,0}$ — the indexer says block $1$ is more relevant to the query than block $0$.

**Cost.** Computing $I_{t,s}$ for one query and one compressed block is one dot product in $\mathbb{R}^{c^I}$ plus one ReLU plus one multiply-add per indexer head — $O(c^I n_h^I)$ FLOPs. For $n/m$ blocks and $n$ queries, total indexer FLOPs are $O(n \cdot \tfrac{n}{m} \cdot c^I n_h^I)$. This is still quadratic in $n$, but with a much smaller constant (FP4, tiny $c^I$) than full attention — the V4 paper measures this as negligible relative to the core attention.

---

## 7. CSA Step 5 — Top-$k$ Sparse Selection

Given the index scores $I_{t,:}$ across all allowed compressed blocks, we keep only the top $k$:

$$
C_t^{\text{SprsComp}} = \left\{\ C_s^{\text{Comp}}\ \Big|\ I_{t,s} \in \text{Top-}k(I_{t,:})\ \right\}.
$$

"Allowed" here means the **causal condition** $s < \lfloor t/m \rfloor$: the query at position $t$ can only see compressed blocks whose rightmost token precedes $t$. For our query $t = 5$ and $m = 2$, $\lfloor 5/2 \rfloor = 2$, so $s \in \{0, 1\}$.

The operator $\lfloor \cdot \rfloor$ is the **floor function** — round down to the nearest integer.

With $k = 1$, we pick the block with the highest score:

$$
C_5^{\text{SprsComp}} = \{\, C_1^{\text{Comp}} \,\} = \{\,[1.747,\ 0.750]\,\}.
$$

The query at $t = 5$ will now perform its core attention against a **set of size $k = 1$** instead of the $n = 6$ raw KV entries. This is where CSA's $O(nk)$ cost comes from — see the [DSA from scratch post](/blog/attention-deepseek-sparse) for the full FLOPs derivation.

---

## 8. CSA Step 6 — Shared-KV Multi-Query Attention

The final stage is **Multi-Query Attention (MQA)** ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)): all query heads share a single key and value stream, which here is $C_t^{\text{SprsComp}}$.

Produce the core attention queries from the same latent $\mathbf{c}_t^Q$ we already computed for the indexer:

$$
[\mathbf{q}_{t,1};\ \ldots;\ \mathbf{q}_{t,n_h}] = \mathbf{c}_t^Q \cdot W^{UQ}, \qquad W^{UQ} \in \mathbb{R}^{d_c \times c n_h}.
$$

Sharing $\mathbf{c}_t^Q$ between the indexer and the core attention is an explicit optimization — the paper calls it out in Section 2.3.1 — because it halves the query-side projection cost.

For our $n_h = 1$, $d_c = 2$, $c = 2$, pick $W^{UQ} = I$. Then $\mathbf{q}_{5,1} = [1, 0]$.

Now perform core attention:

$$
\mathbf{o}_{t,i} = \text{CoreAttn}\!\left(\text{query}=\mathbf{q}_{t,i},\ \text{key}=C_t^{\text{SprsComp}},\ \text{value}=C_t^{\text{SprsComp}}\right).
$$

With $k = 1$, the softmax over one key is trivial (a softmax of a single-element set is $[1]$), so:

$$
\mathbf{o}_{5,1} = 1.000 \cdot C_1^{\text{Comp}} = [1.747,\ 0.750].
$$

Numerical check: with a one-element attention, the output must equal the single value. It does. $\checkmark$

With larger $k$ the softmax is:

$$
\mathbf{o}_{t,i} = \sum_{s \in \text{Top-}k} \text{Softmax}_s\!\left(\frac{\mathbf{q}_{t,i} \cdot C_s^{\text{Comp}}}{\sqrt{c}}\right) \cdot C_s^{\text{Comp}}.
$$

The $\sqrt{c}$ is the **scaled dot-product attention** scaling ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) that keeps logit variance constant as $c$ grows.

**Key/Value sharing.** Notice the key and value in the attention call are the same tensor $C^{\text{SprsComp}}$. This is MQA's defining property: the compressed entry serves as *both* the key (for scoring) and the value (for the weighted sum). The KV cache stores one $c$-dim vector per compressed block — not two, not $n_h$ copies — so the ledger is as small as it could be.

---

## 9. Bringing CSA Together — The Compression Architecture

Here is CSA at one glance:

<svg viewBox="0 0 760 320" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border:1px solid #ddd;border-radius:6px;max-width:100%;height:auto">
  <style>
    .box { fill: #ffffff; stroke: #333; stroke-width: 1.5; }
    .compr { fill: #e8f0ff; stroke: #3060a0; stroke-width: 1.5; }
    .index { fill: #fff0e0; stroke: #c06010; stroke-width: 1.5; }
    .attn { fill: #e0ffe8; stroke: #108040; stroke-width: 1.5; }
    .lbl { font-family: sans-serif; font-size: 12px; fill: #222; }
    .small { font-family: sans-serif; font-size: 10px; fill: #444; }
    .arr { fill: none; stroke: #333; stroke-width: 1.3; marker-end: url(#ah); }
  </style>
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <!-- Hidden states -->
  <rect x="20" y="20" width="120" height="40" class="box"/>
  <text x="80" y="40" class="lbl" text-anchor="middle">Hidden states H</text>
  <text x="80" y="55" class="small" text-anchor="middle">n × d</text>
  <!-- KV streams -->
  <rect x="180" y="0" width="120" height="30" class="compr"/>
  <text x="240" y="20" class="lbl" text-anchor="middle">C^a, Z^a</text>
  <rect x="180" y="50" width="120" height="30" class="compr"/>
  <text x="240" y="70" class="lbl" text-anchor="middle">C^b, Z^b</text>
  <!-- Overlapping softmax -->
  <rect x="340" y="20" width="130" height="40" class="compr"/>
  <text x="405" y="40" class="lbl" text-anchor="middle">Softmax over 2m</text>
  <text x="405" y="54" class="small" text-anchor="middle">+ weighted sum</text>
  <!-- Compressed entries -->
  <rect x="510" y="20" width="120" height="40" class="compr"/>
  <text x="570" y="40" class="lbl" text-anchor="middle">C^Comp</text>
  <text x="570" y="55" class="small" text-anchor="middle">(n/m) × c</text>
  <!-- Query -->
  <rect x="20" y="140" width="120" height="40" class="box"/>
  <text x="80" y="160" class="lbl" text-anchor="middle">Query h_t</text>
  <text x="80" y="175" class="small" text-anchor="middle">1 × d</text>
  <!-- c^Q -->
  <rect x="180" y="140" width="80" height="40" class="box"/>
  <text x="220" y="160" class="lbl" text-anchor="middle">c_t^Q</text>
  <text x="220" y="175" class="small" text-anchor="middle">1 × d_c</text>
  <!-- indexer queries -->
  <rect x="300" y="100" width="120" height="40" class="index"/>
  <text x="360" y="120" class="lbl" text-anchor="middle">Indexer queries</text>
  <text x="360" y="135" class="small" text-anchor="middle">q^I (low-rank)</text>
  <!-- core queries -->
  <rect x="300" y="170" width="120" height="40" class="attn"/>
  <text x="360" y="190" class="lbl" text-anchor="middle">Core queries</text>
  <text x="360" y="205" class="small" text-anchor="middle">q (shared c^Q)</text>
  <!-- index scores -->
  <rect x="460" y="100" width="120" height="40" class="index"/>
  <text x="520" y="120" class="lbl" text-anchor="middle">Index scores I</text>
  <text x="520" y="135" class="small" text-anchor="middle">ReLU · w^I</text>
  <!-- top-k -->
  <rect x="610" y="100" width="130" height="40" class="index"/>
  <text x="675" y="120" class="lbl" text-anchor="middle">Top-k selector</text>
  <text x="675" y="135" class="small" text-anchor="middle">k ≪ n/m entries</text>
  <!-- core attention -->
  <rect x="460" y="240" width="280" height="50" class="attn"/>
  <text x="600" y="260" class="lbl" text-anchor="middle">Shared-KV MQA core attention</text>
  <text x="600" y="276" class="small" text-anchor="middle">output o_t ∈ R^c, then grouped output projection</text>
  <!-- arrows -->
  <path class="arr" d="M140,40 L180,20"/>
  <path class="arr" d="M140,50 L180,60"/>
  <path class="arr" d="M300,40 L340,40"/>
  <path class="arr" d="M470,40 L510,40"/>
  <path class="arr" d="M140,160 L180,160"/>
  <path class="arr" d="M260,160 L300,140"/>
  <path class="arr" d="M260,160 L300,180"/>
  <path class="arr" d="M420,120 L460,120"/>
  <path class="arr" d="M580,120 L610,120"/>
  <path class="arr" d="M630,60 Q690,80 675,100"/>
  <path class="arr" d="M675,140 Q620,200 600,240"/>
  <path class="arr" d="M420,190 L460,260"/>
</svg>

---

## 10. HCA — The Heavily Compressed Extreme

HCA is CSA with three things removed:

1. **One KV stream instead of two.** No overlap.
2. **Larger compression ratio $m' \gg m$.** The paper uses $m' = 128$ against $m = 4$.
3. **No lightning indexer, no top-$k$.** Every query attends to *all* compressed blocks densely.

Formally, given input hidden states $H \in \mathbb{R}^{n \times d}$:

$$
C = H \cdot W^{KV}, \qquad Z = H \cdot W^Z, \qquad W^{KV}, W^Z \in \mathbb{R}^{d \times c}.
$$

Group into non-overlapping blocks of width $m'$:

$$
S_{m'i:m'(i+1)-1} = \text{Softmax}_{\text{row}}(Z_{m'i:m'(i+1)-1} + B),
$$

$$
\boxed{\ C_i^{\text{Comp}} = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j.\ }
$$

This is **exactly** the compression sub-operation of CSA, with the $b$-stream deleted and $m$ replaced by $m'$.

### HCA on our running example

Use $m' = 3$, so $n / m' = 6/3 = 2$ HCA compressed entries. Reuse $W^{KV} = W^{aKV} = I$ and $W^Z = W^{aZ}$, so:

$$
C = H, \qquad Z = \begin{bmatrix} 1 & 0 \\ 0 & 0 \\ 2 & 0 \\ 1 & 0 \\ 0 & 0 \\ 1 & 0 \end{bmatrix}.
$$

### Block $i = 0$ (positions $0, 1, 2$)

Column $0$: $Z_{0:2,0} = [1, 0, 2]$. Softmax denominator: $e^1 + e^0 + e^2 = 2.718 + 1 + 7.389 = 11.107$.

$$
[S_{0,0}, S_{1,0}, S_{2,0}] = \left[\tfrac{2.718}{11.107},\ \tfrac{1}{11.107},\ \tfrac{7.389}{11.107}\right] \approx [0.245,\ 0.090,\ 0.665].
$$

Sum check: $0.245 + 0.090 + 0.665 = 1.000$. $\checkmark$

Column $1$: $Z_{0:2,1} = [0, 0, 0]$, so softmax $= [1/3, 1/3, 1/3]$.

Weighted sum, column $0$:

$$
C_0^{\text{HCA}}[0] = 0.245 \cdot 1 + 0.090 \cdot 0 + 0.665 \cdot 2 = 0.245 + 0 + 1.330 = 1.575.
$$

Weighted sum, column $1$:

$$
C_0^{\text{HCA}}[1] = \tfrac{1}{3} \cdot 0 + \tfrac{1}{3} \cdot 2 + \tfrac{1}{3} \cdot 1 = 1.000.
$$

So $C_0^{\text{HCA}} \approx [1.575,\ 1.000]$.

### Block $i = 1$ (positions $3, 4, 5$)

Column $0$: $Z_{3:5,0} = [1, 0, 1]$. Denominator: $e^1 + e^0 + e^1 = 2.718 + 1 + 2.718 = 6.436$.

$$
[S_{3,0}, S_{4,0}, S_{5,0}] \approx [0.422,\ 0.155,\ 0.422].
$$

Column $1$: all zeros $\to [1/3, 1/3, 1/3]$.

Weighted sum, column $0$:

$$
C_1^{\text{HCA}}[0] = 0.422 \cdot 1 + 0.155 \cdot 0 + 0.422 \cdot 1 = 0.844.
$$

Weighted sum, column $1$:

$$
C_1^{\text{HCA}}[1] = \tfrac{1}{3} \cdot 1 + \tfrac{1}{3} \cdot 1 + \tfrac{1}{3} \cdot 0 = 0.667.
$$

So $C_1^{\text{HCA}} \approx [0.844,\ 0.667]$.

For query $t = 5$ under HCA, the causal condition is $s < \lfloor t/m' \rfloor = \lfloor 5/3 \rfloor = 1$, so only block $0$ is visible. The core attention is a dense softmax over *one* block, which trivially outputs $C_0^{\text{HCA}}$.

In a realistic 1M-context setting with $m' = 128$ and $n = 10^6$, HCA produces $10^6 / 128 \approx 7813$ compressed entries per layer — already $128\times$ smaller than raw KV — and each query attends densely over all of them. There is no sparse selection, so there is no indexer cost, and no top-$k$ kernel.

---

## 11. Unified View — CSA and HCA on One Spectrum

CSA and HCA look like two different mechanisms, but they share a single equation. Define the **general compressed attention operator** parameterized by $(m, k, \text{overlap})$:

$$
\boxed{\ \text{CompAttn}_{m,k,\text{overlap}}(H, t) \ = \ \text{MQA}\!\left(\mathbf{q}_t,\ \text{Top-}k\!\left(\text{Indexer}(t, C^{\text{Comp}}_{m,\text{overlap}})\right)\right).\ }
$$

The three knobs:

| Knob | CSA | HCA |
|---|---|---|
| Compression ratio | $m$ (moderate) | $m' \gg m$ |
| Overlap | yes (2m softmax) | no (m softmax) |
| Top-$k$ selection | $k < n/m$ | $k = n/m'$ (no-op) |

HCA is the special case of the operator with the overlap turned off and the top-$k$ budget set to "everything". CSA is the case with overlap on and a sparse budget. The **indexer** exists in both equations — it just becomes trivial (rank by score, keep all) in HCA, so the implementation omits it.

One framework, two specializations. The mathematical elegance lies in how a single compression-then-attend template, parameterized by two knobs, recovers both mechanisms as points on a continuous spectrum.

---

## 12. The Efficiency Ledger

Now the numbers that motivated everything.

### 12.1 KV cache per layer

Raw GQA8 baseline with head dimension $d_h = 128$, BF16: $n \cdot 8 \cdot 128 \cdot 2 = 2048n$ bytes.

CSA stores one compressed entry per $m$ tokens, each of size $c$, in BF16 (2 bytes): $\tfrac{n}{m} \cdot c \cdot 2$ bytes.

HCA: $\tfrac{n}{m'} \cdot c \cdot 2$ bytes.

With V4-Pro values $c = 512$, $m = 4$, $m' = 128$:

$$
\text{CSA cache per layer} = \tfrac{n}{4} \cdot 512 \cdot 2 = 256 n \text{ bytes},
$$

$$
\text{HCA cache per layer} = \tfrac{n}{128} \cdot 512 \cdot 2 = 8 n \text{ bytes}.
$$

Numerical check against the baseline:

$$
\frac{\text{CSA}}{\text{GQA8}} = \frac{256}{2048} = 0.125 = 12.5\%,
$$

$$
\frac{\text{HCA}}{\text{GQA8}} = \frac{8}{2048} \approx 0.004 = 0.4\%.
$$

For V4-Pro with $L = 61$ layers, roughly half CSA and half HCA (exact ratio depends on the interleaving schedule), the total becomes:

$$
\frac{30 \cdot 256 + 31 \cdot 8}{61 \cdot 2048} = \frac{7680 + 248}{124928} \approx 6.3\%.
$$

The paper's measured number is $10\%$ at $1M$ context; the small gap reflects the sliding-window KV, sink logits, RoPE dimensions, and FP8 mixed precision, none of which we modeled here.

### 12.2 Per-query attention FLOPs

Baseline MHA over $n$ raw tokens: $O(n \cdot n_h \cdot d_h)$ FLOPs per query — the **quadratic wall**.

CSA core attention: $O(k \cdot n_h \cdot c)$ FLOPs per query. With $k = 1024 \ll n = 10^6$ and $c = 512$, this is **independent of $n$** — the length-scaling moves entirely into the indexer and the compression step, both of which are cheaper than the original attention.

CSA indexer per query: $O(\tfrac{n}{m} \cdot n_h^I \cdot c^I)$ FLOPs in FP4. With the paper's $n_h^I = 64$, $c^I = 128$, $m = 4$, this is $\tfrac{n}{4} \cdot 64 \cdot 128 = 2048n$ — same asymptotic class as GQA MHA but in FP4 (which on current hardware is the same peak FLOPs as FP8, but theoretically $1/3$ lower on future hardware per the paper's Section 2.3.4).

Adding everything up, the paper reports single-token inference FLOPs at $1M$ context are $27\%$ of V3.2 for V4-Pro and $10\%$ for V4-Flash. These numbers are the end product of the ledger above.

---

## 13. Summary

DeepSeek-V4's long-context efficiency reduces to one equation applied twice: compress $m$ consecutive tokens into one $c$-dimensional entry via a learned per-coordinate softmax, and then either attend sparsely over the compressed stream (CSA, with an overlap-and-index twist) or densely over an even more compressed stream (HCA). Trading the $n \times d$ raw KV cache for an $(n/m) \times c$ or $(n/m') \times c$ compressed cache is what turns $10^6$-token contexts from a $250$ GB infeasibility into a $25$ GB routine.
