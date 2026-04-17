---
title: "The Efficient Transformer Design Space: Comparing All Variants and the Three Futures of Attention"
description: "A unified view of every efficient attention mechanism — each is a different way of replacing the dense N×N attention matrix (sparsify it, factorize it, approximate the softmax kernel, recurse it, or pool one of its dimensions), every variant computed on a single 8-token example so the differences are visible side by side, ranked on the impossible triangle of training parallelism, low-cost inference, and strong quality, and ending with the three credible futures: hybrid retention, selective state spaces, and latent attention."
date: 2026-04-10
tags: [machine-learning, attention, transformers, efficient-transformers, survey, design-space, retnet, mamba, deepseek-v2, mla, linear-attention, sparse-attention, low-rank, kernel-methods, mixture-of-experts]
order: 1
---

Every efficient attention mechanism is doing one thing: replacing the dense $N \times N$ attention matrix $A$ with something cheaper. There are only five moves available. **Sparsify** $A$ — fill most of it with zeros and only compute the entries that survive. **Factorize** $A$ — write it as the product of two smaller matrices. **Approximate** the softmax kernel itself with a finite-dimensional feature map. **Recurse** $A$ — turn the horizontal scan over $N$ keys into a vertical state update of constant size. Or **pool** one of its dimensions — shrink $N$ before doing the quadratic work. Every variant in the literature picks one of these five moves (sometimes two), and every paper's clever idea lives in *how* it picks.

Once that frame is in place, the rest is bookkeeping. This post puts a single 8-token example on the page, computes the dense $8 \times 8$ attention matrix once, and then reruns every major variant — fixed patterns (Sparse Transformer, Longformer, BigBird), learned patterns (Reformer, Routing), low-rank methods (Linformer, Synthesizer), kernel methods (Performer, Linear Transformer), recurrent reformulations (RetNet, Mamba), and pooling (Perceiver, Nyströmformer) — on those same eight tokens. The differences become side-by-side visible. We then derive a unifying weight-function view that compresses every method into the same equation $o_i = \sum_j w_{ij} v_j$, where the only thing that changes between methods is the recipe for $w$. Finally, we rank everything on the **impossible triangle** — training parallelism, low-cost inference, strong quality — and use the resulting picture to read out the three credible futures of attention: hybrid retention (RetNet), selective state spaces (Mamba and Mamba-2), and latent attention (DeepSeek-V2 MLA).

The core reference for the survey portion is Tay et al. (2022), "Efficient Transformers: A Survey" (arXiv:2009.06732v3). For the three futures we revisit Sun et al. (2023) on RetNet, Gu and Dao (2023) and Dao and Gu (2024) on Mamba and Mamba-2, and DeepSeek-AI (2024) on DeepSeek-V2.

---

## The Running Example

We use the same tiny example throughout the entire post. Let

$$
N = 8, \qquad d_k = d_v = 2, \qquad \text{single head}
$$

with input sequence $X \in \mathbb{R}^{8 \times 2}$ and identity query/key/value projections, so that $Q = K = V = X$. The 8 token vectors are

$$
X = \begin{pmatrix}
1 & 0 \\
0 & 1 \\
1 & 1 \\
2 & 0 \\
0 & 2 \\
1 & -1 \\
-1 & 1 \\
1 & 2
\end{pmatrix}
$$

This choice is deliberate: small integers, 8 rows (enough to illustrate blocks, strides, windows, and clusters), and a 2-dimensional feature space (so every inner product is a two-term sum we can compute by eye).

For model-scale cost analysis we use the parameters shared by the rest of the series:

$$
d_\text{model} = 512, \quad h = 8 \text{ heads}, \quad d_k = d_v = 64, \quad L = 12 \text{ layers}, \quad \text{fp16}
$$

Everything derived in this blog — every sparsity pattern, every factorization, every recurrent state, every kernel approximation — will be shown first on the 8-token example, then scaled up to these model-sized numbers.

---

## 1. Vanilla Attention, from Scratch

### 1.1 The unnormalized attention matrix

We start from the definition given in Vaswani et al. (2017). For a single head,

$$
\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The intermediate object is the **attention score matrix** $S = QK^\top \in \mathbb{R}^{N \times N}$, where $S_{ij} = q_i^\top k_j$. With our running example, $q_i = k_i = x_i$, so $S_{ij} = x_i^\top x_j$.

We compute $S$ entry by entry. Each entry is a two-term inner product. For example, $S_{1,4} = x_1^\top x_4 = 1 \cdot 2 + 0 \cdot 0 = 2$. Filling in all 64 entries by the **distributive law** $(a_1, a_2)^\top (b_1, b_2) = a_1 b_1 + a_2 b_2$,

$$
S = QK^\top = \begin{pmatrix}
1 & 0 & 1 & 2 & 0 & 1 & -1 & 1 \\
0 & 1 & 1 & 0 & 2 & -1 & 1 & 2 \\
1 & 1 & 2 & 2 & 2 & 0 & 0 & 3 \\
2 & 0 & 2 & 4 & 0 & 2 & -2 & 2 \\
0 & 2 & 2 & 0 & 4 & -2 & 2 & 4 \\
1 & -1 & 0 & 2 & -2 & 2 & -2 & -1 \\
-1 & 1 & 0 & -2 & 2 & -2 & 2 & 1 \\
1 & 2 & 3 & 2 & 4 & -1 & 1 & 5
\end{pmatrix}
$$

**Numerical check (row 3).** Row 3 is $x_3 = (1, 1)$. Its entries should be $1 \cdot x_{j,1} + 1 \cdot x_{j,2}$, i.e., the sum of the two coordinates of each $x_j$. Scanning the columns: $1+0 = 1$, $0+1 = 1$, $1+1 = 2$, $2+0 = 2$, $0+2 = 2$, $1-1 = 0$, $-1+1 = 0$, $1+2 = 3$. Row 3 is $(1,1,2,2,2,0,0,3)$. Matches.

This matrix $S$ is the central object of the entire blog. **Every efficient transformer we will see is a different way to avoid materializing, storing, or computing all 64 entries of $S$** (and, at model scale, the $N^2 = $ millions of entries the real thing produces).

### 1.2 The cost, as a ledger

Computing $S = QK^\top$ costs $2 N^2 d_k$ floating-point operations (FLOPs) — one multiply and one add per entry, $d_k$ terms per entry, $N^2$ entries. At model scale per layer per head,

$$
2 \cdot N^2 \cdot 64
$$

FLOPs. The softmax costs $O(N^2)$ more. Multiplying by $V$ costs another $2 N^2 d_v$. Memory is dominated by storing $S$ itself: $N^2$ entries per head per layer. For $N = 4096$ (a short context), $h = 8$, $L = 12$ that is

$$
N^2 \cdot h \cdot L = 4096^2 \cdot 8 \cdot 12 \approx 1.6 \times 10^9 \text{ activations}
$$

in fp16 this is about 3.2 GB *just for the attention matrices*, before we multiply by $V$ and before any gradients. This is the bottleneck that the entire field has been trying to break.

### 1.3 Why softmax forces materialization

This is the part that confuses almost everyone on first reading, so we pin it down precisely. The problem is not the $N^2$ multiplications. The problem is that the softmax row-normalizes, and row normalization requires the full row of $S$ to be available simultaneously. If we tried to reorder the computation $(QK^\top) V \to Q (K^\top V)$, we would save a factor of $N$ in FLOPs (the middle product is only $d_k \times d_v$). But softmax is **nonlinear and row-dependent**, so it cannot commute past $V$. The algebraic obstruction is this:

$$
\text{Softmax}(QK^\top) V \;\neq\; \text{Softmax}(Q) (K^\top V)
$$

The right-hand side is simply not the same function. Every efficient attention method is a different answer to: *how do we get around this obstruction?* The answers fall into eight broad families.

---

## 2. A Taxonomy of Efficient Transformers

We follow the taxonomy of Tay et al. (2022), organized by the *technical innovation* each family employs:

1. **Fixed Patterns (FP)** — sparsify $S$ by zeroing out everything outside a pre-specified pattern (blocks, strides, windows).
2. **Combination of Patterns (CP)** — stack two or more fixed patterns to improve coverage (strided + local, or row + column).
3. **Learnable Patterns (LP)** — still sparsify, but learn which entries to keep (clustering, hashing, sorting).
4. **Neural Memory** — route the interaction through a small set of learned memory tokens that pool the sequence.
5. **Low-Rank Methods** — factorize $S$ or its inputs as a product of $N \times k$ and $k \times N$ matrices.
6. **Kernels** — replace $\exp(q^\top k / \sqrt{d_k})$ with a finite feature map $\phi(q)^\top \phi(k)$ and apply associativity.
7. **Recurrence** — connect blocks via a state that carries across chunks.
8. **Downsampling** — reduce $N$ itself by pooling, striding, or projecting the sequence to a shorter one.

Two further categories sit *orthogonal* to the eight above — they do not change the attention weight function $w_{ij}$ at all, but affect the overall cost along different axes:

9. **Conditional Computation / MoE** — attacks the other half of the FLOP budget (the feed-forward layers) by sparsely activating a subset of parameters per token.
10. **System-Level Optimization (IO-aware attention)** — computes the *exact* softmax attention function but reorders the computation to respect the GPU memory hierarchy, eliminating the $N^2$ HBM bottleneck without any mathematical approximation. This is the family of **FlashAttention** (Dao et al., 2022) and its successors.

We will walk through all ten, each time showing what happens to the 8×8 score matrix $S$ of Section 1. The first eight change the *mathematics* of attention. The last two change the *execution* — MoE at the layer level, FlashAttention at the kernel level — while leaving the attention math untouched (or, in MoE's case, leaving it untouched and changing the FFN around it).

---

## 3. Fixed Patterns — Zeroing Out the Matrix

The earliest and simplest idea: pick a predetermined sparsity pattern, and compute attention only where the pattern is 1.

### 3.1 Blockwise / Local attention

Partition the 8 tokens into non-overlapping blocks of size $b = 4$. Each token attends only within its own block. For blocks $\{1,2,3,4\}$ and $\{5,6,7,8\}$, the mask $M_\text{block}$ is a block-diagonal $8 \times 8$ matrix:

$$
M_\text{block} = \begin{pmatrix}
\mathbf{1} & \mathbf{0} \\
\mathbf{0} & \mathbf{1}
\end{pmatrix}
$$

where each $\mathbf{1}$ and $\mathbf{0}$ is a $4 \times 4$ all-ones or all-zeros block. The masked score matrix is $S \odot M_\text{block}$ (elementwise product), which keeps only the top-left and bottom-right $4 \times 4$ sub-blocks of $S$:

$$
S \odot M_\text{block} =
\left(
\begin{array}{cccc|cccc}
1 & 0 & 1 & 2 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 2 & 2 & 0 & 0 & 0 & 0 \\
2 & 0 & 2 & 4 & 0 & 0 & 0 & 0 \\
\hline
0 & 0 & 0 & 0 & 4 & -2 & 2 & 4 \\
0 & 0 & 0 & 0 & -2 & 2 & -2 & -1 \\
0 & 0 & 0 & 0 & 2 & -2 & 2 & 1 \\
0 & 0 & 0 & 0 & 4 & -1 & 1 & 5
\end{array}
\right)
$$

Only 32 entries survive. The cost drops from $N^2 = 64$ to $(N/b) \cdot b^2 = 2 \cdot 16 = 32$, which is $O(Nb)$. At model scale, for $N = 4096$ and $b = 64$, this is a 64× saving. This is the core idea of **Blockwise Transformer** (Qiu et al., 2019) and **Local Attention** (Parmar et al., 2018, Image Transformer).

**Interpretation.** The model sees only a window around each token. It cannot directly attend from position 1 to position 5. Any long-range interaction must be constructed by stacking layers (each layer passes information across block boundaries via the union of overlapping or shifted blocks). This is exactly the trade-off made by convolutional nets: local receptive fields, depth provides global reach.

### 3.2 Sliding window

A closely related variant uses overlapping windows of radius $w$ centered on each query. For $w = 1$, token $i$ attends to positions $\{i-1, i, i+1\}$. The mask is a tridiagonal band:

$$
M_\text{slide}^{(w=1)} = \begin{pmatrix}
1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 1 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 1
\end{pmatrix}
$$

The cost is $O(N w)$. This is the pattern used in **Longformer** (Beltagy et al., 2020) and — in a dilated form, where the window skips every $d$-th position — for wider receptive fields.

### 3.3 Strided / dilated attention

The other basic fixed pattern: attend only to positions at fixed stride $s$. For $s = 2$, token $i$ attends to $\{i, i-2, i-4, \ldots\}$. This is the strided head of **Sparse Transformer** (Child et al., 2019).

For our $8 \times 8$ grid with $s = 2$, the mask is

$$
M_\text{stride}^{(s=2)}[i,j] = \begin{cases} 1 & \text{if } (i - j) \bmod 2 = 0 \\ 0 & \text{otherwise} \end{cases}
$$

which gives a checkerboard pattern on the lower triangle. Each row has $\lfloor i/s \rfloor$ active entries, so the total cost is $O(N \cdot N/s) = O(N^2 / s)$. Strided attention alone does not reduce complexity asymptotically — it reduces it by a constant factor. To get sub-quadratic scaling, Sparse Transformer combines strided with local (Section 4).

**Numerical check.** Row 3 of $S$ is $(1, 1, 2, 2, 2, 0, 0, 3)$. With $s = 2$ and causal masking, row 3 keeps columns $\{1, 3\}$ (positions $3, 1$): entries $(1, 2)$. The other row-3 entries are zeroed.

### 3.4 Memory-compressed / convolutional pooling

Instead of sparsifying, we can *shrink* the key/value sequence by applying a strided 1D convolution on the length dimension. If we pool $K$ and $V$ from length $N = 8$ to length $N/k = 4$ with kernel size and stride both equal to $k = 2$, we compute

$$
K_\text{comp} \in \mathbb{R}^{4 \times 2}, \qquad V_\text{comp} \in \mathbb{R}^{4 \times 2}
$$

With a trivial mean-pool kernel, $K_\text{comp}$ is the row-wise average of consecutive pairs of $X$:

$$
K_\text{comp} = \begin{pmatrix}
(1+0)/2 & (0+1)/2 \\
(1+2)/2 & (1+0)/2 \\
(0+1)/2 & (2-1)/2 \\
(-1+1)/2 & (1+2)/2
\end{pmatrix} = \begin{pmatrix}
0.5 & 0.5 \\
1.5 & 0.5 \\
0.5 & 0.5 \\
0.0 & 1.5
\end{pmatrix}
$$

The new score matrix $Q K_\text{comp}^\top$ is $8 \times 4$ — half the width of the original. The cost is $O(N \cdot N/k)$, saving a factor of $k$. This is the **Memory Compressed Transformer** (Liu et al., 2018).

---

## 4. Combination of Patterns — Stacking Sparsities

Local alone cannot see far. Strided alone covers long range but misses neighbors. The natural move is to combine them.

### 4.1 Sparse Transformer: local + strided

**Sparse Transformer** (Child et al., 2019) dedicates half the heads to a local pattern and half to a strided pattern. The overall attention pattern is then the *union* of the two masks (OR, not AND), applied across heads. For $b = 4$ and $s = 2$, the combined mask is

$$
M_\text{combo}[i,j] = \mathbf{1}[\lfloor i/b \rfloor = \lfloor j/b \rfloor] \vee \mathbf{1}[(i-j) \bmod s = 0]
$$

The cost is $O(N \sqrt{N})$ when $b = s = \sqrt{N}$ — this is the sub-quadratic result Sparse Transformer is famous for.

**Interpretation.** Each head sees a different view. By the **union bound on attention paths**, any token can reach any other token in at most two hops: one strided hop to a "waypoint" and one local hop from the waypoint. Two layers of combined sparse attention suffice for a global receptive field.

### 4.2 Axial attention

For multidimensional inputs (images, videos), the combination takes a specially clean form. **Axial Transformer** (Ho et al., 2019) views a sequence of length $N = H \times W$ as a 2D grid and applies attention along rows and columns separately:

$$
Y = \text{Attention}_\text{col}(\text{Attention}_\text{row}(X))
$$

Row attention costs $H \cdot W^2$ and column attention costs $W \cdot H^2$. For $H = W = \sqrt{N}$, the total is $2 N \sqrt{N} = O(N^{1.5})$, the same rate as Sparse Transformer.

For our running example, imagine reshaping the 8 tokens as a $2 \times 4$ grid:

$$
\begin{pmatrix}
x_1 & x_2 & x_3 & x_4 \\
x_5 & x_6 & x_7 & x_8
\end{pmatrix}
$$

Row attention runs two independent $4 \times 4$ attention blocks (positions 1–4, and 5–8). This recovers the block-diagonal pattern of Section 3.1 exactly. Then column attention runs four $2 \times 2$ blocks across the rows: $(1,5), (2,6), (3,7), (4,8)$. Composing the two, every pair $(i, j)$ can be reached in two steps (one row step and one column step), **by the coupon collector path argument** used in Child et al. (2019).

### 4.3 BigBird: local + strided + random + global

**BigBird** (Zaheer et al., 2020) adds a third ingredient: random attention. Each query attends to (1) its $w$ neighbors, (2) $r$ random positions, and (3) $g$ global tokens. The rationale is a result from **random-graph theory**: a graph formed by union of a local ring, $r$ random edges per node, and $g$ globally connected hubs is a **universal approximator for sequences**, provided the graph has small diameter. BigBird proves this is still a linear-cost pattern and that stacked BigBird layers can simulate any polynomial-time sequence function.

---

## 5. Learnable Patterns — Data-Driven Sparsity

Fixed patterns are content-agnostic: position 3 attends to the same positions in every input. Learnable patterns flip this. They still end up with a sparse attention matrix, but *which* entries survive is chosen by the data.

### 5.1 Reformer: LSH bucketing

**Reformer** (Kitaev et al., 2020) uses **locality-sensitive hashing (LSH)**. Each $q_i$ and $k_j$ is hashed by a random projection $R \in \mathbb{R}^{d_k \times b/2}$:

$$
h(x) = \arg\max_{\ell} [xR \; ; \; -xR]_\ell
$$

This is the **random-projection hash of Andoni and Indyk (2008)**. Nearby vectors (in dot-product sense) hash to the same bucket with high probability, by the **Johnson–Lindenstrauss lemma**.

After hashing, tokens are sorted by bucket and attention is computed only within each bucket. For our running example, suppose LSH assigns buckets as

$$
h(x_1) = h(x_4) = h(x_6) = A, \quad h(x_2) = h(x_5) = h(x_8) = B, \quad h(x_3) = h(x_7) = C
$$

Then each token only attends inside its own bucket. The attention score matrix becomes a block-diagonal permutation: after reordering rows and columns by bucket, we see only a block of size 3 for $A$, a block of 3 for $B$, and a block of 2 for $C$. The cost is $O(N \log N)$, where the $\log N$ factor comes from the sorting step.

### 5.2 Routing Transformer: online $k$-means

**Routing Transformer** (Roy et al., 2020) clusters query/key vectors using online $k$-means into $\sqrt{N}$ clusters of size $\sqrt{N}$, and restricts attention to within-cluster. The cost is $O(N^{1.5})$. It is Reformer with a different hash function.

### 5.3 Sinkhorn Transformer: block sorting

**Sinkhorn Transformer** (Tay et al., 2020b) sorts blocks of tokens such that after sorting, local attention captures the most relevant long-range pairs. The sorting is differentiable via the **Sinkhorn–Knopp algorithm** (Sinkhorn, 1964), which iteratively row- and column-normalizes a matrix to produce a doubly stochastic matrix; by the **Birkhoff–von Neumann theorem**, doubly stochastic matrices are convex combinations of permutations, so the Sinkhorn output is a *soft* permutation that becomes hard in the limit.

All three of these models share a pattern: **learn a permutation / assignment, then apply block-local attention in the permuted space.** The sparsity is learned, but the fundamental cost reduction is still coming from "only attend to a constant number of neighbors."

---

## 6. Neural Memory — Global Bottleneck Tokens

A radically different idea: instead of sparsifying the $N \times N$ matrix, introduce $m \ll N$ trainable "memory" or "inducing point" tokens, and route all communication through them.

### 6.1 Set Transformer: inducing points

**Set Transformer** (Lee et al., 2019) introduces $m$ **inducing points** $I \in \mathbb{R}^{m \times d}$, trainable parameters. An **Induced Set Attention Block** (ISAB) is defined by

$$
\text{ISAB}_m(X) = \text{MAB}(X, \text{MAB}(I, X))
$$

where $\text{MAB}(A, B) = \text{Attention}(Q{=}A, K{=}B, V{=}B)$. The inner $\text{MAB}(I, X)$ is $m \times N$ attention ($m$ queries, $N$ keys). The outer $\text{MAB}(X, \cdot)$ is $N \times m$. Total cost is $O(mN)$, linear in $N$.

For our running example with $m = 2$ inducing points $I = (I_1, I_2)$, the inducing step produces a $2 \times 2$ matrix of pooled summaries:

$$
H = \text{MAB}(I, X) \in \mathbb{R}^{2 \times 2}
$$

Each row of $H$ is a convex combination of the 8 rows of $X$, weighted by softmax of $I_k \cdot x_j$. Then the output step is

$$
Y = \text{MAB}(X, H) \in \mathbb{R}^{8 \times 2}
$$

Each $y_i$ is a softmax-weighted sum of just two vectors, $H_1$ and $H_2$. The global interaction happens because every $y_i$ depends on every $x_j$ through $H$.

**Interpretation.** The inducing points act as a **low-dimensional summary** of the sequence. Information flows $X \to H \to Y$, with $H$ as a $m$-vector bottleneck. This is the same idea as the $[CLS]$ token, but with $m > 1$ and everywhere in the model.

### 6.2 ETC, Longformer, BigBird: global tokens

**ETC** (Ainslie et al., 2020) and **Longformer** (Beltagy et al., 2020) keep the local/sliding attention of Section 3 but add $g$ global tokens that attend to *and* are attended by every position. The mask becomes

$$
M_\text{etc} = M_\text{local} \vee M_\text{global}
$$

where $M_\text{global}$ has all-ones rows and columns for global indices. This costs $O(N(w + g))$, linear.

### 6.3 Perceiver and Nyströmformer: the same trick twice

**Perceiver** (Jaegle et al., 2021) goes further: it makes the *queries* themselves a set of $m$ latent vectors and attends from the latent queries to the $N$-length key/value sequence. This produces an $m \times d$ output, which is processed by a standard transformer and then — for tasks that need per-token predictions — cross-attended back to the $N$ positions. **Nyströmformer** (Xiong et al., 2021b) uses the same two-stage pooling idea with deterministic landmark positions.

All of these are **two-stage attention**: $(N \to m)$ pool, then $(m \to N)$ broadcast. The total cost is $O(mN)$, linear.

---

## 7. Low-Rank Methods — Factorize the Matrix

If $S$ is approximately low-rank — that is, if a rank-$k$ approximation $S \approx U V^\top$ with $U, V \in \mathbb{R}^{N \times k}$ is accurate — we can avoid materializing $S$ directly.

### 7.1 Linformer: project the length dimension

**Linformer** (Wang et al., 2020c) observes empirically that $S$ is low-rank in practice. It adds two learned length-projection matrices $E, F \in \mathbb{R}^{k \times N}$ and computes

$$
\text{Attention}_\text{lin}(Q, K, V) = \text{Softmax}\!\left(\frac{Q (EK)^\top}{\sqrt{d_k}}\right) (FV)
$$

The projected key matrix $EK \in \mathbb{R}^{k \times d_k}$ has only $k$ rows — the $N$-length key sequence has been compressed to a $k$-length sequence by the linear map $E$. For our running example with $N = 8$ and $k = 2$, let

$$
E = F = \begin{pmatrix}
\tfrac{1}{4} & \tfrac{1}{4} & \tfrac{1}{4} & \tfrac{1}{4} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & \tfrac{1}{4} & \tfrac{1}{4} & \tfrac{1}{4} & \tfrac{1}{4}
\end{pmatrix}
$$

so $EK$ is the row-average of the first four rows stacked with the row-average of the last four rows:

$$
EK = \begin{pmatrix}
(1 + 0 + 1 + 2)/4 & (0 + 1 + 1 + 0)/4 \\
(0 + 1 + 1 - 1)/4 & (2 - 1 + 1 + 2)/4
\end{pmatrix} = \begin{pmatrix} 1 & 0.5 \\ 0.25 & 1 \end{pmatrix}
$$

The new score matrix $Q(EK)^\top \in \mathbb{R}^{8 \times 2}$. For row 3 (where $q_3 = (1, 1)$), the entries are $1 \cdot 1 + 1 \cdot 0.5 = 1.5$ and $1 \cdot 0.25 + 1 \cdot 1 = 1.25$.

The softmax is now over 2 columns, not 8. The output $\text{Softmax}(\cdot) F V$ is still $N \times d_v$. The total cost is $O(Nk)$. The catch: $E$ and $F$ mix across positions, so **Linformer cannot be causally masked** — information from the future leaks through the projection. It is an encoder-only method.

### 7.2 Synthesizer: can we even condition on $X$?

**Synthesizer** (Tay et al., 2020a) asks a radical question: is the $QK^\top$ structure necessary at all? The Random Synthesizer replaces $S$ with a trainable $R \in \mathbb{R}^{N \times N}$ that does not depend on the input:

$$
Y = \text{Softmax}(R) G(X)
$$

Its factorized variant writes $R = R_1 R_2^\top$ with $R_1, R_2 \in \mathbb{R}^{N \times k}$, reducing parameters to $2Nk$. The surprising empirical finding of Synthesizer is that this works almost as well as real attention. The attention pattern is, in a sense, more about mixing positions in a learnable way than about content-based alignment.

---

## 8. Kernel Methods — Associativity, and the Bridge to RNNs

This is the family that dissolves the softmax bottleneck rather than sidestepping it.

### 8.1 The kernel rewrite

The softmax attention output at position $i$ is

$$
o_i = \sum_{j=1}^{N} \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i^\top k_{j'} / \sqrt{d_k})} v_j
$$

Replace the exponential similarity with an arbitrary kernel $\kappa(q, k) = \phi(q)^\top \phi(k)$ where $\phi : \mathbb{R}^{d_k} \to \mathbb{R}^{d_\phi}$ is a feature map. By **Mercer's theorem**, any positive semi-definite kernel admits such a decomposition. Now

$$
o_i = \sum_{j=1}^{N} \frac{\phi(q_i)^\top \phi(k_j)}{\sum_{j'} \phi(q_i)^\top \phi(k_{j'})} v_j = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_{j'} \phi(k_{j'})}
$$

The second equality uses the **linearity of the dot product**: $\phi(q_i)^\top$ is a constant with respect to the sum over $j$, so it pulls outside. Define

$$
S_N = \sum_{j=1}^{N} \phi(k_j) v_j^\top \in \mathbb{R}^{d_\phi \times d_v}, \qquad Z_N = \sum_{j=1}^{N} \phi(k_j) \in \mathbb{R}^{d_\phi}
$$

Then

$$
o_i = \frac{\phi(q_i)^\top S_N}{\phi(q_i)^\top Z_N}
$$

Both $S_N$ and $Z_N$ are computed *once* over the whole sequence, then reused for every query $i$. This is the **associativity trick** (Katharopoulos et al., 2020): we traded the $O(N^2 d_v)$ outer-product path $(\phi(Q)\phi(K)^\top)V$ for the $O(N d_\phi d_v)$ inner-product path $\phi(Q)(\phi(K)^\top V)$.

### 8.2 The causal case: an RNN appears

When we add causal masking, the sums only run up to index $i$:

$$
S_i = \sum_{j=1}^{i} \phi(k_j) v_j^\top, \qquad Z_i = \sum_{j=1}^{i} \phi(k_j)
$$

These satisfy the recurrences

$$
S_i = S_{i-1} + \phi(k_i) v_i^\top, \qquad Z_i = Z_{i-1} + \phi(k_i)
$$

with $S_0 = 0, Z_0 = 0$. This is exactly an **RNN**: a state $(S_i, Z_i)$ updated by a rank-1 outer product at each step. Inference cost per token is $O(d_\phi d_v)$, constant in $N$. The KV cache, famously, disappears — it has been replaced by the fixed-size matrix $S$.

### 8.3 Numerical check on the running example

Use the simplest feature map: $\phi(x) = x$ (the identity). So $\phi(q_i) = q_i = x_i$. We compute $Z_N = \sum_j x_j$:

$$
Z_8 = (1+0+1+2+0+1-1+1, \; 0+1+1+0+2-1+1+2) = (5, \; 6)
$$

And $S_N = \sum_j x_j x_j^\top$ (since $v_j = k_j = x_j$), a $2 \times 2$ matrix:

$$
S_8 = \sum_j x_j x_j^\top
$$

Row-by-row, $x_j x_j^\top$ for each $j$ is a $2\times 2$ matrix; summing:

$$
S_8 = \begin{pmatrix} 1+0+1+4+0+1+1+1 & 0+0+1+0+0-1-1+2 \\ \cdot & 0+1+1+0+4+1+1+4 \end{pmatrix} = \begin{pmatrix} 9 & 1 \\ 1 & 12 \end{pmatrix}
$$

where the lower-left entry equals the upper-right by symmetry ($S_8 = X^\top X$).

Now the linear-attention output for query 3 (with $q_3 = (1, 1)$) is

$$
o_3 = \frac{q_3^\top S_8}{q_3^\top Z_8} = \frac{(1,1) \begin{pmatrix} 9 & 1 \\ 1 & 12 \end{pmatrix}}{(1,1) \cdot (5,6)} = \frac{(10, 13)}{11} = (0.909, \; 1.182)
$$

Only two matrix-vector products and one division — no softmax, no materialized $S$ matrix. The 64-entry score grid that dominated Sections 1–7 has been *erased*.

### 8.4 The Performer: unbiased random-feature approximation of softmax

Using $\phi(x) = x$ (the Linear Transformer of Katharopoulos et al. 2020) is an *instance* of kernel attention, but it is not the same function as softmax. **Performer** (Choromanski et al., 2020) addresses this by choosing $\phi$ such that softmax is recovered *in expectation* — not pointwise, not deterministically, but as an unbiased Monte Carlo estimate. The trick is the **positive random-feature identity**

$$
\exp(q^\top k) = \mathbb{E}_{\omega \sim \mathcal{N}(0, I)}\!\left[\exp\!\left(\omega^\top q - \tfrac{1}{2}\|q\|^2\right) \exp\!\left(\omega^\top k - \tfrac{1}{2}\|k\|^2\right)\right]
$$

which is a rearrangement of the **Gaussian moment generating function**. Drawing $M$ samples of $\omega$ gives an unbiased estimator of the softmax kernel with variance $O(1/M)$; Performer uses orthogonal random features (ORFs) to cut variance further. Two points are worth stressing. First, Performer is a *stochastic approximation*: for any finite $M$ the output is not the softmax attention output but a random variable whose expectation is. Second, while the estimator is unbiased for the unnormalized kernel, the normalization (the softmax denominator) introduces bias in the final attention output, and empirical quality on standard benchmarks consistently lags exact softmax attention unless $M$ is taken fairly large. The Performer is a cleaner approximation than $\phi(x) = \text{elu}(x) + 1$, but it remains an approximation, not an equivalence.

---

## 9. Recurrence — Stitching Blocks Together

Section 3 chopped the sequence into independent blocks. That's great for cost but terrible for reach: information cannot cross block boundaries. Recurrence fixes this.

### 9.1 Transformer-XL: segment-level state

**Transformer-XL** (Dai et al., 2019) processes the sequence in segments of length $\ell$, but at each segment, the keys and values are *concatenated* with the (stop-gradient) keys and values from the previous segment:

$$
\tilde{h}^{n-1}_{\tau+1} = [\text{SG}(h^{n-1}_\tau) \; ; \; h^{n-1}_{\tau+1}]
$$

$$
q = h^{n-1}_{\tau+1} W_q, \quad k = \tilde{h}^{n-1}_{\tau+1} W_k, \quad v = \tilde{h}^{n-1}_{\tau+1} W_v
$$

So each segment's attention has access to the hidden states of the previous segment, effectively doubling the receptive field per layer. With $L$ layers, the receptive field grows to $O(L\ell)$.

**Compressive Transformer** (Rae et al., 2020) extends this with a two-level memory: a fine-grained primary memory and a compressed secondary memory, where a pooling or convolutional compressor squeezes older segments into fewer slots.

**Interpretation.** Transformer-XL is algebraically orthogonal to everything in Sections 3–8. It does not change the attention operator inside a segment; it changes what the segment *sees* by attaching recent history. This is the same move as RNN hidden-state passing, lifted to segments of tokens.

---

## 10. Downsampling — Shrink $N$ Itself

Downsampling attacks the cost by reducing $N$ at some intermediate layer.

- **Funnel Transformer** (Dai et al., 2020) pools the sequence length progressively through the encoder, similar to a convolutional pyramid. Cost drops layer by layer.
- **Perceiver** (Jaegle et al., 2021) projects an $N$-length input into $m$ latent slots once, then runs standard quadratic attention on the $m$-length latent sequence. Cost per latent layer is $O(m^2)$; the only $N$-dependent cost is the initial cross-attention $O(mN)$.
- **Nyströmformer** (Xiong et al., 2021b) approximates the $N \times N$ softmax via landmark sampling, using the **Nyström method for matrix approximation** (Williams and Seeger, 2001): pick $m$ landmark rows, compute $N \times m$ and $m \times m$ and $m \times N$ submatrices, and reconstruct the full matrix as their product. Cost $O(Nm)$.

---

## 11. Conditional Computation — Sparse FFNs

**Mixture-of-Experts** (MoE) models like **Switch Transformer** (Fedus et al., 2021), **GShard** (Lepikhin et al., 2020), and **GLaM** (Du et al., 2021) do not touch attention. They replace the feed-forward layer with a pool of $E$ experts, each of which is an independent FFN, and route each token to the top-$k$ experts by a small gating network.

If top-$k$ routing picks $k = 1$ or $k = 2$ experts per token, and the experts are the same size as a dense FFN, then the FLOP cost per token is $k/E$ times the dense cost while the parameter count grows by a factor of $E$. MoE is a **parameter-to-FLOP-ratio** trick, and it is orthogonal to every efficient attention method above. You can bolt Switch on top of Linformer, or Performer, or Longformer.

For our 8-token running example, suppose $E = 4$ experts and top-1 routing. Each of the 8 tokens activates exactly one expert. If the router assigns $x_1, x_4 \to e_1$; $x_2, x_3 \to e_2$; $x_5, x_7 \to e_3$; $x_6, x_8 \to e_4$, then we compute four independent FFN batches of size 2 instead of one dense FFN batch of size 8. Total FLOPs scale with the number of tokens, not the number of experts.

MoE is the first of two families that change the *execution* of a Transformer without changing the attention weight function at all. The second is more surprising, because it leaves *both* the attention weights and the FFN computation alone. It only changes where the numbers live while the computation is in flight.

---

## 12. System-Level Optimization — FlashAttention and the Memory Hierarchy

Every family in Sections 3–11 attacks the *mathematics* of attention — either the sparsity pattern, the kernel, the recurrence, or the feed-forward layer. **FlashAttention** (Dao et al., 2022) attacks nothing mathematical at all. It computes the *exact* softmax attention output of the original Vaswani et al. (2017) formula, bit-for-bit equivalent to the naive implementation, yet runs 2–4× faster and uses 10–20× less memory for the attention activations. How is this possible?

The answer is that naive attention is bottlenecked on the wrong thing. Section 1.2 counted the FLOPs and concluded that the $N^2$ entries of $S$ dominate. But on modern GPUs, FLOPs are not the scarce resource. **Memory bandwidth** is. A naive attention kernel spends most of its wall-clock time moving the $N \times N$ score matrix between slow off-chip memory and the GPU's arithmetic units, and only a small fraction actually multiplying.

### 12.1 The GPU memory hierarchy

A modern GPU has a two-level memory hierarchy relevant to attention:

- **HBM (High-Bandwidth Memory)**, the large off-chip DRAM. Capacity is measured in tens of gigabytes, bandwidth in the range of 1–3 TB/s. All tensors live here by default.
- **SRAM (Static RAM)**, the on-chip cache / shared memory inside each streaming multiprocessor. Capacity is measured in tens to low hundreds of kilobytes per SM, bandwidth is roughly an order of magnitude higher than HBM (~19 TB/s on an A100).

A naive attention kernel performs the following HBM traffic, for one query block of attention on an input of length $N$ with head dimension $d$:

1. Read $Q$ and $K$ from HBM. Write the full $S = QK^\top \in \mathbb{R}^{N \times N}$ back to HBM.
2. Read $S$ from HBM. Compute row-wise max and sum-of-exponentials. Write the normalized $P = \text{Softmax}(S)$ back to HBM.
3. Read $P$ and $V$ from HBM. Compute $O = PV$. Write $O$ back to HBM.

The HBM traffic is $\Theta(N^2)$, dominated by the two full passes over the $N \times N$ matrix. For $N = 4096$ and head dimension 64 in fp16, this is roughly 32 MB of HBM traffic *per head per layer*, and at long sequence lengths the bandwidth wall hits before the arithmetic wall.

### 12.2 The three improvements of FlashAttention

FlashAttention fuses all three passes above into a single **IO-aware kernel** that never materializes $S$ or $P$ in HBM. It achieves this via three complementary techniques:

**Improvement 1 — Tiling.** Partition $Q$ into row blocks of size $B_r$ and $K, V$ into column blocks of size $B_c$, chosen so that a single $(Q_\text{blk}, K_\text{blk}, V_\text{blk})$ triple fits in SRAM. For each $Q$ block, iterate over all $K, V$ blocks. Inside the innermost loop, the partial score matrix $S_\text{blk} = Q_\text{blk} K_\text{blk}^\top$ and the partial output contribution are computed entirely in SRAM. Only the final per-row output $O$ is written back to HBM. This alone would break the correctness of softmax — which is the next problem.

**Improvement 2 — Online softmax (streaming normalization).** Softmax is row-wise, so a block-at-a-time computation must maintain running statistics that can be updated as each new key block arrives, without ever seeing the full row. The key identity is the **log-sum-exp merge rule**. Given two partial rows with per-block maxima $m^{(1)}, m^{(2)}$ and sum-of-exponentials $\ell^{(1)}, \ell^{(2)}$ (where each $\ell^{(b)}$ is computed after subtracting the block's own max for numerical stability), define the merged max and merged sum as

$$
m^\text{new} = \max(m^{(1)}, m^{(2)}), \qquad \ell^\text{new} = e^{m^{(1)} - m^\text{new}} \, \ell^{(1)} + e^{m^{(2)} - m^\text{new}} \, \ell^{(2)}
$$

This is correct by the **log-sum-exp rescaling identity**: for any shift $c$, $\sum_j e^{s_j - m} = e^{c - m} \sum_j e^{s_j - c}$. The running unnormalized output $O^{(b)} = \sum_{j \in B_b} e^{s_j - m^{(b)}} v_j$ is rescaled the same way:

$$
O^\text{new} = e^{m^{(1)} - m^\text{new}} \, O^{(1)} + e^{m^{(2)} - m^\text{new}} \, O^{(2)}
$$

After all blocks have been processed, the final attention output for the query row is $O^\text{final} / \ell^\text{final}$. Since every rescaling factor is a scalar multiplication and the merge rule is associative, the final result is *exactly* the softmax attention output — not an approximation.

**Numerical check on the running example.** Take row 3 of $S$ from Section 1.1: $(1, 1, 2, 2, 2, 0, 0, 3)$. Split it into two blocks $B_1 = (1, 1, 2, 2)$ and $B_2 = (2, 0, 0, 3)$.

Block 1: $m^{(1)} = 2$, $\ell^{(1)} = e^{1-2} + e^{1-2} + e^{2-2} + e^{2-2} = 2 e^{-1} + 2 \approx 0.7358 + 2 = 2.7358$.

Block 2: $m^{(2)} = 3$, $\ell^{(2)} = e^{2-3} + e^{0-3} + e^{0-3} + e^{3-3} = e^{-1} + 2 e^{-3} + 1 \approx 0.3679 + 0.0996 + 1 = 1.4675$.

Merge: $m^\text{new} = \max(2, 3) = 3$, and

$$
\ell^\text{new} = e^{2-3} \cdot 2.7358 + e^{3-3} \cdot 1.4675 = e^{-1} \cdot 2.7358 + 1 \cdot 1.4675 \approx 1.0064 + 1.4675 = 2.4739
$$

Direct check. The unshifted row softmax denominator with max subtracted at $m = 3$ is

$$
\sum_{j=1}^{8} e^{s_j - 3} = e^{-2} + e^{-2} + e^{-1} + e^{-1} + e^{-1} + e^{-3} + e^{-3} + e^{0}
$$

$$
\approx 0.1353 + 0.1353 + 0.3679 + 0.3679 + 0.3679 + 0.0498 + 0.0498 + 1 \approx 2.4739
$$

The online merge and the direct computation agree to the last printed digit. Tiling plus online softmax is, mathematically, still the same softmax.

**Improvement 3 — Recomputation in the backward pass.** Standard attention stores the $N \times N$ matrix $P$ during the forward pass so the backward pass can reuse it for the gradients. This is the single largest memory cost of training a Transformer. FlashAttention *does not store $P$*. Instead, it saves only the per-row softmax statistics $(m_i, \ell_i) \in \mathbb{R}^2$ — two scalars per query row, an $O(N)$ total memory footprint. In the backward pass, it recomputes $P$ block-by-block in SRAM using the same tiling + online softmax machinery. This is the **checkpoint–recompute trade-off** from Chen et al. (2016), applied surgically to attention: trade a roughly $2\times$ increase in FLOPs for the backward pass in exchange for eliminating the $O(N^2)$ activation memory entirely.

### 12.3 Cost ledger

On a sequence of length $N$ with head dimension $d$, FlashAttention's HBM traffic is $O(N^2 d^2 / M)$ where $M$ is the SRAM size, compared with the naive $O(N^2 + Nd)$. For realistic $M$ (on the order of 100 KB) this is a large asymptotic improvement. In practice on an A100, Dao et al. (2022) report 2–4× wall-clock speedup over a tuned PyTorch attention kernel at $N = 1024$–$4096$, growing as $N$ increases, and 10–20× memory savings on the attention activations (because the $N \times N$ matrix is gone from HBM).

**None of this changes $w_{ij}$.** The attention weights are bit-identical to naive softmax attention. The entry $w_{ij}$ for our running example is exactly what Section 1 would have computed. The speedup is entirely from reorganizing when and where numbers move across the memory hierarchy.

### 12.4 Why this matters for the design space

FlashAttention settles a question that dominated the first wave of efficient transformers: *was the $N^2$ attention matrix actually the bottleneck?* The answer, empirically, is "the $N^2$ **FLOPs** are not; the $N^2$ **HBM traffic** was." This changes the calculus for every other family. Sparse attention, low-rank attention, and linear attention all had to beat the FLOP cost of vanilla attention to be worth their complexity overhead. After FlashAttention, they have to beat the far lower *bandwidth*-optimized cost of FlashAttention, which raised the bar substantially. Several methods that looked asymptotically superior in 2020 (Performer, Linformer, Reformer) turned out in 2022–2023 benchmarks to be slower than FlashAttention at practical sequence lengths.

FlashAttention-2 (Dao, 2023) further reduces non-matmul FLOPs and improves work partitioning across GPU warps. FlashAttention-3 (Shah et al., 2024) adds asynchronous compute and FP8 support for Hopper architecture. The line of work is now deeply entangled with GPU microarchitecture — the kernels are co-designed with the silicon.

The broader lesson is that the design space of efficient attention has a dimension that the 2020 survey taxonomy did not name: **system-level execution**. This axis is orthogonal to everything in the Tay et al. taxonomy, and — unlike most axes — it does not require giving anything up. You keep softmax, you keep $O(N^2)$ FLOPs, and you still win. This is why FlashAttention (not Performer, not Linformer, not Reformer) is the efficient-attention technique that is actually deployed in every production Transformer stack today.

With FlashAttention on the table, we have now seen ten families. Nine of them change what $w_{ij}$ is. One (MoE) changes what $v_j$ is. One (FlashAttention) changes neither — and, remarkably, wins more than any of the others at practical scale. This is the full picture. We are now ready to synthesize it.

---

## 13. The Unified View: One Equation, Six Orthogonal Axes

Ten families, dozens of models, hundreds of ablations. If we stand back, what structure do they actually share? The punchline of the last ten sections is that the design space collapses to a single line. **With one carefully delimited exception for each of MoE and FlashAttention, every attention variant we have seen computes the output at position $i$ as**

$$
\boxed{\,o_i = \sum_{j=1}^{N} w_{ij} \, v_j\,}
$$

The difference between methods is entirely in (a) how $w_{ij}$ is defined, and (b) how the sum is organized in memory and time. Let us enumerate the weight functions first, and then enumerate the orthogonal axes that generate them.

| Method | Weight function $w_{ij}$ |
| --- | --- |
| Vanilla attention | $\dfrac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i^\top k_{j'} / \sqrt{d_k})}$ |
| Blockwise / Local | Same as vanilla, but $w_{ij} = 0$ unless $\lfloor i/b \rfloor = \lfloor j/b \rfloor$ |
| Sliding window | Same as vanilla, but $w_{ij} = 0$ unless $|i - j| \leq w$ |
| Strided | Same as vanilla, but $w_{ij} = 0$ unless $(i - j) \bmod s = 0$ |
| Sparse Transformer | Union of a local and a strided pattern, split across heads |
| Reformer | Vanilla within LSH bucket, zero across buckets |
| Routing | Vanilla within $k$-means cluster, zero across clusters |
| Linformer | $\dfrac{\exp(q_i^\top (EK)_r / \sqrt{d_k})}{\sum_{r'} \exp(q_i^\top (EK)_{r'} / \sqrt{d_k})}$, then mapped back through $F$ |
| Linear Transformer | $\dfrac{\phi(q_i)^\top \phi(k_j)}{\phi(q_i)^\top \sum_{j'} \phi(k_{j'})}$ |
| Performer | Random-feature estimator: $\hat{w}_{ij}$ with $\mathbb{E}[\hat{w}_{ij}] \propto \exp(q_i^\top k_j)$ |
| Transformer-XL | Vanilla, but the sum includes segment-previous cached keys/values |
| Set Transformer / Perceiver | Two-stage: $w^{(1)}_{im}$ then $w^{(2)}_{mj}$, composed |
| Synthesizer (random) | $w_{ij} = \text{Softmax}(R)_{ij}$, independent of input |
| RetNet (retention) | $\gamma^{i-j} \cdot q_i^\top k_j$ (no softmax normalization) |
| MLA (DeepSeek-V2) | Vanilla softmax, but $q_i, k_j$ are reconstructed from a compressed latent $c^{KV}$ |
| Mamba (selective SSM) | Structured masked linear attention, equivalent under SSD |
| FlashAttention | Vanilla softmax, exactly — only the order of computation changes |
| MoE FFN | Not an attention weight — affects the per-position transform $v_j$ instead |

**This is the design space.** Almost every published efficient transformer is a choice of the function $w : \mathbb{R}^{d_k} \times \mathbb{R}^{d_k} \times \{1,\ldots,N\}^2 \to \mathbb{R}_{\geq 0}$ that assigns a weight to each query-key pair, together with a choice of how the sum is executed on hardware. This space decomposes along **six roughly orthogonal axes**. Each axis is a knob that a model designer can turn independently of the others, and most production architectures turn *more than one* knob at once.

1. **Sparsity structure.** Which entries of the weight matrix are nonzero? Dense (vanilla), pattern-based (local, stride, block, axial), random (BigBird), learned (Reformer, Routing, Sinkhorn), or some union of these. Turning this knob reduces FLOPs and memory proportionally to the sparsity, but leaves softmax and content-dependence intact.

2. **Factorization rank.** Can the weight matrix be written as $UV^\top$ with $U, V \in \mathbb{R}^{N \times k}$ for some $k \ll N$? Linformer and Synthesizer sit here, and so does any low-rank attention approximation. This axis is orthogonal to sparsity: a matrix can be simultaneously sparse *and* low-rank (as in Scatterbrain; Chen et al., 2021).

3. **Kernelization.** Is the similarity function $\exp(q^\top k / \sqrt{d_k})$, or is it $\phi(q)^\top \phi(k)$ for some finite feature map $\phi$? The choice of $\phi$ controls the approximation quality (Performer, Random Feature Attention) or the capacity of the resulting RNN state (Linear Transformer, ELU+1). This axis interacts strongly with the recurrence axis, because a kernelized attention with causal masking *becomes* an RNN by the associativity trick of Section 8.

4. **Recurrence / temporal structure.** Is the computation fully parallel over positions (vanilla), fully sequential (plain RNN), chunked (Transformer-XL, RetNet chunkwise, Mamba-2 SSD), or mixed (segment-parallel with recurrent state hand-off)? This axis governs inference cost per token and training parallelism, and is the axis on which the impossible triangle is drawn.

5. **Representation compression.** Are $k$ and $v$ stored directly, or routed through a bottleneck? GQA groups heads, MQA collapses to one KV head, MLA (DeepSeek-V2) projects into a low-rank latent $c^{KV}$ and reconstructs per-head keys via matrix absorption. This axis attacks the KV cache specifically, orthogonal to the attention math itself.

6. **System-level execution.** Given a fixed attention function, how is it mapped to hardware? FlashAttention (Section 12) is the canonical lever on this axis: tiling, online softmax, and recomputation reorder the same $QK^\top V$ computation to fit the GPU memory hierarchy, producing an exact softmax output at a fraction of the wall-clock time and memory traffic. No mathematical change; only IO-awareness.

Every concrete efficient transformer is a point in this six-dimensional space, and production architectures now routinely combine three or more axes: a model might use GQA (axis 5) + FlashAttention (axis 6) + sliding window (axis 1) + MoE in the FFN (a seventh, FFN-parameter-sparsity axis that lies outside the attention computation). The earlier survey waves treated these axes as competing; the modern view treats them as composable.

Content-dependence of $w$ is a seventh, finer-grained axis (data-dependent as in vanilla softmax vs. data-independent as in Synthesizer), and length preservation ($N \to N$ vs. $N \to m \to N$ as in Perceiver) is an eighth. But the six above are the load-bearing ones — the axes on which the winning architectures of 2023–2026 actually differ.

With this axis decomposition in hand, the question of the next section becomes precise. Given that a model can turn any subset of these six knobs, which combinations simultaneously achieve training parallelism, $O(1)$ decode, and strong quality? Historically, no combination had. This is the setup for the impossible triangle.

---

## 14. Ranking Variants on the Impossible Triangle

Sun et al. (2023) formalize a three-way trade-off they call the **impossible triangle** for sequence models:

1. **Training parallelism**: can we train all $N$ positions in parallel (i.e., no sequential dependencies along the sequence dimension)?
2. **Low-cost inference**: is per-token decode $O(1)$ memory and time (no growing KV cache)?
3. **Strong performance**: does the model match vanilla Transformer quality on standard benchmarks?

Historically every architecture got *at most two of three*.

- **Vanilla Transformer**: parallel (yes), strong (yes), cheap inference (no — KV cache grows linearly, attention is $O(N)$ per token).
- **RNN / LSTM**: cheap inference (yes), strong on short sequences (partially), parallel training (no — sequential BPTT).
- **Linear Transformer / Katharopoulos**: parallel (yes), cheap inference (yes — constant RNN state), strong (no — quality lags softmax by several points).
- **Sparse / local attention (Longformer, Sparse Transformer)**: parallel (yes), strong (yes on long-context tasks), cheap inference (no — the local KV cache still grows).

Let us map every variant from Section 13 onto this triangle with a quick audit.

| Variant | Parallel training? | $O(1)$ decode? | Matches MHA quality? |
| --- | --- | --- | --- |
| Vanilla | ✓ | ✗ | ✓ |
| Local / Sliding | ✓ | ✗ | ✗ on long range |
| Sparse Transformer | ✓ | ✗ | ≈ |
| Longformer / ETC / BigBird | ✓ | ✗ | ≈ (encoder-only) |
| Reformer | ✓ | ✗ | ≈ |
| Linformer | ✓ | ✗ (and encoder-only) | ≈ |
| Linear Transformer | ✓ | ✓ | ✗ |
| Performer | ✓ (encoder); ✗ (causal, slow) | ✓ (inference) | ≈ |
| Transformer-XL | ≈ (segment-parallel) | ✗ | ≈ |
| Perceiver | ✓ | ✓ (in latent) | ≈ |
| GQA / MQA | ✓ | better than MHA | ≈ |
| Switch / GShard | ✓ | ✗ (same as vanilla) | ≈ (different axis) |

Looking at this table, **no member of the first wave of efficient transformers achieves all three vertices simultaneously.** Every row has at least one failure, and the failures are not random: they cluster along the axis decomposition of Section 13. The sparsity-based methods (Sparse, Longformer, BigBird, Reformer) keep softmax and therefore keep a growing KV cache — they trade quality against FLOPs, not against the decoder-time state. The factorization and kernel methods (Linformer, Linear Transformer, Performer) successfully collapse the state to constant size but give up either causal use or softmax-level quality. The recurrence methods (Transformer-XL) only give up parallelism partially, and only across segment boundaries. In other words, **the first wave moved freely along one or two of the six axes at a time, but every axis-restricted move bought one vertex of the triangle at the cost of another.** That is the structural reason the triangle held. Escaping it requires moving along *several* axes at once, in a coordinated way that lets the improvements compound instead of cancel.

---

## 15. The Field's Verdict: Design Trends as of 2022

Before jumping to futures, it is worth pausing on what the Tay et al. survey's retrospective concluded. The honest summary from their 2022 update is:

1. **Quadratic attention is still the default.** Despite a dozen "X-formers," most production language and vision models in 2022 still used vanilla softmax attention. The efficient variants either underperformed on standard benchmarks, or required custom kernels that limited them to one hardware stack, or both.
2. **Local attention, done right, is a very tough baseline.** Xiong et al. (2021a) showed that plain sliding-window attention with good hyperparameters beats most learnable-pattern methods on Long Range Arena.
3. **The word "efficient" is overloaded.** An $O(N)$ attention can be slower than an $O(N^2)$ attention on realistic sequence lengths because of constants, memory access patterns, and kernel launch overhead. Dehghani et al. (2021) coined this the "efficiency misnomer."
4. **Sparse MoE is the only clear win.** Switch, GShard, GLaM, ST-MoE have demonstrably reduced compute-per-parameter without quality loss.
5. **Nothing has unseated quadratic attention at scale.** Tay et al. write: "it is then a question of whether that new xformer will still be a Transformer."

That last sentence is the jumping-off point for the next three sections. If the first wave failed because each method moved along only one or two of the six design axes, the next wave should move along several at once — and the movements should be chosen so that the vertices of the impossible triangle stop competing. Specifically, the triangle held because parallelism and $O(1)$ decode seemed to demand opposite execution models: parallelism wants a big $N \times N$ matrix, cheap decode wants a small recurrent state. The way out, if there is one, is a family of formulas that can be *run in two equivalent modes* — as a parallel matrix operation for training, and as a recurrent state update for inference — with strong enough quality that the quality vertex comes along for the ride.

That is exactly the shape of the three futures. **RetNet** adds a scalar decay to linear attention so that the same retention formula is a masked matrix product in parallel mode and a first-order recurrence in decode mode. **Mamba** starts from state space models, adds input-dependent selectivity to recover attention-like quality, and (via the structured state space duality) turns the recurrence back into a masked matrix product for training. **DeepSeek-V2** takes the dual route: keep softmax and the matrix form exactly as they are, but absorb the up-projection into the query so that the effective recurrent state is a low-rank latent rather than a full KV cache. Three different axis combinations, one shared goal. The rest of the blog walks through each in turn, on the same 8-token example, checking numerically that each future does what it claims.

---

## 16. Future 1 — Hybrid Retention (RetNet)

**Retentive Network** (Sun et al., 2023) claims all three vertices of the impossible triangle with a single mechanism. The core idea is that one formula should admit two (actually three) *exactly equivalent* computation modes, so we can train in parallel mode and infer in recurrent mode without any approximation.

### 16.1 Retention, derived

Start from causal linear attention (Section 8): $o_i = \sum_{j \leq i} \phi(q_i)^\top \phi(k_j) v_j$ with $\phi(x) = x$. Add a per-step decay $\gamma \in (0, 1)$ so that older tokens fade geometrically:

$$
o_i = \sum_{j=1}^{i} \gamma^{i - j} \, (q_i^\top k_j) \, v_j
$$

This is the **retention mechanism**. The decay $\gamma^{i-j}$ solves the **unbounded state growth** problem of linear attention (the state $S_i$ would otherwise accumulate forever) and supplies **position encoding** for free (the decay distinguishes positions by distance).

### 16.2 The three forms, equivalent

**Parallel form.** Define the decay matrix $D \in \mathbb{R}^{N \times N}$ by $D_{ij} = \gamma^{i-j}$ for $i \geq j$ and $0$ otherwise. Then

$$
O_\text{par} = (QK^\top \odot D) V
$$

This is exactly attention with a *causal decay mask* in place of softmax. Cost: $O(N^2 d)$ — same as vanilla attention (but parallel across all $i$).

**Recurrent form.** Define the state $S_i \in \mathbb{R}^{d_k \times d_v}$ by the recurrence

$$
S_i = \gamma \, S_{i-1} + k_i v_i^\top, \qquad o_i = q_i^\top S_i
$$

with $S_0 = 0$. Cost: $O(d_k d_v)$ per token, constant in $N$. This is a pure RNN.

**Chunkwise form.** Process the sequence in chunks of size $C$, using the parallel form within each chunk and the recurrent form to pass a state across chunks. Cost per chunk: $O(C^2 d + C d^2)$. For long sequences this is the best of both worlds.

### 16.3 Verification on the running example

Take $\gamma = 0.5$. Compute $o_3$ using both forms.

**Parallel.** The $D$ matrix restricted to row 3 is $(\gamma^2, \gamma^1, \gamma^0, 0, 0, 0, 0, 0) = (0.25, 0.5, 1, 0, 0, 0, 0, 0)$. The row of $QK^\top$ (from Section 1.1) is $(1, 1, 2, 2, 2, 0, 0, 3)$. The masked inner product is

$$
(0.25 \cdot 1, \; 0.5 \cdot 1, \; 1 \cdot 2, \; 0, 0, 0, 0, 0) = (0.25, 0.5, 2, 0, 0, 0, 0, 0)
$$

Multiplying this row vector by $V$ (whose rows are $x_1, \ldots, x_8$):

$$
o_3^\text{par} = 0.25 \, x_1 + 0.5 \, x_2 + 2 \, x_3 = 0.25 (1,0) + 0.5 (0,1) + 2 (1,1) = (2.25, 2.5)
$$

**Recurrent.** We walk forward:

$$
S_1 = 0.5 \cdot 0 + k_1 v_1^\top = x_1 x_1^\top = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}
$$

$$
S_2 = 0.5 \, S_1 + x_2 x_2^\top = \begin{pmatrix} 0.5 & 0 \\ 0 & 0 \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0.5 & 0 \\ 0 & 1 \end{pmatrix}
$$

$$
S_3 = 0.5 \, S_2 + x_3 x_3^\top = \begin{pmatrix} 0.25 & 0 \\ 0 & 0.5 \end{pmatrix} + \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 1.25 & 1 \\ 1 & 1.5 \end{pmatrix}
$$

Then

$$
o_3^\text{rec} = q_3^\top S_3 = (1, 1) \begin{pmatrix} 1.25 & 1 \\ 1 & 1.5 \end{pmatrix} = (2.25, \; 2.5)
$$

Parallel and recurrent match exactly: $(2.25, 2.5) = (2.25, 2.5)$. This is not an approximation; it is the same computation rearranged.

### 16.4 Interpretation

Retention is linear attention with a decay $\gamma$. The decay fixes the two flaws of plain linear attention that the Section 8 derivation left open: unbounded state and missing position information. The training happens in parallel form (GPU-efficient). The inference happens in recurrent form (RAM-efficient, no KV cache growth). Quality, as reported in Sun et al. (2023), is competitive with strong Transformer baselines at the scales they evaluate. **All three vertices of the impossible triangle are addressed by the same underlying equation, computed in different orders.**

---

## 17. Future 2 — Selective State Spaces (Mamba and Mamba-2)

**Mamba** (Gu and Dao, 2023) takes the opposite path: it does not start from attention at all. It starts from state space models — a class of recurrent dynamical systems from control theory — and extends them to match Transformer quality.

### 17.1 The continuous SSM

A **state space model** is the continuous-time linear system

$$
h'(t) = A h(t) + B x(t), \qquad y(t) = C h(t)
$$

with $h \in \mathbb{R}^N$ the latent state, $A \in \mathbb{R}^{N \times N}$ the state matrix, $B \in \mathbb{R}^{N \times 1}$ the input matrix, $C \in \mathbb{R}^{1 \times N}$ the output matrix. This is **Kalman's state space representation** (Kalman, 1960) and is the standard formulation in linear control.

### 17.2 Discretization and the convolution form

For discrete inputs $x_1, \ldots, x_N$, we discretize using the **zero-order hold (ZOH)** rule with step size $\Delta$:

$$
\bar{A} = \exp(\Delta A), \qquad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B
$$

The discretized system is

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \qquad y_t = C h_t
$$

Unrolling, $y_t = \sum_{s=0}^{t} C \bar{A}^{t-s} \bar{B} x_s$, which is a **convolution** of the input with the kernel $K = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots)$. For *time-invariant* $A, B, C$, this convolution can be computed in $O(N \log N)$ with FFT.

### 17.3 The selectivity problem

Prior SSMs (S4, H3, etc.) had time-invariant $A, B, C$ — the dynamics did not depend on the current input. Gu and Dao (2023) showed with a cleanly designed toy task that this is a *fundamental* limitation: the model cannot, in principle, focus on or ignore specific tokens based on content, because its filter is the same at every timestep. This is the part that confuses almost everyone coming from attention-land, because attention *always* had content-based gathering. SSMs before Mamba did not.

### 17.4 Selective SSMs

Mamba fixes this by making $\Delta$, $B$, and $C$ linear functions of the input $x_t$:

$$
B_t = W_B x_t, \qquad C_t = W_C x_t, \qquad \Delta_t = \text{softplus}(W_\Delta x_t)
$$

Now every step has its own dynamics. The recurrence becomes

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad y_t = C_t h_t
$$

with $\bar{A}_t, \bar{B}_t$ derived from $\Delta_t, A, B_t$ via ZOH. The convolution shortcut breaks (the filter is no longer shift-invariant), but the recurrence still runs in $O(N)$. Gu and Dao introduced a **hardware-aware selective scan** that fuses discretization, the recurrence, and the output projection into a single SRAM-resident CUDA kernel — avoiding the memory bandwidth bottleneck that would otherwise kill the recurrent form.

### 17.5 The Mamba-2 duality

**Dao and Gu (2024)** then proved a remarkable result: the selective SSM, written as a matrix $M$ mapping $x \in \mathbb{R}^N$ to $y \in \mathbb{R}^N$ via $y = M x$, is a **1-semiseparable matrix**. And 1-semiseparable matrices are exactly the matrices produced by a certain form of masked linear attention with a structured mask. This is the **structured state space duality (SSD)**:

$$
\underbrace{\text{selective SSM}}_\text{recurrent, } O(N) \;\equiv\; \underbrace{\text{structured masked attention}}_\text{parallel, } O(N^2)
$$

The two are *the same function*, computed by different algorithms. Mamba-2 uses this duality to build a block-decomposition algorithm: quadratic attention within chunks (efficient on GPU matmul units), linear recurrence across chunks, for 2–8× speedup over Mamba-1.

### 17.6 Interpretation

Mamba is the *other* path to the impossible triangle. Where RetNet starts from attention and adds decay to get a recurrence, Mamba starts from a recurrence and adds selectivity to get attention. Mamba-2 shows that these two paths arrive at points in the same space — both are masked linear attention with a structured decay/state evolution. The "attention vs. state space" distinction is algebraic, not conceptual.

---

## 18. Future 3 — Multi-head Latent Attention (DeepSeek-V2)

The third credible future does not abandon softmax attention at all. It keeps it, and attacks only the specific bottleneck that hurts at deployment: the **KV cache**. This is the path of **DeepSeek-V2** (DeepSeek-AI, 2024).

### 18.1 The KV cache ledger

During autoregressive generation, each layer of a Transformer must store the keys and values of every past token to attend against them when producing the next token. For a model with $h$ heads, head dimension $d_h$, and $L$ layers, the per-token KV cache cost is

$$
2 \cdot h \cdot d_h \cdot L
$$

bytes per token (in fp16). For DeepSeek's 67B-scale baseline with $h = 128, d_h = 128, L = 60$, this is 3.9 MB per token. At 32k context, that is 125 GB just for the KV cache of a single sequence — larger than the model itself.

GQA (Ainslie et al., 2023) cuts this by a factor of $h / g$ where $g$ is the number of KV groups. MQA ($g = 1$) cuts it by $h$. Both trade capacity for memory.

### 18.2 Low-rank joint KV compression

DeepSeek-V2 compresses $k$ and $v$ **jointly** into a single low-rank latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$ with $d_c \ll h \cdot d_h$:

$$
c_t^{KV} = W^{DKV} h_t
$$

where $W^{DKV} \in \mathbb{R}^{d_c \times d_\text{model}}$ is a down-projection. At inference time, only $c_t^{KV}$ is cached, not the full $k$ and $v$.

To reconstruct per-head keys and values at attention time, the model applies up-projections:

$$
k_t^{(i)} = W_i^{UK} c_t^{KV}, \qquad v_t^{(i)} = W_i^{UV} c_t^{KV}
$$

### 18.3 Matrix absorption — the key trick

Naively, this would cost a large up-projection matmul for every query step. DeepSeek-V2 uses a beautiful **matrix absorption** trick based on the **associativity of matrix multiplication**. The attention score between query $q_t^{(i)}$ and key $k_s^{(i)}$ is

$$
q_t^{(i) \top} k_s^{(i)} = (W_i^{UQ} c_t^{Q})^\top (W_i^{UK} c_s^{KV}) = c_t^{Q \top} (W_i^{UQ \top} W_i^{UK}) c_s^{KV}
$$

By associativity, we can precompute $W_i^{UQ \top} W_i^{UK}$ once at load time and never materialize the full $k$. The computation stays in the compressed $d_c$-dimensional latent space throughout — we never expand back to $h \cdot d_h$. The same trick absorbs $W_i^{UV}$ into the output projection on the value side.

### 18.4 Decoupled RoPE

Rotary positional embeddings (RoPE, Su et al., 2021) apply a position-dependent rotation to queries and keys. This conflicts with the absorption trick, because RoPE injects a position-dependent factor that cannot be absorbed. DeepSeek-V2 solves this with **decoupled RoPE**: split each head into two parts — a large content part that uses the compressed latent path (absorbed, no RoPE), and a small position part of dimension $d_h^R$ that carries RoPE and is cached directly. The KV cache then stores $d_c + d_h^R$ numbers per token instead of $2 h d_h$.

### 18.5 Cost on the running example

For our model-scale numbers $d_\text{model} = 512$, $h = 8$, $d_h = 64$, $L = 12$:

- Standard MHA cache per token: $2 h d_h L = 2 \cdot 8 \cdot 64 \cdot 12 = 12{,}288$ numbers.
- MLA cache per token with $d_c = 128$, $d_h^R = 32$: $(d_c + d_h^R) L = 160 \cdot 12 = 1{,}920$ numbers.

That is a **6.4× reduction** in KV cache, and at the full DeepSeek-V2 scale (128 heads, 60 layers) the reported reduction reaches roughly 93%. The empirical result in the DeepSeek-V2 paper is that on the benchmarks they evaluate, MLA achieves performance *competitive with or improving on* standard MHA — a notable finding, because most prior KV-compression techniques (MQA, GQA with small $g$) trade quality for memory. The authors attribute the gain to the latent-space bottleneck forcing a more compact, disentangled representation of the key/value information. This should not be read as a universal dominance claim: MLA has been evaluated primarily in DeepSeek's own training recipe, and the head-to-head with MHA under fully matched budgets across architectures and tasks remains an open question. What is clear is that MLA opens a Pareto frontier point that neither MQA nor GQA reached.

### 18.6 Interpretation

MLA is the pragmatist's answer. It does not change the fundamental $O(N)$ per-token attention cost. It does not replace softmax. It only compresses the cache, using low-rank factorization plus associativity. But the compression ratio it achieves is larger than any prior KV-compression technique, and — unlike MQA and aggressive GQA — it does not come with a quality regression in the regimes where it has been measured. MLA is what happens when you take the design-space lessons of the survey — "factorize where you can, keep softmax where it works" — and apply them surgically, while leaving FlashAttention and MoE available to stack on top.

---

## 19. The Three Futures, Side by Side

We now have three credible successors to vanilla multi-head attention. Each one solves the impossible triangle differently.

| Axis | RetNet (Retention) | Mamba-2 (SSD) | DeepSeek-V2 (MLA) |
| --- | --- | --- | --- |
| Starting point | Linear attention + decay | Continuous SSM + selectivity | Softmax MHA + low-rank KV |
| Core formula | $S_i = \gamma S_{i-1} + k_i v_i^\top$ | $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ | $c_t^{KV} = W^{DKV} h_t$ |
| Training cost | $O(N^2 d)$ or $O(N C d)$ chunked | $O(N^2 d)$ in SSD form | $O(N^2 d)$ standard |
| Per-token decode | $O(d^2)$ constant state | $O(N d)$ state | $O(N d_c)$ compressed |
| KV cache | None (state only) | None (state only) | $d_c + d_h^R$ per token |
| Position encoding | $\gamma^{i-j}$ decay | Implicit in $\Delta_t$ | Decoupled RoPE |
| What it preserves | Attention-like parallelism | Attention-like quality | Softmax quality exactly |
| What it sacrifices | Exact softmax (decays instead) | Shift-invariance of filter | Raw cache size (not per-token cost) |

There are three different answers to the same question. RetNet says: keep the attention *structure*, replace the softmax with a decay. Mamba says: keep the recurrence *structure*, add selectivity to match attention quality. DeepSeek-V2 says: keep the softmax *exactly*, factorize the cache.

The Mamba-2 result that selective SSMs are 1-semiseparable masked attention suggests these futures are converging. RetNet is masked linear attention with a scalar decay mask. Mamba-2 is masked linear attention with a structured decay mask. DeepSeek-V2 keeps masked softmax attention but pushes the key/value representation through a low-rank bottleneck. All three are variations on one theme: **compute attention via structured, low-rank, or decaying interactions over a compressed state.**

---

## 20. Summary

The twenty efficient-transformer blogs in this series, synthesized into one arc: softmax attention materializes an $N \times N$ matrix $S = QK^\top$, and every efficient variant is a different way to avoid materializing it — by sparsifying (fixed, combined, or learnable patterns), factorizing (low-rank, kernel, Nyström), recursing (Transformer-XL, linear attention with RNN state, SSMs), pooling (Perceiver, Funnel, Set Transformer), replacing the FFN alongside it (MoE), or — remarkably — leaving the mathematics untouched and only reordering the execution to respect the GPU memory hierarchy (FlashAttention). All of these collapse to a single equation $o_i = \sum_j w_{ij} v_j$ in which each method picks a different weight function $w$, except FlashAttention and MoE, which keep $w$ fixed and operate on orthogonal axes (execution and feed-forward sparsity respectively). The design space factors along six roughly orthogonal axes — sparsity structure, factorization rank, kernelization, recurrence / temporal structure, representation compression, and system-level execution — and every published efficient transformer is a point in this axis-aligned product space, often combining two or three axes at once. As of 2022 the survey's honest retrospective was that none of the early variants had unseated vanilla attention at scale; what has actually won, at production scale, is FlashAttention (the system axis) plus GQA / MLA (the representation axis) plus MoE (the FFN axis), all sitting on top of softmax attention. Since then, three credible *algorithmic* futures have emerged: RetNet closes the impossible triangle by running the same decaying-linear-attention formula in parallel for training and recurrent for inference; Mamba starts from state space models, adds input-dependent selectivity, and — via the structured state space duality — turns out to be computing a masked-attention function by a different algorithm; DeepSeek-V2 keeps softmax unchanged but compresses the KV cache into a low-rank latent with matrix absorption and decoupled RoPE, reporting roughly a 93% cache reduction at production scale with performance competitive with or improving on standard MHA. The design space has converged: future attention is structured, low-rank or decaying, computed over a compressed state, equivalent to its own recurrent form, and executed by an IO-aware kernel that never materializes the $N \times N$ matrix.
