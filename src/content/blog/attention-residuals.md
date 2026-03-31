---
title: "Attention Residuals: Replacing Fixed Skip Connections with Learned Depth-Wise Attention"
description: "Building Attention Residuals from scratch — why standard residuals dilute information, how softmax attention over depth fixes it, the block variant that makes it practical, and the structured-matrix view that unifies everything — all derived step by step with a 4-layer running example"
date: 2026-03-19
tags: ["deep-learning", "transformers", "architecture", "residual-connections", "attention", "linear-algebra"]
---

Residual connections are the backbone of every modern deep network. The update rule $\boldsymbol{h}_l = \boldsymbol{h}_{l-1} + f_{l-1}(\boldsymbol{h}_{l-1})$ is so universal that we rarely question it. But this simplicity hides a rigid design choice: every previous layer's output is accumulated with a fixed weight of 1. There is no mechanism for a later layer to say "I need the embedding more than I need layer 3's output" or "layer 7's contribution is irrelevant to me."

The [Attention Residuals paper](https://arxiv.org/abs/2603.15031) (Kimi Team, 2026) proposes a direct fix: replace the fixed accumulation with learned, input-dependent softmax attention over all previous layer outputs. The idea is clean — apply the same attention mechanism that Transformers use over the sequence dimension, but now over the *depth* dimension.

We will derive everything from scratch using a single running example: a tiny network with $L = 4$ layers and scalar hidden states ($d = 1$). By the end, we will have built up to Full Attention Residuals, Block Attention Residuals, and the structured-matrix view that reveals standard residuals, Highway networks, Hyper-Connections, and AttnRes as points on a single spectrum.

---

## 1. Standard Residual Connections

### 1.1 The Recurrence

A **residual connection** adds the output of a layer to its input, preserving an identity path through the network. The hidden state at layer $l$ is:

$$
\boldsymbol{h}_l = \boldsymbol{h}_{l-1} + f_{l-1}(\boldsymbol{h}_{l-1})
$$

where $f_{l-1}$ is the transformation applied by layer $l-1$ (an attention sub-layer or an MLP sub-layer in a Transformer), and $\boldsymbol{h}_1$ is the token embedding.

Let us define $\boldsymbol{v}_0 = \boldsymbol{h}_1$ (the embedding) and $\boldsymbol{v}_i = f_i(\boldsymbol{h}_i)$ for $i \geq 1$ (each layer's output). Then we can unroll the recurrence.

### 1.2 Unrolling the Recurrence

For our 4-layer network:

$$
\boldsymbol{h}_1 = \boldsymbol{v}_0
$$

$$
\boldsymbol{h}_2 = \boldsymbol{h}_1 + f_1(\boldsymbol{h}_1) = \boldsymbol{v}_0 + \boldsymbol{v}_1
$$

$$
\boldsymbol{h}_3 = \boldsymbol{h}_2 + f_2(\boldsymbol{h}_2) = \boldsymbol{v}_0 + \boldsymbol{v}_1 + \boldsymbol{v}_2
$$

$$
\boldsymbol{h}_4 = \boldsymbol{h}_3 + f_3(\boldsymbol{h}_3) = \boldsymbol{v}_0 + \boldsymbol{v}_1 + \boldsymbol{v}_2 + \boldsymbol{v}_3
$$

The general form is:

$$
\boxed{\boldsymbol{h}_l = \sum_{i=0}^{l-1} \boldsymbol{v}_i = \boldsymbol{v}_0 + \sum_{i=1}^{l-1} f_i(\boldsymbol{h}_i)}
$$

Every layer receives the *uniform sum* of all previous layer outputs. The coefficient on every term is exactly 1 — no more, no less.

### 1.3 Numerical Check

Let our scalar example have $\boldsymbol{v}_0 = 10$, $\boldsymbol{v}_1 = 2$, $\boldsymbol{v}_2 = -1$, $\boldsymbol{v}_3 = 5$. Then:

- $\boldsymbol{h}_1 = 10$
- $\boldsymbol{h}_2 = 10 + 2 = 12$
- $\boldsymbol{h}_3 = 10 + 2 + (-1) = 11$
- $\boldsymbol{h}_4 = 10 + 2 + (-1) + 5 = 16$

Each hidden state is a simple running total. Layer 4 has no choice but to accept the sum $10 + 2 - 1 + 5 = 16$. It cannot "turn down" $\boldsymbol{v}_2 = -1$ or "amplify" $\boldsymbol{v}_0 = 10$.

### 1.4 The Depth Mixing Matrix

We can write the full system as a matrix equation. Define the **depth mixing matrix** $\mathbf{M} \in \mathbb{R}^{L \times L}$ where $\mathbf{M}_{i \to l}$ is the weight that layer $l$ assigns to the output of layer $i$. For standard residuals, $\mathbf{M}_{i \to l} = 1$ for all $i < l$:

$$
\begin{bmatrix} \boldsymbol{h}_1 \\ \boldsymbol{h}_2 \\ \boldsymbol{h}_3 \\ \boldsymbol{h}_4 \end{bmatrix} = \begin{bmatrix} 1 & & & \\ 1 & 1 & & \\ 1 & 1 & 1 & \\ 1 & 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} \boldsymbol{v}_0 \\ \boldsymbol{v}_1 \\ \boldsymbol{v}_2 \\ \boldsymbol{v}_3 \end{bmatrix}
$$

This is an all-ones lower-triangular matrix. Every entry below and including the diagonal is 1. There is zero selectivity.

### 1.5 Numerical Check (Matrix Form)

Using our values $\boldsymbol{v} = [10, 2, -1, 5]^\top$:

$$
\begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} 10 \\ 2 \\ -1 \\ 5 \end{bmatrix} = \begin{bmatrix} 10 \\ 12 \\ 11 \\ 16 \end{bmatrix}
$$

Checking row 4: $1 \times 10 + 1 \times 2 + 1 \times (-1) + 1 \times 5 = 16$. Matches.

---

## 2. Why Fixed Accumulation Is a Problem

### 2.1 The PreNorm Dilution Problem

In practice, modern LLMs use **PreNorm** — applying layer normalization *before* each sub-layer rather than after. PreNorm restores a clean identity path and stabilizes gradients, making it the dominant paradigm.

But PreNorm introduces a subtle problem. Since $\|\boldsymbol{h}_l\| = \|\sum_{i=0}^{l-1} \boldsymbol{v}_i\|$ grows as $O(L)$ with depth (by the unrolled recurrence above), and PreNorm normalizes *before* the transformation, each layer's relative contribution gets progressively diluted. The embedding $\boldsymbol{v}_0$ that was 100% of $\boldsymbol{h}_1$ is only a fraction $\sim 1/L$ of $\boldsymbol{h}_L$.

For our example with $L = 4$: $\boldsymbol{h}_4 = 16$, and $\boldsymbol{v}_0 = 10$ contributes $10/16 = 62.5\%$. With $L = 100$, the embedding would contribute roughly $\sim 1\%$ of the hidden state magnitude.

This has a concrete consequence: deeper layers must learn increasingly larger outputs just to remain influential. Empirically, the paper shows that output magnitudes grow monotonically with depth in baseline models — a direct symptom of this dilution.

### 2.2 Three Limitations of Single-State Recurrence

Whether we use fixed weights (standard residuals) or learned gates (Highway networks), every approach that conditions only on $\boldsymbol{h}_{l-1}$ shares three limitations:

1. **No selective access.** Different layer types (attention vs. MLP) receive the same aggregated state, despite potentially benefiting from different weightings of past layers.

2. **Irreversible loss.** Once information is mixed into the running sum, it cannot be selectively recovered. If $\boldsymbol{v}_3$ partially cancels $\boldsymbol{v}_1$ in the sum, layer 5 cannot "undo" this.

3. **Output growth.** Later layers must learn increasingly larger outputs to influence the accumulated residual, which can destabilize training.

---

## 3. The Time-Depth Duality

This is the central insight of the paper. It is worth slowing down for.

### 3.1 RNNs Compress Over Time

A **recurrent neural network (RNN)** processes a sequence by maintaining a single hidden state $\boldsymbol{s}_t$ that compresses all past tokens:

$$
\boldsymbol{s}_t = \boldsymbol{s}_{t-1} + f(\boldsymbol{s}_{t-1}, \boldsymbol{x}_t)
$$

This is structurally identical to the residual update $\boldsymbol{h}_l = \boldsymbol{h}_{l-1} + f_{l-1}(\boldsymbol{h}_{l-1})$. The RNN compresses over time steps $t$; the residual connection compresses over layers $l$. Both maintain a single state that accumulates all prior contributions with fixed weights.

### 3.2 Attention Replaced RNNs Over Time

The Transformer solved the RNN bottleneck by replacing the fixed recurrence with **attention**: each position can selectively access all previous positions with learned, data-dependent weights. This was the linear-to-softmax transition for the sequence dimension.

### 3.3 AttnRes Applies the Same Fix Over Depth

Attention Residuals propose the exact same transition for depth. Instead of compressing all previous layers into a single running sum, each layer selectively attends to all previous layer outputs with learned, input-dependent weights via softmax.

The analogy is precise:

| | Sequence (RNN → Transformer) | Depth (Residual → AttnRes) |
|---|---|---|
| State | $\boldsymbol{s}_t$ (hidden state) | $\boldsymbol{h}_l$ (hidden state) |
| Sources | Past tokens $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_{t-1}$ | Past layer outputs $\boldsymbol{v}_0, \ldots, \boldsymbol{v}_{l-1}$ |
| Fixed mixing | RNN recurrence | Residual sum |
| Selective mixing | Sequence attention | Depth attention (AttnRes) |

Standard residuals and prior recurrence-based variants can all be shown to perform depth-wise *linear* attention. AttnRes generalizes them to depth-wise *softmax* attention — completing for depth the same linear-to-softmax transition that proved transformative for sequences.

---

## 4. Full Attention Residuals

### 4.1 The Attention Weights

We now define the attention mechanism over depth. For each layer $l$, the **attention weight** that layer $l$ assigns to source $i$ is:

$$
\alpha_{i \to l} = \frac{\phi(\boldsymbol{q}_l, \boldsymbol{k}_i)}{\sum_{j=0}^{l-1} \phi(\boldsymbol{q}_l, \boldsymbol{k}_j)}
$$

where $\phi: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\geq 0}$ is a kernel function. The paper uses:

$$
\phi(\boldsymbol{q}, \boldsymbol{k}) = \exp\!\bigl(\boldsymbol{q}^\top \text{RMSNorm}(\boldsymbol{k})\bigr)
$$

This is the standard softmax attention formula — the **softmax function** ensures the weights sum to 1 across all sources. The RMSNorm inside $\phi$ prevents layers with naturally larger outputs from dominating the attention weights.

### 4.2 Queries, Keys, and Values

For each layer $l$, we define:

**Query:** $\boldsymbol{q}_l = \boldsymbol{w}_l$, a learned $d$-dimensional vector specific to layer $l$. This is a **pseudo-query** — it is a parameter, not a function of the hidden state.

**Keys and Values:**

$$
\boldsymbol{k}_i = \boldsymbol{v}_i = \begin{cases} \boldsymbol{h}_1 & i = 0 \\ f_i(\boldsymbol{h}_i) & 1 \leq i \leq l-1 \end{cases}
$$

The keys and values are identical — both are the individual layer outputs. This is a deliberate design choice: the pseudo-query $\boldsymbol{w}_l$ is decoupled from the forward computation, meaning attention weights for all layers in a group can be computed in parallel.

### 4.3 The Full AttnRes Formula

The input to layer $l$ is then:

$$
\boxed{\boldsymbol{h}_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \boldsymbol{v}_i}
$$

Compare this with the standard residual: $\boldsymbol{h}_l = \sum_{i=0}^{l-1} 1 \cdot \boldsymbol{v}_i$. The only change is replacing the fixed coefficient 1 with the learned attention weight $\alpha_{i \to l}$. But because these weights are softmax-normalized and input-dependent, each layer can now selectively emphasize or suppress any previous layer's contribution.

### 4.4 Numerical Example

Let us work through Full AttnRes for our 4-layer scalar example. Since $d = 1$, each query $\boldsymbol{w}_l$ and each key $\boldsymbol{k}_i$ is a scalar. Suppose:

- Layer outputs: $\boldsymbol{v}_0 = 10$, $\boldsymbol{v}_1 = 2$, $\boldsymbol{v}_2 = -1$, $\boldsymbol{v}_3 = 5$
- Pseudo-queries: $w_1 = 0.5$, $w_2 = 0.3$, $w_3 = 0.8$, $w_4 = 0.1$

For simplicity, let us skip the RMSNorm (in $d = 1$ it just normalizes to $\pm 1$) and compute raw $\phi(q, k) = \exp(q \cdot k)$ directly on unnormalized values for illustration.

**Layer 2** attends over sources $\{0, 1\}$ with query $w_2 = 0.3$:

$$
\phi(0.3, 10) = e^{3.0} = 20.09, \quad \phi(0.3, 2) = e^{0.6} = 1.82
$$

$$
\alpha_{0 \to 2} = \frac{20.09}{20.09 + 1.82} = \frac{20.09}{21.91} = 0.917
$$

$$
\alpha_{1 \to 2} = \frac{1.82}{21.91} = 0.083
$$

$$
\boldsymbol{h}_2 = 0.917 \times 10 + 0.083 \times 2 = 9.17 + 0.17 = 9.34
$$

Compare with the standard residual: $\boldsymbol{h}_2 = 10 + 2 = 12$. AttnRes produces a *weighted* combination that sums to a different value — and crucially, the weights are normalized so the magnitude does not grow uncontrollably.

### 4.5 The Depth Mixing Matrix for Full AttnRes

For Full AttnRes, the mixing matrix $\mathbf{M}$ has entries $\mathbf{M}_{i \to l} = \alpha_{i \to l}$:

$$
\mathbf{M}_{\text{Full}} = \begin{bmatrix} \phi(\boldsymbol{w}_1, \boldsymbol{k}_0) & & & \\ \phi(\boldsymbol{w}_2, \boldsymbol{k}_0) & \phi(\boldsymbol{w}_2, \boldsymbol{k}_1) & & \\ \phi(\boldsymbol{w}_3, \boldsymbol{k}_0) & \phi(\boldsymbol{w}_3, \boldsymbol{k}_1) & \phi(\boldsymbol{w}_3, \boldsymbol{k}_2) & \\ \phi(\boldsymbol{w}_4, \boldsymbol{k}_0) & \phi(\boldsymbol{w}_4, \boldsymbol{k}_1) & \phi(\boldsymbol{w}_4, \boldsymbol{k}_2) & \phi(\boldsymbol{w}_4, \boldsymbol{k}_3) \end{bmatrix}
$$

(Each row is normalized by its row sum.) This is a dense, input-dependent, lower-triangular matrix with rank $L$ — the maximum possible. Contrast this with the all-ones matrix of standard residuals.

### 4.6 Overhead and Feasibility

Full AttnRes requires $O(L^2 d)$ computation and $O(Ld)$ memory to store all layer outputs. The computation cost is modest because $L < 1000$ in practice (unlike sequence length which can reach millions). The memory overlaps entirely with activations already retained for backpropagation in standard training.

However, at scale with pipeline parallelism and activation recomputation, every layer output must be kept alive and transmitted across pipeline stages, making the $O(Ld)$ memory and communication prohibitive. This motivates the Block variant.

---

## 5. Block Attention Residuals

### 5.1 The Idea: Compress Within Blocks, Attend Across Blocks

**Block Attention Residuals** (Block AttnRes) partition the $L$ layers into $N$ blocks of $S = L/N$ layers each. Within each block, layer outputs are accumulated via standard summation. Across blocks, we apply full attention over the $N$ block-level representations.

This reduces memory from $O(Ld)$ to $O(Nd)$ and computation from $O(L^2)$ to $O(N^2)$.

### 5.2 Intra-Block Accumulation

We divide our $L = 4$ layers into $N = 2$ blocks of $S = 2$ layers each. Let $\mathcal{B}_1 = \{1, 2\}$ and $\mathcal{B}_2 = \{3, 4\}$. The **block representation** is the sum of layer outputs within the block:

$$
\boldsymbol{b}_n = \sum_{j \in \mathcal{B}_n} f_j(\boldsymbol{h}_j)
$$

For our example:

$$
\boldsymbol{b}_1 = \boldsymbol{v}_1 + \boldsymbol{v}_2 = 2 + (-1) = 1
$$

$$
\boldsymbol{b}_2 = \boldsymbol{v}_3 + \boldsymbol{v}_4
$$

We also track **partial sums** within each block. Define $\boldsymbol{b}_n^i$ as the partial sum over the first $i$ layers in block $n$:

$$
\boldsymbol{b}_1^1 = \boldsymbol{v}_1 = 2, \quad \boldsymbol{b}_1^2 = \boldsymbol{v}_1 + \boldsymbol{v}_2 = 1 = \boldsymbol{b}_1
$$

### 5.3 Inter-Block Attention

For the first layer in block $n$, the value matrix consists of all previous block representations plus the embedding $\boldsymbol{b}_0 = \boldsymbol{h}_1$:

$$
\mathbf{V} = [\boldsymbol{b}_0, \boldsymbol{b}_1, \ldots, \boldsymbol{b}_{n-1}]^\top
$$

For subsequent layers within block $n$, we additionally include the current block's partial sum $\boldsymbol{b}_n^{i-1}$:

$$
\mathbf{V} = [\boldsymbol{b}_0, \boldsymbol{b}_1, \ldots, \boldsymbol{b}_{n-1}, \boldsymbol{b}_n^{i-1}]^\top
$$

Keys and attention weights follow the same kernel $\phi$ from Eq. 2 and Eq. 3 (Section 4.1–4.2), with the block representations serving as both keys and values.

### 5.4 Walking Through Block AttnRes

For our $L = 4$, $N = 2$ example with $\boldsymbol{b}_0 = \boldsymbol{v}_0 = 10$ and $\boldsymbol{b}_1 = 1$:

**Layer 3** (first layer of block 2) attends over $\{\boldsymbol{b}_0, \boldsymbol{b}_1\} = \{10, 1\}$:

$$
\boldsymbol{h}_3 = \alpha_{0 \to 3} \cdot 10 + \alpha_{1 \to 3} \cdot 1
$$

**Layer 4** (second layer of block 2) attends over $\{\boldsymbol{b}_0, \boldsymbol{b}_1, \boldsymbol{b}_2^1\}$ where $\boldsymbol{b}_2^1 = \boldsymbol{v}_3 = f_3(\boldsymbol{h}_3)$. So it sees $N + 1 = 3$ sources instead of $N = 2$, gaining one extra source for the intra-block partial sum.

### 5.5 The Block AttnRes Mixing Matrix

For our $L = 4$, $N = 2$ example, the mixing matrix is:

$$
\mathbf{M}_{\text{Block}} = \begin{bmatrix} \phi(\boldsymbol{w}_1, \boldsymbol{k}_0) & & & \\ \phi(\boldsymbol{w}_2, \boldsymbol{k}_0) & \phi(\boldsymbol{w}_2, \boldsymbol{k}_1) & & \\ \phi(\boldsymbol{w}_3, \boldsymbol{k}_0) & \phi(\boldsymbol{w}_3, \boldsymbol{k}_1 + \boldsymbol{k}_2) & & \\ \phi(\boldsymbol{w}_4, \boldsymbol{k}_0) & \phi(\boldsymbol{w}_4, \boldsymbol{k}_1 + \boldsymbol{k}_2) & \phi(\boldsymbol{w}_4, \boldsymbol{k}_3) & \end{bmatrix}
$$

Notice the key difference from Full AttnRes: layers within a completed block share the same combined key $\boldsymbol{k}_1 + \boldsymbol{k}_2$. The individual layer-level granularity is lost in exchange for a dramatic reduction in the number of sources from $L$ to approximately $N + S$.

### 5.6 Interpolating Between Extremes

The block count $N$ controls a smooth interpolation:

- $N = L$: Each block has one layer ($S = 1$). Every layer output is its own block. This recovers **Full AttnRes**.
- $N = 1$: All layers are in one block. Intra-block summation reduces to standard addition. This recovers **standard residual connections** with the embedding isolated as $\boldsymbol{b}_0$.

Empirically, $N \approx 8$ recovers most of the gain of Full AttnRes. For the 48B-parameter Kimi Linear model with 54 layers, Block AttnRes uses 6 layers per block, producing 9 blocks plus the token embedding for a total of 10 depth-wise sources.

---

## 6. The Two-Phase Computation Strategy

A naive implementation of Block AttnRes would compute attention at every layer, each requiring a full pass over all preceding blocks — resulting in $O(L \cdot N)$ total memory accesses. The paper introduces a two-phase strategy that exploits a key property: the pseudo-queries $\boldsymbol{w}_l$ are learned parameters decoupled from the forward computation.

### 6.1 Phase 1: Parallel Inter-Block Attention

Because the pseudo-queries are parameters (not functions of $\boldsymbol{h}_l$), we can batch all $S$ queries within a block and compute their attention over all previous blocks simultaneously:

$$
\mathbf{Q} = [\boldsymbol{w}_l]_{l \in \mathcal{B}_n} \in \mathbb{R}^{S \times d}, \quad \mathbf{K} = \mathbf{V} = [\boldsymbol{b}_0; \ldots; \boldsymbol{b}_{n-1}] \in \mathbb{R}^{n \times d}
$$

This single batched attention call returns, for each layer $l$ in the block, the inter-block attention output $\boldsymbol{o}_l^{(1)}$ along with its softmax statistics (the max $m_l^{(1)}$ and log-sum-exp $\ell_l^{(1)}$). This amortizes the memory reads from $S$ reads down to 1 read per block.

### 6.2 Phase 2: Sequential Intra-Block Attention with Online Softmax Merge

Phase 2 processes layers sequentially within the block. For each layer $l$ (after the first in the block), it computes intra-block attention over the evolving partial sum $\boldsymbol{b}_n^{i-1}$, obtaining output $\boldsymbol{o}_l^{(2)}$ with statistics $m_l^{(2)}$ and $\ell_l^{(2)}$.

The two sets of attention outputs are then merged using the **online softmax** algorithm. This is a numerically stable method for combining two softmax computations that were performed independently. The merge computes:

$$
m_l = \max(m_l^{(1)}, m_l^{(2)})
$$

$$
\boldsymbol{h}_l = \frac{e^{m_l^{(1)} - m_l} \cdot \boldsymbol{o}_l^{(1)} + e^{m_l^{(2)} - m_l} \cdot \boldsymbol{o}_l^{(2)}}{e^{m_l^{(1)} - m_l} \cdot \ell_l^{(1)} + e^{m_l^{(2)} - m_l} \cdot \ell_l^{(2)}}
$$

This is mathematically equivalent to computing softmax over all sources jointly, but avoids materializing the full attention matrix. The subtraction of $m_l$ from both exponents prevents numerical overflow — this is the same **log-sum-exp trick** used in FlashAttention and standard stable softmax implementations.

### 6.3 Numerical Check of Online Softmax Merge

Suppose Phase 1 gives $\boldsymbol{o}^{(1)} = 8.5$ with $m^{(1)} = 3.0$, $\ell^{(1)} = 25.0$, and Phase 2 gives $\boldsymbol{o}^{(2)} = 2.0$ with $m^{(2)} = 1.5$, $\ell^{(2)} = 5.0$.

$$
m = \max(3.0, 1.5) = 3.0
$$

$$
\boldsymbol{h} = \frac{e^{3.0 - 3.0} \times 8.5 + e^{1.5 - 3.0} \times 2.0}{e^{3.0 - 3.0} \times 25.0 + e^{1.5 - 3.0} \times 5.0} = \frac{1.0 \times 8.5 + 0.223 \times 2.0}{1.0 \times 25.0 + 0.223 \times 5.0} = \frac{8.5 + 0.446}{25.0 + 1.115} = \frac{8.946}{26.115} = 0.3425
$$

The key property: this gives the same result as if we had computed softmax attention over all sources jointly. The online merge introduces zero approximation error.

### 6.4 Memory Access Cost

The total per-layer memory access cost for Block AttnRes with the two-phase strategy is:

| | Read | Write |
|---|---|---|
| Phase 1 (amortized) | $\frac{N}{S}d$ | $d$ |
| Phase 2 | $3d$ | $d$ |
| **Total** | $(\frac{N}{S} + 3)d$ | $2d$ |

With typical values $L = 128$, $N = 8$, $S = 16$: total reads = $(\frac{8}{16} + 3)d = 3.5d$, total writes = $2d$, for a grand total of $5.5d$. Compare this with standard residuals at $3d$ — the overhead is modest. The end-to-end inference latency overhead is less than 2% on typical workloads.

---

## 7. The Structured-Matrix View: Unifying All Residual Variants

This is the part that ties everything together. We have seen that standard residuals, Highway networks, Hyper-Connections, and AttnRes all compute $\boldsymbol{h}_l = \sum_{i=0}^{l-1} \mathbf{M}_{i \to l} \cdot \boldsymbol{v}_i$ for different choices of the depth mixing matrix $\mathbf{M}$. The paper formalizes this and shows that the variants differ in three properties: whether the weights are fixed or learned, whether they are input-dependent, and the **semiseparable rank** of $\mathbf{M}$.

### 7.1 Standard Residuals: All-Ones Matrix

As derived in Section 1.4:

$$
\mathbf{M}_{\text{Residual}} = \begin{bmatrix} 1 \\ 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \end{bmatrix}
$$

Weights: fixed. Input-dependent: no. The matrix has the simplest possible structure.

### 7.2 Highway Networks: Gated Carry Products

**Highway networks** introduce element-wise gates $g_l \in [0, 1]^d$ that interpolate between the identity path and the transformation:

$$
\boldsymbol{h}_l = (1 - g_l) \odot \boldsymbol{h}_{l-1} + g_l \odot f_{l-1}(\boldsymbol{h}_{l-1})
$$

For scalar clarity, define the **carry product** $\gamma_{i \to l}^\times := \prod_{j=i+1}^{l} (1 - g_j)$. This represents how much of source $i$'s output survives through all subsequent gates to reach layer $l$. The mixing matrix entries are:

$$
\mathbf{M}_{0 \to l} = \gamma_{1 \to l}^\times, \quad \mathbf{M}_{i \to l} = g_{i+1} \cdot \gamma_{i+1 \to l}^\times \text{ for } i \geq 1
$$

### 7.3 Numerical Check of Highway Carry Products

Let $g_1 = 0.3$, $g_2 = 0.5$, $g_3 = 0.2$, $g_4 = 0.6$. Compute the carry products for layer 4:

$$
\gamma_{1 \to 4}^\times = (1 - 0.2)(1 - 0.5)(1 - 0.6) = 0.8 \times 0.5 \times 0.4 = 0.16
$$

Wait — we need to be careful about indexing. Let us redo this with the paper's convention. The gate $g_l$ is applied at layer $l$. The carry product from source $i$ to layer $l$ is:

$$
\gamma_{i \to l}^\times = \prod_{j=i+1}^{l} (1 - g_j)
$$

For $\mathbf{M}_{0 \to 4}$ (embedding reaching layer 4):

$$
\gamma_{1 \to 4}^\times = (1 - g_2)(1 - g_3)(1 - g_4) = 0.5 \times 0.8 \times 0.4 = 0.16
$$

The Highway mixing matrix for our $L = 4$ example:

$$
\mathbf{M}_{\text{Highway}} = \begin{bmatrix} 1 \\ \gamma_{1\to2}^\times & g_2 \\ \gamma_{1\to3}^\times & g_2 \cdot \gamma_{2\to3}^\times & g_3 \\ \gamma_{1\to4}^\times & g_2 \cdot \gamma_{2\to4}^\times & g_3 \cdot \gamma_{3\to4}^\times & g_4 \end{bmatrix}
$$

The key structural property: since the cumulative products factor through scalar gates, $\mathbf{M}$ is **1-semiseparable** — the same rank as the standard residual, but with input-dependent weights. The weights sum to 1 by construction (each row partitions probability mass between "carry" and "transform"), making Highway a softmax-free, depth-wise instance of stick-breaking attention.

### 7.4 (m)Hyper-Connections: Multi-Stream Matrices

**Hyper-Connections** (HC) and their manifold-constrained variant **mHC** widen the recurrence to $m$ parallel streams. The update is:

$$
\mathbf{H}_l = \mathbf{H}_{l-1} \mathbf{A}_l + f_{l-1}(\mathbf{H}_{l-1} \boldsymbol{\alpha}_{l-1}) \boldsymbol{\beta}_{l-1}^\top
$$

where $\mathbf{A}_l \in \mathbb{R}^{m \times m}$ is a learned transition matrix, $\boldsymbol{\alpha}_{l-1} \in \mathbb{R}^m$ mixes streams into a single input for $f_{l-1}$, and $\boldsymbol{\beta}_{l-1} \in \mathbb{R}^m$ distributes the output back across streams.

Unrolling this recurrence gives:

$$
\mathbf{M}_{i \to l} = \boldsymbol{\beta}_i^\top \mathbf{A}_{i+1 \to l}^\times \boldsymbol{\alpha}_l
$$

where $\mathbf{A}_{i+1 \to l}^\times := \prod_{k=i+1}^{l} \mathbf{A}_k$ is the **cumulative matrix product** of transitions. The $m \times m$ transitions render $\mathbf{M}$ $m$-semiseparable. mHC further constrains each $\mathbf{A}_l$ to be **doubly stochastic** (by the **Birkhoff–von Neumann theorem**, every doubly stochastic matrix is a convex combination of permutation matrices), stabilizing the cumulative products across depth.

### 7.5 Full AttnRes: Dense, Input-Dependent

Full AttnRes computes:

$$
\mathbf{M}_{i \to l} = \alpha_{i \to l} = \frac{\phi(\boldsymbol{w}_l, \boldsymbol{k}_i)}{\sum_j \phi(\boldsymbol{w}_l, \boldsymbol{k}_j)}
$$

where $\boldsymbol{k}_i = \boldsymbol{v}_i$ are the layer outputs. This yields a dense, rank-$L$ lower-triangular matrix. Every entry is input-dependent (through the keys) and the weights are softmax-normalized.

### 7.6 Block AttnRes: Controlled Rank

Block AttnRes shares weights within completed blocks: for all $i$ in a completed block $\mathcal{B}_n$, $\mathbf{M}_{i \to l} = \alpha_{n \to l}$ (the same weight for all sources in the block). Within the current block, each layer additionally attends to the evolving partial sum $\boldsymbol{b}_n^{i-1}$. The effective rank of $\mathbf{M}$ lies between $N$ and $N + S$, interpolating between standard residuals ($N = 1$) and Full AttnRes ($N = L$).

### 7.7 The Spectrum

We can now arrange all variants along a spectrum of increasing expressiveness:

| Method | Weight type | Input-dependent? | Rank of $\mathbf{M}$ |
|---|---|---|---|
| Standard Residual | Fixed (all 1s) | No | 1 |
| Highway | Learned (gates) | Yes | 1 |
| (m)HC | Learned (matrices) | Yes | $m$ |
| Block AttnRes | Learned (softmax) | Yes | $N$ to $N+S$ |
| Full AttnRes | Learned (softmax) | Yes | $L$ |

The insight is that these are not separate inventions — they are points on a single axis of increasing rank in the depth mixing matrix, with AttnRes at the maximum.

---

## 8. Prior Residuals as Depth-Wise Linear Attention

This section makes the time-depth duality from Section 3 mathematically precise.

### 8.1 The (m)HC Weight as Linear Attention

Recall the unrolled (m)HC weight from Section 7.4:

$$
\mathbf{M}_{i \to l} = \boldsymbol{\beta}_i^\top \mathbf{A}_{i+1 \to l}^\times \boldsymbol{\alpha}_l
$$

This admits a natural interpretation as **linear attention** over depth. The vector $\boldsymbol{\alpha}_l$ plays the role of a query issued by layer $l$. The vector $\boldsymbol{\beta}_i$ serves as a key summarizing the contribution of layer $i$. The cumulative transition $\mathbf{A}_{i+1 \to l}^\times$ acts as a depth-relative positional operator governing the query-key interaction across intervening layers.

The $m$ parallel streams correspond to state expansion along the depth axis, expanding the recurrent state from $d$ to $d \times m$. This is directly analogous to how multi-head attention expands representation capacity along the sequence axis.

### 8.2 From Linear to Softmax Attention Over Depth

Standard residuals and Highway networks perform depth-wise attention with rank-1 matrices — the simplest case. (m)HC extends this to rank-$m$ linear attention. AttnRes goes further and replaces the linear kernel with softmax normalization via the kernel $\phi(\boldsymbol{q}, \boldsymbol{k}) = \exp(\boldsymbol{q}^\top \text{RMSNorm}(\boldsymbol{k}))$.

This is the same transition that took RNNs (linear attention with state compression) to Transformers (softmax attention with direct access) — but applied to the depth dimension rather than the sequence dimension.

---

## 9. Initialization and Training Dynamics

### 9.1 Zero Initialization of Pseudo-Queries

A critical implementation detail: all pseudo-query vectors $\boldsymbol{w}_l$ must be initialized to zero. When $\boldsymbol{w}_l = \boldsymbol{0}$ for all $l$:

$$
\phi(\boldsymbol{0}, \boldsymbol{k}_i) = \exp(\boldsymbol{0}^\top \text{RMSNorm}(\boldsymbol{k}_i)) = \exp(0) = 1
$$

for all sources $i$. This means:

$$
\alpha_{i \to l} = \frac{1}{l} \quad \text{for all } i
$$

At initialization, every layer assigns equal weight to all previous sources — AttnRes starts as a uniform average, then learns to specialize during training. This prevents training volatility from random initial attention patterns.

### 9.2 How AttnRes Fixes PreNorm Dilution

Recall the dilution problem from Section 2.1: in standard residuals, $\|\boldsymbol{h}_l\|$ grows as $O(L)$ because every layer output is added with weight 1. With AttnRes, the weights $\alpha_{i \to l}$ sum to 1 by the definition of softmax. The hidden state is a *convex combination* of previous outputs rather than their sum:

$$
\|\boldsymbol{h}_l\| = \left\|\sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \boldsymbol{v}_i\right\| \leq \sum_{i=0}^{l-1} \alpha_{i \to l} \|\boldsymbol{v}_i\| \leq \max_i \|\boldsymbol{v}_i\|
$$

The last inequality follows from the **triangle inequality** and the fact that $\sum_i \alpha_{i \to l} = 1$ (this is the **convexity** of the weighted average). The hidden state magnitude is bounded by the largest individual layer output, not their cumulative sum.

For Block AttnRes, the selective aggregation resets at block boundaries, confining the growth within each block. The paper shows empirically that this yields a bounded periodic pattern in output magnitudes — a dramatic improvement over the monotonic growth of the baseline.

### 9.3 Gradient Distribution

With standard residuals, the gradient with respect to an intermediate hidden state is:

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_l} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_L} \cdot \prod_{j=l}^{L-1} \left(\mathbf{I} + \frac{\partial f_j}{\partial \boldsymbol{h}_j}\right)
$$

All residual weights are fixed at 1, so there is no mechanism to regulate gradient flow across depth. This leads to disproportionately large gradients in the earliest layers.

With AttnRes, the learnable softmax weights introduce competition among sources for probability mass. This naturally distributes gradients more uniformly across depth — the paper shows substantially more uniform gradient magnitudes across transformer blocks compared to the baseline.

---

## 10. Learned Attention Patterns

### 10.1 What the Network Actually Learns

The paper visualizes the learned weights $\alpha_{i \to l}$ for a 16-head model with both Full and Block AttnRes. Three patterns emerge:

**Preserved locality.** Each layer attends most strongly to its immediate predecessor — the diagonal of the attention matrix dominates. This makes sense: the standard residual (which is purely local) works well, so the learned weights should stay close to it unless there is reason to deviate.

**Selective long-range connections.** Despite the diagonal dominance, selective off-diagonal concentrations emerge. For example, layer 4 attending to early sources, or layers 15–16 reaching back to the first few layers. These are learned skip connections beyond the standard residual path.

**Embedding persistence.** The token embedding $\boldsymbol{h}_1$ (source 0) retains non-trivial weight throughout the network, especially in pre-attention layers. This is consistent with the embedding carrying fundamental token identity information that remains relevant at all depths.

### 10.2 Attention vs. MLP Specialization

Pre-attention inputs show broader receptive fields (attending to sources across a wider range of depths), while pre-MLP inputs show sharper diagonal reliance on recent representations. This specialization is consistent with attention layers operating more globally across depth while MLPs refine locally — a distinction that standard residuals cannot express.

---

## 11. Experimental Results

### 11.1 Scaling Laws

The paper trains five model sizes (194M to 528M activated parameters) with three variants each: PreNorm baseline, Full AttnRes, and Block AttnRes ($N \approx 8$).

Fitting power-law curves $\mathcal{L} = A \times C^{-\alpha}$ (where $C$ is compute in PFLOP/s-days):

- Baseline: $\mathcal{L} = 1.891 \times C^{-0.057}$
- Block AttnRes: $\mathcal{L} = 1.870 \times C^{-0.058}$
- Full AttnRes: $\mathcal{L} = 1.865 \times C^{-0.057}$

All three variants share similar scaling exponents, but AttnRes achieves consistently lower loss across the entire compute range. At the largest scale (5.6 PFLOP/s-days), Block AttnRes reaches 1.692 versus the baseline's 1.714 — equivalent to a $1.25\times$ compute advantage.

### 11.2 48B Parameter Model

The full-scale model uses Block AttnRes on the Kimi Linear architecture (48B total / 3B activated parameters), pre-trained on 1.4T tokens. Results across 16 benchmarks:

- **General knowledge:** MMLU +1.1, BBH +1.7, GPQA-Diamond +7.5
- **Math and code:** GSM8K +0.7, Math +3.6, CMath +0.4, HumanEval +3.1, MBPP +1.9
- **Chinese:** CMMLU +0.9, C-Eval +2.9

Block AttnRes matches or outperforms the baseline on every benchmark. The improvements are particularly pronounced on multi-step reasoning tasks (GPQA-Diamond, Math), consistent with the hypothesis that improved depth-wise information flow benefits compositional tasks where later layers need to selectively retrieve and build upon earlier representations.

### 11.3 Ablation Highlights

Key ablation findings on a 16-layer model:

| Variant | Loss |
|---|---|
| Baseline (PreNorm) | 1.766 |
| DenseFormer (fixed cross-layer weights) | 1.767 |
| mHC ($m$ streams) | 1.747 |
| Full AttnRes | 1.737 |
| Block AttnRes ($S = 4$) | 1.746 |
| w/ input-dependent query | 1.731 |
| w/ input-independent mixing | 1.749 |
| w/ sigmoid instead of softmax | 1.741 |
| w/o RMSNorm on keys | 1.743 |

DenseFormer grants cross-layer access but with fixed, input-independent coefficients — it shows no gain over the baseline, highlighting the importance of input-dependent weighting. Replacing softmax with sigmoid degrades performance, which the paper attributes to softmax's competitive normalization forcing sharper selection among sources. Removing RMSNorm on keys degrades both Full and Block AttnRes, confirming that preventing large-magnitude layers from dominating the attention weights is essential.

---

## 12. Summary

Standard residual connections accumulate all previous layer outputs with fixed unit weights, producing an all-ones depth mixing matrix that offers zero selectivity and causes hidden-state magnitudes to grow as $O(L)$ under PreNorm — the dilution problem. Attention Residuals replace this fixed accumulation with learned softmax attention over depth, where each layer uses a single pseudo-query vector to selectively weight all previous layer outputs, yielding a dense, input-dependent mixing matrix with maximum rank. Block AttnRes makes this practical at scale by compressing layers into $N$ blocks with standard summation within blocks and full attention across block representations, reducing memory from $O(Ld)$ to $O(Nd)$ while recovering most of the gain with $N \approx 8$ — a two-phase computation strategy with online softmax merging keeps inference overhead below 2%.

---

*Previous: [Mixture of Experts from Scratch — Part 2](/blog/mixture-of-experts-part-2)*  
*Next: [Mathematical Prerequisites for Mixture of Experts — Part 3](/blog/math-prerequisites-for-mixture-of-experts-part-3)*
