---
title: "DeepSeek Sparse Attention: Learned Token Selection from Scratch"
description: "Letting the model choose its own sparse pattern — a lightweight indexer scores every past token, top-k selects the keys that matter for this query, and two-stage training with KL alignment keeps the sparse model faithful to its dense teacher at $O(nk)$ cost."
date: 2026-04-07
tags: [machine-learning, attention, transformers, sparse-attention, deepseek, efficiency]
order: 6
---

The previous two blogs derived two approaches to sparse attention, both living on the same axis: reduce the number of entries each token attends to.

The Sparse Transformer (the Sparse Factorization blog) used fixed factorized patterns — strided and fixed — achieving $O(n\sqrt{n})$. The patterns were rigid: which tokens attend to which was determined entirely by position, independent of content.

Longformer (the Sliding Window blog) used sliding windows plus global attention, achieving $O(n)$. The windows were local and fixed-width, the global tokens were chosen by task (like [CLS] for classification), not by content. This was simpler and faster, but still static — token 7 always attends to tokens 5 through 9 regardless of what those tokens contain.

Both approaches make the same fundamental bet: the *structure* of attention can be decided in advance. Neither approach asks the question: "given what this token actually says, which other tokens in the context are most relevant to it?"

DeepSeek Sparse Attention (DSA), introduced in the DeepSeek-V3.2 paper (DeepSeek-AI, 2025), takes the next step. It learns a lightweight scoring network — the **lightning indexer** — that reads each query token and every preceding token, scores their relevance, and selects only the top-$k$ most relevant tokens for attention. The attention pattern is no longer fixed by position or task. It is determined dynamically, at inference time, by the content of the tokens themselves.

The result: attention complexity drops from $O(n^2)$ to $O(nk)$ where $k \ll n$ is a fixed budget of selected tokens, with no degradation in model quality on standard benchmarks.

We will derive the entire mechanism from scratch, trace every computation by hand, and verify every count numerically.

---

## The Running Example

We continue with the same sequence of $n = 16$ tokens from the previous blogs:

$$
\underbrace{x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_{10}, x_{11}, x_{12}, x_{13}, x_{14}, x_{15}}_{n = 16 \text{ tokens}}
$$

Since DSA is designed for autoregressive (causal) models, each token $t$ can only attend to preceding tokens $s < t$ (plus itself). We focus on computing the attention output for **query token $t = 10$**, which has 10 preceding tokens to choose from: positions 0 through 9.

We fix the following DSA-specific parameters:

- Indexer dimension: $d^I = 2$ (the dimension of indexer query and key vectors)
- Indexer heads: $H^I = 2$ (number of heads in the lightning indexer)
- Selection budget: $k = 3$ (number of tokens selected for attention)

For the main attention mechanism, the same model as always:

- $d_\text{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$, $L = 12$ layers, fp16

In the real DeepSeek-V3.2 deployment, $k = 2{,}048$ tokens are selected from contexts of up to $128{,}000$ tokens. We use $k = 3$ out of 10 preceding tokens so that we can trace every computation by hand.

---

## 1. The Problem with Fixed Sparse Patterns

### 1.1 What the Sparse Transformer and Longformer assume

Both previous approaches define the connectivity set $S_i$ — the set of tokens that position $i$ attends to — using a fixed rule based on position alone:

**Sparse Transformer (Sparse Factorization blog):** $S_i$ is determined by strided or fixed factorization patterns. Token $i$ attends to tokens at positions $\{i, i - 1, i - 2, \ldots\}$ within its stride group and to designated summary positions. The rule depends on $i \bmod \sqrt{n}$.

**Longformer (Sliding Window blog):** $S_i = \{j : |i - j| \leq r\} \cup G$, where $r$ is the window radius and $G$ is a fixed set of global positions. The rule depends on $|i - j|$ and whether $j \in G$.

In both cases, $S_i$ is the same regardless of what the tokens contain. If token 5 is the word "the" or the word "catastrophe," it attends to exactly the same set of positions.

### 1.2 Why this wastes capacity

Real attention patterns in trained transformers are not uniform. Empirically, attention is highly concentrated: for a given query, a small number of key-value pairs receive the vast majority of attention weight, and the rest receive near-zero weight. Clark et al. (2019) and Kovaleva et al. (2019) documented this extensively.

A sliding window of width $w = 512$ computes attention scores for all 512 positions in the window, but many of those scores are near zero after softmax. The computation spent on near-zero entries is wasted.

The ideal approach: for each query, identify the small number of tokens that *would* receive high attention weight under full attention, attend only to those, and skip the rest. This is what DSA does.

### 1.3 The design challenge

The challenge is circular: to know which tokens would receive high attention weight, we need to compute the attention scores — but computing all attention scores is exactly what we are trying to avoid.

DSA resolves this circularity with a two-component design:

1. A **lightning indexer** — a small, cheap scoring network that approximates the full attention distribution
2. A **top-$k$ selector** — picks the highest-scoring tokens according to the indexer

The indexer is much cheaper than full attention, so the total cost (indexer + sparse attention on selected tokens) is less than the cost of full dense attention.

---

## 2. The Lightning Indexer

### 2.1 Definition

The **lightning indexer** is a small multi-head network that computes an **index score** $I_{t,s}$ between a query token at position $t$ and each preceding token at position $s$. This score estimates how relevant token $s$ is to token $t$.

The formula is:

$$
\boxed{I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}\!\left(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I\right)}
$$

where:

- $H^I$ is the number of **indexer heads** (a small number, independent of the main attention heads $h$)
- $\mathbf{q}_{t,j}^I \in \mathbb{R}^{d^I}$ is the **indexer query vector** for token $t$ at indexer head $j$, derived from the query token's hidden state $\mathbf{h}_t$
- $\mathbf{k}_s^I \in \mathbb{R}^{d^I}$ is the **indexer key vector** for token $s$, derived from the preceding token's hidden state $\mathbf{h}_s$
- $w_{t,j}^I \in \mathbb{R}$ is a **scalar weight** for indexer head $j$ at query position $t$, also derived from $\mathbf{h}_t$
- $\text{ReLU}(x) = \max(0, x)$ is the **rectified linear unit**

The dot product $\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I$ measures the alignment between query $t$ and key $s$ in indexer head $j$. The ReLU clips negative alignments to zero — if a token is anti-aligned with the query, it contributes nothing. The scalar weights $w_{t,j}^I$ allow the model to weight different indexer heads differently for each query position.

### 2.2 Why ReLU instead of softmax

The main attention mechanism uses softmax to normalize scores into a probability distribution. The indexer uses ReLU instead. Why?

The indexer does not need normalized scores. Its only job is to produce a ranking — which tokens have the highest scores — so that the top-$k$ selector can pick them. For ranking, unnormalized scores suffice. ReLU is cheaper to compute than softmax (no exponentiation, no summation over all positions), and it can be implemented efficiently in low-precision arithmetic (FP8). Since the indexer runs over every query-key pair — the same $O(n^2)$ number of pairs as full attention — making each operation as cheap as possible is critical.

### 2.3 Tracing for our running example

We compute $I_{10,s}$ for all preceding tokens $s \in \{0, 1, \ldots, 9\}$.

We use $H^I = 2$ indexer heads and $d^I = 2$. Here are the concrete vectors:

**Indexer head 1** ($j = 1$): $\mathbf{q}_{10,1}^I = [1, 2]$, $w_{10,1}^I = 0.5$

**Indexer head 2** ($j = 2$): $\mathbf{q}_{10,2}^I = [2, -1]$, $w_{10,2}^I = 0.3$

**Indexer key vectors** (shared across heads, derived from each preceding token's hidden state):

| Token $s$ | $\mathbf{k}_s^I$ |
|---|---|
| 0 | $[1, 0]$ |
| 1 | $[0, 1]$ |
| 2 | $[-1, 1]$ |
| 3 | $[2, 1]$ |
| 4 | $[0, -1]$ |
| 5 | $[1, 1]$ |
| 6 | $[-1, 0]$ |
| 7 | $[1, -1]$ |
| 8 | $[0, 2]$ |
| 9 | $[2, 0]$ |

### 2.4 Step-by-step computation

**Head 1** ($\mathbf{q} = [1, 2]$, $w = 0.5$):

For each token $s$, we compute the dot product $\mathbf{q}_{10,1}^I \cdot \mathbf{k}_s^I$, apply ReLU, then multiply by $w_{10,1}^I$:

| $s$ | $\mathbf{k}_s^I$ | $\mathbf{q} \cdot \mathbf{k}_s^I$ | $\text{ReLU}(\cdot)$ | $w \cdot \text{ReLU}(\cdot)$ |
|---|---|---|---|---|
| 0 | $[1, 0]$ | $1 \cdot 1 + 2 \cdot 0 = 1$ | $1$ | $0.5$ |
| 1 | $[0, 1]$ | $1 \cdot 0 + 2 \cdot 1 = 2$ | $2$ | $1.0$ |
| 2 | $[-1, 1]$ | $1 \cdot (-1) + 2 \cdot 1 = 1$ | $1$ | $0.5$ |
| 3 | $[2, 1]$ | $1 \cdot 2 + 2 \cdot 1 = 4$ | $4$ | $2.0$ |
| 4 | $[0, -1]$ | $1 \cdot 0 + 2 \cdot (-1) = -2$ | $0$ | $0$ |
| 5 | $[1, 1]$ | $1 \cdot 1 + 2 \cdot 1 = 3$ | $3$ | $1.5$ |
| 6 | $[-1, 0]$ | $1 \cdot (-1) + 2 \cdot 0 = -1$ | $0$ | $0$ |
| 7 | $[1, -1]$ | $1 \cdot 1 + 2 \cdot (-1) = -1$ | $0$ | $0$ |
| 8 | $[0, 2]$ | $1 \cdot 0 + 2 \cdot 2 = 4$ | $4$ | $2.0$ |
| 9 | $[2, 0]$ | $1 \cdot 2 + 2 \cdot 0 = 2$ | $2$ | $1.0$ |

**Head 2** ($\mathbf{q} = [2, -1]$, $w = 0.3$):

| $s$ | $\mathbf{k}_s^I$ | $\mathbf{q} \cdot \mathbf{k}_s^I$ | $\text{ReLU}(\cdot)$ | $w \cdot \text{ReLU}(\cdot)$ |
|---|---|---|---|---|
| 0 | $[1, 0]$ | $2 \cdot 1 + (-1) \cdot 0 = 2$ | $2$ | $0.6$ |
| 1 | $[0, 1]$ | $2 \cdot 0 + (-1) \cdot 1 = -1$ | $0$ | $0$ |
| 2 | $[-1, 1]$ | $2 \cdot (-1) + (-1) \cdot 1 = -3$ | $0$ | $0$ |
| 3 | $[2, 1]$ | $2 \cdot 2 + (-1) \cdot 1 = 3$ | $3$ | $0.9$ |
| 4 | $[0, -1]$ | $2 \cdot 0 + (-1) \cdot (-1) = 1$ | $1$ | $0.3$ |
| 5 | $[1, 1]$ | $2 \cdot 1 + (-1) \cdot 1 = 1$ | $1$ | $0.3$ |
| 6 | $[-1, 0]$ | $2 \cdot (-1) + (-1) \cdot 0 = -2$ | $0$ | $0$ |
| 7 | $[1, -1]$ | $2 \cdot 1 + (-1) \cdot (-1) = 3$ | $3$ | $0.9$ |
| 8 | $[0, 2]$ | $2 \cdot 0 + (-1) \cdot 2 = -2$ | $0$ | $0$ |
| 9 | $[2, 0]$ | $2 \cdot 2 + (-1) \cdot 0 = 4$ | $4$ | $1.2$ |

### 2.5 Combining heads

The total index score for each token $s$ is the sum across both heads:

$$
I_{10,s} = w_{10,1}^I \cdot \text{ReLU}(\mathbf{q}_{10,1}^I \cdot \mathbf{k}_s^I) + w_{10,2}^I \cdot \text{ReLU}(\mathbf{q}_{10,2}^I \cdot \mathbf{k}_s^I)
$$

| Token $s$ | Head 1 contribution | Head 2 contribution | $I_{10,s}$ |
|---|---|---|---|
| 0 | $0.5$ | $0.6$ | $1.1$ |
| 1 | $1.0$ | $0$ | $1.0$ |
| 2 | $0.5$ | $0$ | $0.5$ |
| 3 | $2.0$ | $0.9$ | $\mathbf{2.9}$ |
| 4 | $0$ | $0.3$ | $0.3$ |
| 5 | $1.5$ | $0.3$ | $1.8$ |
| 6 | $0$ | $0$ | $0$ |
| 7 | $0$ | $0.9$ | $0.9$ |
| 8 | $2.0$ | $0$ | $\mathbf{2.0}$ |
| 9 | $1.0$ | $1.2$ | $\mathbf{2.2}$ |

### 2.6 Interpretation

The index scores are content-dependent. Token 3 scores highest ($2.9$) because its key vector $[2, 1]$ aligns well with both indexer heads. Token 6 scores zero because its key vector $[-1, 0]$ is anti-aligned with both indexer queries — ReLU clips both dot products to zero.

This is the fundamental difference from all previous sparse attention methods. In the Sparse Transformer and Longformer, token 6 might or might not be in the connectivity set, but that decision was based on its position, not its content. Here, token 6 is excluded because the indexer determined — based on the actual hidden states — that it is irrelevant to the current query.

Notice that each head captures different relevance patterns. Head 1 (query $[1, 2]$) favors tokens with large second components — tokens 3 and 8, which have key vectors $[2, 1]$ and $[0, 2]$. Head 2 (query $[2, -1]$) favors tokens with large first components and small or negative second components — tokens 9 and 7, with key vectors $[2, 0]$ and $[1, -1]$. The multi-head structure lets the indexer capture multiple notions of relevance simultaneously.

---

## 3. Top-$k$ Token Selection

### 3.1 Definition

Given the index scores $\{I_{t,s}\}$ for all preceding tokens, the **fine-grained token selection mechanism** retrieves only the key-value entries corresponding to the top-$k$ index scores. The **selected set** is:

$$
\boxed{\mathcal{S}_t = \{s \mid I_{t,s} \in \text{Top-}k(I_{t,:})\}}
$$

This is the set of $k$ token positions with the highest index scores. Only these positions participate in the main attention computation.

### 3.2 Selection for our running example

With $k = 3$, we select the 3 tokens with the highest index scores from the table above:

| Rank | Token $s$ | $I_{10,s}$ |
|---|---|---|
| 1 | 3 | $2.9$ |
| 2 | 9 | $2.2$ |
| 3 | 8 | $2.0$ |

So the selected set is:

$$
\mathcal{S}_{10} = \{3, 8, 9\}
$$

Out of 10 preceding tokens, token 10 will attend to only 3. Tokens 0, 1, 2, 4, 5, 6, and 7 are excluded entirely — their key-value entries are never loaded, and no attention scores are computed for them.

### 3.3 The attention computation on selected tokens

The attention output for token $t$ is computed using only the selected key-value entries:

$$
\boxed{\mathbf{u}_t = \text{Attn}\!\left(\mathbf{h}_t, \, \{\mathbf{c}_s \mid s \in \mathcal{S}_t\}\right)}
$$

where $\mathbf{c}_s$ denotes the key-value entry for token $s$, and $\text{Attn}$ is the standard attention mechanism (query-key dot product, softmax, value-weighted sum).

Expanding this for our example: the query vector $\mathbf{q}_{10}$ is computed from $\mathbf{h}_{10}$ as usual. The key vectors $\mathbf{k}_3$, $\mathbf{k}_8$, $\mathbf{k}_9$ and value vectors $\mathbf{v}_3$, $\mathbf{v}_8$, $\mathbf{v}_9$ are computed from (or retrieved from the KV cache for) the selected positions. The attention weights are:

$$
\alpha_{10,s} = \frac{\exp(\mathbf{q}_{10} \cdot \mathbf{k}_s / \sqrt{d_k})}{\sum_{s' \in \{3, 8, 9\}} \exp(\mathbf{q}_{10} \cdot \mathbf{k}_{s'} / \sqrt{d_k})}, \quad s \in \{3, 8, 9\}
$$

The softmax is computed over only 3 entries instead of 10. The output is:

$$
\mathbf{u}_{10} = \alpha_{10,3} \, \mathbf{v}_3 + \alpha_{10,8} \, \mathbf{v}_8 + \alpha_{10,9} \, \mathbf{v}_9
$$

### 3.4 Comparing connectivity sets

Let us place DSA alongside the previous methods for token $t = 10$ in our running example:

| Method | $S_{10}$ | $\|S_{10}\|$ | Selection rule |
|---|---|---|---|
| Full causal | $\{0, 1, 2, \ldots, 10\}$ | $11$ | All preceding tokens |
| Sliding window ($w = 4$) | $\{8, 9, 10\}$ | $3$ | Positions within distance $r = 2$ |
| Longformer ($w = 4$, $G = \{0, 8\}$) | $\{0, 8, 9, 10\}$ | $4$ | Window $\cup$ global |
| **DSA** ($k = 3$) | $\{3, 8, 9\}$ | $3$ | Top-$k$ by content relevance |

DSA selects token 3, which is far outside the sliding window (distance 7 from position 10). No fixed-window method would include it. But the indexer determined that token 3's content is highly relevant to token 10's query — more relevant than nearby tokens 4, 5, 6, or 7.

At the same time, DSA excludes token 10 itself from the selected set (its index score was not in the top 3). In practice, this is handled by always including the current token, or by using a sufficiently large $k$ that nearby tokens are naturally included. We omit this detail to keep the example clean.

### 3.5 Interpretation

The top-$k$ selection is a hard selection — tokens either participate in attention or they do not. There is no soft weighting of excluded tokens. This is different from approaches like Adaptive Span (Sukhbaatar et al., 2019) that use a soft mask to gradually reduce attention to distant tokens.

The hard selection has a practical advantage: excluded tokens' key-value entries are never loaded from memory. In the KV cache during autoregressive generation, this means only $k$ entries are read per query token instead of all preceding entries. For long contexts ($n = 128{,}000$), this is the dominant source of speedup.

---

## 4. Instantiation Under MLA

### 4.1 Why MLA matters

DeepSeek-V3.2 uses Multi-head Latent Attention (MLA), which was introduced in the DeepSeek-V2 paper and derived in an earlier blog in this series. In MLA, the key-value cache stores a single compressed latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ per token, rather than separate key and value vectors for each head. This dramatically reduces the KV cache size.

DSA must work within MLA's framework. The paper implements DSA based on the **MQA (Multi-Query Attention) mode** of MLA, where each latent vector $\mathbf{c}_t^{KV}$ is shared across all query heads. This means the indexer's selected set $\mathcal{S}_t$ applies uniformly to all attention heads — every head attends to the same set of $k$ tokens.

### 4.2 Why shared selection across heads

At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency. If different heads selected different tokens, the memory access pattern would become irregular and difficult to parallelize. By using MQA mode, all heads share the same $k$ selected key-value entries, enabling efficient batched matrix multiplications.

### 4.3 The indexer's projection matrices

The indexer query vectors $\mathbf{q}_{t,j}^I$ and scalar weights $w_{t,j}^I$ are derived from the query token's hidden state $\mathbf{h}_t$ through learned linear projections. The indexer key vectors $\mathbf{k}_s^I$ are derived from the latent vectors $\mathbf{c}_s^{KV}$ (or equivalently, from the hidden states $\mathbf{h}_s$ of the preceding tokens). RoPE (Rotary Position Embedding) is partially applied to the indexer's queries and keys to encode positional information.

The critical point is that all of these projections are small: $d^I$ is much smaller than $d_k$, and $H^I$ is much smaller than $h$. The indexer is intentionally lightweight.

### 4.4 Parameter count of the indexer

Each indexer head requires:
- A query projection: $d_\text{model} \to d^I$, contributing $d_\text{model} \times d^I$ parameters
- A scalar weight projection: $d_\text{model} \to 1$, contributing $d_\text{model}$ parameters

The key projection is shared across heads: $d_\text{model} \to d^I$, contributing $d_\text{model} \times d^I$ parameters.

Total indexer parameters per layer:

$$
H^I \cdot (d_\text{model} \cdot d^I + d_\text{model}) + d_\text{model} \cdot d^I
$$

$$
= d_\text{model} \left[H^I (d^I + 1) + d^I\right]
$$

### 4.5 Numerical check

With our running example values ($d_\text{model} = 512$, $H^I = 2$, $d^I = 2$):

$$
512 \times [2 \times (2 + 1) + 2] = 512 \times [6 + 2] = 512 \times 8 = 4{,}096 \text{ parameters per layer}
$$

Compare to the main attention's QKV projections: $3 \times d_\text{model}^2 = 3 \times 512^2 = 786{,}432$ parameters per layer. The indexer adds $4{,}096 / 786{,}432 \approx 0.5\%$ overhead — negligible.

In practice, the indexer dimensions are larger than our toy example, but the principle holds: the indexer is a tiny fraction of the main model's parameters.

---

## 5. Training Stage 1: Dense Warm-up

### 5.1 The chicken-and-egg problem

DSA is introduced into an existing pretrained model (DeepSeek-V3.1-Terminus) through continued training. At the start, the lightning indexer is randomly initialized — it has no idea which tokens are relevant. If we immediately use the indexer to select tokens and train with sparse attention, the model would attend to random subsets, producing garbage gradients.

The solution is a two-stage training procedure. The first stage — the **dense warm-up** — trains the indexer to match the existing model's full attention distribution while keeping the rest of the model frozen.

### 5.2 Constructing the target distribution

During dense warm-up, the model runs with full (dense) attention as usual. For each query token $t$, the main attention mechanism computes attention scores across all $h$ heads. We need a single target distribution that the indexer should learn to approximate.

The paper constructs this target in two steps:

**Step 1: Aggregate across heads.** For each query token $t$, sum the attention score matrices across all attention heads to get a single relevance score for each key position $s$:

$$
a_{t,s} = \sum_{i=1}^{h} \alpha_{t,s}^{(i)}
$$

where $\alpha_{t,s}^{(i)}$ is the attention weight that head $i$ assigns to key position $s$ when processing query position $t$.

**Step 2: L1-normalize along the sequence dimension.** Divide by the sum to produce a proper probability distribution:

$$
p_{t,s} = \frac{a_{t,s}}{\sum_{s'} a_{t,s'}}
$$

Since each head's attention weights already sum to 1 (by the **softmax normalization property**), the sum across heads is $\sum_s a_{t,s} = \sum_s \sum_i \alpha_{t,s}^{(i)} = \sum_i \sum_s \alpha_{t,s}^{(i)} = \sum_i 1 = h$. We swap the order of summation by **Fubini's theorem** (interchanging finite sums). So the L1 normalization divides by $h$:

$$
p_{t,s} = \frac{1}{h} \sum_{i=1}^{h} \alpha_{t,s}^{(i)}
$$

This is simply the **arithmetic mean** of the attention distributions across heads.

### 5.3 Numerical check

With $h = 8$ heads, suppose the attention weights for query $t = 10$ at key position $s = 3$ across the 8 heads are: $0.15, 0.02, 0.30, 0.05, 0.10, 0.01, 0.20, 0.08$.

$$
a_{10,3} = 0.15 + 0.02 + 0.30 + 0.05 + 0.10 + 0.01 + 0.20 + 0.08 = 0.91
$$

$$
p_{10,3} = \frac{0.91}{8} = 0.11375
$$

The target says: "averaged across all heads, about 11.4% of the attention weight goes to token 3." The indexer should learn to assign a high score to token 3.

### 5.4 The warm-up loss function

The training objective is a **KL divergence** between the target distribution $p_{t,:}$ and the softmax of the indexer scores $I_{t,:}$:

$$
\boxed{\mathcal{L}^I = \sum_t \mathbb{D}_\text{KL}\!\left(p_{t,:} \;\|\; \text{Softmax}(I_{t,:})\right)}
$$

The **KL divergence** (also called **Kullback-Leibler divergence** or **relative entropy**) between two distributions $p$ and $q$ over the same discrete set is:

$$
\mathbb{D}_\text{KL}(p \| q) = \sum_s p_s \log \frac{p_s}{q_s}
$$

By the **logarithm quotient rule** ($\log(a/b) = \log a - \log b$), this is equivalent to:

$$
\mathbb{D}_\text{KL}(p \| q) = \sum_s p_s \log p_s - \sum_s p_s \log q_s
$$

The first term $\sum_s p_s \log p_s$ is the negative **entropy** of $p$, which is a constant with respect to the indexer parameters (since $p$ comes from the frozen main model). So minimizing the KL divergence is equivalent to minimizing:

$$
-\sum_s p_{t,s} \log q_{t,s}
$$

where $q_{t,s} = \text{Softmax}(I_{t,:})_s$. This is the **cross-entropy** between the target distribution $p$ and the indexer's predicted distribution $q$. The gradient flows only through the indexer parameters.

### 5.5 What the loss drives the indexer to do

When $p_{t,s}$ is large (the main model attends heavily to token $s$), the loss penalizes the indexer for assigning a low softmax probability $q_{t,s}$ to that token. The $\log$ amplifies small probabilities — if $q_{t,s}$ is near zero for a token that $p_{t,s}$ values, the loss is very large.

The effect: the indexer learns to produce high scores $I_{t,s}$ for exactly the tokens that the full attention mechanism considers important. After warm-up, the indexer's top-$k$ selection should closely match the set of tokens that receive the most attention weight in the full model.

### 5.6 Warm-up training details

The warm-up stage is deliberately short and focused:

- All model parameters are **frozen** except the lightning indexer
- Learning rate: $10^{-3}$
- Duration: 1,000 steps
- Batch size: 16 sequences of 128K tokens each
- Total tokens: $16 \times 128{,}000 \times 1{,}000 = 2.048 \times 10^9 \approx 2.1\text{B tokens}$

Only 2.1 billion tokens are needed to train the indexer — a tiny fraction of the model's full pretraining budget. This works because the indexer has very few parameters and the target distribution provides a strong supervision signal.

---

## 6. Training Stage 2: Sparse Training

### 6.1 Transitioning from dense to sparse

After the warm-up, the indexer can approximately identify the most relevant tokens. Now the model transitions to actually using sparse attention: for each query, only the top-$k$ tokens selected by the indexer participate in the main attention computation.

In this stage, **all model parameters are unfrozen** — both the main model and the indexer are trained jointly. The main model adapts to receiving only $k$ tokens per query instead of all preceding tokens, and the indexer continues to improve its selection.

### 6.2 The sparse training loss for the indexer

The indexer's loss changes in the sparse stage. During warm-up, the KL divergence was computed over all positions. Now, it is computed only over the selected positions $\mathcal{S}_t$:

$$
\boxed{\mathcal{L}^I = \sum_t \mathbb{D}_\text{KL}\!\left(p_{t,\mathcal{S}_t} \;\|\; \text{Softmax}(I_{t,\mathcal{S}_t})\right)}
$$

where $p_{t,\mathcal{S}_t}$ denotes the target distribution restricted to the selected set $\mathcal{S}_t = \{s \mid I_{t,s} \in \text{Top-}k(I_{t,:})\}$, and $\text{Softmax}(I_{t,\mathcal{S}_t})$ is the softmax of the indexer scores restricted to the same set.

### 6.3 Why restrict to the selected set

During sparse training, the main attention is only computed over the selected tokens. The attention distribution $p_{t,:}$ from the main model now only has entries for positions in $\mathcal{S}_t$ — there are no attention weights for excluded positions because they were never computed.

So the KL divergence can only be evaluated over positions where both distributions are defined: the selected set $\mathcal{S}_t$.

### 6.4 Detaching the indexer input

This is a subtle but important implementation detail. The paper states that the indexer input is **detached from the computational graph** for separate optimization:

> "It is worth noting that we detach the indexer input from the computational graph for separate optimization."

What does this mean? The indexer's input — the hidden states $\mathbf{h}_t$ and $\mathbf{h}_s$ from which the indexer queries and keys are derived — comes from the main model's forward pass. If we allowed gradients to flow from the indexer loss $\mathcal{L}^I$ back through these hidden states, the indexer loss would influence the main model's weights. This would create an undesirable coupling: the main model might distort its representations to make the indexer's job easier, rather than to minimize the language modeling loss.

By detaching, the two training signals are kept separate:

- The **main model** is trained only by the language modeling loss (next-token prediction)
- The **indexer** is trained only by $\mathcal{L}^I$ (matching the main model's attention distribution)

### 6.5 Sparse training details

- Learning rate: $7.3 \times 10^{-6}$ (much lower than warm-up — the main model is being finetuned, not the indexer alone)
- Token selection budget: $k = 2{,}048$ per query token
- Duration: 15,000 steps
- Batch size: 480 sequences of 128K tokens each
- Total tokens: $480 \times 128{,}000 \times 15{,}000 = 9.216 \times 10^{11} \approx 943.7\text{B tokens}$

### 6.6 Numerical check on token budget

In DeepSeek-V3.2's deployment with 128K context, each query token can potentially attend to up to 128,000 preceding positions. The selection budget is $k = 2{,}048$. The fraction of tokens selected:

$$
\frac{k}{n} = \frac{2{,}048}{128{,}000} = 0.016 = 1.6\%
$$

Each query attends to only 1.6% of the context. The remaining 98.4% of key-value entries are never loaded from memory.

For our running example: $k/n = 3/10 = 30\%$. At the small scale of 16 tokens, the savings are modest. The benefit scales with context length.

---

## 7. Complexity Analysis

### 7.1 Full attention baseline

In standard causal attention, each query token $t$ attends to all $t$ preceding tokens plus itself. The total number of attention entries across all positions is:

$$
\sum_{t=0}^{n-1} (t + 1) = \frac{n(n+1)}{2}
$$

This is $O(n^2)$. This is the **triangular number formula** (also called **Gauss's summation formula**).

### 7.2 DSA core attention

With DSA, each query token $t$ attends to at most $k$ selected tokens. The total attention entries are at most:

$$
\sum_{t=0}^{n-1} \min(t + 1, k) = \sum_{t=0}^{k-1} (t + 1) + \sum_{t=k}^{n-1} k = \frac{k(k+1)}{2} + (n - k) \cdot k
$$

For $n \gg k$, the second term dominates:

$$
\boxed{\text{Core attention entries} \approx nk = O(nk)}
$$

### 7.3 Numerical check

With $n = 16$, $k = 3$:

$$
\frac{3 \times 4}{2} + (16 - 3) \times 3 = 6 + 39 = 45
$$

Compare to full causal attention: $\frac{16 \times 17}{2} = 136$. Reduction factor: $136 / 45 \approx 3.0\times$.

At deployment scale ($n = 128{,}000$, $k = 2{,}048$):

$$
\text{Full: } \frac{128{,}000 \times 128{,}001}{2} \approx 8.19 \times 10^9
$$

$$
\text{DSA core: } 128{,}000 \times 2{,}048 = 2.62 \times 10^8
$$

Reduction factor: $8.19 \times 10^9 / 2.62 \times 10^8 \approx 31\times$.

### 7.4 The indexer's own cost

The lightning indexer itself has $O(n^2)$ complexity — it computes a score $I_{t,s}$ for every query-key pair, just like full attention. So DSA does not eliminate quadratic operations entirely.

The key insight: the indexer's per-pair cost is far smaller than the main attention's per-pair cost. The main attention requires:

1. Compute $\mathbf{q}_t \cdot \mathbf{k}_s$ for $d_k$-dimensional vectors: $d_k$ multiply-adds
2. Apply softmax: exponentiation and normalization
3. Compute weighted value sum: $d_v$ multiply-adds

The indexer requires:

1. Compute $\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I$ for $d^I$-dimensional vectors: $d^I$ multiply-adds (with $d^I \ll d_k$)
2. Apply ReLU: a single comparison
3. Multiply by scalar weight: 1 multiply
4. Sum across $H^I$ heads: $H^I$ additions

Since $d^I \ll d_k$ and the indexer can run in FP8 (8-bit floating point) while the main attention typically runs in FP16 or BF16, the indexer's per-pair cost is a small fraction of the main attention's per-pair cost.

### 7.5 Total cost comparison

Let $c_\text{main}$ be the per-pair cost of the main attention and $c_\text{idx}$ be the per-pair cost of the indexer, with $c_\text{idx} \ll c_\text{main}$.

| Method | Total cost |
|---|---|
| Full attention | $n^2 \cdot c_\text{main}$ |
| DSA | $n^2 \cdot c_\text{idx} + nk \cdot c_\text{main}$ |

For the ratio:

$$
\frac{\text{DSA cost}}{\text{Full cost}} = \frac{c_\text{idx}}{c_\text{main}} + \frac{k}{n}
$$

With $c_\text{idx} / c_\text{main} \approx 0.05$ (a rough estimate given the dimension and precision differences) and $k/n = 2{,}048 / 128{,}000 \approx 0.016$:

$$
\frac{\text{DSA cost}}{\text{Full cost}} \approx 0.05 + 0.016 = 0.066
$$

DSA uses roughly 6.6% of the compute of full attention — a $\sim\!15\times$ reduction.

### 7.6 The four-method scaling comparison

| $n$ | Full $\frac{n^2}{2}$ | Sparse $\frac{3}{2}n^{3/2}$ | Window $(w{+}1)n$ | DSA $nk$ |
|---|---|---|---|---|
| 1,024 | 524,288 | 49,152 | 525,312 | 524,288 \* |
| 16,384 | 134,217,728 | 3,145,728 | 8,404,992 | 33,554,432 |
| 128,000 | 8,192,000,000 | 68,567,040 | 65,664,000 | 262,144,000 |

For $w = 512$ (Longformer) and $k = 2{,}048$ (DSA). \*When $n < k$ (as in the $n = 1{,}024$ row), DSA cannot select more tokens than exist — it collapses to full attention, so the entry count equals $\frac{n^2}{2}$. At first glance, DSA's entry count is larger than Longformer's — because $k > w$. But the comparison is misleading: DSA selects the $k$ most relevant tokens from anywhere in the context, while Longformer is restricted to a local window of width $w$. DSA's selected tokens carry more information per entry because they are chosen by content relevance, not by proximity.

The real comparison is quality-adjusted: DSA with $k = 2{,}048$ achieves the same model quality as full attention with $n$ entries. Longformer with $w = 512$ may miss important long-range dependencies that fall outside the window.

---

## 8. Parity Evaluation: Does Sparsity Hurt?

### 8.1 Standard benchmarks

The paper evaluates DeepSeek-V3.2-Exp (the version with DSA) against DeepSeek-V3.1-Terminus (the dense baseline) on a suite of standard benchmarks. The goal: verify that introducing sparse attention does not degrade model quality.

The result: performance is closely matched. On both short-context and long-context tasks, DeepSeek-V3.2-Exp shows no substantial performance degradation compared to the dense baseline. ChatbotArena Elo scores — a measure of human preference — are also closely matched between the two models.

### 8.2 Long-context evaluation

Long-context tasks are the critical test, because they are where sparse attention is most aggressive (selecting $k = 2{,}048$ out of up to $128{,}000$ tokens). On AA-LCR (a long-context reasoning benchmark), DeepSeek-V3.2-Exp scores 4 points higher than DeepSeek-V3.1-Terminus. On Fiction.liveBench, it consistently outperforms the dense baseline across multiple metrics.

This is a striking result. The model that sees only 1.6% of the context per query performs as well as — or better than — the model that sees 100%. The implication: the remaining 98.4% of tokens contribute near-zero attention weight and can be safely ignored.

### 8.3 Interpretation

The parity result validates the core hypothesis: real attention patterns are sparse, and a lightweight indexer can learn to identify the important tokens. The 2.1 billion tokens of warm-up training are sufficient to align the indexer with the full attention distribution, and the 943.7 billion tokens of sparse training are sufficient for the model to adapt to receiving only selected tokens.

---

## 9. Inference Cost Savings

### 9.1 The cost profile of autoregressive generation

During autoregressive generation (decoding), the model generates one token at a time. For each new token, it must:

1. Compute the query vector for the new token
2. Load all previous key-value entries from the KV cache
3. Compute attention scores against all previous entries
4. Produce the attention output

Step 2 is the bottleneck for long sequences — the KV cache grows linearly with sequence length, and loading it from GPU memory is the dominant cost.

### 9.2 DSA's effect on decoding cost

With DSA, step 2 changes: instead of loading all previous KV entries, the model first runs the lightning indexer to score all entries, then loads only the top-$k$ entries.

The indexer's scoring requires loading the indexer keys (which are much smaller than the full KV entries, since $d^I \ll d_k$). After selection, only $k$ full KV entries are loaded.

The paper provides concrete cost comparisons at different sequence positions, estimated from benchmarking on H800 GPUs at \$2 per GPU-hour:

At 128K context during decoding, DeepSeek-V3.2 (with DSA) is significantly cheaper per token than DeepSeek-V3.1-Terminus (dense). The cost curves in Figure 3 of the paper show that the dense model's decoding cost grows linearly with position (as expected — each new token attends to all previous tokens), while DSA's cost grows much more slowly (the indexer cost grows linearly, but the main attention cost is capped at $k$ entries).

### 9.3 Short-context behavior

For short sequences where the context length is less than or close to $k$, DSA provides no benefit — the model would select all tokens anyway. The paper notes that for short-sequence prefilling, they use a masked MHA mode to simulate dense attention, achieving higher efficiency under short-context conditions.

---

## 10. The Unified View: Four Sparse Attention Approaches

We have now seen four approaches to reducing attention complexity, all on the same axis of the taxonomy. Let us place them in a single framework.

### 10.1 The common abstraction

Every approach defines a connectivity set $S_t$ for each query position $t$. The attention output is identical in all cases:

$$
\text{output}_t = \sum_{s \in S_t} \alpha_{t,s} \, \mathbf{v}_s, \qquad \alpha_{t,s} = \frac{\exp(\mathbf{q}_t \cdot \mathbf{k}_s / \sqrt{d_k})}{\sum_{s' \in S_t} \exp(\mathbf{q}_t \cdot \mathbf{k}_{s'} / \sqrt{d_k})}
$$

The only difference is the rule for constructing $S_t$:

| Method | $S_t$ construction | Depends on content? |
|---|---|---|
| Full attention | $S_t = \{0, \ldots, t\}$ | No |
| Sparse Transformer | Fixed factorized patterns | No |
| Longformer | Window $\cup$ global positions | No |
| **DSA** | Top-$k$ by learned indexer | **Yes** |

### 10.2 The progression

These four methods represent a clear progression along two dimensions — cost and adaptivity:

| Property | Full | Sparse Transformer | Longformer | **DSA** |
|---|---|---|---|---|
| Complexity | $O(n^2)$ | $O(n\sqrt{n})$ | $O(n)$ | $O(nk)$ |
| Pattern | Dense | Fixed sparse | Fixed local + global | **Learned sparse** |
| Task adaptation | None | None | Global token choice | **Content-dependent** |
| Long-range access | All tokens | 2-hop | Via global tokens or $L$ hops | **Direct** (if indexer selects) |
| Integration | Native | Train from scratch | Drop-in (pretrain bridge) | **Continued training** |

DSA is the first method in this series where the sparse pattern is determined by content. Every previous method could be fully described before seeing any data — the pattern was a function of positions alone. DSA's pattern is a function of the actual hidden states at runtime.

### 10.3 Interpretation

The trajectory from the Sparse Factorization blog through this blog traces a shift from structural assumptions to learned decisions.

The Sparse Transformer assumed periodicity — strided patterns for images, fixed patterns for text. This was a strong inductive bias that worked well for structured data but could not adapt to the actual content.

Longformer assumed locality — most relevant information is nearby, with a few global exceptions. This was a weaker, more general assumption, and it enabled linear scaling. But the global tokens were still chosen by position (the first token, the question tokens), not by content.

DSA makes no structural assumption at all. It learns, from the model's own attention patterns, which tokens matter for each query. The cost is a small overhead (the indexer) and a two-stage training procedure. The benefit is a sparse attention pattern that is as close to optimal as the indexer can learn — no wasted computation on irrelevant tokens, no missed long-range dependencies that happen to fall outside a fixed window.

---

## Summary

DeepSeek Sparse Attention replaces the fixed connectivity rules of previous sparse methods with a learned, content-dependent token selection mechanism. A lightweight lightning indexer — a small multi-head network using ReLU activations and low-precision arithmetic — scores every preceding token's relevance to each query, and a top-$k$ selector picks the $k$ most relevant tokens for the main attention computation. The indexer is trained in two stages: a dense warm-up that aligns it with the full model's attention distribution via KL divergence (2.1B tokens, 1,000 steps, model frozen), followed by sparse training where both the indexer and the main model adapt jointly (943.7B tokens, 15,000 steps). Instantiated under MLA's MQA mode, DSA reduces the core attention complexity from $O(n^2)$ to $O(nk)$ — at $k = 2{,}048$ and $n = 128{,}000$, each query attends to only 1.6% of the context — with no degradation on standard or long-context benchmarks.

---

*Previous: [Sliding Window Attention: From Local Windows to Global Context](/blog/attention-sliding-window)*  
*Next: [Gated Attention: Replacing Residuals and ReLU with Learned Gates](/blog/attention-gated-attention)*
