---
title: "From Soft Alignment to Queries, Keys, and Values: Deriving the Transformer's Attention"
description: "The Q/K/V abstraction derived from first principles — why Bahdanau's feedforward alignment collapses to a dot product, how the Transformer formalizes queries, keys, and values as separate projections, why we scale by √d_k (with a complete variance proof), what multi-head attention adds, and a full 3-token self-attention numerical walkthrough"
date: 2026-03-31
tags: ["deep-learning", "attention", "transformers", "q-k-v", "self-attention"]
order: 3
---

In the previous post, we derived Bahdanau attention from scratch: a feedforward compatibility function $e_{ij} = \mathbf{v}_a^\top \tanh(W_a s_{i-1} + U_a h_j)$, softmax normalization, and a dynamic context vector $c_i = \sum_j \alpha_{ij} h_j$. It worked. The model learned to align source and target words without any explicit supervision.

But the Transformer paper (Vaswani et al., 2017) replaced the feedforward compatibility function with something far simpler: a dot product. In doing so, it unlocked full parallelization, enabled self-attention, and produced the architecture that underpins every major language model today. In this post we derive why — starting from a single question: what is the simplest compatibility function that still works?

**The running example**: 3 tokens with 2-dimensional embeddings:

$$
\mathbf{x}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{x}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad \mathbf{x}_3 = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}
$$

Think of dimension 1 as encoding "noun-ness" and dimension 2 as encoding "verb-ness." Token 1 is a pure noun, token 2 is a pure verb, token 3 is equally both. We will compute scaled dot-product self-attention on these three tokens entirely by hand.

---

## 1. Reconsidering the Compatibility Function

### 1.1 What Bahdanau's Alignment Model Does

Recall the Bahdanau alignment model:

$$
e_{ij} = \mathbf{v}_a^\top \tanh\!\left(W_a s_{i-1} + U_a h_j\right)
$$

The query ($s_{i-1}$, the decoder state) and the key ($h_j$, the encoder annotation) are each projected into a shared $n'$-dimensional space, summed, passed through $\tanh$, then dotted with $\mathbf{v}_a$. The result is a scalar compatibility score.

This is powerful — a feedforward network can represent any smooth compatibility function. But it has two structural drawbacks:

1. **Sequential computation**: We compute $e_{ij}$ for each $(i, j)$ pair separately. For $T_y$ target positions and $T_x$ source positions, we make $T_y \times T_x$ passes through the alignment network.

2. **Cannot be batched across pairs**: Because of the $\tanh$ nonlinearity on the sum $W_a s_{i-1} + U_a h_j$, we cannot factor the computation into separate operations on $s_{i-1}$ and $h_j$ and then combine them via a simple matrix multiply. The addition inside the $\tanh$ couples the query and key.

### 1.2 The Dot-Product Compatibility Function

The simplest similarity measure between two vectors $\mathbf{q}$ and $\mathbf{k}$ is their **dot product** (also called the **inner product**):

$$
e(\mathbf{q}, \mathbf{k}) = \mathbf{q} \cdot \mathbf{k} = \mathbf{q}^\top \mathbf{k} = \sum_{d=1}^{D} q_d k_d
$$

For unit-norm vectors, this equals the cosine of the angle between them — a direct measure of directional similarity. For general vectors, it captures both magnitude and directional alignment.

The dot product has a key structural advantage: if we pack all queries into a matrix $Q \in \mathbb{R}^{n \times d_k}$ and all keys into a matrix $K \in \mathbb{R}^{m \times d_k}$, then all $n \times m$ pairwise dot products are computed in one matrix multiplication:

$$
S = QK^\top \in \mathbb{R}^{n \times m}
$$

This maps directly onto the matrix multiplication primitives that modern GPUs are built to execute as fast as possible. The feedforward alignment model cannot be parallelized this way — the nonlinearity breaks the factorization.

---

## 2. Queries, Keys, and Values

### 2.1 Why Three Separate Projections?

In Bahdanau's model, the encoder states $h_j$ serve two roles simultaneously:
- They are the things we **score against** (used inside the alignment model to compute $e_{ij}$).
- They are the things we **retrieve** (used directly in the weighted sum $c_i = \sum_j \alpha_{ij} h_j$).

These two roles are conflated. The Transformer paper separates them into three distinct linear projections applied to each token:

- The **query** ($\mathbf{q}_i$): what the current position is looking for. It is what the alignment scores are computed with respect to.
- The **key** ($\mathbf{k}_j$): what each position "offers" for the purpose of being found. It is what the query is compared against.
- The **value** ($\mathbf{v}_j$): the actual content retrieved from position $j$ once it has been selected. It is what is mixed into the output.

Each is obtained by a separate learned linear projection:

$$
Q = X W_Q \in \mathbb{R}^{n \times d_k}, \quad K = X W_K \in \mathbb{R}^{n \times d_k}, \quad V = X W_V \in \mathbb{R}^{n \times d_v}
$$

where $X \in \mathbb{R}^{n \times d_\text{model}}$ is the input token matrix, $W_Q, W_K \in \mathbb{R}^{d_\text{model} \times d_k}$, and $W_V \in \mathbb{R}^{d_\text{model} \times d_v}$ are learned weight matrices.

### 2.2 Why Separate Keys from Values?

The basis for finding relevant information (captured by keys and queries) may differ from the basis for representing the content once found (captured by values). A word might be findable because it plays a particular syntactic role — its key encodes this. But what you actually want to extract from it may be its semantic content — its value encodes that. The separation gives the model flexibility to learn different representations for these two purposes.

In Bahdanau's model, both roles were played by the same vector $h_j$ with no projection. The Transformer adds two degrees of freedom: $W_K$ and $W_V$ project $h_j$ into separate key and value spaces.

### 2.3 Scaled Dot-Product Attention

Putting it together, the full attention computation is:

$$
\boxed{\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V}
$$

This is **scaled dot-product attention** as defined in Vaswani et al. (2017). The factor $1/\sqrt{d_k}$ is the scaling term. We derive exactly why it is needed in the next section.

---

## 3. Why Divide by $\sqrt{d_k}$?

This is the part that confuses almost everyone. The scaling looks like an arbitrary normalization constant. It is not.

### 3.1 The Variance of a Dot Product

Suppose the components of $\mathbf{q}$ and $\mathbf{k}$ are drawn independently from a standard normal distribution:

$$
q_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1), \quad k_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1), \quad i = 1, \ldots, d_k
$$

We want to find the variance of the dot product $\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$.

**Step 1: Mean of each term.** Since $q_i$ and $k_i$ are independent:
$$
\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\, \mathbb{E}[k_i] = 0 \times 0 = 0
$$

By **linearity of expectation**: $\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = 0$.

**Step 2: Variance of each term.** We use $\text{Var}(Z) = \mathbb{E}[Z^2] - (\mathbb{E}[Z])^2$:
$$
\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - 0 = \mathbb{E}[q_i^2]\, \mathbb{E}[k_i^2]
$$

The second equality uses independence of $q_i$ and $k_i$. Since $q_i \sim \mathcal{N}(0,1)$, we have $\mathbb{E}[q_i^2] = \text{Var}(q_i) + (\mathbb{E}[q_i])^2 = 1 + 0 = 1$. Similarly $\mathbb{E}[k_i^2] = 1$. So $\text{Var}(q_i k_i) = 1$.

**Step 3: Variance of the sum.** The terms $q_i k_i$ are independent across $i$ (since the $q_i$ and $k_i$ are all independent). By **additivity of variance for independent random variables**:

$$
\text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k \cdot 1 = d_k
$$

The standard deviation of the dot product is $\sqrt{d_k}$.

### 3.2 Why Large Variance Hurts

This is the core insight from Vaswani et al.: for large $d_k$, the dot products grow large in magnitude. For $d_k = 64$ (a typical head dimension), unscaled dot products have standard deviation 8. The inputs to the softmax span a range of roughly $\pm 24$ (three standard deviations).

Recall that $\text{softmax}(x_i) = e^{x_i} / \sum_j e^{x_j}$. When one input is 24 and the others are near 0, $e^{24} \approx 2.6 \times 10^{10}$ while $e^0 = 1$. The softmax becomes nearly a hard argmax — all weight concentrates on the single largest score. The gradient of the softmax in this regime is nearly zero (the function is saturated). Training with near-zero gradients stalls.

Dividing by $\sqrt{d_k}$ rescales the dot products to have standard deviation 1, keeping the softmax in a well-conditioned regime where gradients are healthy.

### 3.3 Numerical Check

In our running example, $d_k = 2$, so $\sqrt{d_k} = \sqrt{2} \approx 1.4142$.

The unscaled dot product of $\mathbf{x}_1 = [1, 0]$ with itself: $1^2 + 0^2 = 1.0$. After scaling: $1.0 / 1.4142 \approx 0.707$. The unscaled dot product of $\mathbf{x}_3 = [0.5, 0.5]$ with itself: $0.25 + 0.25 = 0.5$. After scaling: $0.5 / 1.4142 \approx 0.354$.

These are small enough that the softmax over them will be well-conditioned — no saturation.

---

## 4. Scaled Dot-Product Attention: Full Numerical Walkthrough

We now carry out the complete computation for our 3-token example. We use $W_Q = W_K = W_V = I$ (the $2 \times 2$ identity matrix) to isolate the attention mechanism from the projection step — equivalent to self-attention with no learned projection.

### 4.1 Step 1: Compute the Score Matrix $S = QK^\top / \sqrt{d_k}$

With $Q = K = X$:

$$
QK^\top = XX^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \end{bmatrix}
$$

Computing each of the 9 entries by the **definition of matrix multiplication** (row of left matrix dotted with column of right matrix):

- $(1,1)$: $[1, 0] \cdot [1, 0] = 1 \cdot 1 + 0 \cdot 0 = 1$
- $(1,2)$: $[1, 0] \cdot [0, 1] = 1 \cdot 0 + 0 \cdot 1 = 0$
- $(1,3)$: $[1, 0] \cdot [0.5, 0.5] = 1 \cdot 0.5 + 0 \cdot 0.5 = 0.5$
- $(2,1)$: $[0, 1] \cdot [1, 0] = 0 \cdot 1 + 1 \cdot 0 = 0$
- $(2,2)$: $[0, 1] \cdot [0, 1] = 0 \cdot 0 + 1 \cdot 1 = 1$
- $(2,3)$: $[0, 1] \cdot [0.5, 0.5] = 0 \cdot 0.5 + 1 \cdot 0.5 = 0.5$
- $(3,1)$: $[0.5, 0.5] \cdot [1, 0] = 0.5 \cdot 1 + 0.5 \cdot 0 = 0.5$
- $(3,2)$: $[0.5, 0.5] \cdot [0, 1] = 0.5 \cdot 0 + 0.5 \cdot 1 = 0.5$
- $(3,3)$: $[0.5, 0.5] \cdot [0.5, 0.5] = 0.5 \cdot 0.5 + 0.5 \cdot 0.5 = 0.25 + 0.25 = 0.5$

$$
QK^\top = \begin{bmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \\ 0.5 & 0.5 & 0.5 \end{bmatrix}
$$

Dividing by $\sqrt{2} \approx 1.4142$:

$$
S = \frac{QK^\top}{\sqrt{2}} = \begin{bmatrix} 0.7071 & 0 & 0.3536 \\ 0 & 0.7071 & 0.3536 \\ 0.3536 & 0.3536 & 0.3536 \end{bmatrix}
$$

### 4.2 Step 2: Apply Row-Wise Softmax

Each row $i$ of $S$ is the score vector for token $i$ attending to all tokens. Softmax is applied independently to each row.

**Row 1** (token 1 attending to all): $[0.7071,\ 0,\ 0.3536]$

$$
\exp(0.7071) = 2.0281, \quad \exp(0) = 1.0000, \quad \exp(0.3536) = 1.4240
$$

$$
Z_1 = 2.0281 + 1.0000 + 1.4240 = 4.4521
$$

$$
A_{1,\cdot} = \left[\frac{2.0281}{4.4521},\; \frac{1.0000}{4.4521},\; \frac{1.4240}{4.4521}\right] = [0.4556,\; 0.2247,\; 0.3199] \approx [0.456,\; 0.225,\; 0.320]
$$

**Row 2** (token 2 attending to all): $[0,\ 0.7071,\ 0.3536]$

The three score values are the same as row 1, just with the large score now at position 2 instead of position 1. By the **permutation equivariance of the softmax function** (softmax is symmetric under permutation of inputs — the same exponentials, just reindexed):

$$
A_{2,\cdot} = [0.225,\; 0.456,\; 0.320]
$$

**Row 3** (token 3 attending to all): $[0.3536,\ 0.3536,\ 0.3536]$

All three scores are identical. When all inputs to the softmax are equal, the **uniform distribution** is the unique output, since equal exponentials divided by $3 \times$ that exponential each give $1/3$. Explicitly: $\exp(0.3536) = 1.4240$ for all three. Sum $= 3 \times 1.4240 = 4.2720$. Each weight $= 1.4240 / 4.2720 = 0.333$.

$$
A_{3,\cdot} = [0.333,\; 0.333,\; 0.333]
$$

The full attention weight matrix:

$$
A = \begin{bmatrix} 0.456 & 0.225 & 0.320 \\ 0.225 & 0.456 & 0.320 \\ 0.333 & 0.333 & 0.333 \end{bmatrix}
$$

### 4.3 Step 3: Multiply by Values

With $V = X$, compute $O = AV$ using the **definition of matrix multiplication**:

$$
O = \begin{bmatrix} 0.456 & 0.225 & 0.320 \\ 0.225 & 0.456 & 0.320 \\ 0.333 & 0.333 & 0.333 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{bmatrix}
$$

**Output for token 1:**

$$
o_1^{(\text{dim 1})} = 0.456 \times 1 + 0.225 \times 0 + 0.320 \times 0.5 = 0.456 + 0 + 0.160 = 0.616
$$

$$
o_1^{(\text{dim 2})} = 0.456 \times 0 + 0.225 \times 1 + 0.320 \times 0.5 = 0 + 0.225 + 0.160 = 0.385
$$

$$
\boxed{\mathbf{o}_1 = [0.616,\; 0.385]}
$$

**Output for token 2** (by symmetry of the weight matrix with rows 1 and 2 swapped):

$$
\boxed{\mathbf{o}_2 = [0.385,\; 0.616]}
$$

**Output for token 3:**

$$
o_3^{(\text{dim 1})} = 0.333 \times 1 + 0.333 \times 0 + 0.333 \times 0.5 = 0.333 + 0 + 0.167 = 0.500
$$

$$
o_3^{(\text{dim 2})} = 0.333 \times 0 + 0.333 \times 1 + 0.333 \times 0.5 = 0 + 0.333 + 0.167 = 0.500
$$

$$
\boxed{\mathbf{o}_3 = [0.500,\; 0.500]}
$$

### 4.4 Interpretation

Token 1 started as a "pure noun" $[1.0, 0.0]$ and became $[0.616, 0.385]$: it absorbed verb flavor from token 2 and mixed flavor from token 3, weighted by its attention to them. Token 2 started as a "pure verb" $[0.0, 1.0]$ and became $[0.385, 0.616]$: symmetric result. Token 3 started as an equal mixture $[0.5, 0.5]$ and remained $[0.5, 0.5]$: because it attended equally to a pure noun and a pure verb, the blend balanced out.

This is self-attention's essential function: each token's output representation is a context-weighted blend of the entire sequence. No token is isolated — each one "sees" all others.

---

## 5. Multi-Head Attention

### 5.1 The Limitation of a Single Head

A single attention head computes one weighted average over the values. This means it can only express one "mode" of attention per layer. If a token should simultaneously attend to its syntactic dependent (for structure) and to a semantically related word (for meaning), a single head must blend these two patterns into one weight vector. The information from each pattern gets mixed into a single average.

### 5.2 Multiple Heads in Parallel

**Multi-head attention** runs $h$ independent attention heads simultaneously, each with its own projection matrices:

$$
\text{head}_i = \text{Attention}\!\left(Q W_i^Q,\; K W_i^K,\; V W_i^V\right)
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

where $W_i^Q \in \mathbb{R}^{d_\text{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_\text{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_\text{model} \times d_v}$ are the per-head projection matrices, and $W^O \in \mathbb{R}^{h d_v \times d_\text{model}}$ is an output projection that combines the heads back into a single representation.

Vaswani et al. use $h = 8$ heads with $d_k = d_v = d_\text{model}/h = 64$ (for $d_\text{model} = 512$). Each head operates in a 64-dimensional subspace. The total computational cost is similar to single-head attention with full dimensionality, because the per-head dimension is smaller by a factor of $h$.

Multi-head attention allows the model to **jointly attend to information from different representation subspaces at different positions**. The visualization in the Transformer paper shows heads specializing in different tasks: anaphora resolution, syntactic structure, positional patterns. A single head would have to average over all of these.

---

## 6. Self-Attention vs. Cross-Attention

The previous post used Bahdanau's **cross-attention**: the query came from the decoder, and the keys and values came from the encoder. The query and the memory it retrieves from are different sequences.

**Self-attention** is the special case where the query, keys, and values all come from the same sequence:

$$
Q = K = V = X \cdot W_{\{Q,K,V\}}
$$

Every token attends to every other token in the same sequence. Each output token's representation is a context-weighted blend of all other tokens in the sequence.

This is the key innovation of the Transformer encoder. A BiRNN encoder can also capture context, but only sequentially: position $j$'s representation is influenced by positions $j-1$, $j-2$, $\ldots$ through the recurrence (and $j+1$, $j+2$, $\ldots$ through the backward pass). With self-attention, every position can directly attend to every other position in a single operation — the "path length" between any two positions is 1.

---

## 7. The Three Uses of Attention in the Transformer

The Transformer architecture uses attention in three distinct ways:

**1. Encoder self-attention**: $Q$, $K$, $V$ all come from the encoder's previous-layer output. Each encoder position attends to all encoder positions. This builds contextual representations where each token encodes information about the full sequence.

**2. Decoder masked self-attention**: $Q$, $K$, $V$ come from the decoder's previous-layer output, but with a **causal mask**. All entries above the diagonal in $S$ are set to $-\infty$ before the softmax. Since $\text{softmax}(-\infty) = 0$, this prevents position $i$ from attending to positions $j > i$. The decoder cannot "look ahead" at future target tokens.

**3. Encoder–decoder attention (cross-attention)**: $Q$ comes from the decoder, and $K$, $V$ come from the encoder output. This is the direct generalization of Bahdanau attention: the decoder queries the full encoder representation at each decoding step.

---

## 8. Positional Encoding: Why Attention Needs It

Self-attention is **permutation equivariant**: if we permute the rows of $X$, the output $O$ is permuted in the same way. This follows from the structure of $S = QK^\top$: permuting the rows of $Q$ permutes the rows of $S$, which permutes the rows of $A$, which permutes the rows of $AV$. There is nothing in the formula that distinguishes "token at position 3" from "the same token appearing at position 7."

This means a Transformer without positional information treats "the cat sat on the mat" identically to "mat the on sat cat the" (up to the final permutation of the output). For language, word order is critical.

The Transformer injects positional information by adding a fixed **positional encoding** to each token embedding before the first attention layer:

$$
\tilde{\mathbf{x}}_t = \mathbf{x}_t + \text{PE}(t)
$$

The sinusoidal positional encoding used by Vaswani et al. is:

$$
\text{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d_\text{model}}}\right), \quad \text{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d_\text{model}}}\right)
$$

where $t$ is the position index and $i$ indexes the dimension. Different dimensions oscillate at different frequencies: low dimensions (small $i$) change rapidly with $t$ (encoding fine-grained local position), and high dimensions change slowly (encoding coarse-grained global position). This is a **Fourier decomposition** of the position index across the embedding dimensions.

The key property of this encoding: for any fixed offset $k$, $\text{PE}(t + k)$ is a linear function of $\text{PE}(t)$ (by the **angle addition identities for trigonometric functions**: $\sin(t + k) = \sin t \cos k + \cos t \sin k$). This means the model can learn to attend by relative offset — "attend to the word 3 positions before me" — without explicit training on relative positions.

---

## 9. Bahdanau vs. Transformer: A Unified View

We can now see Bahdanau attention and Transformer attention as the same mechanism — a soft memory retrieval — with different design choices:

| Property | Bahdanau (2015) | Transformer (2017) |
|---|---|---|
| Compatibility function | Feedforward: $\mathbf{v}^\top \tanh(W_a s + U_a h)$ | Scaled dot product: $\mathbf{q}^\top \mathbf{k} / \sqrt{d_k}$ |
| Keys = Values? | Yes (both are $h_j$) | No (separate $K$ and $V$ projections) |
| Query source | Decoder hidden state $s_{i-1}$ | Learned projection of current token |
| Context encoding | Sequential (RNN) | Parallel (positional encodings) |
| Parallelizable? | No (sequential RNN required) | Yes (pure matrix operations) |
| Attending to same sequence? | No (cross-attention only) | Yes (self-attention and cross-attention) |

The Transformer is not a different thing from Bahdanau attention. It is the same idea — weighted retrieval from memory — with a simpler compatibility function, separate key and value projections, multiple heads, and positional encodings in place of recurrence. Each change is motivated by a concrete limitation of Bahdanau's design.

---

## Summary

Starting from Bahdanau's feedforward alignment model, we derived the Transformer's scaled dot-product attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The $\sqrt{d_k}$ scaling is not cosmetic: dot products grow in variance linearly with dimension ($\text{Var} = d_k$ when components are i.i.d. standard normal), and large variance saturates the softmax to near-zero gradients. Dividing by $\sqrt{d_k}$ restores unit variance.

In our 3-token running example: score matrix diagonal entries at 0.707, attention weights $[0.456, 0.225, 0.320]$ for token 1, and output $[0.616, 0.385]$ — the pure-noun token absorbed verb flavor from its context.

Multi-head attention runs $h$ such computations in parallel with independent projections, allowing each head to specialize on a different relationship type. The Transformer uses this mechanism in three places: encoder self-attention, decoder masked self-attention, and encoder–decoder cross-attention.

In the next post, we count exactly how expensive this is. The attention matrix $A \in \mathbb{R}^{n \times n}$ has $n^2$ entries. For $n = 1024$, that is over one million entries. For $n = 16384$, it is 268 million. The consequences for memory, compute, and inference are the subject of the next post — and they are severe.

---

*Previous: [What Attention is Really Doing](/blog/attention-what-is-it-really)*  
*Next: [Why Vanilla Attention Breaks at Scale](/blog/attention-why-vanilla-breaks)*
