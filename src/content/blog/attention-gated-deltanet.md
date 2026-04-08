---
title: "Targeted Memory: The Delta Rule, Gated DeltaNet, and Kimi Delta Attention"
description: "Building targeted memory management for linear attention from the ground up — why RetNet's uniform decay is a blunt instrument that forgets everything equally, deriving the delta rule as online gradient descent on a reconstruction loss that erases only the key being overwritten, combining the delta rule with gating to get the gated delta rule that can both clear globally and erase surgically, refining the scalar gate to a channel-wise gate in Kimi Delta Attention for per-dimension memory control, the online learning framework that unifies all linear attention variants through their optimization objectives, the WY representation that parallelizes products of Householder matrices for hardware-efficient chunkwise training, the S-NIAH case study showing why neither gating alone nor delta alone suffices, and the Kimi Linear production architecture that interleaves KDA with full attention at a 3:1 ratio using NoPE to outperform pure full-attention models at 48B parameters — all derived step by step with concrete numerical examples."
date: 2026-04-07
tags: [machine-learning, attention, transformers, linear-attention, delta-rule, gated-deltanet, kimi-linear, kda, associative-memory, efficiency]
order: 1
---

The Why Replace Attention blog replaced softmax attention with a linear recurrence: $S_n = S_{n-1} + \phi(k_n)v_n^\top$. The RetNet blog added uniform exponential decay: $S_n = \gamma S_{n-1} + k_n v_n^\top$. Both are hybrid architectures — the same formula computes in parallel (for training) or recurrently (for inference). But both manage memory with a single, blunt tool: accumulate everything (Why Replace Attention) or decay everything uniformly (RetNet).

Consider what happens when a model processes a long document. Early paragraphs establish a topic. Middle paragraphs introduce a character. Late paragraphs contradict information from the middle. The model needs to:

1. **Retain** the topic information from early paragraphs (still relevant)
2. **Erase** the contradicted information from the middle (now wrong)
3. **Store** the new correction from late paragraphs (replacing the old)

Uniform decay cannot do this. With $\gamma = 0.99$, information from 100 steps ago has weight $0.99^{100} = 0.366$ — whether it is the still-relevant topic or the now-contradicted fact. The decay is blind to content.

This blog introduces **targeted memory management**: mechanisms that selectively erase specific key-value associations while preserving others. We derive three progressively refined approaches:

1. **The delta rule** (Schlag et al., 2021; Yang et al., 2024b): $S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$. Erases only the value currently associated with $k_t$, then writes the new value. Derived from online gradient descent on a reconstruction loss.

2. **The gated delta rule** (Yang, Kautz, and Hatamizadeh, 2025): $S_t = \alpha_t(I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$. Adds a scalar forget gate $\alpha_t$ that can clear the entire state when needed — combining global decay with targeted erasure.

3. **Kimi Delta Attention (KDA)** (Kimi Team, 2025): $S_t = (I - \beta_t k_t k_t^\top) \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top$. Replaces the scalar gate with a channel-wise (per-dimension) gate $\alpha_t \in [0,1]^{d_k}$, enabling independent decay control for each feature dimension.

The core papers are Yang et al. (2024b), "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (DeltaNet); Yang, Kautz, and Hatamizadeh (2025), "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025); and Kimi Team (2025), "Kimi Linear: An Expressive, Efficient Attention Architecture."

---

## The Running Example

We continue with the same tiny example from the Why Replace Attention and RetNet blogs:

- $n = 4$ tokens, $d_k = d_v = 2$, single head

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}, \quad K = \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \\ 0 & 2 \end{pmatrix}, \quad V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}
$$

We assume unit-normalized keys for the delta rule (the papers apply L2 normalization to keys). For our example, we normalize each key:

$$
\hat{k}_1 = \frac{1}{\sqrt{0^2+1^2}}\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}0\\1\end{pmatrix}, \quad \hat{k}_2 = \frac{1}{\sqrt{1^2+0^2}}\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}1\\0\end{pmatrix}
$$

$$
\hat{k}_3 = \frac{1}{\sqrt{1^2+1^2}}\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}0.707\\0.707\end{pmatrix}, \quad \hat{k}_4 = \frac{1}{\sqrt{0^2+2^2}}\begin{pmatrix}0\\2\end{pmatrix} = \begin{pmatrix}0\\1\end{pmatrix}
$$

Notice that $\hat{k}_4 = \hat{k}_1$ — tokens 1 and 4 have the same key direction. This is deliberate: it creates a **key collision** that tests how the update rules handle overwriting.

For the writing strength and gating parameters, we fix:

$$
\beta_1 = 0.8, \quad \beta_2 = 0.9, \quad \beta_3 = 0.7, \quad \beta_4 = 0.85
$$

$$
\alpha_1 = 0.95, \quad \alpha_2 = 0.95, \quad \alpha_3 = 0.95, \quad \alpha_4 = 0.95
$$

For channel-wise gating (KDA), we will specify per-dimension $\alpha$ vectors when we reach that section.

For cost analysis, we use the same model parameters from the series:

- $d_\text{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$, $L = 12$ layers, fp16

---

## 1. Associative Memory and the Outer Product

### 1.1 The state as a lookup table

The Why Replace Attention blog derived the state matrix $S_t \in \mathbb{R}^{d_v \times d_k}$ as accumulating outer products: $S_t = \sum_{i=1}^t \phi(k_i) v_i^\top$. The RetNet blog added decay: $S_t = \sum_{i=1}^t \gamma^{t-i} k_i v_i^\top$. In both cases, reading from the state works by multiplying with a query:

$$
o_t = S_t^\top q_t
$$

This is **associative memory** (Smolensky, 1990; Schlag et al., 2021a). The state $S$ stores key-value associations as a sum of outer products, and a query retrieves the value associated with the most similar key. The analogy is a hash table where keys can partially overlap and reads return a weighted blend of all stored values.

### 1.2 The capacity problem

An associative memory built from outer products of $d_k$-dimensional keys can store at most $d_k$ orthogonal key-value pairs perfectly. When we store more associations than $d_k$ dimensions — or when keys are not orthogonal — the stored values interfere with each other. This is the **memory collision** problem.

Let us see this concretely. After storing $v_1 = (1, 0)^\top$ with $\hat{k}_1 = (0, 1)^\top$ and $v_4 = (2, 0)^\top$ with $\hat{k}_4 = (0, 1)^\top$, a linear attention state would contain:

$$
S = \hat{k}_1 v_1^\top + \hat{k}_4 v_4^\top = \begin{pmatrix}0\\1\end{pmatrix}(1 \;\; 0) + \begin{pmatrix}0\\1\end{pmatrix}(2 \;\; 0) = \begin{pmatrix}0 & 0\\1 & 0\end{pmatrix} + \begin{pmatrix}0 & 0\\2 & 0\end{pmatrix} = \begin{pmatrix}0 & 0\\3 & 0\end{pmatrix}
$$

Querying with $\hat{k}_1$:

$$
S^\top \hat{k}_1 = \begin{pmatrix}0 & 3\\0 & 0\end{pmatrix}\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}3\\0\end{pmatrix}
$$

The retrieved value is $(3, 0)$ — neither $v_1 = (1,0)$ nor $v_4 = (2,0)$, but their sum. The memory has **superimposed** the two values and cannot distinguish them. With uniform decay ($\gamma = 0.9$), the old value is attenuated but not erased: $0.9^3 \cdot v_1 + v_4$ still mixes both. The delta rule solves this by erasing the old value before writing the new one.

---

## 2. The Delta Rule: Online Gradient Descent on Reconstruction Loss

### 2.1 The optimization objective

We want the state $S_t$ to act as a memory where querying with key $k_t$ retrieves value $v_t$. Formally, we want to minimize the **reconstruction loss** (Schlag et al., 2021a):

$$
\mathcal{L}_t(S) = \frac{1}{2} \| S^\top k_t - v_t \|^2
$$

This loss is zero when the state perfectly reconstructs the value $v_t$ given the key $k_t$. It is the squared error between what the memory returns and what it should return.

### 2.2 Deriving the update rule

We perform a single step of **stochastic gradient descent** on $\mathcal{L}_t$ with respect to $S$, using the previous state $S_{t-1}$ as the starting point and $\beta_t$ as the learning rate:

$$
S_t = S_{t-1} - \beta_t \nabla_S \mathcal{L}_t(S_{t-1})
$$

The gradient is:

$$
\nabla_S \mathcal{L}_t(S_{t-1}) = \nabla_S \left[ \frac{1}{2} \| S_{t-1}^\top k_t - v_t \|^2 \right]
$$

Expanding: $\| S_{t-1}^\top k_t - v_t \|^2 = (S_{t-1}^\top k_t - v_t)^\top (S_{t-1}^\top k_t - v_t)$. The gradient of $\frac{1}{2}\|Ax - b\|^2$ with respect to $A$ is $(Ax - b)x^\top$. So:

$$
\nabla_S \mathcal{L}_t(S_{t-1}) = (S_{t-1}^\top k_t - v_t) k_t^\top
$$

Wait — let us be more careful. The loss is $\frac{1}{2}\|S^\top k_t - v_t\|^2$ where $S \in \mathbb{R}^{d_v \times d_k}$. Writing $e_t = S_{t-1}^\top k_t - v_t$ (the reconstruction error), we need $\frac{\partial}{\partial S} \frac{1}{2} e_t^\top e_t$.

For a single element $S_{ij}$: $(S^\top k_t)_j = \sum_l S_{lj} (k_t)_l$, so $\frac{\partial (S^\top k_t)_j}{\partial S_{ij}} = (k_t)_i$. Thus $\frac{\partial \mathcal{L}}{\partial S_{ij}} = (S^\top k_t - v_t)_j \cdot (k_t)_i$, giving:

$$
\nabla_S \mathcal{L}_t(S_{t-1}) = k_t (S_{t-1}^\top k_t - v_t)^\top = k_t e_t^\top
$$

Substituting into the SGD update:

$$
S_t = S_{t-1} - \beta_t \, k_t (S_{t-1}^\top k_t - v_t)^\top
$$

Expanding:

$$
S_t = S_{t-1} - \beta_t \, k_t k_t^\top S_{t-1} + \beta_t \, k_t v_t^\top
$$

Factoring:

$$
\boxed{S_t = (I - \beta_t \, k_t k_t^\top) \, S_{t-1} + \beta_t \, k_t v_t^\top}
$$

This is the **delta rule**. The term $(I - \beta_t k_t k_t^\top)$ is a rank-1 update to the identity — a **generalized Householder transformation** (when $\|k_t\| = 1$). It selectively modifies $S_{t-1}$ in the direction of $k_t$ while leaving components orthogonal to $k_t$ untouched.

### 2.3 What the delta rule does geometrically

The matrix $(I - \beta_t k_t k_t^\top)$ applied to a vector $x$ subtracts $\beta_t (k_t^\top x) k_t$ from $x$. This removes $\beta_t$ fraction of the component of $x$ in the direction of $k_t$. When $\beta_t = 1$ and $\|k_t\| = 1$, it is a **projection onto the orthogonal complement** of $k_t$ — complete erasure of the $k_t$-direction.

Applied column-wise to $S_{t-1}$:
- Each column $s_j$ of $S_{t-1}$ becomes $s_j - \beta_t (k_t^\top s_j) k_t$
- The component of $s_j$ along $k_t$ is reduced by factor $(1 - \beta_t \|k_t\|^2)$
- Components orthogonal to $k_t$ are preserved exactly

Then $\beta_t k_t v_t^\top$ writes the new association. The net effect: erase what was stored at $k_t$, write the new value $v_t$. All other stored associations are preserved.

### 2.4 Numerical example

Start with $S_0 = \mathbf{0}$. Using our normalized keys $\hat{k}_t$ and $\beta$ values:

**Step 1** ($\hat{k}_1 = (0, 1)^\top$, $v_1 = (1, 0)^\top$, $\beta_1 = 0.8$):

$$
I - \beta_1 \hat{k}_1 \hat{k}_1^\top = \begin{pmatrix}1 & 0\\0 & 1\end{pmatrix} - 0.8\begin{pmatrix}0\\1\end{pmatrix}(0 \;\; 1) = \begin{pmatrix}1 & 0\\0 & 0.2\end{pmatrix}
$$

$$
S_1 = \begin{pmatrix}1 & 0\\0 & 0.2\end{pmatrix}\begin{pmatrix}0 & 0\\0 & 0\end{pmatrix} + 0.8\begin{pmatrix}0\\1\end{pmatrix}(1 \;\; 0) = \begin{pmatrix}0 & 0\\0.8 & 0\end{pmatrix}
$$

The state stores $(0, 1)^\top \mapsto (1, 0)$ with strength 0.8.

**Step 2** ($\hat{k}_2 = (1, 0)^\top$, $v_2 = (0, 1)^\top$, $\beta_2 = 0.9$):

$$
I - \beta_2 \hat{k}_2 \hat{k}_2^\top = \begin{pmatrix}1 & 0\\0 & 1\end{pmatrix} - 0.9\begin{pmatrix}1\\0\end{pmatrix}(1 \;\; 0) = \begin{pmatrix}0.1 & 0\\0 & 1\end{pmatrix}
$$

$$
S_2 = \begin{pmatrix}0.1 & 0\\0 & 1\end{pmatrix}\begin{pmatrix}0 & 0\\0.8 & 0\end{pmatrix} + 0.9\begin{pmatrix}1\\0\end{pmatrix}(0 \;\; 1) = \begin{pmatrix}0 & 0.9\\0.8 & 0\end{pmatrix}
$$

The first association is preserved (row 2: $(0.8, 0)$), and the new association is stored (row 1: $(0, 0.9)$). Keys $\hat{k}_1$ and $\hat{k}_2$ are orthogonal, so there is no interference.

**Step 3** ($\hat{k}_3 = (0.707, 0.707)^\top$, $v_3 = (1, 1)^\top$, $\beta_3 = 0.7$):

$$
\hat{k}_3 \hat{k}_3^\top = \begin{pmatrix}0.707\\0.707\end{pmatrix}(0.707 \;\; 0.707) = \begin{pmatrix}0.5 & 0.5\\0.5 & 0.5\end{pmatrix}
$$

$$
I - 0.7\begin{pmatrix}0.5 & 0.5\\0.5 & 0.5\end{pmatrix} = \begin{pmatrix}0.65 & -0.35\\-0.35 & 0.65\end{pmatrix}
$$

$$
S_3 = \begin{pmatrix}0.65 & -0.35\\-0.35 & 0.65\end{pmatrix}\begin{pmatrix}0 & 0.9\\0.8 & 0\end{pmatrix} + 0.7\begin{pmatrix}0.707\\0.707\end{pmatrix}(1 \;\; 1)
$$

Computing the matrix product:

$$
\begin{pmatrix}0.65 \cdot 0 + (-0.35)\cdot 0.8 & 0.65 \cdot 0.9 + (-0.35) \cdot 0\\(-0.35)\cdot 0 + 0.65 \cdot 0.8 & (-0.35)\cdot 0.9 + 0.65 \cdot 0\end{pmatrix} = \begin{pmatrix}-0.28 & 0.585\\0.52 & -0.315\end{pmatrix}
$$

Adding the write term: $0.7 \cdot (0.707, 0.707)^\top (1, 1) = (0.495, 0.495)^\top (1, 1) = \begin{pmatrix}0.495 & 0.495\\0.495 & 0.495\end{pmatrix}$

$$
S_3 = \begin{pmatrix}-0.28 + 0.495 & 0.585 + 0.495\\0.52 + 0.495 & -0.315 + 0.495\end{pmatrix} = \begin{pmatrix}0.215 & 1.080\\1.015 & 0.180\end{pmatrix}
$$

**Step 4** ($\hat{k}_4 = (0, 1)^\top$, $v_4 = (2, 0)^\top$, $\beta_4 = 0.85$) — the key collision:

$$
I - 0.85\begin{pmatrix}0\\1\end{pmatrix}(0 \;\; 1) = \begin{pmatrix}1 & 0\\0 & 0.15\end{pmatrix}
$$

$$
S_4 = \begin{pmatrix}1 & 0\\0 & 0.15\end{pmatrix}\begin{pmatrix}0.215 & 1.080\\1.015 & 0.180\end{pmatrix} + 0.85\begin{pmatrix}0\\1\end{pmatrix}(2 \;\; 0)
$$

$$
= \begin{pmatrix}0.215 & 1.080\\0.152 & 0.027\end{pmatrix} + \begin{pmatrix}0 & 0\\1.70 & 0\end{pmatrix} = \begin{pmatrix}0.215 & 1.080\\1.852 & 0.027\end{pmatrix}
$$

Now let us query with $\hat{k}_4 = (0, 1)^\top$ to retrieve the value associated with key direction $(0,1)$:

$$
S_4^\top \hat{k}_4 = \begin{pmatrix}0.215 & 1.852\\1.080 & 0.027\end{pmatrix}\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}1.852\\0.027\end{pmatrix}
$$

The retrieved value is approximately $(1.85, 0.03)$ — close to $v_4 = (2, 0)$, not the old $v_1 = (1, 0)$. The delta rule has largely **overwritten** the old association. Compare this to linear attention, where we would retrieve $(3, 0)$ — the sum of both values, with no way to distinguish old from new.

The overwrite is not perfect (1.85 instead of 2.0) because $\beta_4 = 0.85 < 1$. With $\beta_4 = 1$ and $\|\hat{k}_4\| = 1$, the erasure would be complete and the retrieval exact.

### 2.5 Output computation

The output at each position is:

$$
o_t = S_t^\top q_t
$$

$$
o_1 = S_1^\top q_1 = \begin{pmatrix}0 & 0.8\\0 & 0\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}0\\0\end{pmatrix}
$$

$$
o_2 = S_2^\top q_2 = \begin{pmatrix}0 & 0.8\\0.9 & 0\end{pmatrix}\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}0.8\\0\end{pmatrix}
$$

$$
o_3 = S_3^\top q_3 = \begin{pmatrix}0.215 & 1.015\\1.080 & 0.180\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}1.230\\1.260\end{pmatrix}
$$

$$
o_4 = S_4^\top q_4 = \begin{pmatrix}0.215 & 1.852\\1.080 & 0.027\end{pmatrix}\begin{pmatrix}2\\0\end{pmatrix} = \begin{pmatrix}0.430\\2.160\end{pmatrix}
$$

---

## 3. Why Delta Alone Is Not Enough

### 3.1 The missing forget gate

The delta rule erases the value stored at the current key $k_t$ before writing a new value. But it has no mechanism for **global forgetting** — clearing the entire state or decaying all associations simultaneously.

Consider a context switch: a document changes topic entirely. The model needs to clear out old associations that are no longer relevant. The delta rule can only erase one key direction at a time — it would need to see queries for every old key to erase them all. In the meantime, the old associations persist and interfere with new ones.

This is exactly the scenario tested by the **S-NIAH** (Single Needle In A Haystack) benchmark from RULER (Hsieh et al., 2024). The S-NIAH task has three variants of increasing difficulty:

**S-NIAH-1** (pass-key retrieval with synthetic context): Models memorize a key-value pair embedded in repeated synthetic text. This tests long-term retention with minimal interference. DeltaNet excels (99.0% at 4K) because there is little irrelevant information to manage. Mamba2 degrades beyond 2K (65.4% at 4K) because uniform decay erases the needle too quickly.

**S-NIAH-2** (number in haystack with real-world context): The haystack is real-world essays — dense, information-rich content. The model must store the relevant key-value pair while filtering out thousands of plausible but irrelevant associations. DeltaNet's performance drops sharply (45.6% at 2K, 18.6% at 4K, 14.4% at 8K) — without global forgetting, the memory becomes saturated with irrelevant essay content, causing collisions that bury the needle. Mamba2 does better (98.8% at 2K) because its global decay clears old information, keeping the state clean.

**S-NIAH-3** (UUID in haystack): Values change from numbers to UUIDs — complex patterns that are hard to memorize. Mamba2 degrades (47.6% at 2K) while DeltaNet retains more (85.2% at 1K) thanks to its precise association mechanism.

The pattern is clear:
- **DeltaNet** (delta rule only): strong at precise memorization, weak at filtering irrelevant information
- **Mamba2** (gating only): strong at filtering, weak at precise memorization
- Neither alone suffices

### 3.2 The complementary insight

Gating and the delta rule address different failure modes:

| Mechanism | What it does | Strength | Weakness |
|---|---|---|---|
| Gating ($\alpha_t S_{t-1}$) | Uniform decay of entire state | Clears irrelevant context, prevents saturation | Cannot target specific associations |
| Delta ($I - \beta_t k_t k_t^\top$) | Targeted erasure of one key direction | Precise overwriting, good memorization | Cannot clear global context |

The solution is to combine them.

---

## 4. The Gated Delta Rule

### 4.1 The formula

Yang, Kautz, and Hatamizadeh (2025) propose a simple combination:

$$
\boxed{S_t = \alpha_t \left( I - \beta_t \, k_t k_t^\top \right) S_{t-1} + \beta_t \, k_t v_t^\top}
$$

where $\alpha_t \in (0, 1)$ is a data-dependent scalar gate. This is equation (10) of the Gated DeltaNet paper.

The formula applies two operations in sequence:
1. **Targeted erasure**: $(I - \beta_t k_t k_t^\top) S_{t-1}$ removes the component of the state in the direction of $k_t$
2. **Global decay**: $\alpha_t(\cdot)$ scales the entire result, decaying all associations
3. **Write**: $+ \beta_t k_t v_t^\top$ stores the new association

### 4.2 Interpreting through online learning

From the perspective of the online learning framework introduced by Liu et al. (2024), each linear RNN variant can be understood as the closed-form solution to an optimization problem. The gated delta rule optimizes:

$$
\mathcal{L}_t(S) = \frac{\beta_t}{2} \| S^\top k_t - v_t \|^2 + \frac{1}{2} \| \sqrt{1 - \alpha_t} \, S_{t-1} \|_F^2 - \beta_t \langle S_{t-1}^\top k_t, v_t - \alpha_t S_{t-1}^\top k_t \rangle
$$

The first term is the reconstruction loss (same as the delta rule). The second term is an **$L_2$ regularization** (weight decay) on the state, scaled by $1 - \alpha_t$. When $\alpha_t \to 0$ (aggressive forgetting), the regularization is strong, pulling the state toward zero. When $\alpha_t \to 1$ (no forgetting), the regularization vanishes, reducing to the pure delta rule.

This connects the gated delta rule to a well-known technique in deep learning: **weight decay** (Krogh and Hertz, 1991). The gate $\alpha_t$ controls the strength of weight decay on the fast-weight memory, providing a principled mechanism for memory management.

### 4.3 The unified view

Table 7 of the Kimi Linear paper provides a unified view of all linear attention variants through their online learning objectives and state updates:

| Method | Update Rule |
|---|---|
| Linear Attention | $S_t = S_{t-1} + k_t v_t^\top$ |
| RetNet | $S_t = \alpha S_{t-1} + \beta_t k_t v_t^\top$ |
| Mamba2 | $S_t = \alpha_t S_{t-1} + \beta_t k_t v_t^\top$ |
| GLA | $S_t = \text{Diag}(\alpha_t) S_{t-1} + k_t v_t^\top$ |
| DeltaNet | $S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ |
| Gated DeltaNet | $S_t = \alpha_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ |
| **KDA (ours)** | $S_t = (I - \beta_t k_t k_t^\top) \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top$ |

The progression is clear: each method adds one more degree of control over how the state is updated. Linear attention has no forgetting. RetNet adds a fixed scalar decay. Mamba2 makes the decay data-dependent. GLA makes it per-dimension. DeltaNet adds targeted erasure. Gated DeltaNet combines scalar decay with targeted erasure. KDA combines per-dimension decay with targeted erasure.

### 4.4 Numerical example

Using the same running example with the gated delta rule ($\alpha_t = 0.95$ for all $t$):

**Step 1** ($\hat{k}_1 = (0,1)^\top$, $v_1 = (1,0)^\top$, $\beta_1 = 0.8$, $\alpha_1 = 0.95$):

$$
S_1 = 0.95 \begin{pmatrix}1 & 0\\0 & 0.2\end{pmatrix}\begin{pmatrix}0 & 0\\0 & 0\end{pmatrix} + 0.8\begin{pmatrix}0\\1\end{pmatrix}(1 \;\; 0) = \begin{pmatrix}0 & 0\\0.8 & 0\end{pmatrix}
$$

Same as the delta rule (since $S_0 = 0$, the gate has no effect).

**Step 2** ($\hat{k}_2 = (1,0)^\top$, $v_2 = (0,1)^\top$, $\beta_2 = 0.9$, $\alpha_2 = 0.95$):

$$
S_2 = 0.95\begin{pmatrix}0.1 & 0\\0 & 1\end{pmatrix}\begin{pmatrix}0 & 0\\0.8 & 0\end{pmatrix} + 0.9\begin{pmatrix}1\\0\end{pmatrix}(0 \;\; 1)
$$

$$
= 0.95\begin{pmatrix}0 & 0\\0.8 & 0\end{pmatrix} + \begin{pmatrix}0 & 0.9\\0 & 0\end{pmatrix} = \begin{pmatrix}0 & 0.9\\0.76 & 0\end{pmatrix}
$$

Compare to the pure delta rule where $S_2 = \begin{pmatrix}0 & 0.9\\0.8 & 0\end{pmatrix}$. The gated version has $(0.76)$ instead of $(0.8)$ in position $(2,1)$ — the global decay has slightly reduced the stored association from step 1. This is the gate's effect: a gentle, continuous forgetting that prevents state growth over long sequences.

**Step 3** ($\hat{k}_3 = (0.707, 0.707)^\top$, $v_3 = (1,1)^\top$, $\beta_3 = 0.7$, $\alpha_3 = 0.95$):

$$
S_3 = 0.95\begin{pmatrix}0.65 & -0.35\\-0.35 & 0.65\end{pmatrix}\begin{pmatrix}0 & 0.9\\0.76 & 0\end{pmatrix} + 0.7\begin{pmatrix}0.707\\0.707\end{pmatrix}(1 \;\; 1)
$$

The erasure+decay product:

$$
0.95\begin{pmatrix}-0.266 & 0.585\\0.494 & -0.315\end{pmatrix} = \begin{pmatrix}-0.253 & 0.556\\0.469 & -0.299\end{pmatrix}
$$

Adding the write term $\begin{pmatrix}0.495 & 0.495\\0.495 & 0.495\end{pmatrix}$:

$$
S_3 = \begin{pmatrix}0.242 & 1.051\\0.964 & 0.196\end{pmatrix}
$$

**Step 4** ($\hat{k}_4 = (0,1)^\top$, $v_4 = (2,0)^\top$, $\beta_4 = 0.85$, $\alpha_4 = 0.95$):

$$
S_4 = 0.95\begin{pmatrix}1 & 0\\0 & 0.15\end{pmatrix}\begin{pmatrix}0.242 & 1.051\\0.964 & 0.196\end{pmatrix} + 0.85\begin{pmatrix}0\\1\end{pmatrix}(2 \;\; 0)
$$

$$
= 0.95\begin{pmatrix}0.242 & 1.051\\0.145 & 0.029\end{pmatrix} + \begin{pmatrix}0 & 0\\1.70 & 0\end{pmatrix} = \begin{pmatrix}0.230 & 0.998\\1.838 & 0.028\end{pmatrix}
$$

Querying with $\hat{k}_4$:

$$
S_4^\top \hat{k}_4 = \begin{pmatrix}0.230 & 1.838\\0.998 & 0.028\end{pmatrix}\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}1.838\\0.028\end{pmatrix}
$$

Close to $(2, 0)$ — the gated delta rule successfully overwrites the old association, with the global decay providing additional cleanup of stale information.

### 4.5 S-NIAH results

The Gated DeltaNet paper reports results on the S-NIAH benchmark at 1.3B parameters:

| Model | S-NIAH-1 (1K/2K/4K/8K) | S-NIAH-2 (1K/2K/4K/8K) | S-NIAH-3 (1K/2K/4K) |
|---|---|---|---|
| DeltaNet | 97.4 / 96.8 / 99.0 / 98.8 | 99.4 / 45.6 / 18.6 / 14.4 | 85.2 / 47.0 / 22.4 |
| Mamba2 | 90.2 / 98.8 / 65.4 / 30.4 | 99.4 / 98.8 / 58.2 / 17.0 | 64.4 / 47.6 / 4.6 |
| **Gated DeltaNet** | **98.4 / 88.4 / 91.4 / 91.8** | **100.0 / 99.8 / 92.2 / 29.6** | **86.6 / 84.2 / 27.6** |

Gated DeltaNet combines the strengths of both: it matches or exceeds DeltaNet on memorization tasks (S-NIAH-1, S-NIAH-3) and matches or exceeds Mamba2 on filtering tasks (S-NIAH-2). The combination is strictly better than either component alone.

---

## 5. From Scalar to Channel-Wise Gating: Kimi Delta Attention

### 5.1 The limitation of scalar gating

In Gated DeltaNet, the gate $\alpha_t$ is a single scalar — it decays all dimensions of the state equally. But different feature dimensions may encode different types of information with different lifespans:

- Dimension 1 might encode syntactic structure (short-lived — changes every few tokens)
- Dimension 2 might encode topic identity (long-lived — persists across paragraphs)

A scalar gate forces a single decay rate on both. If $\alpha_t = 0.95$ is chosen to preserve the topic, syntax information accumulates too much. If $\alpha_t = 0.8$ is chosen to clear syntax, topic information decays too fast.

### 5.2 The KDA formula

Kimi Delta Attention (KDA) replaces the scalar gate with a **diagonal matrix** of per-dimension gates:

$$
\boxed{S_t = \left(I - \beta_t k_t k_t^\top\right) \text{Diag}(\alpha_t) \, S_{t-1} + \beta_t k_t v_t^\top}
$$

where $\alpha_t \in [0, 1]^{d_k}$ is now a **vector** with one gate per dimension. $\text{Diag}(\alpha_t)$ is the $d_k \times d_k$ diagonal matrix with $\alpha_t$ on the diagonal.

Note the order of operations: KDA applies the diagonal decay **first**, then the delta rule erasure. In Gated DeltaNet, the scalar gate wraps the entire erasure+state product. In KDA, the per-dimension decay is applied to the previous state before the Householder-style erasure. This ordering has important consequences for parallelization.

### 5.3 Why the ordering matters

In GDN: $S_t = \alpha_t(I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$

In KDA: $S_t = (I - \beta_t k_t k_t^\top) \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top$

The KDA ordering factors into:
1. Apply per-dimension decay: $\tilde{S}_{t-1} = \text{Diag}(\alpha_t) S_{t-1}$
2. Apply delta rule erasure: $S_t = (I - \beta_t k_t k_t^\top) \tilde{S}_{t-1} + \beta_t k_t v_t^\top$

This factored form enables the KDA chunkwise algorithm to bind the DPLR parameters $a = \beta_t k_t$ and $b = k_t \odot \alpha_t$, reducing the number of second-level chunk matrix computations from four to two and eliminating three additional matrix multiplications. The result is roughly $2\times$ faster kernel execution compared to the general DPLR formulation.

### 5.4 Connection to DPLR transition matrices

The KDA recurrence can be rewritten as:

$$
S_t = \left(\text{Diag}(\alpha_t) - \beta_t k_t k_t^\top \text{Diag}(\alpha_t)\right) S_{t-1} + \beta_t k_t v_t^\top
$$

The transition matrix is $A_t = \text{Diag}(\alpha_t) - \beta_t k_t (k_t \odot \alpha_t)^\top$. This has the form $D - a b^\top$ where $D = \text{Diag}(\alpha_t)$, $a = \beta_t k_t$, $b = k_t \odot \alpha_t$. This is a **Diagonal-Plus-Low-Rank (DPLR)** structure — a diagonal matrix plus a rank-1 correction.

The DPLR structure is significant because:
- S4 (Gu et al., 2022) used static DPLR transition matrices, jointly diagonalized into the complex plane
- Mamba2 (Dao and Gu, 2024) used diagonal-only transitions
- KDA uses data-dependent DPLR, gaining expressiveness over Mamba2's diagonal while maintaining efficient parallelization through the shared $a = \beta k$, $b = k \odot \alpha$ parameterization

### 5.5 Numerical example with channel-wise gating

For KDA, we use per-dimension gates. Let us fix:

$$
\alpha_1 = \begin{pmatrix}0.9\\0.99\end{pmatrix}, \quad \alpha_2 = \begin{pmatrix}0.9\\0.99\end{pmatrix}, \quad \alpha_3 = \begin{pmatrix}0.9\\0.99\end{pmatrix}, \quad \alpha_4 = \begin{pmatrix}0.9\\0.99\end{pmatrix}
$$

Dimension 1 decays fast ($\alpha = 0.9$, effective window $\approx 44$ tokens) — capturing local patterns. Dimension 2 decays slowly ($\alpha = 0.99$, effective window $\approx 458$ tokens) — preserving long-range information.

**Step 1** ($\hat{k}_1 = (0,1)^\top$, $v_1 = (1,0)^\top$, $\beta_1 = 0.8$):

$\text{Diag}(\alpha_1) S_0 = \mathbf{0}$, so the delta+write gives the same result:

$$
S_1 = \begin{pmatrix}0 & 0\\0.8 & 0\end{pmatrix}
$$

**Step 2** ($\hat{k}_2 = (1,0)^\top$, $v_2 = (0,1)^\top$, $\beta_2 = 0.9$):

First, apply per-dimension decay:

$$
\text{Diag}(\alpha_2) S_1 = \begin{pmatrix}0.9 & 0\\0 & 0.99\end{pmatrix}\begin{pmatrix}0 & 0\\0.8 & 0\end{pmatrix} = \begin{pmatrix}0 & 0\\0.792 & 0\end{pmatrix}
$$

Then apply delta erasure and write:

$$
S_2 = \begin{pmatrix}0.1 & 0\\0 & 1\end{pmatrix}\begin{pmatrix}0 & 0\\0.792 & 0\end{pmatrix} + 0.9\begin{pmatrix}1\\0\end{pmatrix}(0 \;\; 1) = \begin{pmatrix}0 & 0.9\\0.792 & 0\end{pmatrix}
$$

Compare to GDN ($S_2 = \begin{pmatrix}0 & 0.9\\0.76 & 0\end{pmatrix}$) and pure delta ($S_2 = \begin{pmatrix}0 & 0.9\\0.8 & 0\end{pmatrix}$). KDA preserves more of the first association (0.792 vs 0.76 for GDN) because dimension 2 has a high gate $\alpha = 0.99$. The first association was stored in row 2 (the $\hat{k}_1 = (0,1)$ direction), and the slow-decay dimension preserves it better.

**Step 4** ($\hat{k}_4 = (0,1)^\top$, $v_4 = (2,0)^\top$, $\beta_4 = 0.85$) — after computing through step 3:

For brevity, let us trace just the key collision step. After step 3, KDA produces some state $S_3$. The step-4 update applies:

$$
\text{Diag}(\alpha_4) S_3 = \begin{pmatrix}0.9 & 0\\0 & 0.99\end{pmatrix} S_3
$$

This decays dimension 1 of the state matrix by $0.9$ and dimension 2 by $0.99$, then the delta rule erases in the $\hat{k}_4 = (0,1)$ direction and writes $v_4 = (2,0)$. The per-dimension decay ensures that different features of the stored associations decay at different rates, providing finer-grained memory control than the scalar gate.

---

## 6. KDA as Learnable Position Encoding

### 6.1 The connection to RoPE

Standard softmax attention is permutation-equivariant — it treats all token orderings equally. Position information must be injected externally via position encodings like RoPE (Su et al., 2024). RoPE applies rotation matrices $R_j$ between each position pair:

$$
s_{t,i} = q_t^\top \left(\prod_{j=i+1}^t R_j\right) k_i
$$

The cumulative product of rotations encodes relative position $t - i$. In the attention variants with gating and delta rules, the cumulative product of transition matrices plays the same role:

$$
o_t = \sum_{i=1}^t \left( q_t^\top \left( \prod_{j=i+1}^t A_j \right) k_i \right) v_i
$$

where $A_j = \alpha_j (I - \beta_j k_j k_j^\top)$ for GDN or $A_j = (I - \beta_j k_j k_j^\top) \text{Diag}(\alpha_j)$ for KDA.

The key difference: RoPE's rotation matrices are **orthogonal** and **data-independent** (fixed by position). KDA's transition matrices are **data-dependent** — they adapt to the content of each token. This makes KDA a form of **learnable multiplicative position encoding** that relaxes the orthogonality constraint of RoPE.

### 6.2 RoPE vs KDA: position encoding granularity

A key advantage of RoPE is its **fine-grained** position encoding: different pairs of dimensions rotate at different frequencies, creating a rich multi-scale position signature (analogous to a nonuniform Fourier transform).

Standard GDN uses a per-head scalar gate — a single $\alpha_t$ per head. This is coarser than RoPE's per-dimension-pair encoding. KDA's channel-wise gate $\alpha_t \in [0,1]^{d_k}$ provides **per-dimension** decay rates, matching the fine-grained structure of RoPE. Each dimension can encode position information at a different timescale — fast-decaying dimensions for local position, slow-decaying dimensions for global position.

---

## 7. Parallelizing the Delta Rule: The WY Representation

### 7.1 The problem

The delta rule recurrence involves $(I - \beta_t k_t k_t^\top)$ — a generalized Householder matrix. The product of Householder matrices across a chunk of size $C$:

$$
P_{[t]}^r = \prod_{i=1}^r (I - \beta_{[t]}^i k_{[t]}^i {k_{[t]}^i}^\top) \text{Diag}(\alpha_{[t]}^i)
$$

cannot be naively computed in parallel because each factor depends on the previous. A sequential product of $C$ matrices costs $O(C \cdot d_k^2)$ — acceptable for small $C$ but inefficient for GPU parallelism.

### 7.2 The WY representation

Bischof and Van Loan (1987) showed that products of Householder matrices can be compressed into a **WY representation**: instead of storing all $C$ individual matrices, the product can be expressed as:

$$
P_{[t]}^r = \text{Diag}(\gamma_{[t]}^r) - \sum_{i=1}^r \text{Diag}(\gamma_{[t]}^{i \to r}) k_{[t]}^i {w_{[t]}^i}^\top
$$

where the auxiliary vectors $w_{[t]}^i \in \mathbb{R}^{d_k}$ are computed via the recurrence:

$$
w_{[t]}^r = \beta_{[t]}^r \left( \text{Diag}(\gamma_{[t]}^r) k_{[t]}^r - \sum_{i=1}^{r-1} w_{[t]}^i \left( {k_{[t]}^i}^\top \text{Diag}(\gamma_{[t]}^{i \to r}) k_{[t]}^r \right) \right)
$$

Here $\gamma_{[t]}^{i \to r} = \prod_{j=i}^r \alpha_{[t]}^j$ is the cumulative decay from position $i$ to position $r$ within the chunk.

Similarly, the within-chunk output contribution $H_{[t]}^r$ has its own WY representation:

$$
H_{[t]}^r = \sum_{i=1}^r \text{Diag}(\gamma_{[t]}^{i \to r}) k_{[t]}^i {u_{[t]}^i}^\top
$$

with auxiliary vectors $u_{[t]}^i \in \mathbb{R}^{d_v}$ computed by:

$$
u_{[t]}^r = \beta_{[t]}^r \left( v_{[t]}^r - \sum_{i=1}^{r-1} u_{[t]}^i \left( {k_{[t]}^i}^\top \text{Diag}(\gamma_{[t]}^{i \to r}) k_{[t]}^r \right) \right)
$$

### 7.3 The UT transform for matrix form

To convert the WY recurrence into hardware-efficient matrix operations, the papers use the **UT transform** (Joffrain et al., 2006). Define the matrices $W = T K$ and $U = T V$ where:

$$
T_{[t]} = \left[ I + \text{StrictLower}\left(\text{diag}(\beta_{[t]}) \left(\Gamma_{[t]}^{1 \to C} \odot K_{[t]}\right) \left(\frac{K_{[t]}}{\Gamma_{[t]}^{1 \to C}}\right)^\top\right) \right]^{-1} \text{diag}(\beta_{[t]})
$$

The inverse of the lower-triangular matrix is computed efficiently by forward substitution — a row-wise iterative procedure that avoids explicit matrix inversion. This yields a hardware-efficient chunkwise algorithm:

**State update:**
$$
S_{[t+1]} = \text{Diag}(\gamma_{[t]}^C) S_{[t]} + \left(\Gamma_{[t]}^{1 \to C} \odot K_{[t]}\right)^\top \left(U_{[t]} - W_{[t]} S_{[t]}\right)
$$

**Output computation:**
$$
O_{[t]} = \left(\Gamma_{[t]}^{1 \to C} \odot Q_{[t]}\right) S_{[t]} + \text{Tril}\left( \left(\Gamma_{[t]}^{1 \to C} \odot Q_{[t]}\right) \left(\frac{K_{[t]}}{\Gamma_{[t]}^{1 \to C}}\right)^\top \odot M \right) \left(U_{[t]} - W_{[t]} S_{[t]}\right)
$$

The key insight: the inter-chunk recurrence (state passing between chunks) is $O(d_k^2)$ per chunk, while the intra-chunk computation (within each chunk) is dominated by matrix multiplications that map efficiently to GPU tensor cores. The chunkwise algorithm achieves $O(n \cdot d_k \cdot (C + d_k))$ total cost — the same asymptotic complexity as RetNet's chunkwise form.

### 7.4 Cost comparison

For a single attention head with head dimension $d_h$ and chunk size $C = 64$:

| Operation | FLOPs per token |
|---|---|
| Full attention | $2T d_h$ ($O(T d_h)$ total) |
| KDA chunkwise | $6d_h^2 + 3Cd_h + C^2$ |

With $d_h = 128$ (as used in both papers) and $C = 64$:
- KDA: $6 \times 128^2 + 3 \times 64 \times 128 + 64^2 = 98{,}304 + 24{,}576 + 4{,}096 = 126{,}976$ FLOPs per token
- Full attention at $T = 4096$: $2 \times 4096 \times 128 = 1{,}048{,}576$ FLOPs per token

KDA is $8.3\times$ cheaper per token at 4K context. The advantage grows linearly with sequence length.

---

## 8. The Gated DeltaNet Architecture

### 8.1 Block design

The Gated DeltaNet block follows the Llama macro architecture: token mixer layers with SwiGLU MLP layers, but replacing self-attention with the gated delta rule for token mixing. The block design for the gated delta rule layer:

1. **Input projections**: Linear projections generate $q, k, v$ from the input
2. **Short convolution + SiLU**: Applied to $q$ and $k$ for local context; $v$ uses the same path
3. **L2 normalization**: Applied to $q$ and $k$ for eigenvalue stability (following Yang et al., 2024b)
4. **$\alpha, \beta$ generation**: Separate linear projections with sigmoid activation produce the gate ($\alpha$) and writing strength ($\beta$) as scalars per head
5. **Gated delta rule**: The recurrence $S_t = \alpha_t(I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$
6. **Output normalization + gating**: The output is processed through normalization and a sigmoid-based output gate before the final linear projection

### 8.2 Hybrid architectures

Linear transformers have limitations in modeling local shifts and precise in-context retrieval. Following Griffin (De et al., 2024) and Samba (Ren et al., 2024), the Gated DeltaNet paper develops hybrid architectures that interleave linear recurrent layers with sliding window attention (SWA):

- **GatedDeltaNet-H1**: Gated DeltaNet + SWA layers
- **GatedDeltaNet-H2**: Mamba2 + Gated DeltaNet + SWA layers

The hybrid models achieve the best overall results — combining the efficient long-range modeling of the gated delta rule with the precise local attention of sliding windows.

### 8.3 Experimental results at 1.3B scale

All models trained on 100B tokens from FineWeb-Edu with identical hyperparameters:

**Language modeling and commonsense reasoning** (Table 3 of GDN paper):

| Model | Wiki PPL | LMB PPL | Avg Accuracy |
|---|---|---|---|
| RetNet | 19.08 | 17.27 | 52.02 |
| Mamba2 | 16.56 | 12.56 | 54.89 |
| DeltaNet | 17.71 | 16.88 | 52.14 |
| **Gated DeltaNet** | **16.42** | **12.17** | **55.32** |
| Transformer++ | 18.53 | 18.32 | 52.25 |
| Samba | 16.13 | 13.29 | 54.00 |
| **GatedDeltaNet-H2** | **15.91** | **12.55** | **56.18** |

Gated DeltaNet surpasses all pure recurrent models. The hybrid H2 variant is the overall best.

**Ablation study** (Table S.1 of GDN paper):

| Component removed | Avg PPL | Avg Accuracy |
|---|---|---|
| Full Gated DeltaNet (head dim 128) | 27.35 | 47.26 |
| w/ naive delta rule (no gating) | 30.87 | 45.12 |
| w/o short convolution | 28.95 | 46.16 |
| w/o output gate | 29.12 | 45.46 |
| w/o output norm | 27.55 | 47.07 |

The gating mechanism is the single most important component (+3.52 PPL degradation without it), followed by the output gate and short convolution.

---

## 9. The Kimi Linear Architecture

### 9.1 From Gated DeltaNet to Kimi Linear

Kimi Linear extends Gated DeltaNet in three key ways:

1. **Channel-wise gating**: Replaces scalar $\alpha_t$ with per-dimension $\alpha_t \in [0,1]^{d_k}$
2. **3:1 hybrid ratio**: Interleaves 3 KDA layers with 1 full MLA (Multi-head Latent Attention) layer
3. **NoPE for full attention**: Uses No Position Embedding for the global attention layers, delegating all position encoding to the KDA layers

### 9.2 Neural parameterization

For each head $h$, the KDA inputs are computed from the token representation $x_t \in \mathbb{R}^d$:

$$
q_t^h, k_t^h = \text{L2Norm}(\text{Swish}(\text{ShortConv}(W_{q/k}^h x_t))) \in \mathbb{R}^{d_k}
$$

$$
v_t^h = \text{Swish}(\text{ShortConv}(W_v^h x_t)) \in \mathbb{R}^{d_v}
$$

$$
\alpha_t^h = f(W_\alpha^\uparrow W_\alpha^\downarrow x_t) \in [0,1]^{d_k}
$$

$$
\beta_t^h = \text{Sigmoid}(W_\beta^h x_t) \in [0,1]
$$

The per-channel decay $\alpha_t^h$ is parameterized via a low-rank projection ($W_\alpha^\downarrow$ and $W_\alpha^\uparrow$ with rank equal to the head dimension) and a decay function $f(\cdot)$ — similar to those used in GDN and Mamba. The output uses head-wise RMSNorm and a data-dependent sigmoid gate:

$$
o_t = W_o \left(\text{Sigmoid}(W_g^\uparrow W_g^\downarrow x_t) \odot \text{RMSNorm}(\text{KDA}(q_t, k_t, v_t, \alpha_t, \beta_t))\right)
$$

### 9.3 The 3:1 hybrid ratio

Pure linear attention still struggles with precise memory retrieval and exact copying — tasks where the full $O(n^2)$ attention pattern provides perfect token-to-token lookup. The Kimi Linear architecture addresses this by interleaving KDA with full attention (specifically MLA — Multi-head Latent Attention from DeepSeek-V3):

$$
\underbrace{\text{KDA} \to \text{KDA} \to \text{KDA}}_{3 \text{ layers}} \to \underbrace{\text{MLA}}_{1 \text{ layer}} \to \text{repeat}
$$

The ablation confirms 3:1 is optimal:

| Hybrid Ratio (KDA:MLA) | Training PPL | Validation PPL |
|---|---|---|
| 0:1 (pure MLA) | 9.45 | 5.77 |
| 1:1 | 9.29 | 5.66 |
| **3:1** | **9.23** | **5.65** |
| 7:1 | 9.23 | 5.70 |
| 15:1 | 9.34 | 5.82 |

The 3:1 ratio achieves the best validation PPL while maintaining good training PPL. Lower ratios (more full attention) increase inference cost without improving quality. Higher ratios (less full attention) degrade validation performance, suggesting that periodic full-attention layers provide essential exact retrieval capabilities that KDA alone cannot match.

### 9.4 NoPE for global attention layers

Kimi Linear applies **No Position Embedding (NoPE)** to all full MLA layers. This means the global attention layers have no explicit notion of token order — they treat the input as a set, not a sequence.

Why this works: KDA already encodes position information through its data-dependent transition matrices (Section 6). Adding RoPE to the MLA layers would provide redundant position information with two drawbacks:

1. **RoPE frequency sensitivity**: Models using RoPE can become sensitive to the base frequency, causing degradation at context lengths not seen during training. NoPE avoids this.
2. **Simplified long-context extension**: Without position encodings in the global layers, extending to longer contexts requires no frequency adjustments (no YaRN, no NTK-aware scaling).

The NoPE ablation confirms this (Table 5 of the Kimi paper): Kimi Linear (NoPE) achieves the highest average score (54.5) on long-context benchmarks, outperforming Kimi Linear with RoPE (51.8). NoPE is not just simpler — it is better for long-context performance.

### 9.5 Production-scale results

Kimi Linear is a Mixture-of-Experts (MoE) model:
- **48B total parameters**, **3B activated** per forward pass
- 8 out of 256 experts activated (1 shared + 7 routed)
- Head dimension: $d_k = d_v = 128$ for all attention types
- Trained on **1.4 trillion tokens** from the K2 pretraining corpus

**Pretrain results** (Table 3 of Kimi paper, 1.4T tokens):

| Benchmark | MLA | GDN-H | Kimi Linear |
|---|---|---|---|
| HellaSwag | 81.7 | 82.2 | **82.9** |
| MMLU | 71.6 | 72.2 | **73.8** |
| MMLU-Pro | 47.2 | 47.9 | **51.0** |
| TriviaQA | 68.9 | 70.1 | **71.7** |
| GSM8K | 83.7 | 81.7 | **83.9** |
| MATH | **54.7** | 54.1 | **54.7** |

Kimi Linear consistently outperforms both the full-attention MLA baseline and the hybrid GDN-H baseline across nearly all benchmarks, with identical training recipe and parameter count.

**Long-context results** (128K context, Table 5):

| Benchmark | MLA | GDN-H | Kimi Linear |
|---|---|---|---|
| RULER | 81.3 | 80.5 | **84.3** |
| MRCR | 22.6 | 23.9 | **29.6** |
| HELMET-ICL | 88.0 | 85.5 | **90.0** |
| RepoQA | 63.0 | 63.0 | **68.5** |
| Avg | 52.2 | 51.2 | **54.5** |

Kimi Linear is the strongest on long-context tasks — the per-dimension gating provides fine-grained position encoding that enables better long-range retrieval than both MLA (which relies on RoPE) and GDN-H (which uses coarser scalar gating).

**Inference efficiency** (Figure 7 of Kimi paper):

At 1M tokens decoding:
- Kimi Linear achieves $6.3\times$ faster TPOT (Time Per Output Token) compared to MLA
- Prefilling at 512K: $2.3\times$ faster than MLA
- KV cache reduction: up to 75% (only 1 in 4 layers needs a full KV cache)

### 9.6 Scaling law results

The Kimi team conducted scaling law experiments from 653M to 1.7B activated parameters. The fitted scaling curves:

$$
\text{MLA}: \quad L = 2.3092 \times C^{-0.0536}
$$

$$
\text{Kimi Linear}: \quad L = 2.2879 \times C^{-0.0527}
$$

where $C$ is compute in PFLOP/s-days and $L$ is the loss. At matched compute, Kimi Linear achieves $\sim 1.16\times$ computational efficiency over MLA — producing the same loss with 16% less compute. The scaling exponents are comparable ($-0.0527$ vs $-0.0536$), indicating that the advantage is maintained, not diminishing, as scale increases.

---

## 10. The Online Learning Unification

### 10.1 Why this framework matters

The unified view through online learning objectives (Table 7 of the Kimi paper) is not merely a notational convenience — it reveals the **design space** of linear attention variants and explains why each method makes the tradeoffs it does.

Every linear attention variant can be expressed as:

$$
S_t = S_{t-1} - \nabla_S \mathcal{L}_t(S_{t-1})
$$

where $\mathcal{L}_t$ is an online learning objective. The choice of $\mathcal{L}_t$ determines the update rule and, consequently, the model's memory management behavior.

### 10.2 From loss functions to update rules

**Linear Attention**: $\mathcal{L}_t = -\langle S_{t-1}^\top k_t, v_t \rangle$ (correlation loss — maximize correlation between stored and target values). This gives the pure accumulation rule $S_t = S_{t-1} + k_t v_t^\top$. No forgetting, no erasure.

**RetNet**: Adds $L_2$ regularization with fixed weight: $\mathcal{L}_t = -\beta_t \langle S_{t-1}^\top k_t, v_t \rangle + \frac{1}{2}\|\sqrt{1-\alpha}\, S_{t-1}\|_F^2$. This gives $S_t = \alpha S_{t-1} + \beta_t k_t v_t^\top$ — fixed-rate decay. The regularization strength $1 - \alpha$ is constant (data-independent).

**DeltaNet**: $\mathcal{L}_t = \frac{\beta_t}{2} \| S_{t-1}^\top k_t - v_t \|^2$ (reconstruction loss). This gives $(I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ — targeted erasure via gradient descent on reconstruction error.

**Gated DeltaNet**: $\mathcal{L}_t = \frac{\beta_t}{2}\|S_{t-1}^\top k_t - v_t\|^2 + \frac{1}{2}\|\sqrt{1-\alpha_t}\, S_{t-1}\|_F^2 - \beta_t \langle S_{t-1}^\top k_t, v_t - \alpha_t S_{t-1}^\top k_t \rangle$. Reconstruction loss plus data-dependent $L_2$ regularization — combining targeted erasure with adaptive weight decay.

**KDA**: Same objective family as GDN but with per-dimension regularization via $\text{Diag}(\alpha_t)$, enabling the optimization to apply different regularization strengths to different feature dimensions.

### 10.3 The fast-weight programming interpretation

From the perspective of fast-weight programming (Schlag et al., 2021a; Irie et al., 2022a), the state $S$ is a **fast-weight matrix** — a neural network weight matrix that is updated at every time step, not just during training. The "slow weights" are the projection matrices ($W_Q, W_K, W_V$) learned by gradient descent during training. The "fast weights" $S_t$ are updated at every token by the online learning rule.

Under this interpretation:
- **Linear attention** = fast-weight update by Hebbian learning (correlate inputs with targets)
- **DeltaNet** = fast-weight update by one step of SGD on reconstruction error
- **Gated DeltaNet** = SGD with weight decay on the fast weights
- **KDA** = SGD with per-dimension weight decay on the fast weights

The delta rule's superiority over simple accumulation is the same reason SGD outperforms Hebbian learning: gradient-based updates correct errors rather than merely reinforcing correlations.

---

## Summary

The RetNet blog showed that uniform decay $\gamma$ enables the hybrid architecture pattern — one formula, three computation modes. But decay is a blunt instrument: it forgets everything at the same rate, regardless of content. This blog introduced targeted memory management through the delta rule, derived as one step of online gradient descent on the reconstruction loss $\frac{1}{2}\|S^\top k_t - v_t\|^2$, yielding the update $S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ — a generalized Householder transformation that erases only the component stored at the current key while preserving all other associations (verified numerically: querying with $\hat{k}_4$ after overwriting retrieves $(1.85, 0.03) \approx v_4 = (2,0)$, not the superimposed sum $(3,0)$ that linear attention would return). The S-NIAH benchmark revealed that neither the delta rule (strong memorization, weak filtering) nor gating (strong filtering, weak memorization) alone suffices, motivating the Gated DeltaNet: $S_t = \alpha_t(I - \beta_t k_t k_t^\top)S_{t-1} + \beta_t k_t v_t^\top$, which combines scalar global decay with targeted erasure to achieve the best of both on all three S-NIAH variants. Kimi Delta Attention (KDA) refines this further by replacing the scalar gate with a per-dimension gate: $S_t = (I - \beta_t k_t k_t^\top)\text{Diag}(\alpha_t)S_{t-1} + \beta_t k_t v_t^\top$, enabling fine-grained memory control analogous to RoPE's per-dimension frequency encoding. The WY representation parallelizes the product of Householder matrices for hardware-efficient chunkwise training, and the online learning framework unifies all variants through their optimization objectives — from Hebbian correlation (linear attention) through fixed regularization (RetNet) to adaptive per-dimension SGD with weight decay (KDA). Kimi Linear validates this at production scale: a 48B-parameter MoE model (3B activated) interleaving KDA with full MLA attention at a 3:1 ratio, using NoPE for the global attention layers, trained on 1.4T tokens — outperforming the full-attention MLA baseline on nearly every benchmark while achieving $6.3\times$ faster decoding at 1M tokens and 75% KV cache reduction.

---

*Previous: [Hybrid Architectures: RetNet and the Three Computation Paradigms](/blog/attention-retnet)*

*Next: [The Kernel Zoo: Performers, Fast Weight Programmers, and the Capacity-Approximation Tradeoff in Linear Attention](/blog/attention-linear-attention)*
