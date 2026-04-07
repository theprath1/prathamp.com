---
title: "Hybrid Architectures: RetNet and the Three Computation Paradigms"
description: "Building hybrid sequence architectures from the ground up — why pure transformers and pure RNNs each fail at one vertex of the impossible triangle (training parallelism, low-cost inference, strong performance), the two problems with linear attention's accumulate-only recurrence (unbounded state growth and missing position information), how RetNet's retention mechanism fixes both with exponential decay γ and complex-exponential position encoding, deriving the parallel form as attention with a causal decay mask, verifying the recurrent form produces identical outputs, the chunkwise hybrid that processes chunks in parallel while passing state recurrently, multi-scale retention with different decay rates per head, the swish gate and GroupNorm that complete the architecture, and why this hybrid design — one formula, three computation modes — achieves all three vertices simultaneously — all derived step by step with concrete examples."
date: 2026-04-07
tags: [machine-learning, attention, transformers, retnet, retention, linear-attention, rnn, hybrid-architectures, efficiency]
order: 2
---

The previous twelve blogs modified attention, and Blog 13 replaced it entirely — rewriting the softmax as a kernel and discovering that the result is an RNN with constant-size state. That blog ended with a landscape: the boundary between transformers and RNNs is algebraic, not architectural. But it left an open question: if transformers and RNNs are two views of the same computation, why must we choose one?

Transformers are parallel but expensive at inference ($O(n)$ per token, growing KV cache). RNNs are cheap at inference ($O(1)$ per token, constant state) but sequential during training. Linear attention showed that the same formula can be computed both ways — but its quality lagged behind softmax transformers.

This blog introduces **hybrid architectures**: models designed from the start to have multiple equivalent computation modes. The defining property of a hybrid architecture is that a single mathematical formula admits both a parallel form (for training) and a recurrent form (for inference), producing **identical** outputs — not an approximation, but exact mathematical equivalence. The architecture is neither a transformer nor an RNN. It is both, depending on how you compute it.

We derive the first and cleanest example of this idea: the **retention** mechanism of Sun et al. (2023), the core of the Retentive Network (RetNet). Retention has three computation paradigms — parallel, recurrent, and chunkwise recurrent — all producing identical outputs from a single formula. The parallel form enables GPU-efficient training. The recurrent form enables $O(1)$ inference. The chunkwise form bridges the two for long sequences. Sun et al. call this the **impossible triangle**: training parallelism, low-cost inference, and strong performance. Previous architectures achieved at most two of the three. RetNet claims all three.

The core paper is Sun et al. (2023), "Retentive Network: A Successor to Transformer for Large Language Models."

---

## The Running Example

We continue with the same tiny example from Blog 13:

- $n = 4$ tokens, $d_k = d_v = 2$, single head

with the same query, key, and value matrices:

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}, \quad K = \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \\ 0 & 2 \end{pmatrix}, \quad V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}
$$

For the decay derivations, we fix:

$$
\gamma = 0.9
$$

This means each past token's contribution to the state decays by a factor of $0.9$ per step. A token 10 steps in the past has its influence scaled by $0.9^{10} = 0.349$ — roughly one-third of a token that just arrived.

For cost analysis, we use the same model parameters from the series:

- $d_\text{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$, $L = 12$ layers, fp16

---

## 1. The Two Problems with Linear Attention's Recurrence

### 1.1 Problem 1: Unbounded state accumulation

Blog 13 derived the linear attention recurrence:

$$
S_i = S_{i-1} + \phi(k_i) \, v_i^\top, \qquad z_i = z_{i-1} + \phi(k_i)
$$

Every token adds to the state. Nothing is ever forgotten. Let us trace what happens to the state matrix as tokens accumulate, using the linear attention values from Blog 13 (with $\phi(x) = \text{elu}(x) + 1$):

$$
S_1 = \begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix}, \qquad S_2 = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}, \qquad S_3 = \begin{pmatrix} 3 & 4 \\ 4 & 3 \end{pmatrix}, \qquad S_4 = \begin{pmatrix} 5 & 4 \\ 10 & 3 \end{pmatrix}
$$

The entries of $S$ grow monotonically. To quantify this, we compute the **Frobenius norm** $\|S\|_F = \sqrt{\sum_{i,j} S_{ij}^2}$, which measures the total magnitude of the state:

$$
\|S_1\|_F = \sqrt{1^2 + 0^2 + 2^2 + 0^2} = \sqrt{5} \approx 2.24
$$

$$
\|S_2\|_F = \sqrt{1 + 4 + 4 + 1} = \sqrt{10} \approx 3.16
$$

$$
\|S_3\|_F = \sqrt{9 + 16 + 16 + 9} = \sqrt{50} \approx 7.07
$$

$$
\|S_4\|_F = \sqrt{25 + 16 + 100 + 9} = \sqrt{150} \approx 12.25
$$

The norm grew from 2.24 to 12.25 in just 4 tokens — a $5.5\times$ increase. For a sequence of $n$ tokens, each contributing an outer product of expected magnitude $c$, the state norm grows as $O(n \cdot c)$. At $n = 128{,}000$, the state entries become enormous. The numerical range of $S$ expands without bound, which creates two practical problems:

1. **Precision loss.** In fp16 (the standard training precision), values above 65,504 overflow to infinity. Even before overflow, large values lose precision in the mantissa — small but important contributions from new tokens get rounded away when added to a large accumulated state.

2. **Old information dominates.** Token 1's contribution to $S$ is the same magnitude as token $n$'s, regardless of how far apart they are. In language modeling, a token 100,000 positions ago is almost certainly less relevant than a token 10 positions ago. But the accumulate-only recurrence treats them identically.

### 1.2 Problem 2: No position information

In standard softmax attention, the similarity $\exp(q_i^\top k_j / \sqrt{d_k})$ is typically augmented with position encodings — either absolute (Vaswani et al., 2017) or relative (Su et al., 2021). These encodings allow the model to distinguish "token $j$ is 3 positions before token $i$" from "token $j$ is 300 positions before token $i$."

In linear attention, the kernel similarity $\phi(q_i)^\top \phi(k_j)$ has no position dependence. The feature maps $\phi(q_i)$ and $\phi(k_j)$ depend only on the content of the query and key vectors, not on their positions $i$ and $j$. The recurrent state $S_i = \sum_{j=1}^i \phi(k_j) v_j^\top$ is a sum of outer products where each outer product carries no information about when it was added.

This means the model cannot learn position-dependent patterns like "the verb usually follows the subject within 5 tokens" or "the closing bracket matches the most recent opening bracket." The decay factor in the normalization $z_i = \sum_{j=1}^i \phi(k_j)$ is uniform across all positions — a crude tool that cannot distinguish distances.

### 1.3 What we need

We need two modifications to the linear attention recurrence:

1. **A forgetting mechanism** that decays the contribution of old tokens, keeping the state bounded and prioritizing recent information.
2. **Position encoding** that makes the query-key interaction depend on relative position $(i - j)$, not just content.

RetNet achieves both. The forgetting mechanism is an exponential decay factor $\gamma \in (0, 1)$ applied to the state at every step. The position encoding comes from complex exponentials $e^{i n \theta}$ that rotate the query and key vectors based on their absolute positions, producing a similarity that depends on relative position. We will derive each modification from scratch.

---

## 2. The Retention Recurrence

### 2.1 Adding decay

The simplest fix for unbounded accumulation is to multiply the old state by a scalar $\gamma \in (0, 1)$ at every step. This gives the **retention recurrence**:

$$
\boxed{S_n = \gamma \, S_{n-1} + k_n \, v_n^\top}
$$

$$
o_n = q_n^\top \, S_n
$$

where $S_n \in \mathbb{R}^{d_k \times d_v}$ is the state matrix, $k_n \in \mathbb{R}^{d_k}$ is the key vector for token $n$, $v_n \in \mathbb{R}^{d_v}$ is the value vector, $q_n \in \mathbb{R}^{d_k}$ is the query vector, and $o_n \in \mathbb{R}^{d_v}$ is the output. The initial state is $S_0 = 0$ (the zero matrix).

Compare this to the linear attention recurrence from Blog 13:

$$
S_n^{\text{linear}} = S_{n-1}^{\text{linear}} + \phi(k_n) \, v_n^\top
$$

Two differences:

1. **The factor $\gamma$.** The old state is scaled by $\gamma$ before the new token's contribution is added. When $\gamma = 1$, this reduces to linear attention's accumulation (without the kernel). When $\gamma < 1$, older information exponentially decays.

2. **No kernel feature map.** Retention uses the raw key vector $k_n$, not a transformed version $\phi(k_n)$. There is no elu+1 or any other kernel. This means the query-key product $q_n^\top k_m$ can be negative — retention does not produce non-negative attention weights. The normalization comes from GroupNorm applied to the output (Section 8), not from a denominator like linear attention's $\phi(q_i)^\top z_i$.

### 2.2 Unrolling the recurrence

Let us expand $S_n$ by repeatedly substituting the recurrence. This is the technique of **unrolling a recurrence relation** — replacing each $S_{i}$ with its definition in terms of $S_{i-1}$ until we reach the base case $S_0 = 0$.

$$
S_n = \gamma \, S_{n-1} + k_n \, v_n^\top
$$

Substitute $S_{n-1} = \gamma \, S_{n-2} + k_{n-1} \, v_{n-1}^\top$:

$$
S_n = \gamma (\gamma \, S_{n-2} + k_{n-1} \, v_{n-1}^\top) + k_n \, v_n^\top = \gamma^2 \, S_{n-2} + \gamma \, k_{n-1} \, v_{n-1}^\top + k_n \, v_n^\top
$$

Substitute $S_{n-2} = \gamma \, S_{n-3} + k_{n-2} \, v_{n-2}^\top$:

$$
S_n = \gamma^3 \, S_{n-3} + \gamma^2 \, k_{n-2} \, v_{n-2}^\top + \gamma \, k_{n-1} \, v_{n-1}^\top + k_n \, v_n^\top
$$

The pattern is clear. After $n$ substitutions, we reach $S_0 = 0$ and the $\gamma^n S_0$ term vanishes:

$$
\boxed{S_n = \sum_{m=1}^{n} \gamma^{n-m} \, k_m \, v_m^\top}
$$

Each token $m$ contributes the outer product $k_m v_m^\top$, scaled by $\gamma^{n-m}$. The exponent $n - m$ is the distance from token $m$ to the current position $n$.

The output for token $n$ is:

$$
o_n = q_n^\top S_n = q_n^\top \sum_{m=1}^{n} \gamma^{n-m} \, k_m \, v_m^\top = \sum_{m=1}^{n} \gamma^{n-m} \, (q_n^\top k_m) \, v_m^\top
$$

Since $q_n^\top k_m$ is a scalar and $v_m^\top$ is a row vector, we can write the output as a row vector:

$$
\boxed{o_n^\top = \sum_{m=1}^{n} \gamma^{n-m} \, (q_n^\top k_m) \, v_m^\top}
$$

This says: the output for token $n$ is a weighted sum of all past value vectors $v_1, \ldots, v_n$. The weight on value $v_m$ is the product of two factors: the content similarity $q_n^\top k_m$ (how relevant is token $m$ to query $n$?) and the decay $\gamma^{n-m}$ (how far away is token $m$?).

### 2.3 What exponential decay means

The decay factor $\gamma^{n-m}$ is an **exponentially decaying function** of the distance $n - m$. Let us compute its values for $\gamma = 0.9$:

| Distance $n - m$ | $\gamma^{n-m}$ | Interpretation |
|---|---|---|
| 0 | $0.9^0 = 1.000$ | Current token — full weight |
| 1 | $0.9^1 = 0.900$ | Previous token — 90% |
| 2 | $0.9^2 = 0.810$ | 2 tokens ago — 81% |
| 5 | $0.9^5 = 0.590$ | 5 tokens ago — 59% |
| 10 | $0.9^{10} = 0.349$ | 10 tokens ago — 35% |
| 50 | $0.9^{50} = 0.0052$ | 50 tokens ago — 0.5% |
| 100 | $0.9^{100} = 2.66 \times 10^{-5}$ | 100 tokens ago — negligible |

With $\gamma = 0.9$, tokens more than 50 positions ago contribute less than 1% of their original weight. The model has a soft attention window: it can see all past tokens, but overwhelmingly focuses on recent ones.

The **effective window size** — the distance at which the decay drops to some threshold $\epsilon$ — is:

$$
\gamma^{d} = \epsilon \implies d = \frac{\log \epsilon}{\log \gamma}
$$

This follows by taking the **natural logarithm** of both sides and dividing. For $\gamma = 0.9$ and $\epsilon = 0.01$:

$$
d = \frac{\log 0.01}{\log 0.9} = \frac{-4.605}{-0.105} \approx 43.7
$$

So the effective window is about 44 tokens. The choice of $\gamma$ controls the tradeoff between long-range and short-range attention. Higher $\gamma$ (closer to 1) gives longer effective windows; lower $\gamma$ gives shorter ones. RetNet uses different $\gamma$ values for different heads — we will derive this in Section 7.

### 2.4 The bounded state property

Unlike linear attention, the retention state is bounded. Each entry of $S_n$ is a sum of decaying contributions:

$$
(S_n)_{ij} = \sum_{m=1}^{n} \gamma^{n-m} (k_m)_i (v_m)_j
$$

Assuming each $(k_m)_i (v_m)_j$ is bounded by some constant $c$, the sum is bounded by a **geometric series**:

$$
|(S_n)_{ij}| \leq c \sum_{m=1}^{n} \gamma^{n-m} = c \sum_{d=0}^{n-1} \gamma^d = c \cdot \frac{1 - \gamma^n}{1 - \gamma}
$$

The last equality uses the **geometric series partial sum formula** $\sum_{d=0}^{N-1} \gamma^d = \frac{1 - \gamma^N}{1 - \gamma}$.

As $n \to \infty$, $\gamma^n \to 0$ (since $0 < \gamma < 1$), so:

$$
|(S_n)_{ij}| \leq \frac{c}{1 - \gamma}
$$

For $\gamma = 0.9$: $\frac{c}{1 - 0.9} = 10c$. The state entries are bounded by 10 times the maximum single-token contribution. No matter how long the sequence, the state cannot grow beyond this bound. This is the **bounded geometric series limit**, and it eliminates the precision loss problem of linear attention.

---

## 3. Tracing the Retention Recurrence

### 3.1 Step-by-step computation

Let us trace the retention recurrence for all 4 tokens with $\gamma = 0.9$, using the raw $Q$, $K$, $V$ matrices (no kernel feature map).

**Step $n = 1$:**

$$
S_1 = \gamma \cdot S_0 + k_1 v_1^\top = 0.9 \times 0 + \begin{pmatrix} 0 \\ 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}
$$

$$
o_1^\top = q_1^\top S_1 = \begin{pmatrix} 1 & 0 \end{pmatrix} \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \end{pmatrix}
$$

Token 1 can only attend to itself. The query-key similarity is $q_1^\top k_1 = (1)(0) + (0)(1) = 0$ — query 1 and key 1 are orthogonal. In softmax attention, this would still produce a nonzero output (because $\exp(0) = 1 > 0$). In retention, zero similarity means zero output. The GroupNorm applied later (Section 8) will handle the scaling.

**Step $n = 2$:**

$$
S_2 = 0.9 \times \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix} + \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0.9 & 0 \end{pmatrix} + \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 0.9 & 0 \end{pmatrix}
$$

The old state $S_1$ was decayed by $\gamma = 0.9$. The entry $(S_1)_{21} = 1$ (from token 1's contribution) became $0.9$ in $S_2$. Token 2's contribution $k_2 v_2^\top$ was added at full strength.

$$
o_2^\top = q_2^\top S_2 = \begin{pmatrix} 0 & 1 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 0.9 & 0 \end{pmatrix} = \begin{pmatrix} 0 \cdot 0 + 1 \cdot 0.9 & \;\; 0 \cdot 1 + 1 \cdot 0 \end{pmatrix} = \begin{pmatrix} 0.9 & 0 \end{pmatrix}
$$

**Numerical check.** We can verify this directly from the unrolled formula:

$$
o_2^\top = \gamma^1 (q_2^\top k_1) v_1^\top + \gamma^0 (q_2^\top k_2) v_2^\top
$$

$$
q_2^\top k_1 = (0)(0) + (1)(1) = 1, \qquad q_2^\top k_2 = (0)(1) + (1)(0) = 0
$$

$$
o_2^\top = 0.9 \times 1 \times (1, 0) + 1 \times 0 \times (0, 1) = (0.9, 0) + (0, 0) = (0.9, 0) \quad \checkmark
$$

Token 2 attends to token 1 with similarity 1, decayed by $\gamma^1 = 0.9$. It attends to itself with similarity 0. So the output is dominated by token 1's value vector.

**Step $n = 3$:**

$$
S_3 = 0.9 \times \begin{pmatrix} 0 & 1 \\ 0.9 & 0 \end{pmatrix} + \begin{pmatrix} 1 \\ 1 \end{pmatrix} \begin{pmatrix} 1 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0.9 \\ 0.81 & 0 \end{pmatrix} + \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1.9 \\ 1.81 & 1 \end{pmatrix}
$$

Notice that the entry $(S_1)_{21} = 1$ from token 1 has now decayed to $0.81 = 0.9^2 = \gamma^2$, exactly as predicted by the formula $\gamma^{n-m} = \gamma^{3-1} = 0.81$.

$$
o_3^\top = q_3^\top S_3 = \begin{pmatrix} 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 1.9 \\ 1.81 & 1 \end{pmatrix} = \begin{pmatrix} 1 + 1.81 & \;\; 1.9 + 1 \end{pmatrix} = \begin{pmatrix} 2.81 & 2.9 \end{pmatrix}
$$

**Step $n = 4$:**

$$
S_4 = 0.9 \times \begin{pmatrix} 1 & 1.9 \\ 1.81 & 1 \end{pmatrix} + \begin{pmatrix} 0 \\ 2 \end{pmatrix} \begin{pmatrix} 2 & 0 \end{pmatrix} = \begin{pmatrix} 0.9 & 1.71 \\ 1.629 & 0.9 \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 4 & 0 \end{pmatrix} = \begin{pmatrix} 0.9 & 1.71 \\ 5.629 & 0.9 \end{pmatrix}
$$

$$
o_4^\top = q_4^\top S_4 = \begin{pmatrix} 2 & 0 \end{pmatrix} \begin{pmatrix} 0.9 & 1.71 \\ 5.629 & 0.9 \end{pmatrix} = \begin{pmatrix} 1.8 & 3.42 \end{pmatrix}
$$

### 3.2 Summary of outputs

| Token $n$ | $o_n^\top$ |
|---|---|
| 1 | $(0, \; 0)$ |
| 2 | $(0.9, \; 0)$ |
| 3 | $(2.81, \; 2.9)$ |
| 4 | $(1.8, \; 3.42)$ |

### 3.3 State norm comparison

$$
\|S_1\|_F = 1.00, \quad \|S_2\|_F = \sqrt{0 + 1 + 0.81 + 0} = \sqrt{1.81} \approx 1.35
$$

$$
\|S_3\|_F = \sqrt{1 + 3.61 + 3.276 + 1} = \sqrt{8.886} \approx 2.98
$$

$$
\|S_4\|_F = \sqrt{0.81 + 2.924 + 31.686 + 0.81} = \sqrt{36.23} \approx 6.02
$$

Compare to the linear attention state norms (from Section 1.1): $2.24 \to 3.16 \to 7.07 \to 12.25$. The retention state norms are: $1.00 \to 1.35 \to 2.98 \to 6.02$. The retention state is growing more slowly because the decay factor $\gamma = 0.9$ shrinks old contributions at every step. For long sequences, the retention state converges to a bounded value while the linear attention state grows without bound.

---

## 4. The Parallel Form

### 4.1 From recurrence to matrix form

The recurrent form is ideal for inference (one token at a time), but it is sequential — $S_n$ depends on $S_{n-1}$, which depends on $S_{n-2}$, and so on. During training, we process the entire sequence at once and need a parallel computation.

We already derived the unrolled output:

$$
o_n^\top = \sum_{m=1}^{n} \gamma^{n-m} (q_n^\top k_m) \, v_m^\top
$$

This is a weighted combination of value vectors. The weight on $v_m$ is:

$$
w_{nm} = \gamma^{n-m} (q_n^\top k_m) \quad \text{for } m \leq n, \qquad w_{nm} = 0 \quad \text{for } m > n
$$

The first factor $q_n^\top k_m$ is the $(n, m)$ entry of the matrix $QK^\top$. The second factor $\gamma^{n-m}$ is a function of the distance $n - m$ only, and is zero for $m > n$ (causal masking). We can write this as a single matrix.

### 4.2 The $D$ matrix

Define the **decay matrix** $D \in \mathbb{R}^{n \times n}$ as:

$$
\boxed{D_{nm} = \begin{cases} \gamma^{n-m} & \text{if } n \geq m \\ 0 & \text{if } n < m \end{cases}}
$$

This matrix combines two things into one: **causal masking** (the zero entries above the diagonal ensure token $n$ cannot attend to future tokens $m > n$) and **exponential decay** (the entry $\gamma^{n-m}$ weights past tokens by their distance).

For our running example with $n = 4$ and $\gamma = 0.9$:

$$
D = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0.9 & 1 & 0 & 0 \\ 0.81 & 0.9 & 1 & 0 \\ 0.729 & 0.81 & 0.9 & 1 \end{pmatrix}
$$

The diagonal is all 1's (each token attends to itself with no decay). The first column decays as $1, 0.9, 0.81, 0.729$ — token 1's influence fades as we move forward. The upper triangle is all zeros — no future peeking.

Compare this to the standard causal mask in softmax attention, which is:

$$
M = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{pmatrix}
$$

The standard mask uses 1 for all allowed positions — no decay. $D$ is a generalization: it is a causal mask where the allowed entries are weighted by exponential decay instead of being uniformly 1. When $\gamma = 1$, $D$ reduces to $M$.

### 4.3 The parallel retention formula

The full parallel computation is:

$$
\boxed{\text{Retention}(X) = (QK^\top \odot D) \, V}
$$

where $\odot$ is the **Hadamard product** (element-wise multiplication, defined in Blog 12). This says: compute the $n \times n$ query-key similarity matrix $QK^\top$, multiply it element-wise by the decay matrix $D$ (which simultaneously applies causal masking and exponential decay), then multiply by the value matrix $V$.

### 4.4 Numerical verification

Let us compute the parallel form and verify it matches the recurrent outputs from Section 3.

**Step 1: Compute $QK^\top$.**

$$
QK^\top = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix} \begin{pmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 2 \end{pmatrix}
$$

Row 1: $(1 \cdot 0 + 0 \cdot 1, \; 1 \cdot 1 + 0 \cdot 0, \; 1 \cdot 1 + 0 \cdot 1, \; 1 \cdot 0 + 0 \cdot 2) = (0, 1, 1, 0)$

Row 2: $(0 \cdot 0 + 1 \cdot 1, \; 0 \cdot 1 + 1 \cdot 0, \; 0 \cdot 1 + 1 \cdot 1, \; 0 \cdot 0 + 1 \cdot 2) = (1, 0, 1, 2)$

Row 3: $(1 \cdot 0 + 1 \cdot 1, \; 1 \cdot 1 + 1 \cdot 0, \; 1 \cdot 1 + 1 \cdot 1, \; 1 \cdot 0 + 1 \cdot 2) = (1, 1, 2, 2)$

Row 4: $(2 \cdot 0 + 0 \cdot 1, \; 2 \cdot 1 + 0 \cdot 0, \; 2 \cdot 1 + 0 \cdot 1, \; 2 \cdot 0 + 0 \cdot 2) = (0, 2, 2, 0)$

$$
QK^\top = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 2 \\ 1 & 1 & 2 & 2 \\ 0 & 2 & 2 & 0 \end{pmatrix}
$$

**Step 2: Apply $D$ via Hadamard product.**

$$
QK^\top \odot D = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 2 \\ 1 & 1 & 2 & 2 \\ 0 & 2 & 2 & 0 \end{pmatrix} \odot \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0.9 & 1 & 0 & 0 \\ 0.81 & 0.9 & 1 & 0 \\ 0.729 & 0.81 & 0.9 & 1 \end{pmatrix}
$$

Row 1: $(0 \cdot 1, \; 1 \cdot 0, \; 1 \cdot 0, \; 0 \cdot 0) = (0, 0, 0, 0)$

Row 2: $(1 \cdot 0.9, \; 0 \cdot 1, \; 1 \cdot 0, \; 2 \cdot 0) = (0.9, 0, 0, 0)$

Row 3: $(1 \cdot 0.81, \; 1 \cdot 0.9, \; 2 \cdot 1, \; 2 \cdot 0) = (0.81, 0.9, 2, 0)$

Row 4: $(0 \cdot 0.729, \; 2 \cdot 0.81, \; 2 \cdot 0.9, \; 0 \cdot 1) = (0, 1.62, 1.8, 0)$

$$
QK^\top \odot D = \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0.9 & 0 & 0 & 0 \\ 0.81 & 0.9 & 2 & 0 \\ 0 & 1.62 & 1.8 & 0 \end{pmatrix}
$$

This is the **retention matrix** — the analog of the attention weight matrix in softmax attention. But unlike softmax attention weights, these entries are not normalized to sum to 1, and they can be negative (though in this example they happen to be non-negative because all $q_n^\top k_m$ are non-negative for our particular $Q$ and $K$).

**Step 3: Multiply by $V$.**

$$
(QK^\top \odot D) \, V = \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0.9 & 0 & 0 & 0 \\ 0.81 & 0.9 & 2 & 0 \\ 0 & 1.62 & 1.8 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}
$$

Row 1: $0 \cdot (1, 0) + 0 \cdot (0, 1) + 0 \cdot (1, 1) + 0 \cdot (2, 0) = (0, 0)$

Row 2: $0.9 \cdot (1, 0) + 0 \cdot (0, 1) + 0 \cdot (1, 1) + 0 \cdot (2, 0) = (0.9, 0)$

Row 3: $0.81 \cdot (1, 0) + 0.9 \cdot (0, 1) + 2 \cdot (1, 1) + 0 \cdot (2, 0) = (0.81 + 2, \; 0.9 + 2) = (2.81, 2.9)$

Row 4: $0 \cdot (1, 0) + 1.62 \cdot (0, 1) + 1.8 \cdot (1, 1) + 0 \cdot (2, 0) = (1.8, \; 1.62 + 1.8) = (1.8, 3.42)$

$$
\text{Retention}(X) = \begin{pmatrix} 0 & 0 \\ 0.9 & 0 \\ 2.81 & 2.9 \\ 1.8 & 3.42 \end{pmatrix}
$$

**Verification:** Compare each row with the recurrent outputs from Section 3.2:

| Token | Recurrent $o_n^\top$ | Parallel row $n$ | Match? |
|---|---|---|---|
| 1 | $(0, 0)$ | $(0, 0)$ | $\checkmark$ |
| 2 | $(0.9, 0)$ | $(0.9, 0)$ | $\checkmark$ |
| 3 | $(2.81, 2.9)$ | $(2.81, 2.9)$ | $\checkmark$ |
| 4 | $(1.8, 3.42)$ | $(1.8, 3.42)$ | $\checkmark$ |

Both forms produce identical outputs. The parallel form computed everything at once using matrix operations. The recurrent form computed outputs one at a time using state updates. The mathematics guarantees they are equivalent.

---

## 5. The Hybrid Architecture: One Formula, Multiple Computation Modes

This is the defining section of this blog. Everything we have derived so far leads here: the retention formula is a single mathematical object that can be computed in fundamentally different ways depending on the context. This is what makes RetNet a **hybrid architecture** — not a transformer, not an RNN, but a single model that computes as a transformer during training and as an RNN during inference, with exact equivalence.

### 5.1 Training uses the parallel form

During training, the full sequence is available. The parallel form $\text{Retention}(X) = (QK^\top \odot D) V$ is a sequence of matrix multiplications and a Hadamard product — operations that GPUs execute efficiently in parallel. The cost is:

- $QK^\top$: $(n \times d_k) \times (d_k \times n) = O(n^2 d_k)$
- Hadamard product with $D$: $O(n^2)$
- Multiply by $V$: $(n \times n) \times (n \times d_v) = O(n^2 d_v)$

Total: $O(n^2 d_k)$ per head. This is the same asymptotic cost as softmax attention. The advantage is not in the asymptotic complexity — it is in the simplicity. There is no softmax (which requires a sequential max-subtraction for numerical stability), no exponential, no normalization denominator. Just matrix multiply, Hadamard product, matrix multiply.

### 5.2 Inference uses the recurrent form

During autoregressive generation, we process one token at a time. The recurrent form $S_n = \gamma S_{n-1} + k_n v_n^\top$, $o_n = q_n^\top S_n$ has:

- **State size:** $d_k \times d_v$ per head. With $d_k = d_v = 64$: $64 \times 64 = 4{,}096$ elements per head. Across $h = 8$ heads and $L = 12$ layers, in fp16: $8 \times 12 \times 4{,}096 \times 2 = 786{,}432$ bytes $\approx 0.75$ MB.

- **Compute per token:** One scalar-matrix multiply ($\gamma S_{n-1}$: $O(d_k d_v)$), one outer product ($k_n v_n^\top$: $O(d_k d_v)$), one matrix-vector product ($q_n^\top S_n$: $O(d_k d_v)$). Total: $O(d_k d_v) = O(d_k^2)$ per head.

Both are **constant** — independent of how many tokens have been generated. Compare to softmax attention, where the KV cache grows as $O(n \cdot d_k)$ per head and the compute per token grows as $O(n \cdot d_k)$:

| Property | Softmax attention | Retention (recurrent) |
|---|---|---|
| State per head at step $n$ | $2n \cdot d_k$ elements (KV cache) | $d_k \times d_v$ elements (constant) |
| Compute per token at step $n$ | $O(n \cdot d_k)$ (grows) | $O(d_k^2)$ (constant) |
| State at $n = 128{,}000$, our model | 3.15 GB | 0.75 MB |

The recurrent retention state is $4{,}200\times$ smaller than the KV cache at 128K tokens.

### 5.3 The impossible triangle and why hybrid architectures matter

The parallel and recurrent forms compute the same function. This is the core property of a hybrid architecture: you choose the computation mode based on the hardware context, not the mathematical definition. Training uses the parallel form because GPUs are parallel processors. Inference uses the recurrent form because autoregressive generation is inherently sequential. The same weights, the same function, different execution strategies.

Before hybrid architectures, the field was stuck in a tradeoff. Sun et al. (2023) call it the **impossible triangle**: training parallelism, low-cost inference, and strong performance. Every architecture achieved at most two of the three:

| Architecture | Training parallelism | $O(1)$ inference | Strong performance |
|---|---|---|---|
| Transformer | $\checkmark$ | $\times$ | $\checkmark\checkmark$ |
| Linear Transformer | $\checkmark$ | $\checkmark$ | $\times$ |
| Recurrent NN | $\times$ | $\checkmark$ | $\times$ |
| RWKV | $\times$ | $\checkmark$ | $\checkmark$ |
| H3/S4 | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Hyena | $\checkmark$ | $O(n)$ | $\checkmark$ |
| **RetNet** | $\checkmark$ | $\checkmark$ | $\checkmark\checkmark$ |

RetNet claims all three vertices of the triangle. The rest of this blog derives the additional components — position encoding, chunkwise computation, multi-scale heads, and gating — that make this possible.

---

## 6. Position Encoding via Complex Exponentials

### 6.1 The problem

The retention formula $o_n^\top = \sum_{m=1}^n \gamma^{n-m} (q_n^\top k_m) v_m^\top$ has a position-dependent factor ($\gamma^{n-m}$, which depends on the distance), but the query-key interaction $q_n^\top k_m$ depends only on the content of $q_n$ and $k_m$, not on their positions $n$ and $m$.

Consider two scenarios:
- Query at position 10, key at position 8: $q_{10}^\top k_8$ with decay $\gamma^2$
- Query at position 1000, key at position 998: $q_{1000}^\top k_{998}$ with decay $\gamma^2$

If $q_{10} = q_{1000}$ and $k_8 = k_{998}$ (same content), both scenarios produce the same output. The model cannot learn that position-dependent patterns like "subject-verb agreement" work differently at the start vs the middle of a document. The decay factor provides only a distance-based weighting, not a full relative position encoding.

### 6.2 The general state transition matrix

To add position encoding, we generalize the retention recurrence. Instead of the scalar decay $\gamma$, we use a matrix $A \in \mathbb{R}^{d \times d}$ as the state transition:

$$
s_n = A \, s_{n-1} + k_n \, v_n^\top
$$

Unrolling this gives:

$$
s_n = \sum_{m=1}^{n} A^{n-m} \, k_m \, v_m^\top
$$

The output is:

$$
o_n = q_n^\top s_n = \sum_{m=1}^{n} q_n^\top A^{n-m} k_m \, v_m^\top
$$

Now $A^{n-m}$ is a $d \times d$ matrix raised to the power $(n-m)$. Computing $A^{n-m}$ directly is expensive — matrix exponentiation costs $O(d^3 \log(n-m))$. But if we **diagonalize** $A$, the computation simplifies dramatically.

### 6.3 Diagonalizing $A$

By the **eigendecomposition theorem**, if $A$ has $d$ linearly independent eigenvectors, we can write:

$$
A = \Lambda \, \text{diag}(\lambda_1, \ldots, \lambda_d) \, \Lambda^{-1}
$$

where $\Lambda$ is the matrix of eigenvectors (columns are eigenvectors) and $\lambda_1, \ldots, \lambda_d$ are the eigenvalues. The notation $\text{diag}(\lambda_1, \ldots, \lambda_d)$ denotes the diagonal matrix with $\lambda_i$ on the $i$-th diagonal entry.

The key property of the eigendecomposition is that matrix powers become trivial:

$$
A^{n-m} = \Lambda \, \text{diag}(\lambda_1^{n-m}, \ldots, \lambda_d^{n-m}) \, \Lambda^{-1}
$$

This follows because $A^2 = (\Lambda D_\lambda \Lambda^{-1})(\Lambda D_\lambda \Lambda^{-1}) = \Lambda D_\lambda ({\Lambda^{-1} \Lambda}) D_\lambda \Lambda^{-1} = \Lambda D_\lambda^2 \Lambda^{-1}$, where the $\Lambda^{-1} \Lambda = I$ cancellation in the middle is the crucial step. By induction, $A^k = \Lambda D_\lambda^k \Lambda^{-1}$, and $D_\lambda^k = \text{diag}(\lambda_1^k, \ldots, \lambda_d^k)$ because powers of a diagonal matrix are diagonal with powered entries.

### 6.4 Absorbing $\Lambda$ into the projections

Substituting the eigendecomposition into the output:

$$
o_n^\top = \sum_{m=1}^{n} q_n^\top \Lambda \, \text{diag}(\lambda_i^{n-m}) \, \Lambda^{-1} k_m \, v_m^\top
$$

Define new query and key vectors that absorb the eigenvector matrices:

$$
\tilde{q}_n^\top = q_n^\top \Lambda, \qquad \tilde{k}_m = \Lambda^{-1} k_m
$$

Since $q_n = W_Q^\top x_n$ and $k_m = W_K^\top x_m$ for learned projection matrices $W_Q, W_K$, absorbing $\Lambda$ into the projections means defining:

$$
\tilde{W}_Q = W_Q \Lambda, \qquad \tilde{W}_K = (\Lambda^{-1})^\top W_K
$$

These are just different learned matrices. Since $W_Q$ and $W_K$ are learned from data, absorbing $\Lambda$ changes nothing about the model's expressivity — the optimizer will find appropriate values for $\tilde{W}_Q$ and $\tilde{W}_K$. After this absorption:

$$
o_n^\top = \sum_{m=1}^{n} \tilde{q}_n^\top \, \text{diag}(\lambda_i^{n-m}) \, \tilde{k}_m \, v_m^\top
$$

The matrix $\text{diag}(\lambda_i^{n-m})$ is diagonal, so the product $\tilde{q}_n^\top \, \text{diag}(\lambda_i^{n-m}) \, \tilde{k}_m$ can be written element-wise:

$$
\tilde{q}_n^\top \, \text{diag}(\lambda_i^{n-m}) \, \tilde{k}_m = \sum_{p=1}^{d} (\tilde{q}_n)_p \, \lambda_p^{n-m} \, (\tilde{k}_m)_p
$$

This is a sum of $d$ terms, each involving one eigenvalue $\lambda_p$.

### 6.5 Choosing eigenvalues: $\gamma e^{i\theta}$

RetNet chooses the eigenvalues to be complex numbers of the form:

$$
\lambda_p = \gamma \, e^{i \theta_p}
$$

where $\gamma \in (0, 1)$ is a scalar (the same decay rate for all dimensions within a head) and $\theta_p \in \mathbb{R}$ is a different angle for each dimension $p$. The notation $e^{i\theta}$ is **Euler's formula**: $e^{i\theta} = \cos\theta + i \sin\theta$, where $i = \sqrt{-1}$ is the imaginary unit.

Why this specific form? Because it separates two functions:

1. **The magnitude $\gamma$** controls decay. Since $|\lambda_p| = |\gamma e^{i\theta_p}| = \gamma \cdot |e^{i\theta_p}| = \gamma \cdot 1 = \gamma$, the magnitude of each eigenvalue is $\gamma$. This means $|\lambda_p^{n-m}| = \gamma^{n-m}$ — exponential decay with distance, exactly as before.

2. **The phase $e^{i\theta_p}$** controls rotation. The factor $e^{i(n-m)\theta_p}$ depends on the distance $n - m$ and the dimension-specific angle $\theta_p$. Different dimensions rotate at different frequencies, creating a rich encoding of relative position.

The power $\lambda_p^{n-m}$ factors as:

$$
\lambda_p^{n-m} = (\gamma e^{i\theta_p})^{n-m} = \gamma^{n-m} \, e^{i(n-m)\theta_p}
$$

This is the product of a decay term (real, positive, decreasing) and a rotation term (complex, unit magnitude, oscillating).

### 6.6 The factored form

We can further factor the position-dependent term. Since $\gamma$ is a scalar (same for all dimensions):

$$
\lambda_p^{n-m} = \gamma^{n-m} \, e^{i(n-m)\theta_p} = \gamma^{n-m} \, e^{in\theta_p} \, e^{-im\theta_p}
$$

The last step uses the **exponential product rule** $e^{a+b} = e^a e^b$, applied to $e^{i(n-m)\theta_p} = e^{in\theta_p - im\theta_p} = e^{in\theta_p} e^{-im\theta_p}$.

Substituting into the output:

$$
o_n^\top = \sum_{m=1}^{n} \gamma^{n-m} \left(\sum_{p=1}^{d} (\tilde{q}_n)_p \, e^{in\theta_p} \cdot (\tilde{k}_m)_p \, e^{-im\theta_p}\right) v_m^\top
$$

Define position-encoded queries and keys:

$$
(Q_n)_p = (\tilde{q}_n)_p \, e^{in\theta_p}, \qquad (K_m)_p = (\tilde{k}_m)_p \, e^{-im\theta_p}
$$

In vector notation, using $\odot$ for element-wise multiplication:

$$
\boxed{Q_n = \tilde{q}_n \odot \Theta_n, \qquad K_m = \tilde{k}_m \odot \bar{\Theta}_m}
$$

where $\Theta_n = (e^{in\theta_1}, e^{in\theta_2}, \ldots, e^{in\theta_d})$ and $\bar{\Theta}_m = (e^{-im\theta_1}, e^{-im\theta_2}, \ldots, e^{-im\theta_d})$ is its **complex conjugate** (the complex conjugate of $e^{i\alpha}$ is $e^{-i\alpha}$).

Then the inner sum becomes:

$$
\sum_{p=1}^{d} (Q_n)_p \cdot (K_m)_p = Q_n^\top K_m
$$

where the transpose here is the regular transpose (not conjugate transpose), because the conjugation is already built into $K_m$ through $\bar{\Theta}_m$.

The full output is:

$$
o_n^\top = \sum_{m=1}^{n} \gamma^{n-m} (Q_n^\top K_m) \, v_m^\top
$$

This has exactly the same form as Section 2.2, but now $Q_n$ and $K_m$ carry position information through the complex exponential factors. The parallel form becomes:

$$
\boxed{\text{Retention}(X) = (QK^\top \odot D) \, V}
$$

with the position-encoded $Q = (XW_Q) \odot \Theta$, $K = (XW_K) \odot \bar{\Theta}$, $V = XW_V$, and $D_{nm} = \gamma^{n-m}$ for $n \geq m$, zero otherwise.

### 6.7 The relative position property

This is the crucial observation. The product $Q_n^\top K_m$ expands as:

$$
Q_n^\top K_m = \sum_{p=1}^{d} (\tilde{q}_n)_p \, e^{in\theta_p} \cdot (\tilde{k}_m)_p \, e^{-im\theta_p} = \sum_{p=1}^{d} (\tilde{q}_n)_p \, (\tilde{k}_m)_p \, e^{i(n-m)\theta_p}
$$

The complex exponential $e^{i(n-m)\theta_p}$ depends only on the **relative position** $n - m$, not on the absolute positions $n$ and $m$ separately. This is precisely the property of **relative position encodings** like RoPE (Su et al., 2021) and xPos (Sun et al., 2022). The RetNet paper notes that this formulation is equivalent to xPos — the same mechanism proposed for length-extrapolatable transformers, here derived naturally from the eigendecomposition of the state transition matrix.

### 6.8 Practical implementation

In practice, the complex arithmetic is implemented using real numbers. For each pair of consecutive dimensions $(2p, 2p+1)$, the rotation $e^{i(n-m)\theta_p}$ is applied as a $2 \times 2$ rotation matrix:

$$
\begin{pmatrix} \cos((n-m)\theta_p) & -\sin((n-m)\theta_p) \\ \sin((n-m)\theta_p) & \cos((n-m)\theta_p) \end{pmatrix}
$$

This is the standard technique used by RoPE and xPos: pair up dimensions, rotate each pair by an angle proportional to the position, and different pairs use different base frequencies $\theta_p$.

### 6.9 Numerical example: position encoding effect

To see the effect concretely, consider a single dimension pair with $\theta = \pi/4$. For query at position $n = 4$ and keys at positions $m = 1, 2, 3, 4$:

| Distance $n - m$ | $e^{i(n-m)\theta}$ | $\cos((n-m)\pi/4)$ | $\sin((n-m)\pi/4)$ |
|---|---|---|---|
| 3 | $e^{i \cdot 3\pi/4}$ | $-0.707$ | $0.707$ |
| 2 | $e^{i \cdot 2\pi/4}$ | $0$ | $1$ |
| 1 | $e^{i \cdot \pi/4}$ | $0.707$ | $0.707$ |
| 0 | $e^{i \cdot 0}$ | $1$ | $0$ |

The rotation factor oscillates with distance. A key 3 positions away gets its first dimension component flipped in sign ($\cos(3\pi/4) = -0.707$), while a key 1 position away gets a positive contribution ($\cos(\pi/4) = 0.707$). Combined with the decay $\gamma^{n-m}$, this creates a rich position-dependent similarity landscape: the model can learn to prefer keys at specific relative positions, not just nearby keys.

---

## 7. The Chunkwise Recurrent Form

### 7.1 The motivation

The parallel form has cost $O(n^2 d_k)$ — good for moderate sequences, but quadratic in $n$. The recurrent form has cost $O(n d_k^2)$ — linear in $n$, but sequential (each step depends on the previous state). For long sequences during training, we want the best of both: parallel computation where possible, sequential state passing where necessary.

The **chunkwise recurrent** form divides the sequence into chunks of size $B$. Within each chunk, retention is computed in parallel using the parallel form. Across chunks, the state is passed recurrently. This gives:

- Parallelism within each chunk (GPU-efficient)
- Linear memory across chunks (no $O(n^2)$ matrix)
- Total cost: $O(n \cdot d_k \cdot (B + d_k))$ — when $B$ and $d_k$ are much smaller than $n$, this is linear in $n$

### 7.2 Derivation

Consider chunk $[i]$ containing tokens $(i-1)B + 1$ through $iB$. For notational simplicity, we write $Q_{[i]}, K_{[i]}, V_{[i]}$ for the query, key, and value matrices restricted to this chunk (each is $B \times d$).

The output for a token at position $n$ within chunk $[i]$ has two parts:

1. **Inner-chunk:** Attention to other tokens within the same chunk. This uses the parallel form restricted to the chunk: $(Q_{[i]} K_{[i]}^\top \odot D_\text{chunk}) V_{[i]}$, where $D_\text{chunk}$ is the $B \times B$ decay matrix for positions within the chunk.

2. **Cross-chunk:** Attention to all tokens in previous chunks, summarized by the recurrent state $R_{i-1}$.

The cross-chunk contribution for token at position $j$ (1-indexed) within chunk $[i]$ is:

$$
\text{cross}_j = \gamma^j \cdot q_j^\top R_{i-1}
$$

where $\gamma^j$ accounts for the decay from the end of the previous chunk to position $j$ within the current chunk. In matrix form:

$$
\text{Cross-chunk} = (Q_{[i]} R_{i-1}) \odot \xi
$$

where $\xi$ is a column vector $(\gamma^1, \gamma^2, \ldots, \gamma^B)^\top$ broadcast across the value dimensions.

The state update after processing chunk $[i]$ is:

$$
\boxed{R_i = \gamma^B \, R_{i-1} + K_{[i]}^\top (V_{[i]} \odot \zeta)}
$$

where $\zeta$ is a matrix with row $j$ equal to $(\gamma^{B-j}, \ldots, \gamma^{B-j})$ — the decay from position $j$ within the chunk to the end of the chunk, broadcast across value dimensions. This ensures that the state $R_i$ correctly accumulates contributions from all tokens up to and including chunk $[i]$, with appropriate decay.

The complete chunkwise formula is:

$$
\boxed{\text{Retention}(X_{[i]}) = \underbrace{(Q_{[i]} K_{[i]}^\top \odot D_\text{chunk}) V_{[i]}}_{\text{Inner-chunk}} + \underbrace{(Q_{[i]} R_{i-1}) \odot \xi}_{\text{Cross-chunk}}}
$$

### 7.3 Numerical verification with $B = 2$

Let us split our 4-token sequence into two chunks of $B = 2$:

- Chunk 1: tokens 1, 2
- Chunk 2: tokens 3, 4

**Chunk 1:**

$$
Q_{[1]} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad K_{[1]} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad V_{[1]} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

The chunk-level decay matrix ($B = 2$):

$$
D_\text{chunk} = \begin{pmatrix} 1 & 0 \\ 0.9 & 1 \end{pmatrix}
$$

Inner-chunk computation:

$$
Q_{[1]} K_{[1]}^\top = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
$$

$$
Q_{[1]} K_{[1]}^\top \odot D_\text{chunk} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \odot \begin{pmatrix} 1 & 0 \\ 0.9 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0.9 & 0 \end{pmatrix}
$$

$$
\text{Inner}_1 = \begin{pmatrix} 0 & 0 \\ 0.9 & 0 \end{pmatrix} V_{[1]} = \begin{pmatrix} 0 & 0 \\ 0.9 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0.9 & 0 \end{pmatrix}
$$

Cross-chunk: $R_0 = 0$, so the cross-chunk contribution is zero.

$$
\text{Output chunk 1} = \begin{pmatrix} 0 & 0 \\ 0.9 & 0 \end{pmatrix} \quad \checkmark
$$

State update for chunk 1:

$$
\zeta = \begin{pmatrix} \gamma^{B-1} & \gamma^{B-1} \\ \gamma^{B-2} & \gamma^{B-2} \end{pmatrix} = \begin{pmatrix} 0.9 & 0.9 \\ 1 & 1 \end{pmatrix}
$$

$$
V_{[1]} \odot \zeta = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \odot \begin{pmatrix} 0.9 & 0.9 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 0.9 & 0 \\ 0 & 1 \end{pmatrix}
$$

$$
K_{[1]}^\top (V_{[1]} \odot \zeta) = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0.9 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 0.9 & 0 \end{pmatrix}
$$

$$
R_1 = \gamma^B R_0 + K_{[1]}^\top (V_{[1]} \odot \zeta) = 0 + \begin{pmatrix} 0 & 1 \\ 0.9 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 0.9 & 0 \end{pmatrix}
$$

**Verification:** $R_1$ should equal $S_2$ from the recurrent computation (the state at the end of chunk 1). From Section 3.1: $S_2 = \begin{pmatrix} 0 & 1 \\ 0.9 & 0 \end{pmatrix}$. $\checkmark$

**Chunk 2:**

$$
Q_{[2]} = \begin{pmatrix} 1 & 1 \\ 2 & 0 \end{pmatrix}, \quad K_{[2]} = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}, \quad V_{[2]} = \begin{pmatrix} 1 & 1 \\ 2 & 0 \end{pmatrix}
$$

Inner-chunk:

$$
Q_{[2]} K_{[2]}^\top = \begin{pmatrix} 1 & 1 \\ 2 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 2 & 2 \\ 2 & 0 \end{pmatrix}
$$

$$
Q_{[2]} K_{[2]}^\top \odot D_\text{chunk} = \begin{pmatrix} 2 & 2 \\ 2 & 0 \end{pmatrix} \odot \begin{pmatrix} 1 & 0 \\ 0.9 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 1.8 & 0 \end{pmatrix}
$$

$$
\text{Inner}_2 = \begin{pmatrix} 2 & 0 \\ 1.8 & 0 \end{pmatrix} V_{[2]} = \begin{pmatrix} 2 & 0 \\ 1.8 & 0 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 2 & 0 \end{pmatrix} = \begin{pmatrix} 2 & 2 \\ 1.8 & 1.8 \end{pmatrix}
$$

Cross-chunk:

$$
Q_{[2]} R_1 = \begin{pmatrix} 1 & 1 \\ 2 & 0 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 0.9 & 0 \end{pmatrix} = \begin{pmatrix} 0.9 & 1 \\ 0 & 2 \end{pmatrix}
$$

The decay vector $\xi = (\gamma^1, \gamma^2)^\top = (0.9, 0.81)^\top$, broadcast across columns:

$$
(Q_{[2]} R_1) \odot \xi = \begin{pmatrix} 0.9 & 1 \\ 0 & 2 \end{pmatrix} \odot \begin{pmatrix} 0.9 & 0.9 \\ 0.81 & 0.81 \end{pmatrix} = \begin{pmatrix} 0.81 & 0.9 \\ 0 & 1.62 \end{pmatrix}
$$

Total output chunk 2:

$$
\text{Inner}_2 + \text{Cross}_2 = \begin{pmatrix} 2 & 2 \\ 1.8 & 1.8 \end{pmatrix} + \begin{pmatrix} 0.81 & 0.9 \\ 0 & 1.62 \end{pmatrix} = \begin{pmatrix} 2.81 & 2.9 \\ 1.8 & 3.42 \end{pmatrix} \quad \checkmark
$$

Both tokens match the recurrent and parallel outputs.

State update for chunk 2:

$$
\zeta = \begin{pmatrix} 0.9 & 0.9 \\ 1 & 1 \end{pmatrix}
$$

$$
V_{[2]} \odot \zeta = \begin{pmatrix} 1 & 1 \\ 2 & 0 \end{pmatrix} \odot \begin{pmatrix} 0.9 & 0.9 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 0.9 & 0.9 \\ 2 & 0 \end{pmatrix}
$$

$$
K_{[2]}^\top (V_{[2]} \odot \zeta) = \begin{pmatrix} 1 & 0 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 0.9 & 0.9 \\ 2 & 0 \end{pmatrix} = \begin{pmatrix} 0.9 & 0.9 \\ 4.9 & 0.9 \end{pmatrix}
$$

$$
R_2 = \gamma^B R_1 + K_{[2]}^\top (V_{[2]} \odot \zeta) = 0.81 \begin{pmatrix} 0 & 1 \\ 0.9 & 0 \end{pmatrix} + \begin{pmatrix} 0.9 & 0.9 \\ 4.9 & 0.9 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0 & 0.81 \\ 0.729 & 0 \end{pmatrix} + \begin{pmatrix} 0.9 & 0.9 \\ 4.9 & 0.9 \end{pmatrix} = \begin{pmatrix} 0.9 & 1.71 \\ 5.629 & 0.9 \end{pmatrix}
$$

**Verification:** $R_2$ should equal $S_4$ from the recurrent computation. From Section 3.1: $S_4 = \begin{pmatrix} 0.9 & 1.71 \\ 5.629 & 0.9 \end{pmatrix}$. $\checkmark$

All three computation paradigms — recurrent, parallel, and chunkwise — produce identical outputs and identical final states. They are three views of the same mathematical object.

### 7.4 Chunkwise complexity

The cost of the chunkwise form per chunk:

- Inner-chunk: $Q_{[i]} K_{[i]}^\top$ is $(B \times d_k) \times (d_k \times B) = O(B^2 d_k)$. Multiply by $V$: $O(B^2 d_v)$. Total: $O(B^2 d_k)$.
- Cross-chunk: $Q_{[i]} R_{i-1}$ is $(B \times d_k) \times (d_k \times d_v) = O(B d_k d_v)$.
- State update: $K_{[i]}^\top (V_{[i]} \odot \zeta)$ is $(d_k \times B) \times (B \times d_v) = O(B d_k d_v)$.

There are $n / B$ chunks. Total cost:

$$
\frac{n}{B} \times (B^2 d_k + B d_k d_v) = n B d_k + n d_k d_v = n d_k (B + d_v)
$$

With $B = 512$ and $d_k = d_v = 256$ (RetNet's experimental settings):

$$
O(n \cdot 256 \cdot (512 + 256)) = O(n \cdot 256 \cdot 768) = O(n \cdot 196{,}608)
$$

Compare to the parallel form: $O(n^2 d_k) = O(n^2 \cdot 256)$. The chunkwise form becomes cheaper when $B + d_v < n$, which is $768 < n$ — true for virtually all practical sequences.

---

## 8. Multi-Scale Retention

### 8.1 Different decay rates per head

A single decay rate $\gamma$ creates a single effective window size. But language modeling requires attention at multiple scales — local patterns (syntax, within a phrase) and global patterns (topic, across paragraphs). RetNet assigns a different $\gamma$ value to each attention head using the formula:

$$
\boxed{\gamma = 1 - 2^{-5 - \text{arange}(0, h)} \in \mathbb{R}^h}
$$

where $\text{arange}(0, h) = (0, 1, 2, \ldots, h-1)$ and the formula is applied element-wise. This produces one $\gamma$ per head.

### 8.2 Numerical values for $h = 8$ heads

| Head $i$ | $-5 - i$ | $2^{-5-i}$ | $\gamma_i = 1 - 2^{-5-i}$ | Effective window ($\epsilon = 0.01$) |
|---|---|---|---|---|
| 0 | $-5$ | $1/32 = 0.03125$ | $0.96875$ | 145 |
| 1 | $-6$ | $1/64 = 0.015625$ | $0.984375$ | 292 |
| 2 | $-7$ | $1/128 \approx 0.00781$ | $0.992188$ | 587 |
| 3 | $-8$ | $1/256 \approx 0.00391$ | $0.996094$ | 1{,}177 |
| 4 | $-9$ | $1/512 \approx 0.00195$ | $0.998047$ | 2{,}357 |
| 5 | $-10$ | $1/1024 \approx 0.000977$ | $0.999023$ | 4{,}717 |
| 6 | $-11$ | $1/2048 \approx 0.000488$ | $0.999512$ | 9{,}439 |
| 7 | $-12$ | $1/4096 \approx 0.000244$ | $0.999756$ | 18{,}882 |

The effective window is computed as $d = \log(0.01) / \log(\gamma_i)$, the formula from Section 2.3.

Head 0 has $\gamma = 0.969$ with an effective window of ~145 tokens — it focuses on local patterns. Head 7 has $\gamma = 0.9998$ with an effective window of ~18,882 tokens — it captures long-range dependencies. This is **multi-scale retention** (MSR): the heads automatically specialize at different scales, similar to how multi-resolution wavelets capture patterns at different frequencies.

### 8.3 Why multiple scales help

The ablation in Table 6 of Sun et al. (2023) quantifies the contribution:

| Variant | In-Domain PPL |
|---|---|
| RetNet (full) | **26.05** |
| $-$ $\gamma$ decay (set $\gamma = 1$) | 27.86 |
| $-$ multi-scale decay (same $\gamma$ for all heads) | 27.02 |

Removing decay entirely ($\gamma = 1$) degrades perplexity by 1.81 — this reverts retention to linear attention, confirming that decay is essential. Using a single decay rate across all heads degrades perplexity by 0.97 — confirming that multi-scale specialization provides meaningful improvement beyond the decay mechanism itself.

### 8.4 GroupNorm instead of LayerNorm

Since different heads use different $\gamma$ values, their output magnitudes differ. A head with $\gamma = 0.969$ accumulates less state (more decay) and produces smaller outputs than a head with $\gamma = 0.9998$ (less decay). Applying LayerNorm across all heads would couple their normalization statistics, distorting the relative scales.

RetNet uses **GroupNorm** (Wu and He, 2018) instead, which normalizes each head independently. Formally, if $\text{head}_i \in \mathbb{R}^{n \times d_v}$ is the output of the $i$-th retention head:

$$
Y = \text{GroupNorm}_h(\text{Concat}(\text{head}_1, \ldots, \text{head}_h))
$$

where the GroupNorm has $h$ groups, one per head. Each head is normalized by its own mean and variance, preserving the different scales induced by different $\gamma$ values.

The ablation confirms this: removing GroupNorm degrades perplexity from 26.05 to 27.54 (a 1.49 increase).

An important property of GroupNorm is **scale invariance**: $\text{GroupNorm}(\alpha \cdot \text{head}_i) = \text{GroupNorm}(\text{head}_i)$ for any scalar $\alpha > 0$. This means the retention outputs do not need to be normalized by a denominator (unlike linear attention's $\phi(q_i)^\top z_i$ normalization). The GroupNorm absorbs any global scaling. This is why retention can use raw query-key products — even if the products are large or negative, the GroupNorm handles the scale.

### 8.5 Retention Score Normalization

The scale invariance of GroupNorm also enables additional normalization tricks that improve numerical precision without changing the final output. Sun et al. (2023) apply three normalization factors:

1. Scale $QK^\top$ by $1/\sqrt{d}$ (same as the $1/\sqrt{d_k}$ scaling in standard attention).
2. Normalize the decay matrix: replace $D_{nm}$ with $\tilde{D}_{nm} = D_{nm} / \sqrt{\sum_{i=1}^n D_{ni}}$.
3. Normalize the retention scores: $\tilde{R}_{nm} = R_{nm} / \max(|\sum_{i=1}^n R_{ni}|, 1)$.

These tricks stabilize the numerical flow in both forward and backward passes. Because of GroupNorm's scale invariance, they do not affect the final output or gradients — they only improve intermediate precision.

---

## 9. The Complete RetNet Block

### 9.1 The MSR layer

The multi-scale retention (MSR) module combines the retention heads with a **swish gate**:

$$
\text{head}_i = \text{Retention}(X, \gamma_i)
$$

$$
Y = \text{GroupNorm}_h(\text{Concat}(\text{head}_1, \ldots, \text{head}_h))
$$

$$
\boxed{\text{MSR}(X) = (\text{swish}(X W_G) \odot Y) \, W_O}
$$

where $W_G \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$ and $W_O \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$ are learned parameter matrices. The **swish** activation (Ramachandran et al., 2017) is $\text{swish}(x) = x \cdot \sigma(x)$ where $\sigma$ is the sigmoid function.

The swish gate $\text{swish}(X W_G) \odot Y$ is a multiplicative interaction between the raw input (passed through a linear layer and swish) and the retention output. This is the same gating principle we derived in Blog 12 — learned, per-dimension multiplicative control of information flow. The gate increases the non-linearity of the retention layer, which is important because the retention mechanism itself (without softmax) is a linear function of the values.

The ablation from Sun et al. (2023) confirms: removing the swish gate degrades perplexity from 26.05 to 27.84 (a 1.79 increase). This is the largest single-component degradation in the ablation, even larger than removing decay ($+1.81$) — indicating that the gate is essential for model quality.

### 9.2 The full RetNet block

Each RetNet layer consists of an MSR module and a feed-forward network (FFN), with pre-norm residual connections (the same layout we derived in Blog 12, Section 2.2):

$$
Y^l = \text{MSR}(\text{LN}(X^l)) + X^l
$$

$$
X^{l+1} = \text{FFN}(\text{LN}(Y^l)) + Y^l
$$

where $\text{LN}$ is **LayerNorm** (Ba et al., 2016). The FFN uses GELU activation:

$$
\text{FFN}(X) = \text{gelu}(X W_1) W_2
$$

with $W_1 \in \mathbb{R}^{d_\text{model} \times d_{ff}}$ and $W_2 \in \mathbb{R}^{d_{ff} \times d_\text{model}}$.

### 9.3 Parameter allocation

RetNet re-allocates parameters between the MSR and FFN modules to match the total parameter count of a standard transformer.

In a transformer: self-attention has $4d^2$ parameters ($W_Q, W_K, W_V, W_O$, each $d \times d$), and FFN has $8d^2$ parameters ($W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$). Total: $12d^2$.

In RetNet: MSR has $W_Q, W_K \in \mathbb{R}^{d \times d}$, $W_V \in \mathbb{R}^{d \times 2d}$ (the value head dimension is twice the query/key dimension), $W_G \in \mathbb{R}^{d \times d}$, and $W_O \in \mathbb{R}^{2d \times d}$ (projecting from the widened value dimension back to $d$). That is $d^2 + d^2 + 2d^2 + d^2 + 2d^2 = 8d^2$.

To keep the total at $12d^2$, the FFN intermediate dimension is reduced to $2d$ (from $4d$), giving $4d^2$ for FFN. Total: $8d^2 + 4d^2 = 12d^2$.

### 9.4 Numerical check

With $d_\text{model} = 512$:

- Transformer: $12 \times 512^2 = 3{,}145{,}728$ parameters per layer
- RetNet: $12 \times 512^2 = 3{,}145{,}728$ parameters per layer $\checkmark$

The parameter counts match exactly. Any difference in performance comes from the architecture, not from having more or fewer parameters.

---

## 10. Experimental Results

Sun et al. (2023) evaluate RetNet against Transformers and other efficient architectures across multiple dimensions.

### 10.1 Language modeling

RetNet and Transformer are trained from scratch at three scales (1.3B, 2.7B, 6.7B parameters) on 100B tokens from The Pile, C4, and The Stack. The validation perplexities:

| Model Size | Transformer PPL | RetNet PPL |
|---|---|---|
| 1.3B | ~15.0 | ~14.8 |
| 2.7B | ~13.5 | ~13.3 |
| 6.7B | ~12.8 | ~12.5 |

RetNet achieves comparable or better perplexity at every scale. The gap widens in RetNet's favor as models get larger — a favorable scaling trend.

### 10.2 Zero-shot and few-shot evaluation

On seven downstream tasks (HellaSwag, BoolQ, COPA, PIQA, Winograd, Winogrande, StoryCloze) with the 6.7B model:

| Setting | Transformer Avg | RetNet Avg |
|---|---|---|
| Zero-shot | 66.07 | **69.51** |
| 4-shot | 66.44 | **69.76** |

RetNet outperforms the Transformer on average in both zero-shot and few-shot settings. The improvements are consistent across individual tasks.

### 10.3 Training cost

Training throughput and memory on 8 NVIDIA A100-80GB GPUs with sequence length 8192:

| Model Size | Trm Memory (GB) | RetNet Memory (GB) | Trm Throughput (wps) | RetNet Throughput (wps) |
|---|---|---|---|---|
| 1.3B | 74.8 | 34.5 | 10{,}832 | 73{,}345 |
| 2.7B | 69.6 | 42.0 | 5{,}186 | 38{,}921 |
| 6.7B | 69.0 | 48.0 | 2{,}754 | 17{,}459 |
| 13B | 61.4 | 45.9 | 1{,}209 | 8{,}642 |

RetNet uses 25–54% less memory and achieves 6–7$\times$ higher throughput than vanilla Transformer. Even compared to FlashAttention-optimized Transformers, RetNet is competitive — and RetNet's implementation uses vanilla PyTorch without custom kernels.

### 10.4 Inference cost

At 6.7B scale with 8K sequence length, the recurrent form gives:

- **Memory:** 3.4$\times$ less GPU memory (RetNet's state is constant, Transformer's KV cache grows)
- **Throughput:** 8.4$\times$ higher (words per second)
- **Latency:** 15.6$\times$ lower (milliseconds per token)

RetNet's inference latency is **batch-size invariant** — it stays nearly constant whether processing 1 or 8 sequences simultaneously. Transformer latency grows with batch size because the KV cache competes for GPU memory with the computation.

### 10.5 Comparison with other efficient architectures

At 200M parameters with 16 layers and hidden dimension 1024:

| Method | In-Domain PPL | PG22 | QMSum | GovReport | SummScreen |
|---|---|---|---|---|---|
| RWKV | 30.92 | 51.41 | 28.17 | 19.80 | 25.78 |
| H3 | 29.97 | 49.17 | 24.29 | 19.19 | 25.11 |
| Hyena | 32.08 | 52.75 | 28.18 | 20.55 | 26.51 |
| Linear Transformer | 40.24 | 63.86 | 28.45 | 25.33 | 32.02 |
| **RetNet** | **26.05** | **45.27** | **21.33** | **16.52** | **22.48** |

RetNet outperforms all other efficient architectures on both in-domain and out-of-domain corpora. The Linear Transformer (Blog 13's architecture) is the weakest — confirming that replacing softmax with a simple kernel without decay or position encoding loses too much modeling capacity.

### 10.6 Context length results

RetNet maintains its advantage across different context lengths:

| Model | 512 | 1024 | 2048 |
|---|---|---|---|
| Transformer | 13.55 | 12.56 | 12.35 |
| RetNet | **13.09** | **12.14** | **11.98** |

RetNet consistently achieves lower perplexity, and the gap slightly widens with longer contexts. The exponential decay does not prevent the model from using long-range context — the heads with $\gamma \approx 0.9998$ have effective windows of nearly 19,000 tokens, covering most practical sequence lengths.

---

## 11. The Hybrid Architecture Pattern

### 11.1 What makes an architecture hybrid

We can now define precisely what a **hybrid architecture** means in this context. A hybrid sequence model satisfies three properties:

1. **A single mathematical formula** defines the input-output mapping. There is one function, not two.
2. **Multiple computation modes** implement this formula. Each mode has different cost characteristics (parallel vs sequential, quadratic vs linear memory) suited to different hardware contexts.
3. **Exact equivalence** between modes. The outputs are identical — not approximated, not distilled, not fine-tuned separately. The same trained weights produce the same outputs regardless of which mode is used.

RetNet's retention satisfies all three:

**Parallel** (training): $\text{Retention}(X) = (QK^\top \odot D) V$. Cost: $O(n^2 d_k)$. GPU-efficient matrix operations. Used when the full sequence is available.

**Recurrent** (inference): $S_n = \gamma S_{n-1} + k_n v_n^\top$, $o_n = q_n^\top S_n$. Cost: $O(d_k^2)$ per token. Constant memory, constant compute. Used for autoregressive generation.

**Chunkwise** (long-sequence training): parallel within chunks of size $B$, recurrent across chunks. Cost: $O(n d_k (B + d_k))$. Balances parallelism and memory. Used when sequences are too long for the full parallel form.

We verified numerically that all three produce identical outputs for every token. The equivalence is a mathematical property of the retention formula itself, not an engineering trick.

### 11.2 Why the hybrid pattern is general

The retention mechanism is not the only formula with this property. The mathematical ingredients that enable the hybrid pattern are:

1. **A linear recurrence** with state $S_n = A S_{n-1} + B_n$. Any linear recurrence can be unrolled into a sum (yielding a parallel form) or executed step by step (yielding a recurrent form).
2. **Associativity of matrix multiplication.** The parallel form is just a different parenthesization of the same matrix product — $(QK^\top)V$ vs $Q(K^\top V)$.
3. **Decomposability into chunks.** The sum in the unrolled form can be split at any chunk boundary, giving the chunkwise form.

Any mechanism built on a linear recurrence inherits this hybrid property. This is why the pattern appears repeatedly in the architectures that followed RetNet: Mamba (Gu and Dao, 2023), RWKV (Peng et al., 2023), Griffin (De et al., 2024), and others all have parallel training and recurrent inference modes derived from the same linear recurrence structure. The specific choices — what goes into the state, how the state decays, how position is encoded — differ across architectures, but the hybrid pattern is the same.

### 11.3 From Blog 13 to RetNet

Blog 13 showed that linear attention is an RNN:

$$
S_n^{\text{linear}} = S_{n-1}^{\text{linear}} + \phi(k_n) v_n^\top
$$

This was the first hybrid architecture: it had a parallel form and a recurrent form. But it failed at the third vertex of the impossible triangle — strong performance — because the accumulate-only recurrence lost information and lacked position encoding.

RetNet's retention is a direct modification of this recurrence:

$$
S_n^{\text{retention}} = \gamma S_{n-1}^{\text{retention}} + k_n v_n^\top
$$

The two changes — adding $\gamma$ and dropping the kernel feature map $\phi$ — are small algebraically but large in effect. The decay factor $\gamma$ bounds the state, encodes recency, and (through multi-scale heads) creates a rich set of temporal attention windows. Dropping the kernel and replacing the denominator normalization with GroupNorm gives the model more flexibility — the query-key interaction can be negative, and the normalization is data-adaptive rather than formula-fixed.

The position encoding via complex exponentials ($e^{i n \theta}$) arises naturally from diagonalizing the state transition matrix — it is not an add-on but a structural consequence of the recurrence.

RetNet is the first architecture to convincingly demonstrate that the hybrid pattern can achieve all three vertices of the impossible triangle. It established the template — linear recurrence + exponential decay + multi-scale heads + gating — that subsequent architectures have refined and extended.

---

## Summary

Blog 13's linear attention was the first hybrid architecture: one formula with both a parallel form and a recurrent form. But it failed at quality because the accumulate-only recurrence ($S_n = S_{n-1} + \phi(k_n)v_n^\top$) grows without bound and has no position information. RetNet fixes both problems by adding exponential decay $\gamma$ (bounding the state at $c/(1-\gamma)$ via the geometric series limit) and relative position encoding via complex exponentials ($e^{i(n-m)\theta}$, derived from diagonalizing the state transition matrix $A$). The result is a hybrid architecture with three equivalent computation paradigms — parallel for training ($(QK^\top \odot D)V$, cost $O(n^2 d_k)$), recurrent for inference ($O(d_k^2)$ per token, constant memory), and chunkwise for long sequences ($O(n d_k(B + d_k))$) — all verified to produce identical outputs on a 4-token running example. Multi-scale retention assigns different decay rates per head ($\gamma$ from $0.969$ to $0.9998$ for 8 heads), giving effective attention windows from 145 to 18,882 tokens, and the swish gate plus GroupNorm complete the architecture to match transformer parameter counts while achieving 8.4$\times$ faster inference, 15.6$\times$ lower latency, and competitive-or-better perplexity. This is the hybrid architecture pattern: one formula, multiple computation modes, exact equivalence — the template that Mamba, RWKV, Griffin, and other post-transformer architectures all follow.

---

*Previous: [Why Replace Attention? The Softmax Bottleneck and the Path to Linear Time](/blog/attention-why-replace)*

*Next: [Targeted Memory: The Delta Rule, Gated DeltaNet, and Kimi Delta Attention](/blog/attention-gated-deltanet)*
