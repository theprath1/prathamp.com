---
title: "The Kernel Zoo: Performers, Fast Weight Programmers, and the Capacity-Approximation Tradeoff in Linear Attention"
description: "Navigating the capacity-approximation tradeoff in linear attention feature maps from the ground up — reframing linear transformers as 1990s fast weight programmers that program memory through outer products, deriving the capacity limit that says a state of dimension d_dot can store at most d_dot orthogonal key-value pairs before retrieval breaks, three approaches to choosing the feature map φ (the ELU+1 baseline, the Performer's FAVOR+ that approximates the actual softmax kernel with positive orthogonal random features, and the FWP paper's DPFP that deterministically increases capacity through ReLU-gated quadrant projections), why positive features avoid the variance explosion that makes trigonometric random features unstable for small kernel values, why orthogonal features provably reduce mean squared error over independent samples, sum normalization and why attention normalization can diverge with the delta rule, and the capacity-versus-approximation tradeoff that defines the entire design space — all derived step by step with concrete examples."
date: 2026-04-08
tags: [machine-learning, attention, transformers, linear-attention, kernels, performers, fast-weight-programmers, favor-plus, dpfp, efficiency]
order: 2
---

The Why Replace Attention blog established a powerful framework: replace the softmax similarity $\exp(q^\top k / \sqrt{d_k})$ with a kernel $\phi(q)^\top \phi(k)$ where $\phi : \mathbb{R}^{d_k} \to \mathbb{R}^{d_\phi}$ is a finite-dimensional feature map, apply the associativity of matrix multiplication to change the computation order from $(\phi(Q)\phi(K)^\top)V$ to $\phi(Q)(\phi(K)^\top V)$, and obtain an RNN with constant-size state $S \in \mathbb{R}^{d_\phi \times d_v}$ that computes in linear time. That blog used the simplest possible feature map — $\phi(x) = \text{elu}(x) + 1$ from Katharopoulos et al. (2020) — and left a question open: how should we choose $\phi$?

The RetNet blog and the Gated DeltaNet blog answered a different question: given the recurrent state $S_t$, what is the right update rule? RetNet added exponential decay. DeltaNet added targeted erasure via the delta rule. Gated DeltaNet combined both.

This blog returns to the feature map itself. The update rule determines how the state is managed. The feature map determines what the state can store and how accurately it approximates real attention. These are orthogonal design choices — you can pair any feature map with any update rule. But the feature map is the more fundamental choice, because it determines the **capacity** of the memory (how many associations can be stored) and the **fidelity** of the approximation (how close the linear attention output is to the softmax attention output).

We derive insights from two papers that approach the feature map question from opposite directions:

1. **Schlag, Irie, and Schmidhuber (2021)**, "Linear Transformers Are Secretly Fast Weight Programmers" (ICML 2021): reframes linear attention as a fast weight programmer from the 1990s, derives the capacity limitation that the state can store at most $d_\text{dot}$ orthogonal associations (where $d_\text{dot}$ is the output dimension of $\phi$), and proposes DPFP — a deterministic kernel that increases $d_\text{dot}$ without random sampling.

2. **Choromanski et al. (2021)**, "Rethinking Attention with Performers" (ICLR 2021): asks a different question — instead of designing a new kernel, can we approximate the *actual* softmax kernel with a finite-dimensional feature map? They propose FAVOR+ (Fast Attention Via positive Orthogonal Random features), which uses random projections with provable approximation guarantees.

The first paper says: design $\phi$ to maximize memory capacity. The second says: design $\phi$ to approximate softmax. Both are valid goals, and the tension between them defines the design space for linear attention kernels.

---

## The Running Example

We continue with the same tiny example from the Why Replace Attention, RetNet, and Gated DeltaNet blogs:

- $n = 4$ tokens, $d_k = d_v = 2$, single head

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}, \quad K = \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \\ 0 & 2 \end{pmatrix}, \quad V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}
$$

From the Why Replace Attention blog, the ELU+1 feature maps are:

$$
\phi_\text{elu}(Q) = \begin{pmatrix} 2 & 1 \\ 1 & 2 \\ 2 & 2 \\ 3 & 1 \end{pmatrix}, \quad \phi_\text{elu}(K) = \begin{pmatrix} 1 & 2 \\ 2 & 1 \\ 2 & 2 \\ 1 & 3 \end{pmatrix}
$$

For the capacity analysis, we use the model parameters from the series:

- $d_\text{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$, $L = 12$ layers, fp16

For the random feature demonstrations (FAVOR+), we will need a specific random matrix, which we fix at the start of that section.

---

## 1. Linear Attention Is a Fast Weight Programmer

### 1.1 Fast weight programmers: the 1990s idea

Schlag et al. (2021) make a historical observation: the linear attention recurrence derived in the Why Replace Attention blog is not new. It is a special case of **fast weight programmers** (FWPs), a concept introduced by Schmidhuber in 1991.

The idea of **fast weights** is to make the weights of a neural network change at test time — unlike standard "slow weights" which are fixed after training. A **fast weight programmer** (FWP) is a two-network system: a "slow" network (trained by gradient descent) generates instructions that program the fast weights of a second network. The fast weights change with every input token, while the slow weights remain fixed during inference.

The concept was called **synaptic modulation** by von der Malsburg (1981). Von der Malsburg defined the effective weights as a superposition of conventional, context-independent *slow weights* and fast-changing, context-dependent *fast weights*. Hinton and Plaut (1987) studied the additive superposition of two sets of weights with different learning rates. But before 1991, no network learned by gradient descent to compute the changes of the fast weight storage of another network or of itself.

### 1.2 The FWP recurrence

A fast weight programmer processes a sequence $\{x^{(i)}\}_{i=1}^L$, $x^{(i)} \in \mathbb{R}^d$, and produces an output sequence $\{y^{(i)}\}_{i=1}^L$, $y^{(i)} \in \mathbb{R}^{d_\text{value}}$ as follows. At each step $i$, it:

1. Computes key, value, and query projections:

$$
k^{(i)}, v^{(i)}, q^{(i)} = W_k x^{(i)}, \; W_v x^{(i)}, \; W_q x^{(i)}
$$

2. Updates the fast weight matrix:

$$
W^{(i)} = W^{(i-1)} + v^{(i)} \otimes k^{(i)}
$$

3. Reads from the fast weight matrix:

$$
y^{(i)} = W^{(i)} q^{(i)}
$$

where $\otimes$ denotes the **outer product** and $W^{(0)} = 0$.

The outer product $v^{(i)} \otimes k^{(i)}$ is a $d_\text{value} \times d_k$ matrix — the same shape as the state matrix $S$ from the Why Replace Attention blog. The "programming instruction" at each step is: store the association between key $k^{(i)}$ and value $v^{(i)}$ by adding their outer product to the fast weight matrix.

### 1.3 The equivalence

This is the part that confuses almost everyone when first encountering the FWP perspective, because the notation differs between the two communities.

Let us line up the two recurrences. From the Why Replace Attention blog, linear attention without normalization has:

$$
S_i = S_{i-1} + \phi(k_i) v_i^\top
$$

$$
o_i = S_i^\top q_i
$$

wait — there is an inconsistency in the transpose convention. Let us be precise. The Why Replace Attention blog defines $S_i = \sum_{j=1}^i \phi(k_j) v_j^\top \in \mathbb{R}^{d_\phi \times d_v}$ and computes $o_i = \phi(q_i)^\top S_i \in \mathbb{R}^{1 \times d_v}$.

The FWP defines $W^{(i)} = \sum_{j=1}^i v^{(j)} \otimes k^{(j)}$. Now $v^{(j)} \otimes k^{(j)}$ means the matrix where element $(a, b) = v^{(j)}_a \cdot k^{(j)}_b$. This gives $W^{(i)} \in \mathbb{R}^{d_\text{value} \times d_k}$. And retrieval is $y^{(i)} = W^{(i)} q^{(i)} \in \mathbb{R}^{d_\text{value}}$.

Expanding:

$$
y^{(i)} = W^{(i)} q^{(i)} = \left(\sum_{j=1}^i v^{(j)} (k^{(j)})^\top \right) q^{(i)} = \sum_{j=1}^i (k^{(j)})^\top q^{(i)} \cdot v^{(j)}
$$

This is exactly the linear attention numerator from the Why Replace Attention blog (without the kernel $\phi$): $\sum_{j=1}^i (k_j^\top q_i) v_j$. Setting $\phi = \text{identity}$ (no feature map), the FWP and unnormalized linear attention are identical.

With a feature map $\phi$, linear attention replaces $k^{(j)}$ with $\phi(k^{(j)})$ and $q^{(i)}$ with $\phi(q^{(i)})$:

$$
S_i = \sum_{j=1}^i \phi(k_j) v_j^\top \quad \Longleftrightarrow \quad W^{(i)} = \sum_{j=1}^i v^{(j)} \otimes \phi(k^{(j)})
$$

$$
o_i = \phi(q_i)^\top S_i \quad \Longleftrightarrow \quad y^{(i)} = W^{(i)} \phi(q^{(i)})
$$

These are the same computation with different notational conventions: $S = W^\top$, $\phi(k) = k$ (with $\phi$ applied before storage), and the output is a matrix-vector product either way.

The difference is normalization. The FWP as stated in Schmidhuber (1991) has no normalization — the output is $W^{(i)} q^{(i)}$ directly. Linear attention (Katharopoulos et al., 2020) divides by $\phi(q_i)^\top z_i$, where $z_i = \sum_{j=1}^i \phi(k_j)$. Schlag et al. (2021) call this the difference between the FWP and the "linearised Transformer" — they are the same up to this normalisation.

### 1.4 Numerical verification

Let us trace the FWP for our running example (no feature map, no normalisation, causal masking).

**Step $i = 1$:** $k_1 = (0, 1)^\top$, $v_1 = (1, 0)^\top$, $q_1 = (1, 0)^\top$.

$$
W^{(1)} = W^{(0)} + v_1 \otimes k_1 = 0 + \begin{pmatrix}1\\0\end{pmatrix}(0 \;\; 1) = \begin{pmatrix}0 & 1\\0 & 0\end{pmatrix}
$$

$$
y^{(1)} = W^{(1)} q_1 = \begin{pmatrix}0 & 1\\0 & 0\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}0\\0\end{pmatrix}
$$

**Step $i = 2$:** $k_2 = (1, 0)^\top$, $v_2 = (0, 1)^\top$, $q_2 = (0, 1)^\top$.

$$
W^{(2)} = W^{(1)} + v_2 \otimes k_2 = \begin{pmatrix}0 & 1\\0 & 0\end{pmatrix} + \begin{pmatrix}0\\1\end{pmatrix}(1 \;\; 0) = \begin{pmatrix}0 & 1\\1 & 0\end{pmatrix}
$$

$$
y^{(2)} = W^{(2)} q_2 = \begin{pmatrix}0 & 1\\1 & 0\end{pmatrix}\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}1\\0\end{pmatrix}
$$

Numerical check: with causal masking, $y^{(2)} = \sum_{j=1}^2 (k_j^\top q_2) v_j = (k_1^\top q_2) v_1 + (k_2^\top q_2) v_2 = (0 \cdot 0 + 1 \cdot 1)(1, 0) + (1 \cdot 0 + 0 \cdot 1)(0, 1) = (1, 0) + (0, 0) = (1, 0)$. $\checkmark$

**Step $i = 3$:** $k_3 = (1, 1)^\top$, $v_3 = (1, 1)^\top$, $q_3 = (1, 1)^\top$.

$$
W^{(3)} = W^{(2)} + v_3 \otimes k_3 = \begin{pmatrix}0 & 1\\1 & 0\end{pmatrix} + \begin{pmatrix}1\\1\end{pmatrix}(1 \;\; 1) = \begin{pmatrix}1 & 2\\2 & 1\end{pmatrix}
$$

$$
y^{(3)} = W^{(3)} q_3 = \begin{pmatrix}1 & 2\\2 & 1\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}3\\3\end{pmatrix}
$$

**Step $i = 4$:** $k_4 = (0, 2)^\top$, $v_4 = (2, 0)^\top$, $q_4 = (2, 0)^\top$.

$$
W^{(4)} = W^{(3)} + v_4 \otimes k_4 = \begin{pmatrix}1 & 2\\2 & 1\end{pmatrix} + \begin{pmatrix}2\\0\end{pmatrix}(0 \;\; 2) = \begin{pmatrix}1 & 6\\2 & 1\end{pmatrix}
$$

$$
y^{(4)} = W^{(4)} q_4 = \begin{pmatrix}1 & 6\\2 & 1\end{pmatrix}\begin{pmatrix}2\\0\end{pmatrix} = \begin{pmatrix}2\\4\end{pmatrix}
$$

Now let us verify with the $\phi_\text{elu}$ feature map and normalisation from the Why Replace Attention blog. The causal output at step 4 was computed there as $o_4 = (0.962, 0.577)$. The FWP output at step 4 is $(2, 4)$. These differ because: (a) the FWP uses no feature map ($\phi = \text{identity}$), and (b) the FWP has no normalisation. The FWP perspective is a generalisation — linear attention is the special case with a specific $\phi$ and normalisation.

### 1.5 Interpretation

The formal equivalence is: **linear Transformers are outer product-based Fast Weight Programmers from the 1990s, with the addition of normalisation.** The memories of such FWPs contain key-value associations (stored as outer products), and an FWP learns to reprogram these through sequences of differentiable elementary instructions — which are the additive outer product updates $\phi(k_i) v_i^\top$ invented by the FWP.

This is not merely a historical curiosity. The FWP perspective gives us two concrete analytical tools that the kernel perspective alone does not:

1. **Capacity analysis**: the theory of associative memory and tensor product representations (Smolensky, 1990) tells us exactly when the memory will fail — when we try to store more associations than the dimension allows.

2. **Update rule design**: viewing the state as a programmable memory suggests that the purely additive instruction $S_i = S_{i-1} + \phi(k_i) v_i^\top$ may not be the best programming instruction. This insight motivated the delta rule of the Gated DeltaNet blog.

We now derive the first of these tools.

---

## 2. The Capacity Limit

### 2.1 The memory as a sum of outer products

From the FWP perspective, the state matrix after processing $t$ tokens is:

$$
W^{(t)} = \sum_{i=1}^{t} v^{(i)} \otimes \phi(k^{(i)}) \in \mathbb{R}^{d_v \times d_\text{dot}}
$$

where $d_\text{dot}$ is the output dimension of $\phi$ (the codomain dimension). For $\phi_\text{elu}$, $d_\text{dot} = d_k$. For other feature maps, $d_\text{dot}$ may differ.

This matrix stores associations. Querying with $\phi(k^{(i)})$ retrieves:

$$
W^{(t)} \phi(k^{(i)}) = \sum_{j=1}^{t} v^{(j)} \underbrace{\phi(k^{(j)})^\top \phi(k^{(i)})}_{\text{scalar}}
$$

If we want to retrieve $v^{(i)}$ exactly, we need $\phi(k^{(j)})^\top \phi(k^{(i)}) = 0$ for all $j \neq i$, and $\phi(k^{(i)})^\top \phi(k^{(i)}) = 1$. In other words, the mapped keys $\{\phi(k^{(1)}), \ldots, \phi(k^{(t)})\}$ must be **orthonormal** in $\mathbb{R}^{d_\text{dot}}$.

### 2.2 The orthogonality constraint

In $\mathbb{R}^{d_\text{dot}}$, there can be at most $d_\text{dot}$ mutually orthogonal vectors. This is a fundamental fact from linear algebra — any set of more than $d_\text{dot}$ vectors in $\mathbb{R}^{d_\text{dot}}$ must be linearly dependent, and linearly dependent vectors cannot all be mutually orthogonal.

Therefore:

$$
\boxed{\text{A linear attention state of dimension } d_\text{dot} \text{ can store at most } d_\text{dot} \text{ non-interfering associations.}}
$$

When $t > d_\text{dot}$ — that is, when the sequence length exceeds the state dimension — the model enters an **overcapacity regime**. The mapped keys cannot all be orthogonal, so retrieval produces interference: the retrieved value for key $k^{(i)}$ is contaminated by the values stored with other keys.

### 2.3 Numerical example: orthogonal keys

In our running example, $d_k = 2$ and $\phi_\text{elu}$ preserves the dimension ($d_\text{dot} = d_k = 2$). So the capacity is $d_\text{dot} = 2$. We have $n = 4$ tokens — twice the capacity. Let us see the interference.

After step 2, the state (using $\phi_\text{elu}$ features, no normalisation, causal) is:

$$
S_2 = \phi_\text{elu}(k_1) v_1^\top + \phi_\text{elu}(k_2) v_2^\top = \begin{pmatrix}1\\2\end{pmatrix}(1 \;\; 0) + \begin{pmatrix}2\\1\end{pmatrix}(0 \;\; 1) = \begin{pmatrix}1 & 2\\2 & 1\end{pmatrix}
$$

Query with $\phi_\text{elu}(k_1) = (1, 2)^\top$:

$$
S_2^\top \phi_\text{elu}(k_1) = \begin{pmatrix}1 & 2\\2 & 1\end{pmatrix}\begin{pmatrix}1\\2\end{pmatrix} = \begin{pmatrix}5\\4\end{pmatrix}
$$

We wanted $v_1 = (1, 0)$ but got $(5, 4)$. Even with just 2 stored associations, the retrieval is inaccurate because $\phi_\text{elu}(k_1) = (1, 2)$ and $\phi_\text{elu}(k_2) = (2, 1)$ are not orthogonal: $\phi_\text{elu}(k_1)^\top \phi_\text{elu}(k_2) = 1 \cdot 2 + 2 \cdot 1 = 4 \neq 0$.

The **crosstalk** (Smolensky, 1990) is the interference from the non-orthogonal key. The retrieved value decomposes as:

$$
S_2^\top \phi_\text{elu}(k_1) = \underbrace{\phi_\text{elu}(k_1)^\top \phi_\text{elu}(k_1)}_5 \cdot v_1 + \underbrace{\phi_\text{elu}(k_1)^\top \phi_\text{elu}(k_2)}_4 \cdot v_2 = 5 \cdot (1, 0) + 4 \cdot (0, 1) = (5, 4)
$$

The first term is the desired signal (scaled by 5). The second term is crosstalk (scaled by 4). The signal-to-crosstalk ratio is $5/4 = 1.25$ — barely above 1. With normalisation, the output would be $(5, 4) / (5 + 4) = (0.556, 0.444)$, a blend of $v_1$ and $v_2$.

### 2.4 When keys ARE orthogonal

Now consider what happens if we had keys with orthogonal feature maps. Suppose (hypothetically) $\phi(k_1) = (1, 0)$ and $\phi(k_2) = (0, 1)$. Then:

$$
S_2 = \begin{pmatrix}1\\0\end{pmatrix}(1 \;\; 0) + \begin{pmatrix}0\\1\end{pmatrix}(0 \;\; 1) = \begin{pmatrix}1 & 0\\0 & 1\end{pmatrix}
$$

Query with $\phi(k_1) = (1, 0)^\top$:

$$
S_2^\top \phi(k_1) = \begin{pmatrix}1 & 0\\0 & 1\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}1\\0\end{pmatrix} = v_1 \quad \checkmark
$$

Perfect retrieval, zero crosstalk. But this required orthogonal keys — and we can have at most $d_\text{dot} = 2$ of them.

### 2.5 The capacity formula

Schlag et al. (2021) formalise this using **tensor product representations** (Smolensky, 1990). A tensor product representation stores a structured symbolic expression as a sum of outer products of "role" and "filler" vectors — which translate directly to keys and values in our context.

The capacity result is:

$$
\boxed{\text{Maximum number of perfectly retrievable associations} = d_\text{dot}}
$$

For the feature maps discussed so far:

| Feature map | $d_\text{dot}$ | Capacity |
|---|---|---|
| $\phi_\text{elu}(x) = \text{elu}(x) + 1$ | $d_k$ | $d_k$ |
| No feature map ($\phi = \text{identity}$) | $d_k$ | $d_k$ |

With $d_k = 64$ (our model parameters), the capacity is 64. Any sequence longer than 64 tokens exceeds the capacity, and retrieval errors accumulate.

### 2.6 Numerical check at model scale

With $d_k = 64$ and $h = 8$ heads, each head has capacity 64. The total capacity across all heads is not simply $8 \times 64 = 512$, because the heads operate independently — each head stores its own associations, and they cannot share capacity. But if different heads focus on different associations (which multi-head attention encourages), the effective capacity of the full model at one layer is loosely bounded by $h \times d_k = d_\text{model} = 512$.

For a sequence of $n = 128{,}000$ tokens, the overcapacity ratio is:

$$
\frac{n}{d_k} = \frac{128{,}000}{64} = 2{,}000\times
$$

The model must compress 2,000 tokens worth of information into 1 token's worth of state — per head. This is an extreme compression ratio, and it explains the quality gap between linear and softmax attention on long sequences.

### 2.7 Interpretation

The capacity limit reveals a fundamental tradeoff: to store more associations without interference, we need a larger $d_\text{dot}$. But increasing $d_\text{dot}$ increases the state size ($d_\text{dot} \times d_v$) and the per-step compute ($O(d_\text{dot} \cdot d_v)$). The feature map $\phi$ controls $d_\text{dot}$, so choosing $\phi$ is really choosing a point on the capacity-efficiency curve.

This observation motivates two different strategies:

1. **Increase $d_\text{dot}$ with a deterministic projection** — the DPFP approach from Schlag et al. (2021), which we derive in Section 5.
2. **Approximate the softmax kernel as closely as possible** — the FAVOR+ approach from Choromanski et al. (2021), which we derive in Sections 3 and 4.

We start with the approximation approach, because it answers a natural question: can we have the best of both worlds — linear complexity with the exact attention distribution that made transformers dominant?

---

## 3. Approximating Softmax with Random Features

### 3.1 The goal

The Why Replace Attention blog showed that $\exp(q^\top k / \sqrt{d_k})$ is a valid kernel with an infinite-dimensional feature map. We cannot compute this feature map exactly. But what if we could approximate it with a finite-dimensional feature map $\phi$ such that:

$$
\phi(q)^\top \phi(k) \approx \exp(q^\top k / \sqrt{d_k})
$$

If the approximation is good enough, we get linear complexity (from the associativity trick) with nearly the same attention distribution as softmax.

This is the central idea of Choromanski et al. (2021). They call it **FAVOR+**: **F**ast **A**ttention **V**ia positive **O**rthogonal **R**andom features.

### 3.2 Random feature maps: the general framework

The connection between kernels and random features comes from Rahimi and Recht (2007). A **random feature map** is a function of the form:

$$
\phi(x) = \frac{h(x)}{\sqrt{m}} \begin{pmatrix} f_1(\omega_1^\top x), \ldots, f_1(\omega_m^\top x), \ldots, f_l(\omega_1^\top x), \ldots, f_l(\omega_m^\top x) \end{pmatrix}
$$

where $\omega_1, \ldots, \omega_m$ are random vectors drawn independently from some distribution $\mathcal{D}$ over $\mathbb{R}^d$, $f_1, \ldots, f_l : \mathbb{R} \to \mathbb{R}$ are deterministic functions, and $h : \mathbb{R}^d \to \mathbb{R}$ is a deterministic scaling function. The output $\phi(x)$ has dimension $r = l \cdot m$.

This is Equation 5 of Choromanski et al. (2021). The idea is that if $\mathcal{D}$, $f$, and $h$ are chosen correctly, then:

$$
\text{K}(x, y) = \mathbb{E}[\phi(x)^\top \phi(y)]
$$

where K is the target kernel. This is a consequence of **Bochner's theorem** for shift-invariant kernels (or its generalisation for other kernel families): any positive-definite kernel can be written as an expectation of products of random features.

### 3.3 The softmax kernel

The softmax kernel (omitting the $\sqrt{d}$ scaling for clarity — it can be absorbed into the keys) is:

$$
\text{SM}(x, y) \overset{\text{def}}{=} \exp(x^\top y)
$$

We want to find $\phi$ such that $\phi(x)^\top \phi(y) \approx \exp(x^\top y)$.

Choromanski et al. (2021) start with the observation that $\exp(x^\top y)$ can be decomposed. Using the **exponential identity** $\exp(a + b) = \exp(a)\exp(b)$:

$$
\exp(x^\top y) = \exp\!\left(\frac{\|x\|^2}{2}\right) \exp\!\left(\frac{\|y\|^2}{2}\right) \exp\!\left(-\frac{\|x - y\|^2}{2}\right)
$$

Let us verify this identity. Expanding $\|x - y\|^2 = \|x\|^2 - 2x^\top y + \|y\|^2$:

$$
\exp\!\left(-\frac{\|x - y\|^2}{2}\right) = \exp\!\left(-\frac{\|x\|^2}{2} + x^\top y - \frac{\|y\|^2}{2}\right)
$$

Multiplying by $\exp(\|x\|^2/2) \exp(\|y\|^2/2)$:

$$
\exp\!\left(\frac{\|x\|^2}{2}\right) \exp\!\left(\frac{\|y\|^2}{2}\right) \exp\!\left(-\frac{\|x\|^2}{2} + x^\top y - \frac{\|y\|^2}{2}\right)
$$

The $\|x\|^2/2$ and $-\|x\|^2/2$ cancel. The $\|y\|^2/2$ and $-\|y\|^2/2$ cancel. What remains is $\exp(x^\top y)$. $\checkmark$

Now $\exp(-\|x - y\|^2/2)$ is the **Gaussian kernel** $\text{K}_\text{gauss}(x, y)$, which is a standard shift-invariant kernel. The classical random feature construction for the Gaussian kernel uses random Fourier features (Rahimi and Recht, 2007) with trigonometric functions.

### 3.4 The trigonometric approach (and why it fails)

The classical approach uses $h(x) = 1$, $l = 2$, $f_1 = \sin$, $f_2 = \cos$, and $\mathcal{D} = \mathcal{N}(0, I_d)$. This gives:

$$
\phi^\text{trig}(x) = \frac{\exp(\|x\|^2/2)}{\sqrt{m}} \begin{pmatrix} \sin(\omega_1^\top x), \ldots, \sin(\omega_m^\top x), \cos(\omega_1^\top x), \ldots, \cos(\omega_m^\top x) \end{pmatrix}
$$

The $\exp(\|x\|^2/2)$ factor absorbs the Gaussian-to-softmax conversion from Section 3.3.

The problem is that $\sin$ and $\cos$ output values in $[-1, 1]$. This means $\phi^\text{trig}(x)^\top \phi^\text{trig}(y)$ can be **negative**. In the attention context, a negative similarity score means a negative attention weight, which breaks the interpretation of attention as a weighted average. More practically, negative weights cause the normaliser $\sum_j \phi(q_i)^\top \phi(k_j)$ to become very small or even negative, leading to catastrophic numerical instability.

Choromanski et al. (2021) demonstrate this empirically: trigonometric random features lead to NaN values during training, and when they do converge, the model underperforms significantly compared to positive features. The issue is worst for critical regions where the kernel value $\text{SM}(x, y)$ is small (i.e., when $x$ and $y$ are far apart) — these are exactly the entries where the attention weights should be near zero, but the trigonometric estimator has high variance and can produce large positive or negative values.

### 3.5 Numerical illustration of the variance problem

Let us illustrate with our running example. Take $q_1 = (1, 0)$ and $k_4 = (0, 2)$. The true softmax similarity (without $\sqrt{d_k}$ scaling) is:

$$
\text{SM}(q_1, k_4) = \exp(q_1^\top k_4) = \exp(1 \cdot 0 + 0 \cdot 2) = \exp(0) = 1
$$

With one random vector $\omega = (0.5, -0.3)$ (drawn from $\mathcal{N}(0, I_2)$), the trigonometric estimator gives:

$$
\omega^\top q_1 = 0.5, \quad \omega^\top k_4 = -0.6
$$

$$
\phi^\text{trig}(q_1) = \exp(\|q_1\|^2/2) \cdot (\sin(0.5), \cos(0.5)) = \exp(0.5) \cdot (0.479, 0.878) = (0.790, 1.447)
$$

$$
\phi^\text{trig}(k_4) = \exp(\|k_4\|^2/2) \cdot (\sin(-0.6), \cos(-0.6)) = \exp(2) \cdot (-0.565, 0.825) = (-4.171, 6.089)
$$

$$
\phi^\text{trig}(q_1)^\top \phi^\text{trig}(k_4) = 0.790 \times (-4.171) + 1.447 \times 6.089 = -3.295 + 8.811 = 5.516
$$

The true value is 1. The single-sample estimate is 5.516 — off by a factor of 5.5. Worse, with a different $\omega$, the estimate could be negative. The high variance comes from the $\exp(\|x\|^2/2)$ scaling factor, which amplifies the oscillations of the trigonometric functions.

---

## 4. FAVOR+: Positive Orthogonal Random Features

### 4.1 The key insight: positive random features

Choromanski et al. (2021) observe that the softmax kernel admits a decomposition using **positive** random features — no trigonometric functions needed. This is Lemma 1 of their paper, and it is the theoretical foundation of FAVOR+.

**Lemma 1 (Positive Random Features for Softmax).** For $x, y \in \mathbb{R}^d$, $z = x + y$, and $\omega \sim \mathcal{N}(0, I_d)$:

$$
\text{SM}(x, y) = \mathbb{E}_{\omega \sim \mathcal{N}(0, I_d)}\!\left[\exp\!\left(\omega^\top x - \frac{\|x\|^2}{2}\right) \exp\!\left(\omega^\top y - \frac{\|y\|^2}{2}\right)\right]
$$

Let us prove this. Define:

$$
\Lambda = \exp\!\left(-\frac{\|x\|^2 + \|y\|^2}{2}\right)
$$

Then the expected value becomes:

$$
\Lambda \cdot \mathbb{E}_\omega\!\left[\exp(\omega^\top x) \exp(\omega^\top y)\right] = \Lambda \cdot \mathbb{E}_\omega\!\left[\exp(\omega^\top (x + y))\right] = \Lambda \cdot \mathbb{E}_\omega\!\left[\exp(\omega^\top z)\right]
$$

where $z = x + y$. We used the **exponential product rule**: $\exp(a)\exp(b) = \exp(a + b)$.

Now we need the **moment generating function of a Gaussian**. If $\omega \sim \mathcal{N}(0, I_d)$, then $\omega^\top z \sim \mathcal{N}(0, \|z\|^2)$ (since $\omega^\top z$ is a linear combination of independent standard normals with coefficients $z_1, \ldots, z_d$, giving variance $\sum_i z_i^2 = \|z\|^2$). For a random variable $X \sim \mathcal{N}(0, \sigma^2)$:

$$
\mathbb{E}[\exp(X)] = \exp\!\left(\frac{\sigma^2}{2}\right)
$$

This is the **moment generating function of the normal distribution** evaluated at $t = 1$. Applying it with $\sigma^2 = \|z\|^2$:

$$
\mathbb{E}_\omega[\exp(\omega^\top z)] = \exp\!\left(\frac{\|z\|^2}{2}\right) = \exp\!\left(\frac{\|x + y\|^2}{2}\right)
$$

Expanding $\|x + y\|^2 = \|x\|^2 + 2x^\top y + \|y\|^2$:

$$
\exp\!\left(\frac{\|x\|^2 + 2x^\top y + \|y\|^2}{2}\right)
$$

Multiplying by $\Lambda = \exp(-(\|x\|^2 + \|y\|^2)/2)$:

$$
\exp\!\left(-\frac{\|x\|^2 + \|y\|^2}{2}\right) \exp\!\left(\frac{\|x\|^2 + 2x^\top y + \|y\|^2}{2}\right)
$$

The $\|x\|^2/2$ terms cancel. The $\|y\|^2/2$ terms cancel. What remains:

$$
\exp(x^\top y) = \text{SM}(x, y) \quad \checkmark
$$

### 4.2 The positive feature map

Lemma 1 tells us that the softmax kernel decomposes as $\mathbb{E}[\psi(x) \cdot \psi(y)]$ where:

$$
\psi_\omega(x) = \exp\!\left(\omega^\top x - \frac{\|x\|^2}{2}\right)
$$

This function is always **positive**: the exponential function maps $\mathbb{R} \to \mathbb{R}_{>0}$. No trigonometric oscillations, no negative values, no sign cancellations.

With $m$ random samples $\omega_1, \ldots, \omega_m \sim \mathcal{N}(0, I_d)$, the feature map is:

$$
\boxed{\phi^+(x) = \frac{1}{\sqrt{m}} \exp\!\left(-\frac{\|x\|^2}{2}\right) \begin{pmatrix} \exp(\omega_1^\top x) \\ \exp(\omega_2^\top x) \\ \vdots \\ \exp(\omega_m^\top x) \end{pmatrix}}
$$

The output dimension is $d_\text{dot} = m$, and every component is positive.

The approximation is:

$$
\phi^+(q)^\top \phi^+(k) = \frac{1}{m} \exp\!\left(-\frac{\|q\|^2 + \|k\|^2}{2}\right) \sum_{i=1}^m \exp(\omega_i^\top q) \exp(\omega_i^\top k)
$$

By the law of large numbers, as $m \to \infty$, this converges to $\text{SM}(q, k) = \exp(q^\top k)$.

### 4.3 Numerical example

Let us compute $\phi^+$ for our running example with $m = 4$ random projections. We fix:

$$
\omega_1 = \begin{pmatrix}0.5\\-0.3\end{pmatrix}, \quad \omega_2 = \begin{pmatrix}-0.8\\1.2\end{pmatrix}, \quad \omega_3 = \begin{pmatrix}0.1\\0.7\end{pmatrix}, \quad \omega_4 = \begin{pmatrix}-0.4\\-0.9\end{pmatrix}
$$

These are concrete realisations from $\mathcal{N}(0, I_2)$, chosen for easy arithmetic.

**For $q_1 = (1, 0)$:** $\|q_1\|^2 = 1$, so $\exp(-\|q_1\|^2/2) = \exp(-0.5) = 0.607$.

$$
\omega_1^\top q_1 = 0.5, \quad \omega_2^\top q_1 = -0.8, \quad \omega_3^\top q_1 = 0.1, \quad \omega_4^\top q_1 = -0.4
$$

$$
\phi^+(q_1) = \frac{0.607}{\sqrt{4}} \begin{pmatrix} \exp(0.5) \\ \exp(-0.8) \\ \exp(0.1) \\ \exp(-0.4) \end{pmatrix} = \frac{0.607}{2} \begin{pmatrix} 1.649 \\ 0.449 \\ 1.105 \\ 0.670 \end{pmatrix} = \begin{pmatrix} 0.500 \\ 0.136 \\ 0.335 \\ 0.203 \end{pmatrix}
$$

Every component is positive. $\checkmark$

**For $k_1 = (0, 1)$:** $\|k_1\|^2 = 1$, so $\exp(-\|k_1\|^2/2) = 0.607$.

$$
\omega_1^\top k_1 = -0.3, \quad \omega_2^\top k_1 = 1.2, \quad \omega_3^\top k_1 = 0.7, \quad \omega_4^\top k_1 = -0.9
$$

$$
\phi^+(k_1) = \frac{0.607}{2} \begin{pmatrix} \exp(-0.3) \\ \exp(1.2) \\ \exp(0.7) \\ \exp(-0.9) \end{pmatrix} = \frac{0.607}{2} \begin{pmatrix} 0.741 \\ 3.320 \\ 2.014 \\ 0.407 \end{pmatrix} = \begin{pmatrix} 0.225 \\ 1.008 \\ 0.611 \\ 0.123 \end{pmatrix}
$$

**For $k_2 = (1, 0)$:** $\|k_2\|^2 = 1$, so $\exp(-\|k_2\|^2/2) = 0.607$. Same projections as $q_1$:

$$
\phi^+(k_2) = \phi^+(q_1) = \begin{pmatrix} 0.500 \\ 0.136 \\ 0.335 \\ 0.203 \end{pmatrix}
$$

(Since $k_2 = q_1 = (1,0)$, their feature maps are identical.)

**Approximation check for $\text{SM}(q_1, k_1)$:**

$$
\phi^+(q_1)^\top \phi^+(k_1) = 0.500 \times 0.225 + 0.136 \times 1.008 + 0.335 \times 0.611 + 0.203 \times 0.123
$$

$$
= 0.113 + 0.137 + 0.205 + 0.025 = 0.480
$$

True value: $\text{SM}(q_1, k_1) = \exp(q_1^\top k_1) = \exp(1 \cdot 0 + 0 \cdot 1) = \exp(0) = 1.000$.

The estimate is $0.480$ — off by a factor of 2. With only $m = 4$ random projections, this level of error is expected. As $m$ increases, the estimate converges to the true value.

**Approximation check for $\text{SM}(q_1, k_2)$:**

$$
\phi^+(q_1)^\top \phi^+(k_2) = 0.500^2 + 0.136^2 + 0.335^2 + 0.203^2 = 0.250 + 0.018 + 0.112 + 0.041 = 0.421
$$

True value: $\exp(q_1^\top k_2) = \exp(1 \cdot 1 + 0 \cdot 0) = \exp(1) = 2.718$.

The estimate is $0.421$ vs the true $2.718$. With $m = 4$, the approximation is rough. The Performers paper recommends $m = O(d_k \log d_k)$ — for $d_k = 64$, this gives $m \approx 64 \times \log(64) \approx 64 \times 4.16 \approx 266$.

### 4.4 Why positive features have lower variance

This is the part that matters most, and it is easy to gloss over. The key claim of Choromanski et al. (2021) is that positive features are not just "nicer" (no negative weights) but provably **lower variance** than trigonometric features in the critical regime.

**Lemma 2 (Choromanski et al., 2021).** For independent random samples $\omega_i$:

$$
\text{MSE}(\widehat{\text{SM}}_m^\text{trig}(x, y)) = \frac{1}{2m} \exp(\|x + y\|^2) \cdot \text{SM}^{-2}(x, y) \cdot (1 - \exp(-\|x - y\|^2))^2
$$

$$
\text{MSE}(\widehat{\text{SM}}_m^+(x, y)) = \frac{1}{m} \exp(\|x + y\|^2) \cdot \text{SM}^2(x, y) \cdot (1 - \exp(-\|x - y\|^2))
$$

where MSE stands for the **mean squared error**.

The critical comparison: when $\text{SM}(x, y) \to 0$ (meaning $x^\top y \to -\infty$, so the tokens are very dissimilar), the trigonometric MSE contains $\text{SM}^{-2}(x, y)$, which **blows up** to infinity. The positive MSE contains $\text{SM}^2(x, y)$, which **shrinks** to zero.

In plain language: for tokens that should have near-zero attention weight, the trigonometric estimator has high variance (it wildly over- or under-estimates), while the positive estimator has low variance (it estimates near-zero accurately). Since most token pairs in a long sequence are dissimilar (attention is typically sparse), the positive estimator is dramatically better in practice.

### 4.5 Interpretation

The positive random feature map $\phi^+$ is an **unbiased estimator** of the softmax kernel: $\mathbb{E}[\phi^+(x)^\top \phi^+(y)] = \text{SM}(x, y)$. The trigonometric estimator is also unbiased. But unbiased estimators can still have very different variances — and in the attention context, high variance is catastrophic because it cascades through the model's layers, amplified by the **Lipschitz constant** of the subsequent MLP and normalisation layers.

---

## 5. Orthogonal Random Features

### 5.1 The variance reduction idea

The positive random feature map from Section 4 uses $m$ independent random vectors $\omega_1, \ldots, \omega_m \sim \mathcal{N}(0, I_d)$. Choromanski et al. (2021) propose a further improvement: instead of sampling the $\omega_i$ independently, make them **exactly orthogonal**.

An **orthogonal random feature** (ORF) construction (Yu et al., 2016) works as follows. Draw a random matrix $M \in \mathbb{R}^{m \times d}$ with i.i.d. Gaussian entries, then apply the **Gram-Schmidt orthogonalisation** procedure to obtain an orthogonal matrix $Q \in \mathbb{R}^{m \times d}$ (with $m \leq d$). The rows of $Q$ are the orthogonal random vectors $\omega_1, \ldots, \omega_m$.

Each individual $\omega_i$ still has the correct marginal distribution (each row of the orthogonalised matrix is distributed as a random vector on the sphere of radius $\sqrt{d}$, which matches the direction of a Gaussian vector), but the rows are correlated — they are constrained to be mutually orthogonal. This is a form of **antithetic sampling**, a classical variance reduction technique from Monte Carlo methods.

### 5.2 The variance reduction guarantee

**Theorem 2 (Choromanski et al., 2021).** Let $\widehat{\text{SM}}_m^{\text{ort}+}(x, y)$ denote the positive softmax estimator with orthogonal random features, and $\widehat{\text{SM}}_m^+(x, y)$ the same estimator with independent (IID) random features. Then for any $d > 0$:

$$
\text{MSE}(\widehat{\text{SM}}_m^{\text{ort}+}(x, y)) \leq \text{MSE}(\widehat{\text{SM}}_m^+(x, y)) - \frac{2(m-1)}{m(d+2)}\left(\text{SM}(x, y) - \exp\!\left(-\frac{\|x\|^2 + \|y\|^2}{2}\right)\right)^2
$$

The second term is always non-negative (since $(a - b)^2 \geq 0$), which means:

$$
\boxed{\text{MSE}(\text{orthogonal}) \leq \text{MSE}(\text{IID})}
$$

Orthogonal features are **always** at least as good as independent features, and strictly better whenever $\text{SM}(x, y) \neq \exp(-(\|x\|^2 + \|y\|^2)/2)$ — which is the case unless $x^\top y = 0$.

The improvement factor $\frac{2(m-1)}{m(d+2)}$ grows with $m/d$, reaching approximately $2/d$ when $m = d$. For $d = 64$, this gives an improvement of about $2/66 \approx 3\%$ per entry — small per entry, but compounded across the $n^2$ entries of the attention matrix, the cumulative effect is significant.

### 5.3 Numerical illustration

For our running example with $d = 2$, we construct orthogonal random features by orthogonalising our first two random vectors $\omega_1 = (0.5, -0.3)$ and $\omega_2 = (-0.8, 1.2)$.

**Step 1: Normalise $\omega_1$.**

$$
\|\omega_1\| = \sqrt{0.25 + 0.09} = \sqrt{0.34} = 0.583
$$

$$
\hat{\omega}_1 = \frac{\omega_1}{\|\omega_1\|} = \frac{(0.5, -0.3)}{0.583} = (0.857, -0.515)
$$

**Step 2: Subtract the projection of $\omega_2$ onto $\hat{\omega}_1$** (this is the **Gram-Schmidt process**).

$$
\text{proj} = (\omega_2^\top \hat{\omega}_1) \hat{\omega}_1 = ((-0.8)(0.857) + (1.2)(-0.515)) \cdot (0.857, -0.515)
$$

$$
= (-0.686 - 0.618) \cdot (0.857, -0.515) = -1.304 \cdot (0.857, -0.515) = (-1.117, 0.672)
$$

$$
\omega_2' = \omega_2 - \text{proj} = (-0.8 + 1.117, \; 1.2 - 0.672) = (0.317, 0.528)
$$

**Step 3: Normalise $\omega_2'$.**

$$
\|\omega_2'\| = \sqrt{0.100 + 0.279} = \sqrt{0.379} = 0.616
$$

$$
\hat{\omega}_2 = \frac{\omega_2'}{0.616} = (0.515, 0.857)
$$

**Step 4: Scale both to have the correct norm.** For the ORF construction, the rows are scaled to have norm $\sqrt{d} = \sqrt{2} = 1.414$:

$$
\omega_1^\text{ort} = 1.414 \cdot (0.857, -0.515) = (1.212, -0.728)
$$

$$
\omega_2^\text{ort} = 1.414 \cdot (0.515, 0.857) = (0.728, 1.212)
$$

Verification of orthogonality: $\omega_1^{\text{ort}\top} \omega_2^\text{ort} = 1.212 \times 0.728 + (-0.728) \times 1.212 = 0.882 - 0.882 = 0$. $\checkmark$

Now computing $\phi^{\text{ort}+}$ for $q_1 = (1, 0)$ with these two orthogonal projections ($m = 2$):

$$
\omega_1^{\text{ort}\top} q_1 = 1.212, \quad \omega_2^{\text{ort}\top} q_1 = 0.728
$$

$$
\phi^{\text{ort}+}(q_1) = \frac{0.607}{\sqrt{2}} \begin{pmatrix} \exp(1.212) \\ \exp(0.728) \end{pmatrix} = 0.429 \begin{pmatrix} 3.360 \\ 2.071 \end{pmatrix} = \begin{pmatrix} 1.442 \\ 0.889 \end{pmatrix}
$$

The components are still positive, as guaranteed. The orthogonal construction ensures that the two random projections "cover" the 2D space more efficiently than two independent random directions would.

### 5.4 The complete FAVOR+ mechanism

Putting together positive features (Section 4.2) and orthogonal features (Section 5.1), the FAVOR+ feature map is:

$$
\boxed{\phi^{\text{FAVOR+}}(x) = \frac{1}{\sqrt{m}} \exp\!\left(-\frac{\|x\|^2}{2}\right) \begin{pmatrix} \exp(\omega_1^{\text{ort}\top} x) \\ \vdots \\ \exp(\omega_m^{\text{ort}\top} x) \end{pmatrix}}
$$

where $\omega_1^\text{ort}, \ldots, \omega_m^\text{ort}$ are orthogonal random vectors. The output dimension is $d_\text{dot} = m$.

With this feature map, the linear attention approximation becomes:

$$
o_i = \frac{\phi^{\text{FAVOR+}}(q_i)^\top S_i}{\phi^{\text{FAVOR+}}(q_i)^\top z_i}
$$

where $S_i = \sum_{j=1}^i \phi^{\text{FAVOR+}}(k_j) v_j^\top$ and $z_i = \sum_{j=1}^i \phi^{\text{FAVOR+}}(k_j)$, exactly as in the recurrent form from the Why Replace Attention blog.

The capacity of the FAVOR+ feature map is $d_\text{dot} = m$. The Performers paper recommends $m = O(d_k \log d_k)$, giving a capacity of about $d_k \log d_k$ — larger than the $d_k$ capacity of $\phi_\text{elu}$ by a factor of $\log d_k$.

### 5.5 The Performer architecture

A **Performer** is a Transformer where every attention layer is replaced by FAVOR+. All other components — the MLP layers, residual connections, layer norms, positional encodings — remain identical. This means:

1. A Performer can be initialised from a pretrained Transformer by transferring all weights except the attention mechanism.
2. After a small amount of fine-tuning, the Performer recovers most of the pretrained model's accuracy (Choromanski et al., 2021, Figure 5).

This **backward compatibility** is a significant practical advantage: it means FAVOR+ can be deployed as a drop-in replacement for softmax attention in existing models, without retraining from scratch.

### 5.6 Redrawing random features

One subtlety: during training, the random matrix $\Omega = [\omega_1, \ldots, \omega_m]^\top$ is redrawn periodically (e.g., every few hundred gradient steps). This serves two purposes:

1. It prevents the model from overfitting to a specific set of random features.
2. It improves the overall approximation quality across the training run.

Choromanski et al. (2021) show that redrawing is crucial for achieving performance matching the regular Transformer on larger datasets (PG-19), while on smaller datasets (LM1B), even a fixed random matrix works well.

---

## 6. DPFP: Deterministic Parameter-Free Projection

### 6.1 A different philosophy

FAVOR+ asks: how can we approximate the softmax kernel? Schlag et al. (2021) ask a different question: how can we increase the memory capacity $d_\text{dot}$ without random sampling?

Their answer is the **Deterministic Parameter-Free Projection** (DPFP): a feature map $\phi$ that deterministically projects $d_k$-dimensional keys into a higher-dimensional space of dimension $d_\text{dot} > d_k$, increasing the number of orthogonal directions available for storing associations.

### 6.2 Design goals

From the capacity analysis (Section 2) and the kernel requirements (Section 5.1 of the FWP paper), the feature map $\phi$ should satisfy:

1. **Positivity**: all components of $\phi(x)$ must be non-negative, so that $\phi(q)^\top \phi(k) \geq 0$ and attention weights are non-negative.

2. **Increased dimension**: $d_\text{dot} > d_k$, to increase the capacity beyond $d_k$.

3. **Orthogonality promotion**: keys that are different in $\mathbb{R}^{d_k}$ should map to vectors that are as orthogonal as possible in $\mathbb{R}^{d_\text{dot}}$, minimizing crosstalk.

4. **Determinism**: no random sampling required, avoiding the variance inherent in FAVOR+.

5. **Efficiency**: the function should be cheap to compute, ideally parallelisable across dimensions.

### 6.3 The construction

The DPFP construction works in two steps.

**Step 1: Concatenate and ReLU.** Given an input vector $k \in \mathbb{R}^{d_k}$, form the concatenation $[k, -k] \in \mathbb{R}^{2d_k}$ and apply the element-wise **rectifier function** $r(a) = \max(0, a)$ (also known as **ReLU**):

$$
\tilde{k} = r\!\left(\begin{bmatrix} k \\ -k \end{bmatrix}\right) \in \mathbb{R}^{2d_k}_{\geq 0}
$$

This vector has the property that for each dimension $m$ of the original key, exactly one of $\tilde{k}_m = r(k_m)$ or $\tilde{k}_{m + d_k} = r(-k_m)$ is non-zero (assuming $k_m \neq 0$). Positive components land in the first $d_k$ positions, negative components land in the second $d_k$ positions (with their sign flipped).

**Step 2: Pairwise products with circular shifts.** The hyperparameter $\nu \in \{1, 2, \ldots, 2d_k - 1\}$ controls the capacity. For shift value $s \in \{1, \ldots, \nu\}$, define a shifted copy:

$$
\tilde{k}^{(s)}_i = \tilde{k}_{(i + s) \bmod 2d_k}
$$

Then the DPFP-$\nu$ feature map is the concatenation of element-wise products for all shifts:

$$
\boxed{\phi_\text{DPFP-\nu}(k) = \text{concat}\!\left(\tilde{k} \odot \tilde{k}^{(1)}, \; \tilde{k} \odot \tilde{k}^{(2)}, \; \ldots, \; \tilde{k} \odot \tilde{k}^{(\nu)}\right) \in \mathbb{R}^{2d_k \nu}_{\geq 0}}
$$

where $\odot$ denotes the **Hadamard (element-wise) product**. The output dimension is:

$$
d_\text{dot} = 2d_k \nu
$$

### 6.4 Why this promotes orthogonality

The key insight is that the pairwise products $\tilde{k}_i \cdot \tilde{k}_{i+s}$ partition the input space into non-overlapping regions. For a given shift $s$, the product $\tilde{k}_i \cdot \tilde{k}_{i+s}$ is non-zero only if both $\tilde{k}_i > 0$ and $\tilde{k}_{i+s} > 0$. Since $\tilde{k} = r([k, -k])$, this means both the $i$-th and $(i+s)$-th elements of $[k, -k]$ must be positive.

For two keys $k^{(a)}$ and $k^{(b)}$, the product $\phi(k^{(a)})^\top \phi(k^{(b)})$ involves terms like $\tilde{k}^{(a)}_i \tilde{k}^{(a)}_{i+s} \cdot \tilde{k}^{(b)}_i \tilde{k}^{(b)}_{i+s}$. This is zero unless all four factors are positive — meaning both keys have the same sign pattern in the relevant dimensions. If $k^{(a)}$ and $k^{(b)}$ differ in the sign of any component involved in a product, that product contributes zero to the dot product. This is a much stronger orthogonality condition than the raw dot product $k^{(a)\top} k^{(b)}$.

### 6.5 Numerical example

Let us compute DPFP-$1$ for our running example. With $d_k = 2$ and $\nu = 1$, the output dimension is $d_\text{dot} = 2 \cdot 2 \cdot 1 = 4$.

**For $k_3 = (1, 1)$:** The concatenation is $[k_3, -k_3] = (1, 1, -1, -1)$.

$$
\tilde{k}_3 = r(1, 1, -1, -1) = (1, 1, 0, 0)
$$

Circular shift by 1: $\tilde{k}_3^{(1)} = (0, 0, 1, 1) \to$ shifted: take element at positions $(1, 2, 3, 0)$ of $(1, 1, 0, 0)$, giving $(1, 0, 0, 1)$.

$$
\phi_\text{DPFP-1}(k_3) = \tilde{k}_3 \odot \tilde{k}_3^{(1)} = (1, 1, 0, 0) \odot (1, 0, 0, 1) = (1, 0, 0, 0)
$$

**For $k_1 = (0, 1)$:** $[k_1, -k_1] = (0, 1, 0, -1)$.

$$
\tilde{k}_1 = r(0, 1, 0, -1) = (0, 1, 0, 0)
$$

Shifted: $(1, 0, 0, 0)$.

$$
\phi_\text{DPFP-1}(k_1) = (0, 1, 0, 0) \odot (1, 0, 0, 0) = (0, 0, 0, 0)
$$

**For $k_2 = (1, 0)$:** $[k_2, -k_2] = (1, 0, -1, 0)$.

$$
\tilde{k}_2 = r(1, 0, -1, 0) = (1, 0, 0, 0)
$$

Shifted: $(0, 0, 0, 1)$.

$$
\phi_\text{DPFP-1}(k_2) = (1, 0, 0, 0) \odot (0, 0, 0, 1) = (0, 0, 0, 0)
$$

**For $k_4 = (0, 2)$:** $[k_4, -k_4] = (0, 2, 0, -2)$.

$$
\tilde{k}_4 = r(0, 2, 0, -2) = (0, 2, 0, 0)
$$

Shifted: $(2, 0, 0, 0)$.

$$
\phi_\text{DPFP-1}(k_4) = (0, 2, 0, 0) \odot (2, 0, 0, 0) = (0, 0, 0, 0)
$$

### 6.6 Why our example is degenerate — and the lesson

Three of our four keys map to the zero vector under DPFP-$1$. This is not a bug — it reveals an important property of the DPFP construction: **it requires keys with mixed-sign components to produce non-zero features.** A component pair $(k_m, k_{m+1})$ produces a non-zero DPFP-1 output only when both $r(k_m)$ and $r(k_{m+1})$ (or their negated counterparts) are positive, which requires consecutive positive entries in $[k, -k]$.

Our running example has all non-negative key components — $(0,1)$, $(1,0)$, $(1,1)$, $(0,2)$ — so the only key with two consecutive positive entries in $[k, -k]$ is $k_3 = (1,1)$, which has $\tilde{k}_3 = (1, 1, 0, 0)$.

In practice, keys are the output of the projection $W_k x$, which produces both positive and negative values. To demonstrate DPFP properly, consider keys with mixed signs:

$$
k_A = (1.5, -0.8), \quad k_B = (-0.5, 1.2)
$$

**For $k_A$:** $[k_A, -k_A] = (1.5, -0.8, -1.5, 0.8)$. $\tilde{k}_A = (1.5, 0, 0, 0.8)$. Shifted: $(0, 0, 0.8, 1.5)$.

$$
\phi_\text{DPFP-1}(k_A) = (1.5, 0, 0, 0.8) \odot (0, 0, 0.8, 1.5) = (0, 0, 0, 1.2)
$$

**For $k_B$:** $[k_B, -k_B] = (-0.5, 1.2, 0.5, -1.2)$. $\tilde{k}_B = (0, 1.2, 0.5, 0)$. Shifted: $(1.2, 0.5, 0, 0)$.

$$
\phi_\text{DPFP-1}(k_B) = (0, 1.2, 0.5, 0) \odot (1.2, 0.5, 0, 0) = (0, 0.6, 0, 0)
$$

Now $\phi(k_A) = (0, 0, 0, 1.2)$ and $\phi(k_B) = (0, 0.6, 0, 0)$ are orthogonal:

$$
\phi(k_A)^\top \phi(k_B) = 0 \cdot 0 + 0 \cdot 0.6 + 0 \cdot 0 + 1.2 \cdot 0 = 0
$$

The DPFP has mapped two non-orthogonal keys ($k_A^\top k_B = 1.5 \times (-0.5) + (-0.8) \times 1.2 = -0.75 - 0.96 = -1.71$) to orthogonal feature vectors. This is precisely the orthogonality promotion that increases capacity.

### 6.7 Capacity comparison

| Feature map | $d_\text{dot}$ for $d_k = 64$ | Capacity |
|---|---|---|
| ELU+1 | $64$ | 64 |
| DPFP-1 | $128$ | 128 |
| DPFP-2 | $256$ | 256 |
| DPFP-3 | $384$ | 384 |
| FAVOR+ ($m = 266$) | $266$ | 266 |

Schlag et al. (2021) verify experimentally that these capacity limits are tight: on a synthetic key-value retrieval task, each model fails exactly when the number of unique keys exceeds $d_\text{dot}$ (Figure 2 of the FWP paper). Linear Attention ($d_\text{dot} = 64$) begins to fail at 60 keys. DPFP-1, -2, -3 fail at 128, 256, and 384 keys respectively. Softmax attention handles over 500 keys without failing, confirming its theoretically infinite capacity.

---

## 7. Sum Normalisation

### 7.1 The problem with attention normalisation

The standard normalisation for linear attention divides by $\phi(q_i)^\top z_i$ (the denominator from the Why Replace Attention blog). This is **attention normalisation**: the output is a weighted average of stored values, with weights summing to 1.

Schlag et al. (2021) identify a problem with this normalisation when combined with the delta update rule (the Gated DeltaNet blog). The accumulator $z^{(i)} = z^{(i-1)} + \phi(k^{(i)})$ grows monotonically — it only ever increases, because $\phi(k^{(i)})$ is non-negative. But the delta update rule removes old associations from the state $W^{(i)}$. This means the numerator $W^{(i)} \phi(q^{(i)})$ can decrease (as old values are erased), while the denominator $z^{(i)} \cdot \phi(q^{(i)})$ continues to grow. The result: the output shrinks over time, approaching zero as the denominator accumulates.

### 7.2 Sum normalisation

Schlag et al. (2021) propose a simpler alternative: divide each feature vector by the sum of its components before using it. For the query:

$$
\phi'(q^{(i)}) = \frac{\phi(q^{(i)})}{\sum_{j=1}^{d_\text{dot}} \phi(q^{(i)})_j}
$$

and similarly for the key. Since all components of $\phi$ are non-negative (by design), the sum is positive, and the normalised vector has components that sum to 1. The output of the matrix-vector multiplication $W^{(i)} \phi'(q^{(i)})$ is then a weighted sum of the columns of $W^{(i)}$ where the weights sum to 1 — a proper convex combination.

### 7.3 Numerical example

For $q_1 = (1, 0)$ with $\phi_\text{elu}$: $\phi_\text{elu}(q_1) = (2, 1)$. Sum: $2 + 1 = 3$.

$$
\phi'_\text{elu}(q_1) = \left(\frac{2}{3}, \frac{1}{3}\right)
$$

For $k_1 = (0, 1)$ with $\phi_\text{elu}$: $\phi_\text{elu}(k_1) = (1, 2)$. Sum: $1 + 2 = 3$.

$$
\phi'_\text{elu}(k_1) = \left(\frac{1}{3}, \frac{2}{3}\right)
$$

Normalised similarity: $\phi'(q_1)^\top \phi'(k_1) = \frac{2}{3} \cdot \frac{1}{3} + \frac{1}{3} \cdot \frac{2}{3} = \frac{2}{9} + \frac{2}{9} = \frac{4}{9} = 0.444$.

The un-normalised similarity was $\phi(q_1)^\top \phi(k_1) = 2 \cdot 1 + 1 \cdot 2 = 4$. The normalisation reduces the magnitude but preserves the relative ordering of similarities.

### 7.4 Why sum normalisation works with the delta rule

With sum normalisation, the output is:

$$
y^{(i)} = W^{(i)} \phi'(q^{(i)})
$$

This is a weighted sum of the columns of $W^{(i)}$, where the weights are the components of $\phi'(q^{(i)})$ and they sum to 1. There is no separate denominator that can diverge. When the delta rule modifies $W^{(i)}$ (erasing and rewriting), the output responds directly to the modified state — there is no growing denominator to dampen the signal.

Schlag et al. (2021) verify experimentally that sum normalisation outperforms attention normalisation on both synthetic retrieval tasks and language modelling. On WikiText-103, the best configuration uses sum normalisation without attention normalisation and without absolute positional encoding (Table 3 of the FWP paper): validation perplexity 28.1, test perplexity 31.1.

---

## 8. The Unified View: Comparing Feature Maps

### 8.1 The design space

We have now encountered four feature maps for linear attention:

| Feature map | Type | $d_\text{dot}$ | Key property |
|---|---|---|---|
| Identity ($\phi = I$) | Deterministic | $d_k$ | No transformation; FWP baseline |
| ELU+1 | Deterministic | $d_k$ | Simple, element-wise, preserves dimension |
| DPFP-$\nu$ | Deterministic | $2d_k\nu$ | Increases capacity by ReLU-gated quadrant projection |
| FAVOR+ | Stochastic | $m$ | Approximates softmax with provable guarantees |

These span two axes of the design space:

**Axis 1: Deterministic vs. stochastic.** DPFP and ELU+1 produce the same output every time. FAVOR+ depends on the random matrix $\Omega$, introducing variance into the model's output. During training, this variance is managed by periodic redrawing. During inference, the random features are fixed once.

**Axis 2: Approximation vs. capacity.** FAVOR+ is designed to approximate softmax attention as closely as possible — its quality improves with $m$ and converges to exact softmax as $m \to \infty$. DPFP makes no claim about approximating softmax. Instead, it increases the capacity of the associative memory, allowing the model to store more associations without interference. The two goals are related (softmax attention has infinite capacity) but distinct (a high-capacity kernel can still produce very different attention distributions from softmax).

### 8.2 Experimental evidence: synthetic retrieval

Schlag et al. (2021) test all feature maps on a synthetic key-value retrieval task (Section 6.1 of the FWP paper). The model must memorise a sequence of key-value pairs and retrieve the correct value when queried. The results (Figure 2 of the FWP paper):

- **Softmax attention**: perfect retrieval up to 500+ keys (limited only by training, not capacity)
- **Linear Attention** (ELU+1, $d_\text{dot} = 64$): fails at approximately 60 keys
- **FAVOR+** with 64 random features: fails to achieve zero loss at any sequence length
- **FAVOR+** with 128 random features: fails at approximately 60 keys (same as Linear Attention, since capacity is unrelated to the approximation quality of FAVOR+ in this setting)
- **FAVOR+** with 512 random features: slight improvement but still limited
- **DPFP-1** ($d_\text{dot} = 128$): fails at approximately 128 keys
- **DPFP-2** ($d_\text{dot} = 256$): fails at approximately 256 keys
- **DPFP-3** ($d_\text{dot} = 384$): fails at approximately 384 keys

The results confirm the capacity analysis exactly: each model fails when the number of keys exceeds $d_\text{dot}$.

### 8.3 Experimental evidence: machine translation

On the WMT14 English-to-German translation task (Table 1 of the FWP paper), with $d_k = 64$ and 8 heads:

| Model | $d_\text{dot}$ | Test BLEU |
|---|---|---|
| Standard Transformer | — | 27.7 |
| Linear Transformer (ELU+1) | 64 | 26.8 |
| Performer ($m = 256$) | 256 | 25.3 |
| Performer ($m = 512$) | 512 | **27.7** |
| DPFP (ours) | 256 | 26.9 |
| DPFP (ours) | 512 | 27.1 |

The Performer matches the standard Transformer when $m$ is large enough ($m = 512$, so $d_\text{dot} = 512$). DPFP outperforms the Linear Transformer and reaches 27.1 BLEU at $d_\text{dot} = 512$. But this comes at a cost: the state matrix $S$ is $512 \times 64$ instead of $64 \times 64$ — 8 times larger.

### 8.4 Experimental evidence: language modelling

On WikiText-103 language modelling (Table 2 of the FWP paper), comparing update rules. The small configuration has $D = 128$, $L = 256$ (40M parameters) and the medium has $D = 256$, $L = 384$ (90M parameters). Both are in the overcapacity regime:

| Model | Update rule | Small (test PPL) | Medium (test PPL) |
|---|---|---|---|
| Transformer | — | 34.1 | 29.6 |
| Linear Transformer | sum | 38.3 | 33.0 |
| Delta Network | delta | **35.5** | **31.5** |
| Performer | sum | 39.6 | 33.8 |
| Performer | delta | **37.2** | **31.8** |

Two observations. First, the delta update rule from the Gated DeltaNet blog improves both the Linear Transformer and the Performer in both configurations. For the Linear Transformer, the delta rule reduces test perplexity from 38.3 to 35.5 (small) and from 33.0 to 31.5 (medium). For the Performer, it reduces perplexity from 39.6 to 37.2 (small) and from 33.8 to 31.8 (medium). Second, the Performer is slightly worse than the Linear Transformer (ELU+1) in both configurations (39.6 vs 38.3 in small, 33.8 vs 33.0 in medium), suggesting that for language modelling in the overcapacity regime, the simplicity of ELU+1 may outweigh the theoretical elegance of softmax approximation.

### 8.5 The capacity-approximation tradeoff

The experimental evidence reveals a tension:

- **When capacity is the bottleneck** (short sequences, synthetic tasks): increasing $d_\text{dot}$ via DPFP or large $m$ gives the clearest improvements. The feature map matters more than the update rule.

- **When approximation quality matters** (real-world tasks, long sequences): the Performer with large $m$ can match softmax attention on translation. But on language modelling, the simpler ELU+1 kernel is competitive, suggesting that the model learns to work with whatever kernel it is given.

- **The update rule matters independently**: regardless of the feature map, the delta update rule improves over the simple sum update. This confirms the insight from the Gated DeltaNet blog — the feature map ($\phi$) and the update rule (sum vs. delta vs. gated delta) are orthogonal design choices that compound.

---

## 9. Backward Compatibility and Practical Considerations

### 9.1 Performers as drop-in replacements

A distinctive feature of the Performer is **backward compatibility** with pretrained Transformers. Because FAVOR+ approximates the actual softmax kernel, a Performer initialised with a pretrained Transformer's weights produces an output that approximates the original model's output.

Choromanski et al. (2021) demonstrate this (Figure 5 of the Performers paper): transferring weights from a pretrained Transformer into a Performer produces an initial non-zero accuracy (0.07 on LM1B), and after fine-tuning for a fraction of the original training steps, the Performer recovers accuracy close to the original Transformer. This is impossible with ELU+1 or DPFP, because those kernels produce different attention distributions that bear no approximation relationship to softmax.

### 9.2 Complexity comparison

All methods have the same asymptotic complexity: $O(n \cdot d_\text{dot} \cdot d_v)$ per layer, linear in sequence length $n$. The practical differences are in the constant:

| Method | $d_\text{dot}$ | State size per head | Extra cost |
|---|---|---|---|
| ELU+1 | $d_k$ | $d_k \times d_v$ | None |
| DPFP-$\nu$ | $2d_k\nu$ | $2d_k\nu \times d_v$ | ReLU + element-wise products |
| FAVOR+ ($m$) | $m$ | $m \times d_v$ | Random projections ($m \times d_k$ matrix multiply) |

For the small language modelling configuration from the FWP paper ($D = 128$, $H = 8$ heads, $d_k = 16$), the wall-clock speeds are:

- Linear Transformer (sum, no delta): 66K words/sec
- Linear Transformer (delta, no attention norm): 63K words/sec

The delta update rule adds only a 5% overhead. For the DPFP and Performer with larger $d_\text{dot}$, the state is proportionally larger and the per-step compute increases, but the model is still faster than softmax attention for sequences longer than $d_\text{dot}$.

### 9.3 Training without truncating context

Linear attention models can process arbitrarily long sequences because the state size is constant. Schlag et al. (2021) demonstrate this by training a Delta Network on WikiText-103 without truncating the context window (Table 4 of the FWP paper). They carry the fast weight memory from one training segment to the next, while still limiting the backpropagation span.

The results show that the Delta Network achieves validation perplexity 27.8 and test perplexity 29.4 — better than its truncated-context counterpart (29.7 / 31.5) and competitive with a Transformer-XL that uses 6.29M state size (validation 24.6, test 25.5). The Delta Network achieves this with a state size of only 0.13M — 48 times smaller.

---

## Summary

The Why Replace Attention blog established the kernel framework: replace softmax with $\phi(q)^\top \phi(k)$ to get linear attention. This blog asked: what should $\phi$ be? The Fast Weight Programmer perspective (Schlag et al., 2021) reframes the question as memory capacity — the state $S \in \mathbb{R}^{d_\text{dot} \times d_v}$ can store at most $d_\text{dot}$ orthogonal key-value associations, and the feature map determines $d_\text{dot}$, so choosing $\phi$ is choosing a point on the capacity-efficiency curve. The Performer perspective (Choromanski et al., 2021) reframes the question as softmax approximation — if $\phi$ can approximate $\exp(q^\top k)$ well enough, we inherit the quality of softmax attention at linear cost. FAVOR+ achieves this with positive orthogonal random features that are provably unbiased and lower-variance than trigonometric alternatives. DPFP achieves increased capacity deterministically through ReLU-gated quadrant projections. The experimental evidence shows that the choice of feature map interacts with but is independent of the update rule: the delta rule (Gated DeltaNet blog) improves all feature maps, and the capacity analysis correctly predicts retrieval failures across all kernels — confirming that linear attention is, at its core, a fast weight programmer writing to a finite-capacity associative memory.

---

*Previous: [Targeted Memory: The Delta Rule, Gated DeltaNet, and Kimi Delta Attention](/blog/attention-gated-deltanet)*

*Next: [Mamba and Mamba-2: Selective State Spaces and Structured State Space Duality](/blog/attention-mamba)*
