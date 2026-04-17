---
title: "Mathematical Prerequisites for Linear Attention"
description: "Building the foundations for linear attention — matrix multiplication associativity as the source of the O(nd) speedup, the outer product as the building block of the recurrent state, associative memory with its orthogonality-based capacity bound, and the kernel feature map that replaces softmax — all derived step by step with one consistent two-token example."
date: 2026-04-08
tags: [machine-learning, attention, transformers, linear-attention, kernels, associative-memory, mathematics]
order: 4
---

The next two posts in this series are [Why Replace Attention](/blog/attention-why-replace) and [The Kernel Zoo](/blog/attention-linear-attention). Both rest on one algebraic trick: the expression $(QK^\top)V$ can be reassociated as $Q(K^\top V)$, turning an $O(n^2 d)$ computation into an $O(n d^2)$ one. That single reassociation is the entire reason linear attention exists.

But the trick only works once we strip the softmax, and once we do, a state matrix appears that is a *sum of outer products* — an **associative memory** with a hard capacity limit determined by dimension. To understand both posts, we need four tools: **matrix multiplication associativity**, the **outer product**, the **capacity bound** of an associative memory, and the **kernel feature map** that replaces softmax.

By the end of this post, every boxed result in the two follow-ups — from the cost comparison that motivates linear attention to the capacity analysis that limits it — will read as an application of one of these four tools.

Matrix multiplication, the dot product, and the independence of coordinate axes were established in [Mathematical Prerequisites for the Attention Series](/blog/math-prerequisites-for-attention). We will use them here without re-deriving them.

---

## The Running Example

Two tokens, each with a 2-dimensional key, query, and value. We stack them into matrices:

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \qquad
K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \qquad
V = \begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix}
$$

Row $i$ of $Q$ is the query for token $i$, row $i$ of $K$ is the key, row $i$ of $V$ is the value. The keys are the two coordinate axes of $\mathbb{R}^2$ — they are orthonormal. We will use this single setup for every derivation: matrix associativity, outer products, capacity checks, and kernel replacements.

---

## 1. Matrix Multiplication Associativity

### 1.1 Motivation

Standard attention without softmax computes

$$
O = (QK^\top) V
$$

where $Q, K \in \mathbb{R}^{n \times d_k}$ and $V \in \mathbb{R}^{n \times d_v}$. The intermediate $QK^\top$ has shape $n \times n$ — quadratic in the sequence length. The linear-attention papers rearrange this to $Q(K^\top V)$, where the intermediate $K^\top V$ has shape $d_k \times d_v$ — no $n$ at all. For this to be mathematically valid, the two parenthesizations must give the same answer. That is the content of **associativity**.

### 1.2 Formal statement

For any three matrices $A \in \mathbb{R}^{m \times p}$, $B \in \mathbb{R}^{p \times q}$, $C \in \mathbb{R}^{q \times r}$, matrix multiplication is **associative**:

$$
\boxed{(AB)C = A(BC)}
$$

Both sides produce the same $m \times r$ matrix. The only requirement is that the inner dimensions match — $A$'s column count equals $B$'s row count, and $B$'s column count equals $C$'s row count.

### 1.3 Derivation

Let $L = (AB)C$ and $R = A(BC)$. We show $L_{ij} = R_{ij}$ for every $(i,j)$ by writing each side as a triple sum and using the fact that addition and multiplication of scalars commute.

Entry $(i,j)$ of the product $AB$ is $\sum_{\ell=1}^{p} A_{i\ell} B_{\ell j}$ (the definition of matmul). So

$$
L_{ij} = \sum_{k=1}^{q} (AB)_{ik} \cdot C_{kj} = \sum_{k=1}^{q} \left( \sum_{\ell=1}^{p} A_{i\ell} B_{\ell k} \right) C_{kj} = \sum_{k=1}^{q} \sum_{\ell=1}^{p} A_{i\ell} B_{\ell k} C_{kj}
$$

The last equality distributes $C_{kj}$ through the inner sum — this is plain scalar distributivity.

Now expand $R_{ij}$ the other way:

$$
R_{ij} = \sum_{\ell=1}^{p} A_{i\ell} \cdot (BC)_{\ell j} = \sum_{\ell=1}^{p} A_{i\ell} \left( \sum_{k=1}^{q} B_{\ell k} C_{kj} \right) = \sum_{\ell=1}^{p} \sum_{k=1}^{q} A_{i\ell} B_{\ell k} C_{kj}
$$

The two expressions for $L_{ij}$ and $R_{ij}$ are sums over the same set of triples $(A_{i\ell}, B_{\ell k}, C_{kj})$, just in different order. Finite sums can be reordered freely, so $L_{ij} = R_{ij}$ for every $(i,j)$. That is associativity.

### 1.4 Numerical check with the running example

Using the running matrices, with the softmax dropped:

$$
QK^\top = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

$$
(QK^\top) V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix} = \begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix}
$$

Now the other order:

$$
K^\top V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix} = \begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix}
$$

$$
Q(K^\top V) = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix} = \begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix}
$$

Both parenthesizations give $\begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix}$. $\checkmark$

### 1.5 The cost difference

Associativity says the two sides are *mathematically* equal. They are not *computationally* equal. Using the FLOP-count formula $2mkn$ from [Mathematical Prerequisites for the Cost of Vanilla Attention](/blog/math-prerequisites-for-vanilla-attention-cost):

- $(QK^\top) V$: first multiply costs $2 n d_k n = 2 d_k n^2$; second costs $2 n n d_v = 2 d_v n^2$. Total: $O(n^2 d)$.
- $Q(K^\top V)$: first multiply costs $2 d_k n d_v$; second costs $2 n d_k d_v$. Total: $O(n d^2)$.

For $n \gg d$, the right parenthesization wins. This is the entire cost argument in [Why Replace Attention](/blog/attention-why-replace). Associativity gives us the freedom; FLOP counting tells us which choice saves.

### 1.6 What associativity does NOT give us

Associativity holds unconditionally. But standard attention has a softmax sandwiched between $QK^\top$ and $V$:

$$
O = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The softmax is applied row-wise and is nonlinear. So the quantity we care about is not $(QK^\top) V$ but $\text{softmax}(QK^\top) V$, and there is no associativity through a nonlinearity. **This is why linear attention has to drop or replace the softmax** — section 4 returns to this. Associativity is necessary; removing the nonlinear obstruction to it is what the kernel feature map does.

---

## 2. The Outer Product

### 2.1 Motivation

When we reassociate to $K^\top V$, what we get is a small matrix — shape $d_k \times d_v$. Where does it come from structurally? The answer is a sum of **outer products**, one per token. This view is the bridge between linear attention and the 1990s **Fast Weight Programmer** idea that [The Kernel Zoo](/blog/attention-linear-attention) builds on.

### 2.2 Definition

For column vectors $u \in \mathbb{R}^{m}$ and $v \in \mathbb{R}^{n}$, the **outer product** $u v^\top$ is the $m \times n$ matrix

$$
(u v^\top)_{ij} = u_i v_j
$$

Equivalently, $u v^\top$ is the matrix whose $(i,j)$ entry is the product of the $i$-th entry of $u$ and the $j$-th entry of $v$. The result has **rank 1**: every row is a scalar multiple of $v^\top$, and every column is a scalar multiple of $u$.

### 2.3 Derivation: $K^\top V$ as a sum of outer products

Write $K \in \mathbb{R}^{n \times d_k}$ as a matrix whose $i$-th row is $k_i^\top$ (so $k_i \in \mathbb{R}^{d_k}$ is a column vector), and $V$ similarly with rows $v_i^\top$ (so $v_i \in \mathbb{R}^{d_v}$). Then $K^\top$ has columns $k_1, k_2, \ldots, k_n$ and $V$ has rows $v_1^\top, v_2^\top, \ldots, v_n^\top$.

The matrix product $K^\top V$ expands column-by-row:

$$
K^\top V = \sum_{i=1}^{n} k_i \cdot v_i^\top
$$

This is a standard identity (the **column-row expansion of a matrix product**) and follows from the matmul definition: entry $(p, q)$ of $K^\top V$ is $\sum_{i=1}^{n} (K^\top)_{pi} V_{iq} = \sum_{i=1}^{n} K_{ip} V_{iq} = \sum_{i=1}^{n} (k_i)_p (v_i)_q$, which is exactly $\sum_i (k_i v_i^\top)_{pq}$. Both sides agree entry-wise, so:

$$
\boxed{K^\top V = \sum_{i=1}^{n} k_i v_i^\top \in \mathbb{R}^{d_k \times d_v}}
$$

The linear-attention state is **a sum of $n$ rank-1 matrices**.

### 2.4 Numerical check with the running example

With $k_1 = (1, 0)^\top$, $k_2 = (0, 1)^\top$, $v_1 = (2, 3)^\top$, $v_2 = (5, 7)^\top$:

$$
k_1 v_1^\top = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 2 & 3 \end{pmatrix} = \begin{pmatrix} 2 & 3 \\ 0 & 0 \end{pmatrix}
$$

$$
k_2 v_2^\top = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \begin{pmatrix} 5 & 7 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 5 & 7 \end{pmatrix}
$$

Summing:

$$
k_1 v_1^\top + k_2 v_2^\top = \begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix}
$$

This matches $K^\top V$ from Section 1.4 exactly. $\checkmark$

### 2.5 Interpretation

The outer product view says: each token $i$ contributes one rank-1 matrix $k_i v_i^\top$ to the shared state. When we query with $q$, we compute $(\sum_i k_i v_i^\top)^\top q$ or equivalently $\sum_i v_i (k_i^\top q)$ — a weighted sum of values, with weights $k_i^\top q$. That is linear attention written as a read from a memory built by writing rank-1 updates. The recurrence follows immediately: if $S_t = \sum_{i \leq t} k_i v_i^\top$, then $S_t = S_{t-1} + k_t v_t^\top$.

---

## 3. Associative Memory and the Capacity Bound

### 3.1 Motivation

Sections 1 and 2 show that linear attention's state $S = \sum_i k_i v_i^\top$ is an **associative memory**: writing token $i$ means adding $k_i v_i^\top$, reading with query $q$ means computing $S^\top q$. A natural question — and the central question of [The Kernel Zoo](/blog/attention-linear-attention) — is: how many key–value pairs can we stuff into $S$ before retrieval starts returning the wrong thing?

### 3.2 What perfect retrieval looks like

We write the state after storing $t$ pairs:

$$
S_t = \sum_{i=1}^{t} k_i v_i^\top
$$

To read the value associated with a particular stored key $k_j$, we compute $S_t^\top k_j$. Expanding:

$$
S_t^\top k_j = \sum_{i=1}^{t} v_i (k_i^\top k_j)
$$

For this to return $v_j$ exactly, we would need $k_i^\top k_j = 0$ for $i \neq j$ and $k_j^\top k_j = 1$. In other words, the keys must be **orthonormal**.

### 3.3 The capacity bound

How many orthonormal vectors fit in $\mathbb{R}^{d_k}$? At most $d_k$ — this is a standard fact of linear algebra (any orthonormal set is linearly independent, and a set of linearly independent vectors in $\mathbb{R}^{d_k}$ has at most $d_k$ elements). So:

$$
\boxed{\text{capacity of an associative memory with key dimension } d_k = d_k \text{ pairs}}
$$

Storing more than $d_k$ pairs forces some keys to have nonzero inner products with each other, and the retrieval of $v_j$ becomes contaminated by other $v_i$'s. This is the **overcapacity regime** that [The Kernel Zoo](/blog/attention-linear-attention) analyzes.

### 3.4 Numerical check with the running example (at capacity)

Our running example has $d_k = 2$ and $t = 2$ — exactly at capacity. The keys $k_1 = (1,0)$ and $k_2 = (0,1)$ are orthonormal. Retrieve with $k_1$:

$$
S_2^\top k_1 = \sum_{i=1}^{2} v_i (k_i^\top k_1) = v_1 \cdot (k_1^\top k_1) + v_2 \cdot (k_2^\top k_1) = v_1 \cdot 1 + v_2 \cdot 0 = v_1 = \begin{pmatrix} 2 \\ 3 \end{pmatrix}
$$

Perfect retrieval. Retrieving with $k_2$ gives $v_2 = (5, 7)$ by the same argument. $\checkmark$

### 3.5 Numerical check with the running example (over capacity)

Now push to $t = 3$ with an extra key that cannot be orthogonal to both of the first two. Take $k_3 = \frac{1}{\sqrt{2}}(1, 1)^\top$ and $v_3 = (1, 1)^\top$. Then $k_3^\top k_1 = 1/\sqrt{2}$ and $k_3^\top k_2 = 1/\sqrt{2}$ — nonzero inner products with both stored keys.

Update: $S_3 = S_2 + k_3 v_3^\top$. Retrieve with $k_1$:

$$
S_3^\top k_1 = v_1 (k_1^\top k_1) + v_2 (k_2^\top k_1) + v_3 (k_3^\top k_1) = v_1 \cdot 1 + v_2 \cdot 0 + v_3 \cdot \tfrac{1}{\sqrt{2}}
$$

$$
= \begin{pmatrix} 2 \\ 3 \end{pmatrix} + \tfrac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix} \approx \begin{pmatrix} 2.707 \\ 3.707 \end{pmatrix}
$$

The correct answer was $v_1 = (2, 3)$; we got $(2.707, 3.707)$. The third stored pair leaks into the retrieval with weight $k_3^\top k_1 = 1/\sqrt{2}$. **Interference**. $\checkmark$ — this is exactly the overcapacity failure mode the follow-up post names.

### 3.6 Interpretation

The capacity bound is structural: it is not a property of which keys you chose, but of the ambient space. The only way to raise the capacity is to raise the key dimension — which increases the state size $d_k \times d_v$ and per-step compute. Every feature-map choice in [The Kernel Zoo](/blog/attention-linear-attention) — from ELU to DPFP to FAVOR+ — is a way of mapping the $d_k$-dimensional keys into a higher-dimensional space where more orthogonal directions exist. Capacity and kernel design are the same problem.

---

## 4. The Kernel Feature Map

### 4.1 Motivation

Section 1 noted that reassociation only works if we remove the softmax. But softmax is what makes standard attention expressive: the similarity $\exp(q^\top k / \sqrt{d_k})$ sharply peaks when keys align with queries. Dropping it is too blunt. Linear attention replaces softmax with a **kernel** — a fixed nonlinear feature map applied before the reassociation, so that the structure is restored while the $(QK^\top)V = Q(K^\top V)$ trick still applies.

### 4.2 What a kernel is (in one sentence)

A **kernel** is a function $K(q, k)$ of two inputs that behaves like a similarity score. A kernel is called **linearizable** if there exists a feature map $\phi : \mathbb{R}^{d} \to \mathbb{R}^{d_\phi}$ such that

$$
\boxed{K(q, k) = \phi(q)^\top \phi(k)}
$$

This is the **Mercer factorization** of the kernel: the similarity between $q$ and $k$ is a plain dot product, provided we first apply $\phi$ to both. The feature map can map into a higher-dimensional space than the inputs live in.

### 4.3 Replacing softmax with a kernel

Vanilla attention for query $q_i$ writes

$$
o_i = \sum_{j=1}^{n} \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i^\top k_{j'} / \sqrt{d_k})} v_j
$$

Replace the exponential kernel by $\phi(q)^\top \phi(k)$ for some feature map $\phi$:

$$
o_i = \sum_{j=1}^{n} \frac{\phi(q_i)^\top \phi(k_j)}{\sum_{j'} \phi(q_i)^\top \phi(k_{j'})} v_j = \frac{\phi(q_i)^\top \sum_{j} \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_{j} \phi(k_j)}
$$

The last equality used linearity to pull $\phi(q_i)^\top$ out of both the numerator sum and the denominator sum. The numerator has exactly the shape of Section 2: a sum of outer products $\phi(k_j) v_j^\top$, which is a $d_\phi \times d_v$ state matrix $S$. The denominator is a sum of vectors $\phi(k_j)$, which is a $d_\phi$-vector $z$.

$$
\boxed{o_i = \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}, \qquad S = \sum_j \phi(k_j) v_j^\top, \qquad z = \sum_j \phi(k_j)}
$$

Both $S$ and $z$ are state variables that can be updated incrementally as a new token arrives — that is the $O(1)$-per-token inference of linear attention.

### 4.4 Numerical check with the identity kernel

The simplest feature map is $\phi(x) = x$ (the identity). With $\phi$ identity, $S = K^\top V$ and $z = K^\top \mathbf{1}$. Using the running example, $S = \begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix}$ (from Section 2.4) and

$$
z = k_1 + k_2 = \begin{pmatrix} 1 \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

For query $q_1 = (1, 0)^\top$:

$$
\phi(q_1)^\top S = (1, 0)\begin{pmatrix} 2 & 3 \\ 5 & 7 \end{pmatrix} = (2, 3), \qquad \phi(q_1)^\top z = (1, 0)\begin{pmatrix} 1 \\ 1 \end{pmatrix} = 1
$$

$$
o_1 = \frac{(2, 3)}{1} = (2, 3)
$$

Since $k_1$ is exactly aligned with $q_1$ and the keys are orthonormal, we recover $v_1 = (2, 3)$ exactly. This is the same answer Section 3.4 gave — the associative-memory retrieval and the kernel-attention view are the same computation. $\checkmark$

### 4.5 Softmax vs identity kernel: same formula, different kernel

This is the cleanest distinction in the post. **Softmax attention** uses the exponential kernel $K_\exp(q, k) = \exp(q^\top k / \sqrt{d_k})$. **Linear attention** with feature map $\phi$ uses the kernel $K_\phi(q, k) = \phi(q)^\top \phi(k)$. Same functional form $\sum_j K(q_i, k_j) v_j / \sum_j K(q_i, k_j)$. Different kernel $K$.

The exponential kernel is *not* linearizable with a finite-dimensional $\phi$: every finite feature map gives an approximation, not an equality. [The Kernel Zoo](/blog/attention-linear-attention) catalogs specific choices — ELU+1, DPFP, FAVOR+ — each a different approximation with different capacity and variance properties. The choice of $\phi$ is the design variable, but the skeleton formula is the one boxed above.

### 4.6 Interpretation

Kernelization is what lets the associativity trick of Section 1 apply without killing expressivity entirely. The feature map $\phi$ is the lever: it controls the dimension $d_\phi$ (which Section 3 linked to capacity), the shape of the similarity function (which controls what "attending to" a key means), and the faithfulness to softmax (which controls how closely the linear model mimics the expensive one). Every paper in the linear-attention literature is a choice of $\phi$.

---

## Summary

We built four tools in sequence. **Matrix associativity** $(AB)C = A(BC)$ is the reason $QK^\top V$ can be computed two ways with the same answer but different cost — $O(n^2 d)$ one way, $O(n d^2)$ the other. The **outer product** $u v^\top$ is the rank-1 building block of the linear-attention state: $K^\top V = \sum_i k_i v_i^\top$, and the recurrence $S_t = S_{t-1} + k_t v_t^\top$ falls out for free. The **capacity bound** says this outer-product memory can perfectly recall at most $d_k$ orthonormal key–value pairs; beyond that, retrievals interfere. The **kernel feature map** $\phi$ replaces softmax with a linearizable similarity $K(q, k) = \phi(q)^\top \phi(k)$, restoring the associativity path and turning the design of linear attention into the design of $\phi$.

With these tools in hand, we are ready for [Why Replace Attention](/blog/attention-why-replace), where associativity is wielded to derive linear-time attention, and [The Kernel Zoo](/blog/attention-linear-attention), where capacity and kernel design are traded against each other to explain why different feature maps exist in the first place.
