---
title: "Mathematical Prerequisites for DeepSeek-V2"
description: "Building the linear algebra foundations for Multi-head Latent Attention — matrix-vector multiplication, transpose, associativity, non-commutativity, the bilinear identity, linearity over sums, low-rank factorization, and the dot product of concatenated vectors — all derived step by step with one consistent 3-dimensional example."
date: 2026-04-05
tags: [machine-learning, linear-algebra, mathematics, attention, deepseek]
order: 3
---

The [DeepSeek-V2 blog](/blog/deepseek-v2-mla-and-deepseekmoe) derives Multi-head Latent Attention (MLA), an attention mechanism that compresses keys and values into a shared low-rank latent vector and then recovers them through up-projection matrices. The central computational trick — matrix absorption — rewrites the attention formula so that keys and values are never explicitly computed for cached tokens. Every step of that trick relies on basic linear algebra identities: the transpose, associativity, non-commutativity, and linearity of matrix multiplication.

These identities were not needed in any previous blog, so none of our earlier prerequisite posts derived them. This post builds every one from scratch. By the end, you will have all the tools required to follow the matrix absorption derivation and the low-rank compression argument in the DeepSeek-V2 blog.

The dot product was fully derived in [Mathematical Prerequisites for the Attention Series](/blog/math-prerequisites-for-attention). We will use it without re-deriving it.

---

## The Running Example

We work with a 3-dimensional vector and a $2 \times 3$ matrix throughout. These dimensions are small enough to compute by hand but large enough to show the structure of every operation.

The vector:

$$
\mathbf{x} = \begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}
$$

The matrix:

$$
A = \begin{pmatrix} 1 & 0 & -2 \\ 3 & 1 & 4 \end{pmatrix}
$$

In the DeepSeek-V2 blog, $\mathbf{x}$ plays the role of the hidden state $\mathbf{h}_t$ (512-dimensional in practice), and $A$ plays the role of a projection matrix like $W^{DKV}$ that compresses the hidden state into a smaller latent vector. Here, $A$ compresses from 3 dimensions to 2 — in the blog, it compresses from 512 to 128.

---

## 1. Matrix-Vector Multiplication

In MLA, every projection — from hidden state to latent vector, from latent vector to keys, from latent vector to values — is a matrix-vector multiplication. We need to know exactly what this operation does and what dimensions it produces.

**Matrix-vector multiplication** takes a matrix $A \in \mathbb{R}^{m \times n}$ and a vector $\mathbf{x} \in \mathbb{R}^n$ and produces a new vector $\mathbf{y} = A\mathbf{x} \in \mathbb{R}^m$. Each entry of $\mathbf{y}$ is the dot product of one row of $A$ with $\mathbf{x}$:

$$
\boxed{y_i = \sum_{j=1}^{n} A_{ij} \, x_j}
$$

The matrix has $m$ rows and $n$ columns. The vector has $n$ entries. The number of columns of the matrix must equal the number of entries in the vector — this is the **dimension compatibility rule**. The result has $m$ entries, one per row of the matrix.

### Numerical check

$A$ is $2 \times 3$ and $\mathbf{x}$ is $3 \times 1$. The inner dimensions match (both 3), so the product is defined and produces a $2 \times 1$ vector.

Row 1 of $A$ is $(1, 0, -2)$. Its dot product with $\mathbf{x} = (2, -1, 3)$ is:

$$
y_1 = 1 \cdot 2 + 0 \cdot (-1) + (-2) \cdot 3 = 2 + 0 - 6 = -4
$$

Row 2 of $A$ is $(3, 1, 4)$:

$$
y_2 = 3 \cdot 2 + 1 \cdot (-1) + 4 \cdot 3 = 6 - 1 + 12 = 17
$$

So:

$$
A\mathbf{x} = \begin{pmatrix} 1 & 0 & -2 \\ 3 & 1 & 4 \end{pmatrix} \begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix} = \begin{pmatrix} -4 \\ 17 \end{pmatrix}
$$

We went from a 3-dimensional vector to a 2-dimensional vector. In MLA, this is exactly what the down-projection $W^{DKV}$ does: it takes the 512-dimensional hidden state and produces a 128-dimensional latent vector.

---

## 2. Matrix Transpose

The transpose appears constantly in the DeepSeek-V2 derivations — in attention scores ($\mathbf{q}^T \mathbf{k}$), in the matrix absorption trick ($(W^{UK})^T$), and in the key identity that makes absorption work.

The **transpose** of a matrix $A \in \mathbb{R}^{m \times n}$ is the matrix $A^T \in \mathbb{R}^{n \times m}$ obtained by swapping rows and columns:

$$
\boxed{(A^T)_{ij} = A_{ji}}
$$

Row $i$ of $A^T$ is column $i$ of $A$. Column $j$ of $A^T$ is row $j$ of $A$. The shape flips from $m \times n$ to $n \times m$.

### Numerical check

$$
A = \begin{pmatrix} 1 & 0 & -2 \\ 3 & 1 & 4 \end{pmatrix} \quad \Longrightarrow \quad A^T = \begin{pmatrix} 1 & 3 \\ 0 & 1 \\ -2 & 4 \end{pmatrix}
$$

$A$ is $2 \times 3$. $A^T$ is $3 \times 2$. Entry $(1,2)$ of $A^T$ is $3$, which is entry $(2,1)$ of $A$. Entry $(3,1)$ of $A^T$ is $-2$, which is entry $(1,3)$ of $A$. Every entry checks out.

### Transpose of a vector

A column vector $\mathbf{x} \in \mathbb{R}^n$ is a $n \times 1$ matrix. Its transpose $\mathbf{x}^T$ is a $1 \times n$ row vector. This is why the attention score $\mathbf{q}^T \mathbf{k}$ is a scalar: $\mathbf{q}^T$ is $1 \times d$ and $\mathbf{k}$ is $d \times 1$, so the product is $1 \times 1$.

### The dot product as a matrix product

In the attention blogs, we write the dot product as $\mathbf{q}^T \mathbf{k}$. This is exactly the matrix product of a $1 \times d$ row vector with a $d \times 1$ column vector:

$$
\mathbf{q}^T \mathbf{k} = \sum_{i=1}^{d} q_i k_i
$$

This is the same dot product formula from the [attention prerequisites](/blog/math-prerequisites-for-attention), but written in matrix notation. The two notations are interchangeable.

---

## 3. Matrix-Matrix Multiplication

Before we can derive associativity, we need the general rule for multiplying two matrices.

Given $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, their product $C = AB \in \mathbb{R}^{m \times p}$ has entries:

$$
\boxed{C_{ij} = \sum_{k=1}^{n} A_{ik} \, B_{kj}}
$$

Each entry $C_{ij}$ is the dot product of row $i$ of $A$ with column $j$ of $B$. The inner dimensions must match: $A$ has $n$ columns, $B$ has $n$ rows.

### Numerical check

We need a second matrix to multiply with $A^T$. Let us define:

$$
B = \begin{pmatrix} 5 & -1 \\ 2 & 3 \end{pmatrix}
$$

Then $A^T B$ has $A^T \in \mathbb{R}^{3 \times 2}$ and $B \in \mathbb{R}^{2 \times 2}$, so $A^T B \in \mathbb{R}^{3 \times 2}$.

Entry $(1,1)$: row 1 of $A^T$ is $(1, 3)$, column 1 of $B$ is $(5, 2)$. Dot product: $1 \cdot 5 + 3 \cdot 2 = 5 + 6 = 11$.

Entry $(1,2)$: row 1 of $A^T$ is $(1, 3)$, column 2 of $B$ is $(-1, 3)$. Dot product: $1 \cdot (-1) + 3 \cdot 3 = -1 + 9 = 8$.

Entry $(2,1)$: row 2 of $A^T$ is $(0, 1)$, column 1 of $B$ is $(5, 2)$. Dot product: $0 \cdot 5 + 1 \cdot 2 = 2$.

Entry $(2,2)$: row 2 of $A^T$ is $(0, 1)$, column 2 of $B$ is $(-1, 3)$. Dot product: $0 \cdot (-1) + 1 \cdot 3 = 3$.

Entry $(3,1)$: row 3 of $A^T$ is $(-2, 4)$, column 1 of $B$ is $(5, 2)$. Dot product: $(-2) \cdot 5 + 4 \cdot 2 = -10 + 8 = -2$.

Entry $(3,2)$: row 3 of $A^T$ is $(-2, 4)$, column 2 of $B$ is $(-1, 3)$. Dot product: $(-2) \cdot (-1) + 4 \cdot 3 = 2 + 12 = 14$.

$$
A^T B = \begin{pmatrix} 11 & 8 \\ 2 & 3 \\ -2 & 14 \end{pmatrix}
$$

This is $3 \times 2$, matching our dimension prediction. We will use this result when we verify associativity below.

---

## 4. Associativity of Matrix Multiplication

This is the property that makes the matrix absorption trick in MLA work. In the DeepSeek-V2 blog, we need to rewrite $(W_i^{UK})^T (W_i^{UQ} \mathbf{c}_t^Q)$ as $((W_i^{UK})^T W_i^{UQ}) \mathbf{c}_t^Q$. This is only valid if matrix multiplication is associative.

**Associativity** states that for any matrices $A$, $B$, $C$ of compatible dimensions:

$$
\boxed{(AB)C = A(BC)}
$$

We can group the multiplication in either order and get the same result. This means we can precompute $AB$ as a single matrix and then multiply by $C$, or we can first compute $BC$ and then multiply by $A$. The answer is identical.

### Proof

Let $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$, $C \in \mathbb{R}^{p \times q}$. Both $(AB)C$ and $A(BC)$ are in $\mathbb{R}^{m \times q}$. We show that entry $(i, j)$ is the same on both sides.

**Left side.** $(AB)C$ means: first compute $D = AB$, where $D_{ik} = \sum_{l=1}^{n} A_{il} B_{lk}$. Then compute $(DC)_{ij} = \sum_{k=1}^{p} D_{ik} C_{kj}$. Substituting:

$$
((AB)C)_{ij} = \sum_{k=1}^{p} \left( \sum_{l=1}^{n} A_{il} B_{lk} \right) C_{kj}
$$

We distribute $C_{kj}$ into the inner sum:

$$
= \sum_{k=1}^{p} \sum_{l=1}^{n} A_{il} B_{lk} C_{kj}
$$

**Right side.** $A(BC)$ means: first compute $E = BC$, where $E_{lj} = \sum_{k=1}^{p} B_{lk} C_{kj}$. Then compute $(AE)_{ij} = \sum_{l=1}^{n} A_{il} E_{lj}$. Substituting:

$$
(A(BC))_{ij} = \sum_{l=1}^{n} A_{il} \left( \sum_{k=1}^{p} B_{lk} C_{kj} \right) = \sum_{l=1}^{n} \sum_{k=1}^{p} A_{il} B_{lk} C_{kj}
$$

Both sides equal $\sum_{k=1}^{p} \sum_{l=1}^{n} A_{il} B_{lk} C_{kj}$. The double sum is a finite sum of real numbers, and the order of summation does not matter (this is **commutativity of addition**). The two expressions are identical. $\square$

### Numerical check

We verify $(A^T B) \mathbf{x}  = A^T (B \mathbf{x})$ using our running example. Here $A^T \in \mathbb{R}^{3 \times 2}$, $B \in \mathbb{R}^{2 \times 2}$, and we need a 2-dimensional vector. Let us define:

$$
\mathbf{z} = \begin{pmatrix} 4 \\ -3 \end{pmatrix}
$$

**Left side: $(A^T B) \mathbf{z}$.** We already computed $A^T B = \begin{pmatrix} 11 & 8 \\ 2 & 3 \\ -2 & 14 \end{pmatrix}$. Multiply by $\mathbf{z}$:

$$
(A^T B) \mathbf{z} = \begin{pmatrix} 11 \cdot 4 + 8 \cdot (-3) \\ 2 \cdot 4 + 3 \cdot (-3) \\ (-2) \cdot 4 + 14 \cdot (-3) \end{pmatrix} = \begin{pmatrix} 44 - 24 \\ 8 - 9 \\ -8 - 42 \end{pmatrix} = \begin{pmatrix} 20 \\ -1 \\ -50 \end{pmatrix}
$$

**Right side: $A^T (B \mathbf{z})$.** First compute $B\mathbf{z}$:

$$
B \mathbf{z} = \begin{pmatrix} 5 & -1 \\ 2 & 3 \end{pmatrix} \begin{pmatrix} 4 \\ -3 \end{pmatrix} = \begin{pmatrix} 5 \cdot 4 + (-1)(-3) \\ 2 \cdot 4 + 3 \cdot (-3) \end{pmatrix} = \begin{pmatrix} 20 + 3 \\ 8 - 9 \end{pmatrix} = \begin{pmatrix} 23 \\ -1 \end{pmatrix}
$$

Then multiply by $A^T$:

$$
A^T (B\mathbf{z}) = \begin{pmatrix} 1 & 3 \\ 0 & 1 \\ -2 & 4 \end{pmatrix} \begin{pmatrix} 23 \\ -1 \end{pmatrix} = \begin{pmatrix} 1 \cdot 23 + 3 \cdot (-1) \\ 0 \cdot 23 + 1 \cdot (-1) \\ (-2) \cdot 23 + 4 \cdot (-1) \end{pmatrix} = \begin{pmatrix} 20 \\ -1 \\ -50 \end{pmatrix}
$$

Both sides give $(20, -1, -50)$. Associativity holds.

In MLA, this is exactly how the matrix absorption trick works. The attention score involves $(W_i^{UK})^T$ applied to the query, then dotted with the cached latent. By associativity, we can precompute $\tilde{W}_i^Q = (W_i^{UK})^T W_i^{UQ}$ once, and then apply the combined matrix to each query — avoiding the per-token up-projection entirely.

---

## 5. Non-Commutativity of Matrix Multiplication

In the DeepSeek-V2 blog, the RoPE incompatibility problem rests on one fact: matrix multiplication does not commute. We cannot swap the order of multiplication and expect the same result. This is why RoPE (a rotation matrix applied after the up-projection) cannot be "pushed inside" the up-projection.

**Non-commutativity** means that in general, $AB \neq BA$. This is true even when both products are defined (i.e., when $A$ and $B$ are both square matrices of the same size).

### Numerical check

We use two $2 \times 2$ matrices to demonstrate. Let:

$$
P = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}, \qquad R = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
$$

$R$ is a 90-degree rotation matrix (exactly the kind of operation RoPE performs). Let us compute both $PR$ and $RP$.

$$
PR = \begin{pmatrix} 1 \cdot 0 + 2 \cdot 1 & 1 \cdot (-1) + 2 \cdot 0 \\ 0 \cdot 0 + 1 \cdot 1 & 0 \cdot (-1) + 1 \cdot 0 \end{pmatrix} = \begin{pmatrix} 2 & -1 \\ 1 & 0 \end{pmatrix}
$$

$$
RP = \begin{pmatrix} 0 \cdot 1 + (-1) \cdot 0 & 0 \cdot 2 + (-1) \cdot 1 \\ 1 \cdot 1 + 0 \cdot 0 & 1 \cdot 2 + 0 \cdot 1 \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 2 \end{pmatrix}
$$

$PR \neq RP$. The $(1,1)$ entry is 2 versus 0. The $(2,2)$ entry is 0 versus 2. The order matters.

This is precisely the problem with RoPE and MLA. The up-projection $W^{UK}$ and the RoPE rotation $R_t$ (which depends on position $t$) do not commute. So $R_t(W^{UK} \mathbf{c}_t) \neq W^{UK}(R_t \mathbf{c}_t)$ in general. We cannot apply RoPE to the cached latent vector and then up-project — we would get the wrong answer. That is why DeepSeek-V2 decouples RoPE into a separate set of queries and keys.

---

## 6. The Bilinear Identity (Transpose Trick)

This is the identity that enables the key absorption in Section 5.1 of the DeepSeek-V2 blog. The blog rewrites $(\mathbf{q}^C_{t,i})^T W_i^{UK} \mathbf{c}_j^{KV}$ as $((W_i^{UK})^T \mathbf{q}^C_{t,i})^T \mathbf{c}_j^{KV}$. That is this identity.

For any vector $\mathbf{a} \in \mathbb{R}^m$, matrix $M \in \mathbb{R}^{m \times n}$, and vector $\mathbf{b} \in \mathbb{R}^n$:

$$
\boxed{\mathbf{a}^T M \mathbf{b} = (M^T \mathbf{a})^T \mathbf{b}}
$$

### Derivation

Start with the left side. $\mathbf{a}^T M \mathbf{b}$ is a scalar (a $1 \times 1$ matrix). The transpose of a scalar is itself:

$$
\mathbf{a}^T M \mathbf{b} = (\mathbf{a}^T M \mathbf{b})^T
$$

Now we apply the **transpose of a product** rule. For any matrices $X$ and $Y$ of compatible dimensions, $(XY)^T = Y^T X^T$. Let us prove this rule before using it.

### Detour: Transpose of a Product

We claim $(XY)^T = Y^T X^T$ for any $X \in \mathbb{R}^{m \times n}$ and $Y \in \mathbb{R}^{n \times p}$. Both sides are in $\mathbb{R}^{p \times m}$. Entry $(j, i)$ of $(XY)^T$ equals entry $(i, j)$ of $XY$:

$$
((XY)^T)_{ji} = (XY)_{ij} = \sum_{k=1}^{n} X_{ik} Y_{kj}
$$

Entry $(j, i)$ of $Y^T X^T$:

$$
(Y^T X^T)_{ji} = \sum_{k=1}^{n} (Y^T)_{jk} (X^T)_{ki} = \sum_{k=1}^{n} Y_{kj} X_{ik}
$$

Each term in the sum is $X_{ik} Y_{kj}$ (by **commutativity of real number multiplication**). So the two sums are identical. $\square$

### Completing the derivation

We apply the transpose-of-a-product rule to $\mathbf{a}^T M \mathbf{b}$. Grouping as $(\mathbf{a}^T M) \mathbf{b}$, we have two factors: $\mathbf{a}^T M$ (a $1 \times n$ row vector) and $\mathbf{b}$ (an $n \times 1$ column vector). Their product is $1 \times 1$. Taking the transpose:

$$
((\mathbf{a}^T M) \mathbf{b})^T = \mathbf{b}^T (\mathbf{a}^T M)^T
$$

Now apply the transpose-of-a-product rule to $\mathbf{a}^T M$:

$$
(\mathbf{a}^T M)^T = M^T (\mathbf{a}^T)^T = M^T \mathbf{a}
$$

Here we used the fact that $(\mathbf{a}^T)^T = \mathbf{a}$ — transposing twice returns the original.

Substituting back:

$$
\mathbf{a}^T M \mathbf{b} = \mathbf{b}^T (M^T \mathbf{a})
$$

But $\mathbf{b}^T (M^T \mathbf{a})$ is the dot product of $\mathbf{b}$ and $M^T \mathbf{a}$, which equals $(M^T \mathbf{a})^T \mathbf{b}$ (dot products are commutative: $\mathbf{u}^T \mathbf{v} = \mathbf{v}^T \mathbf{u}$). Therefore:

$$
\mathbf{a}^T M \mathbf{b} = (M^T \mathbf{a})^T \mathbf{b} \quad \square
$$

### Interpretation

The left side, $\mathbf{a}^T M \mathbf{b}$, says: "Apply $M$ to $\mathbf{b}$, then dot with $\mathbf{a}$." The right side, $(M^T \mathbf{a})^T \mathbf{b}$, says: "Apply $M^T$ to $\mathbf{a}$, then dot with $\mathbf{b}$." The two give the same scalar. The matrix can be "moved" from one side to the other, at the cost of transposing it.

In MLA, this means the up-projection $W_i^{UK}$ that would normally be applied to every cached key can instead be applied (transposed) to the single current query. We move the work from the $T$-many cached tokens (expensive) to the one current token (cheap).

### Numerical check

Let $\mathbf{a} = A\mathbf{x} = (-4, 17)^T$ (computed in Section 1), $M = B = \begin{pmatrix} 5 & -1 \\ 2 & 3 \end{pmatrix}$, and $\mathbf{b} = \mathbf{z} = (4, -3)^T$.

**Left side: $\mathbf{a}^T M \mathbf{b}$.**

First, $M \mathbf{b} = B \mathbf{z} = (23, -1)^T$ (computed in Section 4).

Then $\mathbf{a}^T (M\mathbf{b}) = (-4) \cdot 23 + 17 \cdot (-1) = -92 - 17 = -109$.

**Right side: $(M^T \mathbf{a})^T \mathbf{b}$.**

$M^T = B^T = \begin{pmatrix} 5 & 2 \\ -1 & 3 \end{pmatrix}$.

$M^T \mathbf{a} = \begin{pmatrix} 5 \cdot (-4) + 2 \cdot 17 \\ (-1)(-4) + 3 \cdot 17 \end{pmatrix} = \begin{pmatrix} -20 + 34 \\ 4 + 51 \end{pmatrix} = \begin{pmatrix} 14 \\ 55 \end{pmatrix}$.

$(M^T \mathbf{a})^T \mathbf{b} = 14 \cdot 4 + 55 \cdot (-3) = 56 - 165 = -109$.

Both sides give $-109$. The identity holds.

---

## 7. Linearity of Matrix Multiplication

In Section 5.2 of the DeepSeek-V2 blog, we pull the value up-projection matrix $W_i^{UV}$ out of a weighted sum: $\sum_j \alpha_j (W_i^{UV} \mathbf{c}_j) = W_i^{UV} \sum_j \alpha_j \mathbf{c}_j$. This is linearity.

**Linearity of matrix multiplication** means that for any matrix $W$, vectors $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$ of compatible dimension, and scalars $\alpha_1, \alpha_2, \ldots, \alpha_T$:

$$
\boxed{W \left( \sum_{j=1}^{T} \alpha_j \mathbf{x}_j \right) = \sum_{j=1}^{T} \alpha_j (W \mathbf{x}_j)}
$$

### Derivation

It suffices to prove two properties and then combine them.

**Distributivity over addition.** For any matrix $W \in \mathbb{R}^{m \times n}$ and vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:

$$
(W(\mathbf{u} + \mathbf{v}))_i = \sum_{k=1}^{n} W_{ik}(u_k + v_k) = \sum_{k=1}^{n} W_{ik} u_k + \sum_{k=1}^{n} W_{ik} v_k = (W\mathbf{u})_i + (W\mathbf{v})_i
$$

The first equality uses the definition of matrix-vector multiplication. The second uses **distributivity of real number multiplication over addition**: $a(b + c) = ab + ac$. Since this holds for every entry $i$, we have $W(\mathbf{u} + \mathbf{v}) = W\mathbf{u} + W\mathbf{v}$.

**Compatibility with scalar multiplication.** For any scalar $\alpha$:

$$
(W(\alpha \mathbf{u}))_i = \sum_{k=1}^{n} W_{ik}(\alpha u_k) = \alpha \sum_{k=1}^{n} W_{ik} u_k = \alpha (W\mathbf{u})_i
$$

The second equality uses the fact that $\alpha$ is a common factor in every term of the sum. So $W(\alpha \mathbf{u}) = \alpha (W \mathbf{u})$.

**Combining.** Apply these two rules repeatedly to the sum $\sum_j \alpha_j \mathbf{x}_j$:

$$
W\!\left(\sum_{j=1}^{T} \alpha_j \mathbf{x}_j\right) = \sum_{j=1}^{T} W(\alpha_j \mathbf{x}_j) = \sum_{j=1}^{T} \alpha_j (W \mathbf{x}_j) \quad \square
$$

The first equality uses distributivity over addition (applied $T - 1$ times). The second uses compatibility with scalar multiplication (applied $T$ times).

### Numerical check

Let $W = A = \begin{pmatrix} 1 & 0 & -2 \\ 3 & 1 & 4 \end{pmatrix}$. Define two 3-dimensional vectors:

$$
\mathbf{x}_1 = \begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}, \qquad \mathbf{x}_2 = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}
$$

with weights $\alpha_1 = 0.6$ and $\alpha_2 = 0.4$.

**Left side: $W(\alpha_1 \mathbf{x}_1 + \alpha_2 \mathbf{x}_2)$.**

First, $\alpha_1 \mathbf{x}_1 + \alpha_2 \mathbf{x}_2 = 0.6 \cdot (2, -1, 3) + 0.4 \cdot (1, 0, -1) = (1.2, -0.6, 1.8) + (0.4, 0, -0.4) = (1.6, -0.6, 1.4)$.

Then $W \cdot (1.6, -0.6, 1.4)^T$:

$$
y_1 = 1 \cdot 1.6 + 0 \cdot (-0.6) + (-2) \cdot 1.4 = 1.6 - 2.8 = -1.2
$$

$$
y_2 = 3 \cdot 1.6 + 1 \cdot (-0.6) + 4 \cdot 1.4 = 4.8 - 0.6 + 5.6 = 9.8
$$

Left side: $(-1.2, 9.8)$.

**Right side: $\alpha_1 (W \mathbf{x}_1) + \alpha_2 (W \mathbf{x}_2)$.**

We know $W \mathbf{x}_1 = (-4, 17)$ from Section 1.

$W \mathbf{x}_2 = \begin{pmatrix} 1 \cdot 1 + 0 \cdot 0 + (-2)(-1) \\ 3 \cdot 1 + 1 \cdot 0 + 4 \cdot (-1) \end{pmatrix} = \begin{pmatrix} 3 \\ -1 \end{pmatrix}$.

$\alpha_1 (W\mathbf{x}_1) + \alpha_2 (W\mathbf{x}_2) = 0.6 \cdot (-4, 17) + 0.4 \cdot (3, -1) = (-2.4, 10.2) + (1.2, -0.4) = (-1.2, 9.8)$.

Both sides give $(-1.2, 9.8)$.

In MLA, this identity lets us compute the weighted sum of cached latent vectors $\sum_j \alpha_j \mathbf{c}_j^{KV}$ first, and then apply the output projection once — instead of applying the up-projection to each of the $T$ cached vectors individually.

---

## 8. Low-Rank Factorization

This is the structural idea behind MLA's KV compression. The term "low-rank" appears in Section 2.3 of the DeepSeek-V2 blog. We need to understand what it means.

A matrix $W \in \mathbb{R}^{m \times n}$ has a **low-rank factorization** if it can be written as:

$$
\boxed{W = U V}
$$

where $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{r \times n}$ with $r < \min(m, n)$. The number $r$ is called the **rank** of the factorization (or an upper bound on the rank of $W$).

### What this means

Without factorization, $W$ has $m \times n$ entries. With the factorization, we store $U$ ($m \times r$ entries) and $V$ ($r \times n$ entries), for a total of $r(m + n)$ entries. When $r \ll m$ and $r \ll n$, this is much less than $mn$.

But the key insight for MLA is not about storage of the weight matrices — it is about the **intermediate representation**. When we compute $W\mathbf{x} = U(V\mathbf{x})$, the intermediate vector $\mathbf{c} = V\mathbf{x}$ lives in $\mathbb{R}^r$. If $r$ is small, this intermediate vector is a compressed representation of the input.

### Numerical check

In our running example, $A \in \mathbb{R}^{2 \times 3}$ maps from 3 dimensions to 2. We can factor $A$ as the product of a $2 \times 1$ matrix $U$ and a $1 \times 3$ matrix $V$... but only if $A$ has rank 1. Let us check. The rows of $A$ are $(1, 0, -2)$ and $(3, 1, 4)$. Is $(3, 1, 4)$ a scalar multiple of $(1, 0, -2)$? That would require $3/1 = 1/0$, which is undefined. So the rows are linearly independent, and $A$ has rank 2 — it cannot be factored through a 1-dimensional intermediate.

Instead, let us construct a rank-1 example that mirrors MLA's structure. Define:

$$
W = \begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix} \begin{pmatrix} 1 & 0 & -2 \end{pmatrix} = \begin{pmatrix} 2 & 0 & -4 \\ -1 & 0 & 2 \\ 3 & 0 & -6 \end{pmatrix}
$$

Here $U = (2, -1, 3)^T \in \mathbb{R}^{3 \times 1}$ and $V = (1, 0, -2) \in \mathbb{R}^{1 \times 3}$. This is a $3 \times 3$ matrix with 9 entries, but it is fully determined by the $3 + 3 = 6$ entries of $U$ and $V$.

When we compute $W \mathbf{x}$, we can first compute the intermediate scalar $c = V\mathbf{x} = 1 \cdot 2 + 0 \cdot (-1) + (-2) \cdot 3 = -4$, and then compute $U \cdot c = (-4) \cdot (2, -1, 3)^T = (-8, 4, -12)^T$.

Alternatively, compute $W\mathbf{x}$ directly:

$$
W\mathbf{x} = \begin{pmatrix} 2 \cdot 2 + 0 \cdot (-1) + (-4) \cdot 3 \\ (-1) \cdot 2 + 0 \cdot (-1) + 2 \cdot 3 \\ 3 \cdot 2 + 0 \cdot (-1) + (-6) \cdot 3 \end{pmatrix} = \begin{pmatrix} -8 \\ 4 \\ -12 \end{pmatrix}
$$

Both methods give $(-8, 4, -12)^T$. The intermediate representation was a single scalar $c = -4$. In MLA, $V$ plays the role of $W^{DKV}$ (down-projection), $c$ plays the role of the latent vector $\mathbf{c}_t^{KV}$, and $U$ plays the role of $W^{UK}$ or $W^{UV}$ (up-projection). The crucial point: we only need to cache $c$, not the full output $W\mathbf{x}$.

---

## 9. Dot Product of Concatenated Vectors

In Section 4.3 of the DeepSeek-V2 blog, the full query and key are formed by concatenating a content part and a RoPE part: $\mathbf{q}_i = [\mathbf{q}_i^C; \mathbf{q}_i^R]$ and $\mathbf{k}_i = [\mathbf{k}_i^C; \mathbf{k}_i^R]$. Their dot product splits into two independent terms. We derive why.

The **concatenation** of $\mathbf{a} \in \mathbb{R}^m$ and $\mathbf{b} \in \mathbb{R}^n$ is the vector $[\mathbf{a}; \mathbf{b}] \in \mathbb{R}^{m+n}$ formed by stacking $\mathbf{b}$ below $\mathbf{a}$:

$$
[\mathbf{a}; \mathbf{b}] = (a_1, \ldots, a_m, b_1, \ldots, b_n)^T
$$

The dot product of two concatenated vectors, $[\mathbf{a}; \mathbf{b}]$ and $[\mathbf{c}; \mathbf{d}]$ (with $\mathbf{a}, \mathbf{c} \in \mathbb{R}^m$ and $\mathbf{b}, \mathbf{d} \in \mathbb{R}^n$), is:

$$
[\mathbf{a}; \mathbf{b}]^T [\mathbf{c}; \mathbf{d}] = \sum_{i=1}^{m} a_i c_i + \sum_{j=1}^{n} b_j d_j
$$

The first sum is $\mathbf{a}^T \mathbf{c}$ and the second is $\mathbf{b}^T \mathbf{d}$. Therefore:

$$
\boxed{[\mathbf{a}; \mathbf{b}]^T [\mathbf{c}; \mathbf{d}] = \mathbf{a}^T \mathbf{c} + \mathbf{b}^T \mathbf{d}}
$$

### Derivation

By the definition of the dot product:

$$
[\mathbf{a}; \mathbf{b}]^T [\mathbf{c}; \mathbf{d}] = \sum_{k=1}^{m+n} ([\mathbf{a}; \mathbf{b}])_k \cdot ([\mathbf{c}; \mathbf{d}])_k
$$

The first $m$ entries of $[\mathbf{a}; \mathbf{b}]$ are $a_1, \ldots, a_m$ and the last $n$ entries are $b_1, \ldots, b_n$. Similarly for $[\mathbf{c}; \mathbf{d}]$. We split the sum at $k = m$:

$$
= \sum_{k=1}^{m} a_k c_k + \sum_{k=1}^{n} b_k d_k = \mathbf{a}^T \mathbf{c} + \mathbf{b}^T \mathbf{d} \quad \square
$$

The second equality uses the definition of the dot product applied to each pair.

### Numerical check

Let $\mathbf{a} = (2, -1)^T$, $\mathbf{b} = (3,)$ (a 1-dimensional vector), $\mathbf{c} = (0, 4)^T$, $\mathbf{d} = (-2,)$.

**Direct computation.** $[\mathbf{a}; \mathbf{b}] = (2, -1, 3)^T$ and $[\mathbf{c}; \mathbf{d}] = (0, 4, -2)^T$.

$$
[\mathbf{a}; \mathbf{b}]^T [\mathbf{c}; \mathbf{d}] = 2 \cdot 0 + (-1) \cdot 4 + 3 \cdot (-2) = 0 - 4 - 6 = -10
$$

**Split computation.** $\mathbf{a}^T \mathbf{c} = 2 \cdot 0 + (-1) \cdot 4 = -4$. $\mathbf{b}^T \mathbf{d} = 3 \cdot (-2) = -6$. Sum: $-4 + (-6) = -10$.

Both give $-10$.

### Interpretation

The dot product of concatenated vectors decomposes cleanly into independent terms. No cross-talk: the content entries of the query only interact with the content entries of the key, and the RoPE entries only interact with the RoPE entries. This is why the DeepSeek-V2 blog writes:

$$
\mathbf{q}_{t,i}^T \mathbf{k}_{j,i} = (\mathbf{q}_{t,i}^C)^T \mathbf{k}_{j,i}^C + (\mathbf{q}_{t,i}^R)^T \mathbf{k}_j^R
$$

The content similarity and the positional similarity contribute additively. They can be computed and understood independently.

---

## Summary

We built eight linear algebra tools. **Matrix-vector multiplication** takes a matrix and a vector and produces a new vector whose dimension equals the number of rows of the matrix — this is how every projection in MLA works. **The transpose** swaps rows and columns, and the **transpose of a product** reverses the order: $(XY)^T = Y^T X^T$. **Matrix-matrix multiplication** extends the row-dot-column rule to two matrices. **Associativity** says $(AB)C = A(BC)$ — this lets us precompute combined projection matrices in MLA. **Non-commutativity** says $AB \neq BA$ in general — this is why RoPE cannot be pushed inside the up-projection. The **bilinear identity** $\mathbf{a}^T M \mathbf{b} = (M^T \mathbf{a})^T \mathbf{b}$ moves a matrix from one side of a dot product to the other — this is the heart of the key absorption trick. **Linearity** lets us pull a matrix out of a weighted sum — this is how the value absorption works. **Low-rank factorization** writes $W = UV$ with a small intermediate dimension — this is the compression that makes MLA's cache so small. And the **dot product of concatenated vectors** decomposes into independent terms — this is why decoupled RoPE works.

With these tools in hand, we are ready for the [DeepSeek-V2 blog](/blog/deepseek-v2-mla-and-deepseekmoe), where we put them all together to derive Multi-head Latent Attention and DeepSeekMoE from scratch.
