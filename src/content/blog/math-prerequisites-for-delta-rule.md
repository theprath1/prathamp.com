---
title: "Mathematical Prerequisites for the Delta Rule"
description: "Building the foundations for the delta rule and gated attention — the Frobenius norm and squared reconstruction loss, the gradient of a quadratic loss with respect to a matrix state, online stochastic gradient descent as a one-step update, and the chain rule for gated identity paths — all derived step by step with one consistent two-dimensional key-value example."
date: 2026-04-07
tags: [machine-learning, attention, transformers, delta-rule, deltanet, gated-attention, online-learning, mathematics]
order: 5
---

The next two posts in this series, [Targeted Memory: The Delta Rule, Gated DeltaNet, and Kimi Delta Attention](/blog/attention-gated-deltanet) and [Gated Attention: Replacing Residuals and ReLU with Learned Gates](/blog/attention-gated-attention), both rest on the same idea: an attention update can be derived as **one step of stochastic gradient descent on a reconstruction loss**. The delta rule is

$$
S_t = (I - \beta_t k_t k_t^\top)\, S_{t-1} + \beta_t k_t v_t^\top
$$

and is presented in the gated-deltanet paper as the closed-form result of taking one SGD step on $\mathcal{L}_t = \frac{1}{2}\|S^\top k_t - v_t\|^2$ with learning rate $\beta_t$. Without matrix calculus the appearance of $(I - \beta_t k_t k_t^\top)$ from this loss reads as magic. With it, every term is forced.

This post builds four tools. The **Frobenius norm** measures the size of a matrix as the square root of the sum of squared entries. The **squared reconstruction loss** is the natural objective for associative memory. **Matrix calculus** lets us differentiate a scalar loss with respect to a matrix variable, and the **online SGD update** with one matrix-valued step turns the gradient into the delta rule. The fifth tool is the **chain rule for a gated identity path** — needed to analyze the gradient flow through the residual-replacing gates in the gated-attention blog.

Before we get to norms and gradients, we need one small piece of associative-memory algebra. The delta rule updates a matrix state $S_t$ that stores key-value pairs, so we first derive what a write $k_t v_t^\top$ means and what a read $S_t^\top q$ returns. These are the only structural facts we need, and we derive them here so this post can stand on its own.

---

## The Running Example

A 2-dimensional associative memory storing a single key–value pair. The state is a $2 \times 2$ matrix $S$. The key and value are

$$
k = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \qquad v = \begin{pmatrix} 3 \\ 4 \end{pmatrix}, \qquad S_0 = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}
$$

with learning rate $\beta = 1$. We will compute the reconstruction loss at $S_0$, take its gradient with respect to $S$, apply one SGD step, and verify that retrieval $S_1^\top k$ now returns $v$ exactly. Same setup, throughout.

---

## 1. Associative Memory Refresher

### 1.1 Motivation

The delta rule is an update for a matrix state $S_t$. To understand why the loss is $\tfrac{1}{2}\|S^\top k_t - v_t\|^2$, we first need to know how $S_t$ stores key-value pairs and how a query reads them back. That starts with the **outer product**.

### 1.2 Outer Product

For column vectors $u \in \mathbb{R}^{m}$ and $v \in \mathbb{R}^{n}$, the **outer product** $u v^\top$ is the $m \times n$ matrix

$$
(u v^\top)_{ij} = u_i v_j
$$

Every column of $u v^\top$ is a scalar multiple of $u$, and every row is a scalar multiple of $v^\top$, so the result has rank 1.

### 1.3 Derivation: a sum of outer products is an associative memory

Suppose we store key-value pairs $(k_i, v_i)$ in the state

$$
S_t = \sum_{i=1}^{t} k_i v_i^\top
$$

Querying that state with a vector $q$ gives

$$
S_t^\top q = \left(\sum_{i=1}^{t} k_i v_i^\top \right)^\top q = \left(\sum_{i=1}^{t} v_i k_i^\top \right) q = \sum_{i=1}^{t} v_i (k_i^\top q)
$$

The first step used $(ab^\top)^\top = ba^\top$, and the second used distributivity of matrix multiplication over addition. So the readout is a weighted combination of stored values, where each weight is the inner product between the query $q$ and a stored key $k_i$.

### 1.4 Exact retrieval and interference

If we query with one of the stored keys, say $q = k_j$, then

$$
S_t^\top k_j = \sum_{i=1}^{t} v_i (k_i^\top k_j)
$$

This returns $v_j$ exactly only when the keys are **orthonormal**: $k_i^\top k_j = 0$ for $i \neq j$ and $k_j^\top k_j = 1$. In that case every other stored value disappears and we get

$$
S_t^\top k_j = v_j
$$

If the keys are not orthonormal, other values leak into the readout. That leakage is exactly the reconstruction error that the delta rule will try to reduce.

### 1.5 Numerical check with the running example

With the running key and value,

$$
S_1 = k v^\top = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 3 & 4 \end{pmatrix} = \begin{pmatrix} 3 & 4 \\ 0 & 0 \end{pmatrix}
$$

Read with the same key:

$$
S_1^\top k = \begin{pmatrix} 3 & 0 \\ 4 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 4 \end{pmatrix} = v
$$

So a single outer-product write stores a single key-value pair exactly. $\checkmark$

### 1.6 Interpretation

This is the baseline memory model. The delta rule keeps the same readout $S_t^\top k_t$, but instead of always writing a full additive outer product, it takes one SGD step that both writes the new pair and corrects whatever the current state gets wrong about it.

---

## 2. The Frobenius Norm

### 2.1 Motivation

The reconstruction loss compares two vectors and produces a scalar. Internally it computes a vector difference and then "the size of the difference vector". To define that size — and later to extend to matrices when we differentiate — we need the **Frobenius norm**. It is the matrix analogue of the squared Euclidean norm of a vector, and it specializes to the vector case for $1 \times n$ matrices.

### 2.2 Definition for vectors

For a vector $u \in \mathbb{R}^n$, the **squared Euclidean norm** is

$$
\|u\|^2 = \sum_{i=1}^{n} u_i^2 = u^\top u
$$

The square root $\|u\|$ is the length of $u$. The squared form $\|u\|^2$ is what appears in least-squares losses because it is differentiable everywhere and its gradient has a clean closed form.

### 2.3 Definition for matrices

For a matrix $M \in \mathbb{R}^{m \times n}$, the **Frobenius norm** is the square root of the sum of squared entries:

$$
\boxed{\|M\|_F^2 = \sum_{i=1}^{m} \sum_{j=1}^{n} M_{ij}^2 = \text{tr}(M^\top M)}
$$

The trace identity is a useful rewrite: $\text{tr}(M^\top M) = \sum_j (M^\top M)_{jj} = \sum_j \sum_i M_{ij}^2$, the same sum.

### 2.4 Numerical check with the running example

Let $e = S_0^\top k - v$ (the reconstruction error at $S_0$). With $S_0 = 0$:

$$
e = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}^\top \begin{pmatrix} 1 \\ 0 \end{pmatrix} - \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} - \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} -3 \\ -4 \end{pmatrix}
$$

By the squared-norm definition:

$$
\|e\|^2 = (-3)^2 + (-4)^2 = 9 + 16 = 25
$$

So $\|e\| = 5$ — the classical 3-4-5 triangle. $\checkmark$

### 2.5 Interpretation

The Frobenius norm extends the vector squared-length to matrices by treating the matrix as a flattened vector. It is the simplest matrix norm, the one that gives matrix-valued gradient descent its closed-form updates, and the norm under which every objective in this post is defined.

---

## 3. The Reconstruction Loss for Associative Memory

### 3.1 Motivation

Section 1 showed that for an associative memory

$$
S = \sum_i k_i v_i^\top
$$

the readout for query $k$ is

$$
S^\top k = \sum_i v_i (k_i^\top k)
$$

This returns $v_j$ exactly only when the stored keys are orthonormal, so that the query for one key does not pick up contributions from the others. When the keys are *not* orthonormal, retrieval is imperfect — there is a **reconstruction error**. The delta rule is derived by minimizing the squared size of this error.

### 3.2 Formal definition

For a single query–answer pair $(k_t, v_t)$ at step $t$ and a state matrix $S$, the **reconstruction loss** is

$$
\boxed{\mathcal{L}_t(S) = \tfrac{1}{2}\, \| S^\top k_t - v_t \|^2}
$$

The factor of $\tfrac{1}{2}$ is conventional and exists only to cancel the 2 that comes out when we differentiate the square. The objective is to choose $S$ so that retrieving with key $k_t$ returns a vector close to $v_t$.

### 3.3 Numerical check with the running example

At $S_0 = 0$ the error is $e = (-3, -4)^\top$ from Section 2.4, so

$$
\mathcal{L}_t(S_0) = \tfrac{1}{2} \cdot 25 = 12.5
$$

The loss is large because the empty memory cannot reconstruct anything — that is exactly the regime where the first SGD step (Section 5) will produce a large update. $\checkmark$

### 3.4 Interpretation

The reconstruction loss is a differentiable surrogate for "did we store this pair?" — small loss means the memory returns the right value when queried. Section 4 differentiates this loss with respect to the matrix $S$; Section 5 turns the gradient into a one-step update. The two together *are* the delta rule.

---

## 4. Matrix Calculus: Gradient of the Reconstruction Loss

### 4.1 Motivation

Section 3 wrote the loss as a function of a matrix $S$. To do gradient descent we need $\frac{\partial \mathcal{L}}{\partial S}$ — a matrix of the same shape as $S$, whose $(i,j)$ entry is the partial derivative of the scalar $\mathcal{L}$ with respect to the entry $S_{ij}$. This is **matrix calculus**.

### 4.2 The convention

For a scalar function $f(S)$ of a matrix $S \in \mathbb{R}^{m \times n}$, the gradient is the matrix

$$
\left(\frac{\partial f}{\partial S}\right)_{ij} = \frac{\partial f}{\partial S_{ij}}
$$

The output has the same shape as the input.

### 4.3 Derivation: gradient of $\mathcal{L}_t$ with respect to $S$

Start from the definition:

$$
\mathcal{L}_t = \tfrac{1}{2}\, \| S^\top k_t - v_t \|^2 = \tfrac{1}{2}\, (S^\top k_t - v_t)^\top (S^\top k_t - v_t)
$$

Let $e = S^\top k_t - v_t$ (the reconstruction error). Note $e$ depends linearly on $S$: changing $S$ to $S + \Delta S$ changes $e$ by $(\Delta S)^\top k_t$.

Expand the loss:

$$
\mathcal{L}_t = \tfrac{1}{2} e^\top e
$$

Differentiate using the **product rule** on $e^\top e$. A small perturbation $S \mapsto S + \Delta S$ produces a first-order change in the loss

$$
\delta \mathcal{L}_t = \tfrac{1}{2}\big[ (\delta e)^\top e + e^\top (\delta e) \big] = e^\top (\delta e)
$$

where the last step uses that both terms are scalars and equal each other (both are $\sum_i e_i (\delta e)_i$). Substitute $\delta e = (\Delta S)^\top k_t$:

$$
\delta \mathcal{L}_t = e^\top (\Delta S)^\top k_t = (k_t^\top \Delta S\, e)^\top = k_t^\top (\Delta S)\, e
$$

The two transposes flipped because the quantity is a scalar (a $1 \times 1$ matrix is its own transpose). Now we want to read off the gradient. By the **inner-product identity for matrices**, $\langle A, B \rangle_F = \text{tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}$, and

$$
k_t^\top (\Delta S)\, e = \text{tr}(e\, k_t^\top (\Delta S)) = \text{tr}((k_t e^\top)^\top \Delta S) = \langle k_t e^\top,\, \Delta S \rangle_F
$$

The first equality uses the **cyclic property of the trace** ($\text{tr}(ABC) = \text{tr}(CAB)$ when shapes allow). So $\delta \mathcal{L}_t = \langle k_t e^\top,\, \Delta S \rangle_F$, which means the gradient is the matrix that pairs with $\Delta S$ under the Frobenius inner product:

$$
\boxed{\frac{\partial \mathcal{L}_t}{\partial S} = k_t e^\top = k_t (S^\top k_t - v_t)^\top = k_t k_t^\top S - k_t v_t^\top}
$$

The last equality just expanded $e$.

### 4.4 Numerical check with the running example

At $S_0 = 0$, the error is $e = (-3, -4)^\top$ (from Section 2.4). The gradient is

$$
\frac{\partial \mathcal{L}_t}{\partial S}\bigg|_{S_0} = k\, e^\top = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} -3 & -4 \end{pmatrix} = \begin{pmatrix} -3 & -4 \\ 0 & 0 \end{pmatrix}
$$

Verify by entry-wise partial derivatives. Write $S = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$. Then $S^\top k = \begin{pmatrix} a \\ b \end{pmatrix}$ (since $k = (1, 0)^\top$ picks out the first row of $S^\top$, which is the first column of $S$). The error is $e = (a - 3,\, b - 4)^\top$ and the loss is

$$
\mathcal{L} = \tfrac{1}{2}\big[(a-3)^2 + (b-4)^2\big]
$$

The partials are $\partial \mathcal{L}/\partial a = a - 3$, $\partial \mathcal{L}/\partial b = b - 4$, and $\partial \mathcal{L}/\partial c = \partial \mathcal{L}/\partial d = 0$. At $S_0 = 0$ these evaluate to $-3, -4, 0, 0$, so

$$
\frac{\partial \mathcal{L}}{\partial S}\bigg|_{S_0} = \begin{pmatrix} -3 & -4 \\ 0 & 0 \end{pmatrix}
$$

which matches the boxed formula. $\checkmark$

### 4.5 Interpretation

The gradient is a **rank-1 matrix** — it is the outer product of the key with the error. This is what makes the delta rule's update structurally simple: it modifies $S$ only in the **column direction $k_t$** (every column of $k_t e^\top$ is a multiple of $k_t$) and only changes the entries that affect the key being trained. Components of $S$ orthogonal to $k_t$ are untouched. The next section turns this gradient into the delta rule by taking one SGD step.

---

## 5. Online Stochastic Gradient Descent: One Step Becomes the Delta Rule

### 5.1 Motivation

Sections 3 and 4 gave us a loss and its gradient. The natural way to use them is **gradient descent**: move $S$ in the direction of decreasing loss, with a step size $\beta$. **Online** SGD takes one such step *per data point*, never visiting the same example twice. For a sequence of key–value pairs, this gives one update per step — exactly the form of a recurrent update.

### 5.2 The SGD update

For a loss $\mathcal{L}(S)$ and learning rate $\beta > 0$, one SGD step is

$$
\boxed{S_\text{new} = S_\text{old} - \beta\, \frac{\partial \mathcal{L}}{\partial S}\bigg|_{S_\text{old}}}
$$

The minus sign moves $S$ *down* the loss surface (descent). The step size $\beta$ controls how far.

### 5.3 Derivation: applying SGD to the reconstruction loss

Substitute the gradient from Section 4:

$$
S_t = S_{t-1} - \beta_t \, \frac{\partial \mathcal{L}_t}{\partial S}\bigg|_{S_{t-1}} = S_{t-1} - \beta_t (k_t k_t^\top S_{t-1} - k_t v_t^\top)
$$

Distribute the minus sign and group the $S_{t-1}$ terms:

$$
S_t = S_{t-1} - \beta_t k_t k_t^\top S_{t-1} + \beta_t k_t v_t^\top
$$

Factor the first two terms by pulling $S_{t-1}$ out on the right:

$$
\boxed{S_t = (I - \beta_t k_t k_t^\top)\, S_{t-1} + \beta_t k_t v_t^\top}
$$

This is the **delta rule**. It is one step of online SGD on the reconstruction loss with learning rate $\beta_t$ — nothing more, nothing less. The mysterious $(I - \beta_t k_t k_t^\top)$ factor, which the gated-deltanet paper calls a **generalized Householder transformation**, is just the identity minus the gradient's "erase old value" component.

### 5.4 Numerical check with the running example

With $\beta = 1$ and the gradient from Section 4.4:

$$
S_1 = S_0 - 1 \cdot \begin{pmatrix} -3 & -4 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 3 & 4 \\ 0 & 0 \end{pmatrix}
$$

Verify by the boxed factored form. With $\|k\|^2 = 1$ (so $k k^\top = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$):

$$
I - \beta k k^\top = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} - \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}
$$

$$
(I - \beta k k^\top)\, S_0 = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}
$$

$$
\beta k v^\top = 1 \cdot \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 3 & 4 \end{pmatrix} = \begin{pmatrix} 3 & 4 \\ 0 & 0 \end{pmatrix}
$$

Summing: $S_1 = \begin{pmatrix} 3 & 4 \\ 0 & 0 \end{pmatrix}$. Both forms agree. $\checkmark$

Now query: $S_1^\top k = \begin{pmatrix} 3 & 0 \\ 4 & 0 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 4 \end{pmatrix} = v$. Perfect retrieval after one step. $\checkmark$

### 5.5 Delta rule vs plain additive write: same problem, two solutions

This is the cleanest distinction between the delta rule and plain linear attention. **Plain additive write** $S_t = S_{t-1} + k_t v_t^\top$ is the gradient of the loss $-\langle S, k_t v_t^\top \rangle_F$ — it just *adds* the rank-1 update without checking what was already there. **Delta rule** $S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ is the gradient of the *reconstruction loss* — it first **erases** the component of $S_{t-1}$ that corresponds to $k_t$, then writes the new $v_t$. Same memory structure (rank-1 updates, outer-product state). Different objective. Different update.

### 5.6 Interpretation

The delta rule is online SGD on a reconstruction loss. That sentence subsumes everything: where the $(I - \beta k k^\top)$ factor comes from (the gradient of $\|S^\top k - v\|^2$ with respect to $S$), why it has the form of "identity minus a rank-1 erase" (because the gradient of a quadratic is rank-1 in $k$), and why the rule is sometimes called *targeted memory* (it changes only the column direction of the current key). Every variant in [Targeted Memory](/blog/attention-gated-deltanet) — the gated delta rule, KDA, Kimi Linear — is derived by changing the loss and re-applying the same SGD recipe.

---

## 6. The Chain Rule for a Gated Identity Path

### 6.1 Motivation

[Gated Attention](/blog/attention-gated-attention) replaces a fixed residual $y = x + f(x)$ with a learned gate $y = (1 - g(x)) x + g(x) f(x)$, then traces what happens to the gradient as $g$ varies. To follow that trace we need the **chain rule** for a gated identity path — specifically, the gradient of $y$ with respect to $x$ when $y$ is a convex combination of $x$ and $f(x)$ with coefficients that themselves depend on $x$.

### 6.2 The chain rule (one-variable, refresher)

For composed scalar functions $y = f(g(x))$:

$$
\frac{dy}{dx} = f'(g(x)) \cdot g'(x)
$$

The derivative of a composition is the product of derivatives. Generalizes to vector inputs and outputs via the **Jacobian** — the matrix of partial derivatives.

### 6.3 The product rule (refresher)

For two scalar functions $u(x)$ and $v(x)$:

$$
\frac{d(uv)}{dx} = u'(x)\, v(x) + u(x)\, v'(x)
$$

Each factor is differentiated in turn while the other is held fixed; the results are summed.

### 6.4 Derivation: gradient of a gated identity

Consider the scalar form (the vector form is identical entry by entry). Let

$$
y = (1 - g(x)) \cdot x + g(x) \cdot f(x)
$$

where $g, f$ are differentiable. Apply the product rule to each of the two terms:

$$
\frac{dy}{dx} = \underbrace{-g'(x) \cdot x + (1 - g(x)) \cdot 1}_{\text{first term}} + \underbrace{g'(x) \cdot f(x) + g(x) \cdot f'(x)}_{\text{second term}}
$$

Collect:

$$
\boxed{\frac{dy}{dx} = (1 - g(x)) + g(x) f'(x) + g'(x) (f(x) - x)}
$$

Three pieces. The first $(1 - g(x))$ is the **identity-path gradient** — the gradient of $x$ flowing through, weighted by how much of the gate is "open to identity". The second $g(x) f'(x)$ is the **transformed-path gradient** — the gradient through $f$, weighted by the gate. The third $g'(x)(f(x) - x)$ comes from the gate itself depending on $x$; it vanishes when the gate is constant.

### 6.5 Numerical check with the running example

We re-use the running setup as a 1D scalar instance. Take $x = 1$, $g(x) = 1/2$ (constant for simplicity, so $g'(x) = 0$), and $f(x) = x^2$ so $f'(x) = 2x$. Then by the boxed formula:

$$
\frac{dy}{dx}\bigg|_{x=1} = (1 - \tfrac{1}{2}) + \tfrac{1}{2} \cdot 2 \cdot 1 + 0 = \tfrac{1}{2} + 1 = \tfrac{3}{2}
$$

Verify directly. With these choices, $y = \tfrac{1}{2}\, x + \tfrac{1}{2}\, x^2$, so $\frac{dy}{dx} = \tfrac{1}{2} + x$. At $x = 1$: $\tfrac{1}{2} + 1 = \tfrac{3}{2}$. $\checkmark$

### 6.6 Interpretation

The two paths sum because the gate distributes a fraction of the gradient through each. When $g \to 0$ the identity path dominates and the layer behaves like a pure skip connection (gradient flows unattenuated). When $g \to 1$ the transformed path dominates and the layer behaves like a non-residual transformation. This is why gated attention can interpolate between "pure residual" and "no residual" — and why the gate's gradient itself (the $g'(x)(f(x) - x)$ term) vanishes precisely at the points where the two paths agree. Every gradient-flow argument in [Gated Attention](/blog/attention-gated-attention) is an application of this single decomposition.

---

## Summary

We first established the associative-memory picture behind the delta rule: writing a pair means adding a rank-1 outer product $k_i v_i^\top$, and reading with a query $q$ returns $S^\top q = \sum_i v_i (k_i^\top q)$. Exact retrieval happens only when the stored keys are orthonormal; otherwise there is interference, which is why a reconstruction loss makes sense in the first place.

We then built five tools, each used in a follow-up post. The **Frobenius norm** $\|M\|_F^2 = \sum_{ij} M_{ij}^2$ measures matrix size and reduces to the squared Euclidean norm for vectors. The **reconstruction loss** $\mathcal{L}_t = \tfrac{1}{2}\|S^\top k_t - v_t\|^2$ measures how badly the memory misremembers a stored pair. **Matrix calculus** gave us $\frac{\partial \mathcal{L}_t}{\partial S} = k_t (S^\top k_t - v_t)^\top$ — a rank-1 outer product of the key and the error. **Online SGD** applied this gradient with step $\beta_t$ produces the **delta rule** $S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$, deriving the gated-deltanet update from the loss in three lines. Finally, the **chain rule for a gated identity path** decomposed the gradient of $y = (1 - g) x + g f(x)$ into identity, transformed, and gate-derivative pieces — the lens through which the gated-attention blog analyzes residual replacement.

With these tools in hand, we are ready for [Targeted Memory: The Delta Rule, Gated DeltaNet, and Kimi Delta Attention](/blog/attention-gated-deltanet), where every linear-RNN variant becomes the closed-form solution to a different optimization problem, and [Gated Attention](/blog/attention-gated-attention), where the gradient flow through learned gates is traced through exactly the decomposition above.
