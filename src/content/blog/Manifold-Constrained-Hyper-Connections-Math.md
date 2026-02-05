---
title: "Manifold-Constrained Hyper-Connections: Stabilizing Deep Networks Beyond ResNets (with the actual math)"
description: "From residual identity paths to Hyper-Connections and mHC — now with the paper's exact equations, fully unrolled products, and n=2,C=2 numeric examples"
date: 2026-02-04
tags: ["deep-learning", "neural-networks", "linear-algebra", "transformers", "architecture", "optimization"]
---

In my previous version of this post, I focused on intuition: why residual connections work, why Hyper-Connections (HC) are appealing, and why they become unstable. That intuition is still correct — but I didn't explain the actual equations from the mHC paper.

This update fills that gap. We'll go from basics all the way to the exact multi-layer expansion equation in the paper, and we'll explain every term with concrete examples (especially n = 2, C = 2).

Paper: [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)

## 1. The Signal Propagation Problem in Deep Networks

Before residual connections, a deep network repeatedly applies transformations. Even tiny deviations from "do nothing" compound.

### 1.1 A Simple Deep Network

Consider:

$$x_{l+1} = w_l \cdot x_l$$

With $x_0 = 10$ and $w = 0.9$ for three layers:

Layer 1: $x_1 = 0.9 \times 10 = 9$

Layer 2: $x_2 = 0.9 \times 9 = 8.1$

Layer 3: $x_3 = 0.9 \times 8.1 = 7.29$

The signal shrinks: vanishing.

### 1.2 The Opposite Problem: Exploding Signals

With $w = 1.1$:

Layer 1: $x_1 = 11$

Layer 2: $x_2 = 12.1$

Layer 3: $x_3 = 13.31$

The signal grows: exploding.

The key point: repeated multiplication amplifies tiny errors.

## 2. Residual Networks: The Identity Path Solution

Residual connections hard-code a safe path: identity mapping.

### 2.1 Residual Connection (Paper Eq. 1)

The paper writes a residual block as:

$$x_{l+1} = x_l + F(x_l, W_l) \tag{1}$$

$x_l \in \mathbb{R}^{C}$ is the hidden state

$F(\cdot)$ is the attention/MLP block

$W_l$ are the parameters of that block

The important part is the literal $x_l$ term.

### 2.2 Why Identity Mapping Matters (Paper Eq. 2)

If you expand Eq. (1) across multiple layers, the paper gets:

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i) \tag{2}$$

This is why ResNets/Transformers are trainable at depth:

The term $x_l$ survives unchanged to depth $L$.

The network learns "corrections" via the sum of $F(\cdot)$ terms.

## 3. Hyper-Connections: What the Paper Actually Does

HC changes the residual stream itself.

### 3.1 The Key Change: the residual stream becomes n streams

In HC, the paper expands:

from $x_l \in \mathbb{R}^{C}$

to $x_l \in \mathbb{R}^{n \times C}$

So for n = 2, we literally have:

$$x_l = \begin{bmatrix} x_{l,0} \\\\ x_{l,1} \end{bmatrix}, \quad x_{l,0}, x_{l,1} \in \mathbb{R}^C$$

Think: two parallel residual "lanes".

### 3.2 Single-layer Hyper-Connections (Paper Eq. 3)

The paper defines one HC layer as:

$$x_{l+1} = H^{res}_l x_l + H^{post\top}_l F(H^{pre}_l x_l, W_l) \tag{3}$$

This looks scary until you interpret shapes.

**Shapes (this is non-negotiable)**

$x_l \in \mathbb{R}^{n\times C}$

$H^{res}_l \in \mathbb{R}^{n\times n}$ (mixes streams)

$H^{pre}_l \in \mathbb{R}^{1\times n}$ (reads from streams into a single $C$-dim input)

$H^{post}_l \in \mathbb{R}^{1\times n}$ (writes layer output back into all streams)

So HC has three learnable mappings: pre / post / res.

## 4. Eq. (3) With an Actual Numeric Example (n = 2, C = 2)

Let's take:

$$x_l = \begin{bmatrix} 1 & 2 \\\\ 3 & 4 \end{bmatrix}$$

So:

stream 0 = $[1,2]$

stream 1 = $[3,4]$

### 4.1 Pre-mapping: collapse 2 streams → 1 vector

For n = 2:

$$H^{pre}_l = [\alpha, \; 1-\alpha]$$

Pick:

$$H^{pre}_l = [0.6, \; 0.4]$$

Then:

$$H^{pre}_l x_l = 0.6[1,2] + 0.4[3,4] = [1.8, 2.8]$$

Now the Transformer block can run on a normal $C$-dim vector.

### 4.2 The layer function F produces new features

We'll just choose:

$$F([1.8, 2.8]) = [10, 20]$$

That's the "new information" produced by the layer.

### 4.3 Post-mapping: write output back into the 2-stream residual

For n = 2:

$$H^{post}_l = [\beta, \; 1-\beta]$$

Pick:

$$H^{post}_l = [0.7, \; 0.3]$$

Then:

$$H^{post\top}_l F(\cdot) = \begin{bmatrix} 0.7[10,20] \\\\ 0.3[10,20] \end{bmatrix} = \begin{bmatrix} [7,14] \\\\ [3,6] \end{bmatrix}$$

This is literally "copy the same layer output into both streams with different weights".

### 4.4 Residual mixing: mix the streams

For n = 2:

$$H^{res}_l = \begin{bmatrix} a & b \\\\ c & d \end{bmatrix}$$

HC is unconstrained (the dangerous part)

Example HC choice:

$$H^{res}_l = \begin{bmatrix} 2 & -1 \\\\ 1 & 1 \end{bmatrix}$$

Compute:

new stream 0 = $2\cdot[1,2] + (-1)\cdot[3,4] = [-1,0]$

new stream 1 = $1\cdot[1,2] + 1\cdot[3,4] = [4,6]$

So:

$$H^{res}_l x_l = \begin{bmatrix} -1 & 0 \\\\ 4 & 6 \end{bmatrix}$$

### 4.5 Full Eq. (3) output

Eq. (3) says:

$$x_{l+1} = H^{res}_l x_l + H^{post\top}_l F(H^{pre}_l x_l, W_l)$$

So numerically:

$$x_{l+1} = \begin{bmatrix} -1 & 0 \\\\ 4 & 6 \end{bmatrix} + \begin{bmatrix} 7 & 14 \\\\ 3 & 6 \end{bmatrix} = \begin{bmatrix} 6 & 14 \\\\ 7 & 12 \end{bmatrix}$$

That's HC, exactly.

## 5. The Hidden Problem: Multi-layer HC Breaks Identity Mapping

In ResNets, "identity mapping" stays identity because $I^k = I$.

In HC, identity is replaced by a product of learned matrices.

### 5.1 The Multi-layer Expansion (Paper Eq. 4)

The paper expands Eq. (3) across depth and gets:

$$x_L = \left(\prod_{i=1}^{L-l} H^{res}_{L-i}\right) x_l + \sum_{i=l}^{L-1} \left(\prod_{j=1}^{L-1-i} H^{res}_{L-j}\right) H^{post\top}_i F(H^{pre}_i x_i, W_i) \tag{4}$$

This is the central equation you must explain if you want the real math in your blog.

Let's break it into two parts.

## 6. The Two Parts of Eq. (4)

### 6.1 The "identity / carry" term

$$\left(\prod_{i=1}^{L-l} H^{res}_{L-i}\right) x_l$$

Meaning:

Start with $x_l$

Apply every residual mixing matrix from layer $l$ up to $L-1$

Whatever you get is what "identity mapping" became in HC

In ResNet this would just be $x_l$.

### 6.2 The "injection" term (what every layer adds)

$$\sum_{i=l}^{L-1} \left(\prod_{j=1}^{L-1-i} H^{res}_{L-j}\right) H^{post\top}_i F(H^{pre}_i x_i, W_i)$$

Meaning:

Each layer $i$ produces new features via $F(\cdot)$

$H^{post\top}_i$ injects them into the stream

The product after that transports them forward through the remaining residual mixings

Then all layer contributions are summed

## 7. What "i" and "j" Mean (and what the product actually does)

This is the part that confuses almost everyone.

### 7.1 The outer index i (SUM) = where the features are created

In the injection term:

i chooses the layer that generated the features.

So the term for i = 0 means "features created at layer 0".
The term for i = 1 means "features created at layer 1".
…and so on.

### 7.2 The inner index j (PRODUCT) = how many residual mixings happen after that layer

The product:

$$\prod_{j=1}^{L-1-i} H^{res}_{L-j}$$

means:

take the features injected at layer i, and apply the residual mixing matrices of all later layers until you reach layer L.

So it's not "multiple times per layer". It's:

once per layer boundary,

repeated for however many layers remain.

## 8. Fully Unrolling Eq. (4) for a Tiny Network

Let's take a concrete tiny depth:

start: $l = 0$

end: $L = 3$

Layers: 0 → 1 → 2 → 3

Residual matrices:

$H^{res}_0$ (0→1)

$H^{res}_1$ (1→2)

$H^{res}_2$ (2→3)

### 8.1 Identity/carry term becomes

$$\left(\prod_{i=1}^{3} H^{res}_{3-i}\right) x_0 = H^{res}_2 H^{res}_1 H^{res}_0 x_0$$

### 8.2 Injection term becomes three contributions

From layer 0 (must pass through layers 1 and 2 and 3):

$$H^{res}_2 H^{res}_1 H^{post\top}_0 F(H^{pre}_0 x_0)$$

From layer 1 (must pass through layers 2 and 3):

$$H^{res}_2 H^{post\top}_1 F(H^{pre}_1 x_1)$$

From layer 2 (second last, no mixing left):

$$H^{post\top}_2 F(H^{pre}_2 x_2)$$

So the total is:

$$x_3 = H^{res}_2 H^{res}_1 H^{res}_0 x_0 + H^{res}_2 H^{res}_1 H^{post\top}_0 F(\cdot) + H^{res}_2 H^{post\top}_1 F(\cdot) + H^{post\top}_2 F(\cdot)$$

That's Eq. (4), but with all indices removed.

## 9. Why HC Becomes Unstable (Now It's Obvious)

Look at the carry term again:

$$H^{res}_{L-1} H^{res}_{L-2} \cdots H^{res}_l \; x_l$$

If $H^{res}$ is unconstrained, then:

repeated multiplication can explode (gain >> 1)

or vanish (gain << 1)

and gradients also become unstable because backward depends on transposes and products

The paper explicitly points out that this breaks the "identity mapping property" of residual connections.

## 10. mHC: Constrain the Residual Mixing to Restore Stability

mHC keeps the same HC structure but restricts where $H^{res}$ is allowed to be.

### 10.1 The Constraint: Doubly Stochastic Matrices (Paper Eq. 6)

The paper constrains $H^{res}_l$ to the Birkhoff polytope:

$$\mathcal{P}_{\mathcal{M}^{res}}(H^{res}_l) = \{H^{res}_l \in \mathbb{R}^{n \times n} \;|\; H^{res}_l \mathbf{1}_n = \mathbf{1}_n, \; \mathbf{1}_n^\top H^{res}_l = \mathbf{1}_n^\top, \; H^{res}_l \geq 0\} \tag{6}$$

This means:

all entries are non-negative

each row sums to 1

each column sums to 1

So residual mixing becomes convex averaging, not amplification.

### 10.2 The most concrete interpretation (n = 2)

For n = 2, any doubly stochastic matrix has the form:

$$H^{res} = \begin{bmatrix} p & 1-p \\\\ 1-p & p \end{bmatrix} \quad 0 \leq p \leq 1$$

Pick $p = 0.75$:

$$H^{res} = \begin{bmatrix} 0.75 & 0.25 \\\\ 0.25 & 0.75 \end{bmatrix}$$

Apply to our C=2 streams:

$$x_l = \begin{bmatrix} [1,2] \\\\ [3,4] \end{bmatrix}$$

Result:

stream 0:

$$0.75[1,2] + 0.25[3,4] = [1.5, 2.5]$$

stream 1:

$$0.25[1,2] + 0.75[3,4] = [2.5, 3.5]$$

No sign flip, no explosion. Just mixing.

## 11. How mHC Produces These Matrices in Practice

HC learns $H^{pre},H^{post},H^{res}$ dynamically from the input (and static biases). mHC keeps that idea but then projects onto constraints.

### 11.1 Parameterization (Paper Eq. 7)

The paper flattens $x_l \in \mathbb{R}^{n\times C}$ into a vector and computes unconstrained mappings:

$$\begin{aligned}
\bar{x}_l &= \text{RMSNorm}(\text{vec}(x_l)) \\\\
\tilde{H}^{pre}_l &= \alpha^{pre}_l (\bar{x}_l \phi^{pre}_l) + b^{pre}_l \\\\
\tilde{H}^{post}_l &= \alpha^{post}_l (\bar{x}_l \phi^{post}_l) + b^{post}_l \\\\
\tilde{H}^{res}_l &= \alpha^{res}_l \text{mat}(\bar{x}_l \phi^{res}_l) + b^{res}_l
\end{aligned} \tag{7}$$

Key points:

$\alpha$ are gating scalars initialized small (so you start near safe behavior)

$\phi$ are learnable projections

$b$ are learned biases

$\tilde{H}^{res}_l$ is still unconstrained at this point

### 11.2 Manifold projection (Paper Eq. 8)

mHC enforces constraints by projecting:

$$\begin{aligned}
H^{pre}_l &= \sigma(\tilde{H}^{pre}_l) \\\\
H^{post}_l &= 2\sigma(\tilde{H}^{post}_l) \\\\
H^{res}_l &= \text{Sinkhorn-Knopp}(\tilde{H}^{res}_l)
\end{aligned} \tag{8}$$

Sigmoid makes pre/post non-negative

Sinkhorn-Knopp makes $H^{res}$ (approximately) doubly stochastic

### 11.3 Sinkhorn-Knopp (Paper Eq. 9)

Start with a positive matrix $M^{(0)} = \exp(\tilde{H}^{res})$ and alternate column and row normalization:

$$M^{(t)} = T_r(T_c(M^{(t-1)})) \tag{9}$$

After enough iterations, rows and columns are close to sum 1 → effectively doubly stochastic.

## 12. Why This Fix Works (in the exact Eq. (4) sense)

mHC doesn't just "regularize"; it changes what repeated products can become.

In Eq. (4), every dangerous term is a product of residual matrices:

the carry term has a product of many $H^{res}$

every injection term also has a product after injection

If every $H^{res}$ is doubly stochastic:

products remain well-conditioned (no blowup)

the "identity mapping" behavior is restored in the sense of conserving global signal intensity across streams

and the same reasoning applies consistently at every depth (between any two layers)

This is exactly the design goal stated by the paper: restore identity mapping behavior while keeping the expressivity of multi-stream mixing.

## 13. Summary (Updated)

| Architecture | Residual stream | Skip/carry mapping | Stability |
|-------------|-----------------|-------------------|-----------|
| Plain deep net | 1 stream | repeated transform | unstable |
| ResNet/Transformer | 1 stream | fixed identity | stable |
| HC | n streams | learned $H^{res}$ (unconstrained) | unstable at scale |
| mHC | n streams | learned $H^{res}$ projected to doubly stochastic | stable + expressive |

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Identity Mappings in Deep Residual Networks*. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)

3. Zhu, D., Huang, H., Huang, Z., Zeng, Y., Mao, Y., Wu, B., Min, Q., & Zhou, X. (2024). *Hyper-Connections*. [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)

4. Xie, Z., Wei, Y., Cao, H., et al. (2026). *mHC: Manifold-Constrained Hyper-Connections*. [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
