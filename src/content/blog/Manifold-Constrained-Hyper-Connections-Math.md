---
title: "Manifold-Constrained Hyper-Connections: Stabilizing Deep Networks Beyond ResNets (with the actual math)"
description: "From residual identity paths to Hyper-Connections and mHC — now with the paper's exact equations, fully unrolled products, and concrete numeric examples"
date: 2026-02-05
tags: ["deep-learning", "neural-networks", "linear-algebra", "transformers", "architecture", "optimization"]
---

In my previous version of this post, I focused on intuition: why residual connections work, why Hyper-Connections (HC) are appealing, and why they become unstable. Now, in this article, I'll be diving into the actual equations from the mHC paper.

This update fills that gap. I'll go from basics all the way to the exact multi-layer expansion equation in the paper, and I'll explain every term with concrete examples.

## 1. Residual Connection (Paper Eq. 1)

The paper writes a residual block as:

$$x_{l+1} = x_l + F(x_l, W_l) \tag{1}$$

Here $x_l \in \mathbb{R}^{C}$ represents the hidden state at layer $l$, where $C$ is the dimension of the feature vector. The function $F(\cdot)$ denotes whatever transformation the layer performs—this could be a self-attention mechanism, an MLP block, or any other differentiable operation. The parameters $W_l$ are the learnable weights of that specific layer.

The critical insight is the literal $x_l$ term on the right-hand side. This term passes through completely unchanged, regardless of what $F$ does. Even if $F$ produces garbage or zeros, the original signal $x_l$ survives intact. This is the identity path that makes residual networks trainable at extreme depths.

### 1.1 Why Identity Mapping Matters (Paper Eq. 2)

If you expand Eq. (1) across multiple layers, the paper gets:

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i) \tag{2}$$

This expansion reveals why ResNets and Transformers remain trainable even at depths of hundreds or thousands of layers. The term $x_l$ survives unchanged all the way to depth $L$—it doesn't get multiplied, transformed, or attenuated by any intermediate layer. The network's output at layer $L$ is simply the original input $x_l$ plus a sum of corrections from each intermediate layer.

This means each layer learns what to *add* to the existing representation, not how to *replace* it. The sum of $F(\cdot)$ terms represents accumulated refinements, while the base signal flows through a protected channel. Gradients during backpropagation also benefit from this structure, since they can flow directly back through the identity path without being multiplied by weight matrices at every step.

## 2. Hyper-Connections: What the Paper Actually Does

HC changes the residual stream itself. Rather than having a single residual path, HC introduces multiple parallel streams that can interact and mix with each other through learned transformations.

### 2.1 The Key Change: the residual stream becomes n streams

In HC, the paper expands the hidden state from a single vector to multiple vectors:

from $x_l \in \mathbb{R}^{C}$

to $x_l \in \mathbb{R}^{n \times C}$

So for n = 2, we literally have:

$$x_l = \begin{bmatrix} x_{l,0} \\\\ x_{l,1} \end{bmatrix}, \quad x_{l,0}, x_{l,1} \in \mathbb{R}^C$$

Think of this as two parallel residual "lanes" running through the network. Each lane carries a $C$-dimensional feature vector, and the lanes can exchange information with each other at every layer. This is fundamentally different from simply making the hidden dimension larger—we're creating structured parallelism where streams maintain separate identities while being able to communicate.

### 2.2 Single-layer Hyper-Connections (Paper Eq. 3)

The paper defines one HC layer as:

$$x_{l+1} = H^{res}_l x_l + H^{post\top}_l F(H^{pre}_l x_l, W_l) \tag{3}$$

This equation looks intimidating at first glance, but it becomes clear once you understand the shape of each component. The key is that HC introduces three separate learnable mappings that control how information flows: one for reading from streams, one for writing back to streams, and one for mixing streams directly.

**Shapes (this is non-negotiable)**

The hidden state $x_l \in \mathbb{R}^{n\times C}$ is now a matrix where each row is one stream. The residual mixing matrix $H^{res}_l \in \mathbb{R}^{n\times n}$ controls how the $n$ streams blend together—it's a square matrix that mixes streams without changing the feature dimension $C$. The pre-mapping $H^{pre}_l \in \mathbb{R}^{1\times n}$ is a row vector that collapses all $n$ streams into a single $C$-dimensional vector for input to the transformer block. The post-mapping $H^{post}_l \in \mathbb{R}^{1\times n}$ is another row vector that distributes the layer's output back into all $n$ streams.

So HC has three learnable mappings: pre (how to read), post (how to write), and res (how to mix). This gives the network fine-grained control over information routing that ResNets simply don't have.

## 3. Eq. (3) With an Actual Numeric Example (n = 2, C = 2)

Let's work through Eq. (3) with concrete numbers to see exactly what happens. We'll use two streams (n = 2) where each stream has two features (C = 2):

$$x_l = \begin{bmatrix} 1 & 2 \\\\ 3 & 4 \end{bmatrix}$$

In this matrix, stream 0 is $[1,2]$ (the first row) and stream 1 is $[3,4]$ (the second row). Each stream carries a 2-dimensional feature vector representing some learned representation of the input.

### 3.1 Pre-mapping: collapse 2 streams → 1 vector

The pre-mapping takes multiple streams and combines them into a single vector that the transformer block can process. For n = 2, this is simply a weighted combination:

$$H^{pre}_l = [\alpha, \; 1-\alpha]$$

Let's pick specific values to make this concrete:

$$H^{pre}_l = [0.6, \; 0.4]$$

Applying this to our streams computes a weighted average:

$$H^{pre}_l x_l = 0.6[1,2] + 0.4[3,4] = [1.8, 2.8]$$

The result is a single $C$-dimensional vector that blends information from both streams. Stream 0 contributes 60% and stream 1 contributes 40%. Now the Transformer block can run on this normal $C$-dim vector using standard attention and MLP operations, completely unaware that multiple streams exist.

### 3.2 The layer function F produces new features

For this example, we'll simply choose an output value to illustrate the mechanics:

$$F([1.8, 2.8]) = [10, 20]$$

In practice, $F$ would be a full transformer layer with self-attention and feed-forward networks. The specific values don't matter for understanding the HC mechanism—what matters is that $F$ takes a $C$-dimensional input and produces a $C$-dimensional output representing the "new information" this layer has computed.

### 3.3 Post-mapping: write output back into the 2-stream residual

The post-mapping takes the layer's single output vector and distributes it back into all streams. For n = 2:

$$H^{post}_l = [\beta, \; 1-\beta]$$

Let's pick:

$$H^{post}_l = [0.7, \; 0.3]$$

The transpose operation turns this row vector into a column vector, which then multiplies the layer output:

$$H^{post\top}_l F(\cdot) = \begin{bmatrix} 0.7[10,20] \\\\ 0.3[10,20] \end{bmatrix} = \begin{bmatrix} [7,14] \\\\ [3,6] \end{bmatrix}$$

This is literally "copy the same layer output into both streams with different weights". Stream 0 receives 70% of the new information while stream 1 receives 30%. The network can learn to route new features preferentially to certain streams, creating specialized pathways through the architecture.

### 3.4 Residual mixing: mix the streams

The residual mixing matrix $H^{res}$ controls how the existing stream contents blend together before adding the new features. For n = 2, this is a 2×2 matrix:

$$H^{res}_l = \begin{bmatrix} a & b \\\\ c & d \end{bmatrix}$$

In unconstrained HC, these entries can be any real numbers—and this is precisely where the danger lies. Let's pick values that illustrate the problem:

$$H^{res}_l = \begin{bmatrix} 2 & -1 \\\\ 1 & 1 \end{bmatrix}$$

Computing the matrix-vector product:

new stream 0 = $2\cdot[1,2] + (-1)\cdot[3,4] = [-1,0]$

new stream 1 = $1\cdot[1,2] + 1\cdot[3,4] = [4,6]$

So:

$$H^{res}_l x_l = \begin{bmatrix} -1 & 0 \\\\ 4 & 6 \end{bmatrix}$$

Notice what happened: the matrix amplified some values and flipped the sign of others. The entry "2" in position (1,1) doubled stream 0's contribution to itself, while the "-1" subtracted stream 1's values. This kind of unconstrained mixing can cause signals to grow or shrink rapidly across layers.

### 3.5 Full Eq. (3) output

Putting it all together, Eq. (3) says:

$$x_{l+1} = H^{res}_l x_l + H^{post\top}_l F(H^{pre}_l x_l, W_l)$$

Substituting our computed values:

$$x_{l+1} = \begin{bmatrix} -1 & 0 \\\\ 4 & 6 \end{bmatrix} + \begin{bmatrix} 7 & 14 \\\\ 3 & 6 \end{bmatrix} = \begin{bmatrix} 6 & 14 \\\\ 7 & 12 \end{bmatrix}$$

That's HC in action. The final output combines the mixed residual streams with the newly computed features. Each stream's new value depends on a learned combination of all previous streams plus a learned fraction of the layer's output.

## 4. The Hidden Problem: Multi-layer HC Breaks Identity Mapping

In ResNets, the "identity mapping" property is what makes training stable: $I^k = I$ for any power $k$. No matter how many layers you stack, the identity path remains exactly identity.

In HC, this beautiful property is destroyed. The identity is replaced by a product of learned matrices, and matrix products don't preserve nice properties the way scalar multiplication by 1 does.

### 4.1 The Multi-layer Expansion (Paper Eq. 4)

The paper expands Eq. (3) across depth and derives the central equation:

$$x\_L = \left(\prod\_{i=1}^{L-l} H^{\text{res}}\_{L-i}\right) x\_l + \sum\_{i=l}^{L-1} \left(\prod\_{j=1}^{L-1-i} H^{\text{res}}\_{L-j}\right) H^{\text{post}\top}\_i F(H^{\text{pre}}\_i x\_i, W\_i) \tag{4}$$

This equation is the mathematical heart of the paper, and understanding it is essential for grasping why HC becomes unstable. It shows that the output at layer $L$ depends on products of all the residual mixing matrices encountered along the way. Let's break it into two parts to understand what each term represents.

## 5. The Two Parts of Eq. (4)

### 5.1 The "identity / carry" term

$$\left(\prod_{i=1}^{L-l} H^{res}_{L-i}\right) x_l$$

This term represents what happened to the "identity mapping" in HC. In ResNet, the corresponding term would simply be $x_l$—the original input preserved unchanged. But in HC, the original input $x_l$ gets multiplied by the product of every residual mixing matrix from layer $l$ all the way to layer $L-1$.

Start with $x_l$, apply $H^{res}_l$, then apply $H^{res}_{l+1}$, and so on until you've applied $H^{res}_{L-1}$. Whatever emerges from this chain of matrix multiplications is what "identity mapping" has become in HC. The input is no longer preserved—it's transformed by a potentially long sequence of learned linear maps.

### 5.2 The "injection" term (what every layer adds)

$$\sum_{i=l}^{L-1} \left(\prod_{j=1}^{L-1-i} H^{res}_{L-j}\right) H^{post\top}_i F(H^{pre}_i x_i, W_i)$$

This term captures how each layer's computed features propagate to the final output. Each layer $i$ produces new features through $F(\cdot)$, which get injected into the streams via $H^{post\top}_i$. But these features don't stay put—they must pass through all subsequent residual mixing matrices before reaching the output at layer $L$.

The outer sum adds up contributions from every layer: layer $l$ contributes features that pass through $L-l-1$ mixing matrices, layer $l+1$ contributes features that pass through $L-l-2$ mixing matrices, and so on. Layer $L-1$ contributes features that go directly to the output with no additional mixing.

## 6. What "i" and "j" Mean (and what the product actually does)

This is the part that confuses almost everyone when first reading the paper. The indices $i$ and $j$ serve completely different purposes, and conflating them leads to misunderstanding.

### 6.1 The outer index i (SUM) = where the features are created

In the injection term, the index $i$ iterates over layers, selecting which layer's contribution we're currently considering. When $i = 0$, we're looking at features created at layer 0 and how they propagate forward. When $i = 1$, we're looking at features created at layer 1. Each value of $i$ gives one term in the sum, representing one layer's contribution to the final output.

So the term for i = 0 means "features created at layer 0".
The term for i = 1 means "features created at layer 1".
…and so on through all layers.

### 6.2 The inner index j (PRODUCT) = how many residual mixings happen after that layer

The product indexed by $j$ computes how many residual mixing matrices must be applied to transport features from their creation point to the final layer:

$$\prod_{j=1}^{L-1-i} H^{res}_{L-j}$$

This product takes features injected at layer $i$ and applies every subsequent residual mixing matrix until reaching layer $L$. The number of matrices in the product depends on how far layer $i$ is from the output—earlier layers have their features mixed more times.

It's not "multiple applications per layer" or some form of iteration within a layer. It's simply: once per layer boundary, applied for however many layer boundaries remain between the injection point and the output.

## 7. Fully Unrolling Eq. (4) for a Tiny Network

Abstract equations become clear with concrete examples. Let's take a tiny network and write out every term explicitly:

start: $l = 0$

end: $L = 3$

Layers: 0 → 1 → 2 → 3

This means we have three residual mixing matrices:

$H^{res}_0$ governs the transition from layer 0 to layer 1

$H^{res}_1$ governs the transition from layer 1 to layer 2

$H^{res}_2$ governs the transition from layer 2 to layer 3

### 7.1 Identity/carry term becomes

$$\left(\prod_{i=1}^{3} H^{res}_{3-i}\right) x_0 = H^{res}_2 H^{res}_1 H^{res}_0 x_0$$

The input $x_0$ gets multiplied by all three residual matrices in sequence. In ResNet, this would just be $x_0$. In HC, the input has been transformed three times before reaching the output—and each transformation can amplify, shrink, or rotate the signal in ways that compound dangerously.

### 7.2 Injection term becomes three contributions

From layer 0 (features must pass through all subsequent mixing matrices):

$$H^{res}_2 H^{res}_1 H^{post\top}_0 F(H^{pre}_0 x_0)$$

Features computed at layer 0 get injected via $H^{post\top}_0$, then must traverse $H^{res}_1$ and $H^{res}_2$ before reaching the output.

From layer 1 (features must pass through two subsequent mixing matrices):

$$H^{res}_2 H^{post\top}_1 F(H^{pre}_1 x_1)$$

Features computed at layer 1 get injected via $H^{post\top}_1$, then traverse only $H^{res}_2$ before reaching the output.

From layer 2 (features go directly to output, no mixing left):

$$H^{post\top}_2 F(H^{pre}_2 x_2)$$

Features computed at the second-to-last layer go straight to the output with no additional residual mixing.

So the total is:

$$x_3 = H^{res}_2 H^{res}_1 H^{res}_0 x_0 + H^{res}_2 H^{res}_1 H^{post\top}_0 F(\cdot) + H^{res}_2 H^{post\top}_1 F(\cdot) + H^{post\top}_2 F(\cdot)$$

That's Eq. (4) with all indices removed and every term written explicitly. You can now see exactly how the original input and each layer's features flow through the network.

## 8. Why HC Becomes Unstable (Now It's Obvious)

Look at the carry term again:

$$H^{\text{res}}\_{L-1} H^{\text{res}}\_{L-2} \cdots H^{\text{res}}\_l \; x\_l$$

If $H^{res}$ is unconstrained, then this product of matrices can behave in pathological ways. When matrices have eigenvalues greater than 1, repeated multiplication causes exponential growth—the signal explodes. When eigenvalues are less than 1, repeated multiplication causes exponential decay—the signal vanishes.

The same instability affects gradients during backpropagation. The backward pass involves transposes and products of these same matrices, so if forward signals explode, backward gradients will too. The paper explicitly points out that this breaks the "identity mapping property" that made residual connections stable in the first place.

In ResNet, $I \cdot I \cdot I = I$. In HC, $H^{res}_2 \cdot H^{res}_1 \cdot H^{res}_0$ could be anything—and "anything" usually means "unstable."

## 9. mHC: Constrain the Residual Mixing to Restore Stability

mHC keeps the same HC structure—multiple streams, learned pre/post mappings, residual mixing—but restricts where $H^{res}$ is allowed to live. Instead of letting it be any matrix, mHC forces it onto a manifold of "safe" matrices.

### 9.1 The Constraint: Doubly Stochastic Matrices (Paper Eq. 6)

The paper constrains $H^{res}_l$ to the Birkhoff polytope, the set of all doubly stochastic matrices:

$$\mathcal{P}_{\mathcal{M}^{res}}(H^{res}_l) = \{H^{res}_l \in \mathbb{R}^{n \times n} \;|\; H^{res}_l \mathbf{1}_n = \mathbf{1}_n, \; \mathbf{1}_n^\top H^{res}_l = \mathbf{1}_n^\top, \; H^{res}_l \geq 0\} \tag{6}$$

This formal definition encodes three requirements: all entries must be non-negative (no sign flips), each row must sum to 1 (outputs are convex combinations of inputs), and each column must sum to 1 (information is neither created nor destroyed). Together, these constraints ensure that residual mixing becomes convex averaging rather than amplification or attenuation.

### 9.2 The most concrete interpretation (n = 2)

For n = 2, the doubly stochastic constraint dramatically simplifies what matrices are allowed. Any valid $H^{res}$ must have the form:

$$H^{res} = \begin{bmatrix} p & 1-p \\\\ 1-p & p \end{bmatrix} \quad 0 \leq p \leq 1$$

This is a one-parameter family of matrices, controlled entirely by the mixing coefficient $p$. When $p = 1$, you get identity (no mixing). When $p = 0$, you get the swap matrix (complete exchange). When $p = 0.5$, you get perfect averaging.

Let's pick $p = 0.75$ and apply it to our earlier example:

$$H^{res} = \begin{bmatrix} 0.75 & 0.25 \\\\ 0.25 & 0.75 \end{bmatrix}$$

Apply to our C=2 streams:

$$x_l = \begin{bmatrix} [1,2] \\\\ [3,4] \end{bmatrix}$$

Result:

stream 0: $0.75[1,2] + 0.25[3,4] = [1.5, 2.5]$

stream 1: $0.25[1,2] + 0.75[3,4] = [2.5, 3.5]$

No sign flip, no explosion, no vanishing. Just smooth mixing between streams. Each output is a weighted average of inputs, and the total "mass" of information is conserved. This is the kind of operation that remains stable no matter how many times you repeat it.

## 10. How mHC Produces These Matrices in Practice

HC learns $H^{pre}$, $H^{post}$, and $H^{res}$ dynamically from the input, allowing the routing to adapt based on what the network is processing. mHC keeps this adaptive behavior but adds a projection step that forces the learned values onto the constraint manifold.

### 10.1 Parameterization (Paper Eq. 7)

The paper flattens $x_l \in \mathbb{R}^{n\times C}$ into a vector and computes unconstrained mappings:

$$\begin{aligned}
\bar{x}_l &= \text{RMSNorm}(\text{vec}(x_l)) \\\\
\tilde{H}^{pre}_l &= \alpha^{pre}_l (\bar{x}_l \phi^{pre}_l) + b^{pre}_l \\\\
\tilde{H}^{post}_l &= \alpha^{post}_l (\bar{x}_l \phi^{post}_l) + b^{post}_l \\\\
\tilde{H}^{res}_l &= \alpha^{res}_l \text{mat}(\bar{x}_l \phi^{res}_l) + b^{res}_l
\end{aligned} \tag{7}$$

The gating scalars $\alpha$ are initialized to small values, ensuring the network starts near safe, default behavior before learning to deviate. The projections $\phi$ are learnable weight matrices that map the normalized input to the space of routing parameters. The biases $b$ provide learned offsets that don't depend on the input.

At this stage, $\tilde{H}^{res}_l$ is still unconstrained—it's just the raw output of a linear transformation. The magic happens in the next step.

### 10.2 Manifold projection (Paper Eq. 8)

mHC enforces constraints by projecting the unconstrained outputs onto valid manifolds:

$$\begin{aligned}
H^{pre}_l &= \sigma(\tilde{H}^{pre}_l) \\\\
H^{post}_l &= 2\sigma(\tilde{H}^{post}_l) \\\\
H^{res}_l &= \text{Sinkhorn-Knopp}(\tilde{H}^{res}_l)
\end{aligned} \tag{8}$$

The sigmoid function $\sigma$ squashes pre and post mappings to be non-negative and bounded, ensuring they represent valid mixing weights. The factor of 2 on $H^{post}$ allows it to amplify slightly, giving the network more expressive range.

The Sinkhorn-Knopp algorithm is the key innovation for $H^{res}$. It takes any real matrix and iteratively normalizes rows and columns until the result is (approximately) doubly stochastic. This projection is differentiable, so gradients flow through it during backpropagation.

### 10.3 Sinkhorn-Knopp (Paper Eq. 9)

The algorithm starts with a positive matrix $M^{(0)} = \exp(\tilde{H}^{res})$ and alternates between normalizing columns and normalizing rows:

$$M^{(t)} = T_r(T_c(M^{(t-1)})) \tag{9}$$

Here $T_c$ divides each column by its sum, and $T_r$ divides each row by its sum. After enough iterations, both rows and columns sum to 1, giving an effectively doubly stochastic matrix. The algorithm converges quickly—typically just a few iterations suffice—and the entire process is differentiable, allowing end-to-end training.

## 11. Why This Fix Works (in the exact Eq. (4) sense)

mHC doesn't just "regularize" or "add a penalty term"—it fundamentally changes what the products in Eq. (4) can become. Every dangerous term in that equation involves a product of residual matrices, and constraining those matrices to be doubly stochastic changes the mathematical character of those products.

In Eq. (4), the carry term contains a product of many $H^{res}$ matrices, and every injection term also contains a product of the $H^{res}$ matrices encountered after injection. If every single $H^{res}$ in these products is doubly stochastic, then several powerful guarantees follow.

First, products remain well-conditioned. The product of doubly stochastic matrices is itself doubly stochastic, so no matter how deep the network goes, the carry term cannot explode or vanish. Second, the "identity mapping" behavior is restored in a meaningful sense: the total signal intensity across streams is conserved, even though the signal gets redistributed among streams. Third, the same reasoning applies to gradients during backpropagation, since transposes of doubly stochastic matrices are also doubly stochastic.

This is exactly the design goal stated by the paper: restore identity mapping behavior while keeping the expressivity of multi-stream mixing. The network can still learn sophisticated routing patterns—it just can't learn patterns that would cause numerical instability.

## 12. Summary

| Architecture | Residual Stream | Carry/Skip Mapping | Multi-layer Product | Stability |
|-------------|-----------------|-------------------|---------------------|-----------|
| ResNet/Transformer | 1 stream ($x_l \in \mathbb{R}^C$) | Fixed identity $I$ | $I^L = I$ | Stable |
| HC | n streams ($x_l \in \mathbb{R}^{n \times C}$) | Learned $H^{res}$ (unconstrained) | $\prod H^{res}$ can explode/vanish | Unstable at scale |
| mHC | n streams ($x_l \in \mathbb{R}^{n \times C}$) | Learned $H^{res}$ ∈ Birkhoff polytope | Product stays doubly stochastic | Stable + expressive |

The progression from ResNet to mHC tells a clear mathematical story. ResNets achieve stability through the trivial observation that $I^L = I$, but sacrifice flexibility by hard-coding the skip path. Hyper-Connections gain flexibility by learning the skip path, but the product $\prod H^{res}$ can grow or shrink exponentially, reintroducing the instability that residual connections were designed to solve. mHC threads the needle by constraining $H^{res}$ to doubly stochastic matrices, ensuring that products remain well-behaved while still allowing learned, input-dependent routing.

The mathematical elegance lies in recognizing that doubly stochastic matrices form a convex set closed under multiplication—a "safe manifold" where learned skip paths can live without causing the pathologies that plague unconstrained HC.

## References

1. Xie, Z., Wei, Y., Cao, H., et al. (2026). *mHC: Manifold-Constrained Hyper-Connections*. [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
