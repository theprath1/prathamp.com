---
title: "Manifold-Constrained Hyper-Connections: Stabilizing Deep Networks Beyond ResNets"
description: "A deep dive into why residual connections work, how Hyper-Connections generalize them, and why constraining learned skip paths to doubly stochastic matrices solves the instability problem"
date: 2026-01-30
tags: ["deep-learning", "neural-networks", "linear-algebra", "transformers", "architecture", "optimization"]
---

In this article, we'll trace the evolution from basic neural networks to residual networks, then to Hyper-Connections, and finally to manifold-constrained Hyper-Connections (mHC). The goal is to understand not just *what* these techniques do, but *why* each step was necessary and what problems it solves.

## 1. The Signal Propagation Problem in Deep Networks

Before diving into residual connections, let's understand the fundamental problem they were designed to solve.

### 1.1 A Simple Deep Network

Consider the simplest possible neural network where each layer multiplies the input by a weight:

$$x_{l+1} = w_l \cdot x_l$$

Let's trace what happens with an input $x_0 = 10$ and weights $w = 0.9$ across three layers.

**Forward pass (layer by layer):**

- Layer 1: $x_1 = 0.9 \times 10 = 9$
- Layer 2: $x_2 = 0.9 \times 9 = 8.1$
- Layer 3: $x_3 = 0.9 \times 8.1 = 7.29$

The signal shrinks at every layer. After many layers, it approaches zero—this is the **vanishing signal problem**.

### 1.2 The Opposite Problem: Exploding Signals

What if the weight is slightly larger? With $w = 1.1$:

- Layer 1: $x_1 = 1.1 \times 10 = 11$
- Layer 2: $x_2 = 1.1 \times 11 = 12.1$
- Layer 3: $x_3 = 1.1 \times 12.1 = 13.31$

The signal explodes—this is the **exploding signal problem**.

The key insight here is that any weight not exactly equal to 1 will cause problems when applied repeatedly across many layers. This makes training very deep networks extremely difficult—you're essentially trying to balance on a knife's edge where weights need to be *exactly* right to maintain signal magnitude.

## 2. Residual Networks: The Identity Path Solution

ResNets introduced an elegantly simple fix: add a skip connection that preserves the original input.

### 2.1 The Residual Connection Formula

Instead of $x_{l+1} = w \cdot x_l$, ResNets use:

$$x_{l+1} = x_l + F(x_l)$$

Where $F(x_l)$ represents any transformation (convolution, MLP, etc.). The critical addition is the $x_l$ term—the **identity path**.

### 2.2 Why the Identity Path Works

Let's trace through with the same setup: $x_0 = 10$, $w = 0.9$, using the residual form $x_{l+1} = x_l + w \cdot x_l$.

**Forward pass (layer by layer):**

- Layer 1: $x_1 = 10 + 0.9 \times 10 = 19$
- Layer 2: $x_2 = 19 + 0.9 \times 19 = 36.1$
- Layer 3: $x_3 = 36.1 + 0.9 \times 36.1 = 68.59$

This looks like growth, but it's **controlled growth**, not collapse. The identity path ensures the original signal is always present. Even if the learned transformation $F(x)$ produces small or noisy outputs, the core information flows through unchanged.

### 2.3 The Two-Path Architecture

A residual block has two parallel paths:

```
        ┌─────────────┐
x_l ───►│ identity (x)│──┐
        └─────────────┘  │
                         ├──► x_{l+1}
        ┌─────────────┐  │
x_l ───►│  F(x)       │──┘
        └─────────────┘
```

Path 1 (Identity) sends $x_l \rightarrow x_l$ with no change. Path 2 (Residual) applies the learned transformation $x_l \rightarrow F(x_l)$. The outputs are summed. That straight line at the top is the identity path—it's what makes deep training stable.

### 2.4 Why Information is Preserved

In a network without skip connections, when $x_0 = 10$ becomes $x_1 = 9$, the original 10 is **gone forever**. The signal is replaced by a transformed version. With skip connections, the original information flows through unchanged while the residual path learns what to add. The network learns *corrections* rather than *replacements*. This is a subtle but profound shift in how we think about what each layer does.

## 3. Hyper-Connections: Learning the Skip Path

ResNets made a fixed choice: always add the full identity. But what if different features should be passed with different strengths? What if the optimal "skip behavior" varies across the network?

### 3.1 The Motivation

In very deep transformers, each layer has multiple residual streams:

- Attention streams
- MLP streams
- Skip connections across blocks
- Sometimes dozens of interacting paths ResNet's fixed identity $x_{l+1} = x_l + F(x_l)$ treats all features equally. Hyper-Connections ask a natural question: *What if we could learn how much each previous stream contributes?*

### 3.2 The Hyper-Connection Formula

Instead of a fixed identity, Hyper-Connections introduce a **learned mixing matrix** $H$:

$$x_{l+1} = H \cdot x_l + W \cdot x_l$$

Here $H$ replaces the identity path (which was previously just $I$), and $W$ is the usual learned transformation.

### 3.3 What is $H$?

$H$ is a **learned matrix that replaces the identity path**. In ResNet, identity equals $I$ and is fixed. In Hyper-Connections, identity becomes $H$ and is learned. When $H = I$, you get exact ResNet behavior. When $H \neq I$, you get learned routing.

For a 2-neuron example, $H$ might look like:

$$H = \begin{bmatrix} 0.8 & 0.2 \\\\ 0.1 & 0.9 \end{bmatrix}$$

This means neuron 1 mostly keeps its value (0.8) with some mixing from neuron 2 (0.2), and neuron 2 mostly keeps its value (0.9) with some mixing from neuron 1 (0.1). The network can now learn cross-feature interactions in the skip path itself.

### 3.4 Why $H$ is Learnable

In a neural network, anything multiplied with the input can be learned through backpropagation. For two neurons, we have $x_{l+1} = H \cdot x_l$ where:

$$H = \begin{bmatrix} h_{11} & h_{12} \\\\ h_{21} & h_{22} \end{bmatrix}$$

Each $h_{ij}$ is just a scalar parameter—stored like any other weight, updated by gradient descent. There's nothing special about it; it's simply another weight matrix that happens to sit in the skip path.

### 3.5 How $H$ is Initialized

At the start of training, we want $H \approx I$. Why?

- Identity is stable
- Training starts safely
- The model behaves like ResNet initially A common initialization strategy is:

$$H = \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix} + \epsilon$$

Where $\epsilon$ is small random noise, giving something like:

$$H = \begin{bmatrix} 1.01 & -0.02 \\\\ 0.01 & 0.98 \end{bmatrix}$$

The matrix starts very close to identity, then learning adjusts it based on what the task requires.

**Why not initialize randomly?** If you start with something like:

$$H = \begin{bmatrix} 0.7 & 0.4 \\\\ 0.3 & 0.6 \end{bmatrix}$$

Then even before training begins, $H^{20}$ will cause explosion or collapse. The eigenvalues of this matrix aren't equal to 1, so repeated multiplication across 20 layers amplifies small deviations into catastrophic instability. This is why HC initialization must be identity-biased.

### 3.6 The Appeal of Hyper-Connections

Hyper-Connections allow:

- **Adaptive routing**: Different features can take different paths
- **Dynamic information flow**: The network decides what to preserve
- **Richer expressiveness**: More flexibility than fixed skip connections The idea is compelling: *let the network decide how identity should behave* rather than hard-coding it.

## 4. The Hidden Problem: Why Hyper-Connections Are Unstable

Here's where things get interesting. Despite the appealing flexibility, Hyper-Connections have a fundamental mathematical problem.

### 4.1 The Core Issue

Residual connections work because $I^n = I$. The identity matrix raised to any power is still the identity. Applying it across 100 layers changes nothing.

But with Hyper-Connections, $H^n \neq H$. After many layers, even if $H$ starts close to identity, repeated multiplication causes one dimension to dominate while another vanishes, and gradients explode or die. This happens even with just 2 neurons.

### 4.2 Why ResNet Doesn't Have This Problem

ResNet **never learns the identity**. It's fixed: $x \rightarrow x$. Hyper-Connections learn it: $x \rightarrow Hx$. Learning identity is **numerically dangerous** when repeated across many layers because any deviation from perfect identity gets amplified exponentially.

### 4.3 What Goes Wrong During Training

Even if $H_0 \approx I$ at initialization, training updates the matrix:

$$H \leftarrow H - \eta \nabla_H \mathcal{L}$$

Let me break down what this gradient notation means. The term $\nabla_H \mathcal{L}$ represents the derivative of the loss $\mathcal{L}$ with respect to the matrix $H$. Since $H$ is a matrix of numbers:

$$H = \begin{bmatrix} h_{11} & h_{12} \\\\ h_{21} & h_{22} \end{bmatrix}$$

We have four derivatives: $\frac{\partial \mathcal{L}}{\partial h_{11}}$, $\frac{\partial \mathcal{L}}{\partial h_{12}}$, $\frac{\partial \mathcal{L}}{\partial h_{21}}$, $\frac{\partial \mathcal{L}}{\partial h_{22}}$

Each derivative answers a simple question: *If I slightly change this number, does the loss go up or down?* For example, if $\frac{\partial \mathcal{L}}{\partial h_{12}} = -0.1$, it means increasing $h_{12}$ decreases the loss—a good direction to move.

We write this as a matrix to match the shape of $H$:

$$\nabla_H \mathcal{L} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial h_{11}} & \frac{\partial \mathcal{L}}{\partial h_{12}} \\\\ \frac{\partial \mathcal{L}}{\partial h_{21}} & \frac{\partial \mathcal{L}}{\partial h_{22}} \end{bmatrix}$$

We subtract because the gradient points uphill, and we want to go downhill to minimize loss.

### 4.4 The Eigenvalue Problem

Small updates from gradient descent can nudge eigenvalues slightly above 1 or slightly below 1. After $L$ layers, $H^L x$ either explodes or vanishes. This isn't a training bug—it's math. Any matrix with eigenvalues not exactly equal to 1 will cause problems when raised to large powers. The instability is baked into the structure of the problem.

## 5. Manifold-Constrained Hyper-Connections (mHC)

The solution isn't to abandon learned skip paths, but to constrain them to a "safe space" of matrices that remain stable under repeated multiplication.

### 5.1 The Core Idea

mHC doesn't change backpropagation. Instead, it changes **where $H$ is allowed to live**. The principle is simple: *Learn freely, then project back to a safe space.*

Mathematically, this means (1) taking an unconstrained gradient step, then (2) projecting onto the set of safe matrices.

### 5.2 What Matrices Are "Safe" to Repeat?

We want a matrix $H$ such that:

- It doesn't amplify values
- It doesn't shrink values
- Repeating it many times stays stable The safest operation on numbers is **averaging**. Consider $\text{avg}(10, 20) = 15$. Averaging never explodes, never vanishes—it just redistributes.

### 5.3 Averaging with Matrices

A matrix performs averaging if all entries are non-negative and each output is a weighted average of inputs. This happens when **rows sum to 1**. For example:

$$H = \begin{bmatrix} 0.7 & 0.3 \\\\ 0.4 & 0.6 \end{bmatrix}$$

Apply this to $x = \begin{bmatrix} 10 \\\\ 20 \end{bmatrix}$:

$$Hx = \begin{bmatrix} 0.7(10) + 0.3(20) \\\\ 0.4(10) + 0.6(20) \end{bmatrix} = \begin{bmatrix} 13 \\\\ 16 \end{bmatrix}$$

Each output is a weighted average—no scaling, no explosion. The total "mass" of information is preserved.

### 5.4 Why Rows Alone Aren't Enough

The forward pass uses $H$, but the backward pass (for computing gradients) uses $H^T$. So row sums equaling 1 gives us a stable forward pass, but we also need column sums equaling 1 for a stable backward pass. To protect both directions, we need **both constraints**.

## 6. Doubly Stochastic Matrices

### 6.1 Definition

A matrix is **doubly stochastic** if:

1. All entries are greater than or equal to zero
2. Each row sums to 1
3. Each column sums to 1

A simple example:

$$D = \begin{bmatrix} 0.5 & 0.5 \\\\ 0.5 & 0.5 \end{bmatrix}$$

This is perfect averaging—forward stable and backward stable.

### 6.2 Why Doubly Stochastic Matrices Behave Like Identity

These matrices have remarkable properties:

- The largest eigenvalue equals 1
- All other eigenvalues are at most 1
- The set is closed under multiplication (the product of two doubly stochastic matrices is itself doubly stochastic)

This means $D^L$ does not explode or vanish regardless of how large $L$ becomes. Repeated application behaves like identity plus smoothing—exactly what we want for skip paths.

### 6.3 The Birkhoff Polytope

The **Birkhoff polytope** is simply the set of all doubly stochastic matrices. For 2×2 matrices, it forms a diamond-shaped region in parameter space.

A beautiful fact from linear algebra is that **every doubly stochastic matrix is a weighted average of permutation matrices**. This gives us deep intuition about what these matrices actually do.

### 6.4 What Are Permutation Matrices?

A permutation matrix just reorders neurons without changing magnitudes. For 2 neurons, there are only two possibilities.

The identity permutation does nothing:

$$P_1 = \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix}$$

This sends Neuron 1 to Neuron 1 and Neuron 2 to Neuron 2.

The swap permutation exchanges them:

$$P_2 = \begin{bmatrix} 0 & 1 \\\\ 1 & 0 \end{bmatrix}$$

This sends Neuron 1 to Neuron 2 and Neuron 2 to Neuron 1.

Permutation matrices never change magnitude—they only move information around. This is why they're perfectly stable.

### 6.5 Doubly Stochastic = Soft Permutation

Take the two permutation matrices and compute a weighted average with $\alpha = 0.7$:

$$0.7 P_1 + 0.3 P_2 = 0.7 \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix} + 0.3 \begin{bmatrix} 0 & 1 \\\\ 1 & 0 \end{bmatrix}$$

$$= \begin{bmatrix} 0.7 & 0 \\\\ 0 & 0.7 \end{bmatrix} + \begin{bmatrix} 0 & 0.3 \\\\ 0.3 & 0 \end{bmatrix} = \begin{bmatrix} 0.7 & 0.3 \\\\ 0.3 & 0.7 \end{bmatrix}$$

This is doubly stochastic. The beautiful intuition is that **every safe $H$ is a soft permutation of features**. It's not doing hard routing (this neuron goes there), but soft routing (70% of this neuron stays here, 30% goes there).

## 7. The Projection Algorithm: Sinkhorn-Knopp

### 7.1 The Problem

Gradient descent gives us $\tilde{H} = H - \eta \nabla_H \mathcal{L}$, but $\tilde{H}$ is not doubly stochastic. Entries may be negative, and row/column sums are wrong. We need to push it back into the Birkhoff polytope. That push is accomplished by the **Sinkhorn-Knopp algorithm**.

### 7.2 The Modified Update Rule

Instead of the standard update $H \leftarrow H - \eta \nabla_H \mathcal{L}$, mHC uses:

$$\boxed{H \leftarrow \Pi_{\mathcal{D}}\left(H - \eta \nabla_H \mathcal{L}\right)}$$

Where $\Pi_{\mathcal{D}}$ represents projection onto doubly stochastic matrices. This is the entire mathematical change—everything else remains standard.

### 7.3 Sinkhorn-Knopp: Step by Step

The algorithm turns any positive matrix into a doubly stochastic one through iterative normalization.

**Step 0** makes entries positive using exponentiation: $A_{ij} = e^{\tilde{H}_{ij}}$. Now all entries are greater than 0.

**Step 1** normalizes rows by dividing each row by its sum.

**Step 2** normalizes columns by dividing each column by its sum.

**Step 3** repeats steps 1 and 2, alternating between row and column normalization until convergence.

After a few iterations, rows sum to approximately 1 and columns sum to approximately 1. The algorithm:

- Converges fast
- Is differentiable (so we can backpropagate through it)
- Is computationally efficient

### 7.4 Numeric Example

Start with:

$$A = \begin{bmatrix} 2 & 1 \\\\ 1 & 2 \end{bmatrix}$$

Row normalize to get:

$$\begin{bmatrix} 2/3 & 1/3 \\\\ 1/3 & 2/3 \end{bmatrix}$$

Now check column sums: Column 1 sums to $2/3 + 1/3 = 1$, and Column 2 sums to $1/3 + 2/3 = 1$. We're already doubly stochastic after one iteration. In practice, it usually takes just a handful of iterations to converge to machine precision.

## 8. The Complete mHC Pipeline

Putting it all together, the full stabilization pipeline works as follows:

1. Learn $H$ via standard gradient descent
2. Exponentiate to ensure positivity
3. Apply Sinkhorn normalization to project onto doubly stochastic matrices
4. Use the projected $H$ in the network's forward pass

Mathematically:

$$H \leftarrow \text{Sinkhorn}\left(e^{H - \eta \nabla_H \mathcal{L}}\right)$$

This single line captures the essence of the approach: take a gradient step in unconstrained space, then project back to the manifold of safe matrices.

## 9. Why This Works?

The key insight is that we've separated two concerns. 

- **Expressiveness**: $H$ can still learn—it's not fixed like in ResNets. 
- **Stability**: The doubly stochastic constraint ensures safe repeated multiplication.

Because doubly stochastic matrices:

- Have bounded eigenvalues
- Preserve signal magnitude on average
- Are stable in both forward and backward passes

The network gains the flexibility of Hyper-Connections without the instability.

Skip paths become **learned weighted averages**. Forward signals are preserved, backward gradients are preserved, and repeated depth produces redistribution rather than scaling. The identity is no longer fixed—but its stability properties are preserved through geometric constraints on the parameter space.

## 10. Summary

| Architecture | Skip Path | Stability | Flexibility |
|-------------|-----------|-----------|-------------|
| Plain Network | None | Unstable | N/A |
| ResNet | Fixed $I$ | Stable | None |
| Hyper-Connections | Learned $H$ | Unstable | High |
| mHC | Learned $H$ ∈ Birkhoff | Stable | High |

The progression tells a clear story. Plain networks suffer from vanishing/exploding signals—the fundamental problem. ResNets fix this with identity skip paths, but sacrifice flexibility by hard-coding the skip behavior. Hyper-Connections learn the skip path, gaining flexibility but reintroducing instability. Finally, mHC constrains learned skip paths to doubly stochastic matrices, achieving both stability and flexibility.

The mathematical elegance lies in recognizing that by constraining $H$ to lie on the manifold of doubly stochastic matrices (the Birkhoff polytope), we get the best of both worlds—learned routing that remains numerically stable across arbitrary depth.

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Identity Mappings in Deep Residual Networks*. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)

3. Zhu, D., Huang, H., Huang, Z., Zeng, Y., Mao, Y., Wu, B., Min, Q., & Zhou, X. (2024). *Hyper-Connections*. [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)

4. Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., Dai, D., Gao, H., et al. (2025). *mHC: Manifold-Constrained Hyper-Connections*. [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)

5. Sinkhorn, R., & Knopp, P. (1967). *Concerning nonnegative matrices and doubly stochastic matrices*. [Pacific Journal of Mathematics](https://msp.org/pjm/1967/21-2/p14.xhtml)

6. LeetArxiv. (2024). *Sinkhorn-Knopp Algorithm*. [Substack](https://leetarxiv.substack.com/p/sinkhorn-knopp-algorithm-24d)
