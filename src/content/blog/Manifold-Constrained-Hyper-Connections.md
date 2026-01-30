# Manifold-Constrained Hyper-Connections: Stabilizing Deep Networks Beyond ResNets

Deep neural networks have revolutionized machine learning, but training them effectively remains a nuanced challenge. This article explores the evolution from Residual Networks (ResNets) to Hyper-Connections, and finally to Manifold-Constrained Hyper-Connections (mHC)—a mathematically elegant solution to a fundamental instability problem.

Throughout this article, we will use a single consistent example: **a network with two neurons and an input vector $x = \begin{bmatrix} 10 \\ 20 \end{bmatrix}$**. This allows us to trace the same numbers through every concept.

---

## Part 1: Understanding ResNets and the Identity Path

### The Problem with Deep Networks

Consider a simple neural network where each layer multiplies the input by a weight. For a single neuron with weight $w = 0.9$:

$$x_{l+1} = w \cdot x_l$$

Starting with $x_0 = 10$ (the first component of our example vector):

**Forward pass (layer by layer):**
- Layer 1: $x_1 = 0.9 \times 10 = 9$
- Layer 2: $x_2 = 0.9 \times 9 = 8.1$
- Layer 3: $x_3 = 0.9 \times 8.1 = 7.29$

The signal shrinks with every layer. After many layers, it approaches zero—this is the **vanishing gradient problem**.

**What if the weight is slightly larger?**

With $w = 1.1$:
- Layer 1: $x_1 = 1.1 \times 10 = 11$
- Layer 2: $x_2 = 1.1 \times 11 = 12.1$
- Layer 3: $x_3 = 1.1 \times 12.1 = 13.31$

The signal explodes—this is the **exploding gradient problem**.

### The Residual Solution

ResNets introduce a simple but profound change:

$$x_{l+1} = x_l + w \cdot x_l$$

Using the same $x_0 = 10$ and $w = 0.9$:

**Forward pass:**
- Layer 1: $x_1 = 10 + 0.9 \times 10 = 19$
- Layer 2: $x_2 = 19 + 0.9 \times 19 = 36.1$
- Layer 3: $x_3 = 36.1 + 0.9 \times 36.1 = 68.59$

This is controlled growth, not collapse. The identity path ensures the original signal is always present.

### Visualizing the Identity Path

```
        ┌─────────────┐
x_l ───►│ identity (x)│──┐
        └─────────────┘  │
                          ├──► x_{l+1}
        ┌─────────────┐  │
x_l ───►│ weight (wx) │──┘
        └─────────────┘
```

The straight line at the top is the identity path—it passes information unchanged. In a standard network without this path, our original 10 becomes 9 after one layer, and the original value is lost forever. The residual connection preserves it.

---

## Part 2: The Rise (and Problem) of Hyper-Connections

### Why Hyper-Connections Were Introduced

In very deep transformers, each layer has multiple residual streams:
- Attention stream
- MLP stream
- Skip connections across blocks
- Sometimes dozens of paths

Standard ResNet architecture only allows:

$$x_{l+1} = x_l + F(x_l)$$

Hyper-Connections generalize this by asking: *What if we could learn how much each previous stream contributes?*

### The Hyper-Connection Formulation

Instead of hard-coding the identity, Hyper-Connections introduce a learnable mixing matrix $H$. Now we move to our full two-neuron example with $x = \begin{bmatrix} 10 \\ 20 \end{bmatrix}$.

The update becomes:

$$x_{l+1} = Hx_l + Wx_l$$

Here, $H$ is a learned matrix that replaces the identity path:
- When $H = I$: exact ResNet behavior
- When $H \neq I$: learned routing

Consider our example matrix:

$$H = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$$

This means neuron 1 keeps 70% of its value and mixes in 30% from neuron 2, and vice versa.

Applied to our input:

$$Hx = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix} \begin{bmatrix} 10 \\ 20 \end{bmatrix} = \begin{bmatrix} 0.7(10) + 0.3(20) \\ 0.3(10) + 0.7(20) \end{bmatrix} = \begin{bmatrix} 13 \\ 17 \end{bmatrix}$$

The values are mixed but preserved—no explosion, no vanishing.

### The Appeal of Hyper-Connections

Hyper-Connections allow:
- Adaptive routing
- Dynamic information flow
- Richer expressiveness

The idea is elegant: let the network decide how identity should behave.

### The Hidden Instability

Here's the critical insight that motivates the paper. With ResNets:

$$I^n = I$$

The identity matrix raised to any power remains the identity. But with Hyper-Connections:

$$H^n \neq H$$

Let's see what happens when we apply our $H$ matrix repeatedly. After two layers:

$$H^2 = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}^2 = \begin{bmatrix} 0.58 & 0.42 \\ 0.42 & 0.58 \end{bmatrix}$$

The mixing increases. After 20 layers:

$$H^{20} \approx \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix}$$

All information about which neuron had which value is lost—both neurons converge to the average.

Now consider a slightly different matrix where one eigenvalue exceeds 1:

$$H' = \begin{bmatrix} 0.8 & 0.3 \\ 0.3 & 0.8 \end{bmatrix}$$

After many layers, $(H')^{20} x$ explodes because the dominant eigenvalue is $1.1 > 1$.

**Why didn't ResNet have this problem?** Because ResNet never learns the identity—it is fixed as $x \rightarrow x$. Hyper-Connections learn it as $x \rightarrow Hx$. Learning identity is numerically dangerous when repeated across many layers.

---

## Part 3: Understanding the Learnable Matrix H

### What Does "Learnable" Mean?

In neural networks, anything multiplied with the input can be learned through backpropagation. For our two neurons:

$$x_{l+1} = Hx_l$$

Where:

$$H = \begin{bmatrix} h_{11} & h_{12} \\ h_{21} & h_{22} \end{bmatrix}$$

Each $h_{ij}$ is a scalar parameter stored like any other weight and updated by backpropagation.

### Motivation for Learning H

ResNet hard-codes the identity, assuming every layer should pass exactly the same information forward. Hyper-Connections ask: *What if some features should pass more strongly, others less, or be mixed?*

This allows:
- Feature re-weighting
- Feature mixing
- Dynamic routing across layers

Conceptually, it's **learning how to skip**.

### Initialization Strategy

The goal at initialization is:

$$H \approx I$$

Why? Identity is stable, training starts safely, and the model behaves like ResNet initially.

For our 2×2 case, we initialize near identity with small noise:

$$H = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} + \epsilon = \begin{bmatrix} 1.01 & -0.02 \\ 0.01 & 0.98 \end{bmatrix}$$

Applied to $x = \begin{bmatrix} 10 \\ 20 \end{bmatrix}$:

$$Hx = \begin{bmatrix} 1.01(10) + (-0.02)(20) \\ 0.01(10) + 0.98(20) \end{bmatrix} = \begin{bmatrix} 9.7 \\ 19.7 \end{bmatrix}$$

Close to the original—good starting point.

### What Goes Wrong During Training

Even with $H_0 \approx I$, training updates the matrix:

$$H \leftarrow H - \eta \nabla_H \mathcal{L}$$

Small updates can make eigenvalues slightly greater than 1 or slightly less than 1. After many layers:

$$H^L x \Rightarrow \text{explodes or vanishes}$$

This is not a training bug—it's mathematics.

### Understanding the Gradient

The notation $\nabla_H \mathcal{L}$ means the derivative of the loss $\mathcal{L}$ with respect to matrix $H$. For our two neurons, this means four derivatives:

$$\frac{\partial \mathcal{L}}{\partial h_{11}}, \frac{\partial \mathcal{L}}{\partial h_{12}}, \frac{\partial \mathcal{L}}{\partial h_{21}}, \frac{\partial \mathcal{L}}{\partial h_{22}}$$

Each derivative answers: "If I slightly change this number, does the loss go up or down?"

The gradient is written as a matrix to match the shape of $H$:

$$\nabla_H \mathcal{L} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial h_{11}} & \frac{\partial \mathcal{L}}{\partial h_{12}} \\ \frac{\partial \mathcal{L}}{\partial h_{21}} & \frac{\partial \mathcal{L}}{\partial h_{22}} \end{bmatrix}$$

We subtract the gradient because it points uphill, and we want to go downhill (minimize loss).

---

## Part 4: The mHC Solution

### Core Problem Statement

Hyper-Connections replace the identity skip with a learned matrix $H$, but repeated multiplication of an unconstrained matrix across many layers causes explosion or vanishing. Everything in mHC exists to fix this single problem.

### What Matrices Are Safe to Repeat?

We want a matrix $H$ such that:
- It does not amplify values
- It does not shrink values
- Repeating it many times stays stable

The safest operation on numbers is **averaging**. For example, $\text{avg}(10, 20) = 15$. Averaging never explodes, never vanishes—it just redistributes.

### Averaging with Matrices

A matrix can perform averaging if:
- All entries are non-negative
- Each output is a weighted average of inputs

This happens when **rows sum to 1**. Our example matrix has this property:

$$H = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$$

Row 1: $0.7 + 0.3 = 1$. Row 2: $0.3 + 0.7 = 1$.

Applied to $x = \begin{bmatrix} 10 \\ 20 \end{bmatrix}$:

$$Hx = \begin{bmatrix} 0.7(10) + 0.3(20) \\ 0.3(10) + 0.7(20) \end{bmatrix} = \begin{bmatrix} 13 \\ 17 \end{bmatrix}$$

Each output is a weighted average of the inputs—no scaling occurs. The total "mass" is preserved: $10 + 20 = 30$ and $13 + 17 = 30$.

### Why Row Constraints Alone Are Insufficient

The forward pass uses $H$, but the backward pass uses $H^T$:

$$H^T = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$$

In our example, $H = H^T$ (the matrix is symmetric), so both directions are stable. But in general:
- Row sums = 1 ensures stable forward pass
- Column sums = 1 ensures stable backward pass

To protect both directions, we need rows AND columns to sum to 1.

### Doubly Stochastic Matrices

A matrix is **doubly stochastic** if:
- All entries are greater than or equal to 0
- Each row sums to 1
- Each column sums to 1

Our example matrix $H = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$ is doubly stochastic:
- Row 1: $0.7 + 0.3 = 1$
- Row 2: $0.3 + 0.7 = 1$
- Column 1: $0.7 + 0.3 = 1$
- Column 2: $0.3 + 0.7 = 1$

This provides perfect averaging with forward and backward stability.

### Why Doubly Stochastic Matrices Behave Like Identity

Key properties:
- Largest eigenvalue = 1
- All other eigenvalues are less than or equal to 1
- Closed under multiplication

For our $H$, the eigenvalues are $\lambda_1 = 1$ and $\lambda_2 = 0.4$. Since both are at most 1, $H^L$ does not explode. Repeated application behaves like identity plus smoothing—exactly what we need for skip paths.

---

## Part 5: The Birkhoff Polytope and Permutation Matrices

### The Birkhoff Polytope

The **Birkhoff polytope** is simply the set of all doubly stochastic matrices. For 2×2 matrices, it forms a diamond-shaped region in parameter space.

A fundamental theorem states: *Every doubly stochastic matrix is a weighted average of permutation matrices.*

### What Is a Permutation Matrix?

A permutation matrix just reorders neurons. For our 2 neurons, there are only two possibilities:

**Identity permutation (do nothing):**

$$P_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

Applied to $x = \begin{bmatrix} 10 \\ 20 \end{bmatrix}$:

$$P_1 x = \begin{bmatrix} 10 \\ 20 \end{bmatrix}$$

Neuron 1 stays as neuron 1, neuron 2 stays as neuron 2.

**Swap permutation:**

$$P_2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

Applied to $x = \begin{bmatrix} 10 \\ 20 \end{bmatrix}$:

$$P_2 x = \begin{bmatrix} 20 \\ 10 \end{bmatrix}$$

Neuron 1 and neuron 2 are swapped.

Permutation matrices never change magnitude—they only move information around.

### Constructing Our Example Matrix from Permutations

Here's the beautiful insight. Our matrix $H = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$ can be written as:

$$H = 0.7 P_1 + 0.3 P_2$$

Let's verify:

$$0.7 \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} + 0.3 \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0.7 & 0 \\ 0 & 0.7 \end{bmatrix} + \begin{bmatrix} 0 & 0.3 \\ 0.3 & 0 \end{bmatrix} = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$$

This means: **every safe $H$ is a soft permutation of features**. Our $H$ is "70% keep in place, 30% swap"—a probabilistic mixture of routing decisions.

---

## Part 6: The mHC Algorithm

### The Core Idea

mHC does not change backpropagation. Instead, it changes **where $H$ is allowed to live**:

*"Learn freely, then project back to a safe space."*

Mathematically:
1. Perform an unconstrained gradient step
2. Project onto doubly stochastic matrices

### The Modified Update Rule

Instead of:

$$H \leftarrow H - \eta \nabla_H \mathcal{L}$$

mHC performs:

$$\boxed{H \leftarrow \Pi_{\mathcal{D}}\left(H - \eta \nabla_H \mathcal{L}\right)}$$

Where $\Pi_{\mathcal{D}}$ denotes projection onto the set of doubly stochastic matrices $\mathcal{D}$.

This is the entire mathematical change.

### What Projection Means

Suppose gradient descent produces an invalid matrix:

$$\tilde{H} = \begin{bmatrix} 0.8 & 0.4 \\ 0.2 & 0.9 \end{bmatrix}$$

This is not doubly stochastic (rows sum to 1.2 and 1.1, columns sum to 1.0 and 1.3).

Projection means: "Take this matrix and gently push it back so it obeys the constraints." The result might be:

$$H = \begin{bmatrix} 0.65 & 0.35 \\ 0.35 & 0.65 \end{bmatrix}$$

Now rows and columns all sum to 1. This push is accomplished using **Sinkhorn normalization**.

---

## Part 7: The Sinkhorn-Knopp Algorithm

The Sinkhorn-Knopp algorithm transforms any positive matrix into a doubly stochastic one.

### Step 0: Ensure Positivity

Start with gradient descent output. Use exponentiation to ensure all entries are positive:

$$A_{ij} = e^{\tilde{H}_{ij}}$$

For our running example, suppose we start with:

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

All entries are positive (good), but row sums are 3 and column sums are 3 (not 1).

### Step 1: Normalize Rows

Divide each row by its sum:

$$A' = \begin{bmatrix} 2/3 & 1/3 \\ 1/3 & 2/3 \end{bmatrix} = \begin{bmatrix} 0.667 & 0.333 \\ 0.333 & 0.667 \end{bmatrix}$$

Now rows sum to 1. Check columns: $0.667 + 0.333 = 1$ and $0.333 + 0.667 = 1$.

### Step 2: Normalize Columns

Divide each column by its sum. In this case, columns already sum to 1, so we're done.

### Convergence

In general, you alternate row and column normalization until convergence. For our symmetric example, one iteration sufficed.

The final result:

$$H = \begin{bmatrix} 0.667 & 0.333 \\ 0.333 & 0.667 \end{bmatrix}$$

This is doubly stochastic and can be written as $0.667 P_1 + 0.333 P_2$—a soft permutation.

The algorithm is:
- Fast to converge (typically 5-10 iterations)
- Fully differentiable
- Computationally efficient

---

## Part 8: The Complete Stabilization Pipeline

Putting it all together, the mHC pipeline for each training step is:

1. Compute gradient $\nabla_H \mathcal{L}$
2. Take gradient step: $\tilde{H} = H - \eta \nabla_H \mathcal{L}$
3. Exponentiate to ensure positivity
4. Apply Sinkhorn normalization to project onto Birkhoff polytope
5. Use this stable $H$ in the network

Mathematically:

$$H \leftarrow \text{Sinkhorn}(H - \eta \nabla_H \mathcal{L})$$

### Complete Numerical Example

Let's trace through with our consistent example.

**Start:** $H = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$, $x = \begin{bmatrix} 10 \\ 20 \end{bmatrix}$

**Forward pass:** $Hx = \begin{bmatrix} 13 \\ 17 \end{bmatrix}$

**After gradient update:** Suppose $\tilde{H} = \begin{bmatrix} 0.75 & 0.35 \\ 0.28 & 0.72 \end{bmatrix}$ (rows sum to 1.1 and 1.0—invalid)

**After Sinkhorn projection:** $H_{new} = \begin{bmatrix} 0.68 & 0.32 \\ 0.32 & 0.68 \end{bmatrix}$ (doubly stochastic)

**Next forward pass:** $H_{new} x = \begin{bmatrix} 13.2 \\ 16.8 \end{bmatrix}$

The network learns while staying in the stable region.

### Why This Stabilizes Hyper-Connections

With mHC:
- Skip paths become weighted averages
- Forward signals are preserved
- Backward gradients are preserved
- Repeated depth causes redistribution, not scaling

**The identity is no longer fixed—but its stability is preserved.**

---

## Conclusion

Manifold-Constrained Hyper-Connections represent an elegant mathematical solution to a fundamental problem in deep learning. By constraining the learned routing matrices to lie on the manifold of doubly stochastic matrices, mHC achieves the expressiveness of Hyper-Connections while maintaining the stability guarantees that made ResNets successful.

The key insight is that stability in deep networks requires more than good initialization—it requires constraining the optimization trajectory to remain in a mathematically safe region. The Birkhoff polytope of doubly stochastic matrices provides exactly this: a space where learned routing behaves like "soft permutations" that redistribute information without amplification or decay.

As we traced through our example with $x = \begin{bmatrix} 10 \\ 20 \end{bmatrix}$ and the matrix $H = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}$, we saw how the same values flow through ResNets, Hyper-Connections, and finally mHC—each building on the last to achieve greater expressiveness without sacrificing stability.

This approach opens new possibilities for designing very deep architectures with learned skip connections, potentially enabling more sophisticated information routing in future transformer and neural network architectures.

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)

3. Zhu, Y., et al. (2024). Hyper-Connections. [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)

4. Zhang, K., et al. (2024). Manifold-Constrained Hyper-Connections for Stable Deep Learning. [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
