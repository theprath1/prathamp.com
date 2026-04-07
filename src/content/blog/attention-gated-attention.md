---
title: "Gated Attention: Replacing Residuals and ReLU with Learned Gates"
description: "Building gated transformer blocks from the ground up — why standard residual connections and ReLU activations leave performance on the table, identity map reordering (pre-norm), five gating variants from input gating to GRU-type gates, gated identity initialization, GLU and its variants (SwiGLU, GEGLU, ReGLU, Bilinear), the 2/3 parameter budget trick, and the unified view of gating as multiplicative control — all derived step by step with a 4-dimensional running example."
date: 2026-04-07
tags: [machine-learning, attention, transformers, gating, gtrxl, swiglu, geglu, glu, efficiency]
order: 1
---

The previous three blogs derived methods for reducing the attention pattern — which tokens attend to which. Sparse Transformer (Blog 9) used fixed factorized patterns at $O(n\sqrt{n})$. Longformer (Blog 10) used sliding windows plus global tokens at $O(n)$. DeepSeek Sparse Attention (Blog 11) learned the pattern itself at $O(nk)$.

All three blogs lived on Axis 3 of our taxonomy: the attention pattern. They modified the mask $M$ in the attention formula. The attention mechanism itself — the projections, the softmax, the weighted sum — remained unchanged. And the wrapper around the attention block — the residual connection, the normalization, the feed-forward network — was untouched entirely.

This blog moves to Axis 5: the layer-level architecture. We keep the attention computation exactly as it is and instead modify two things that surround it:

1. **The residual connection** — replacing the fixed skip connection $y = x + f(x)$ with a learned gate $y = g(x, f(x))$ that controls how much of the submodule output to let through
2. **The FFN activation** — replacing the ReLU activation with a gated linear unit that multiplies two parallel linear transformations, one of which acts as a learned gate

The first modification comes from the Gated Transformer-XL (GTrXL) paper (Parisotto, Song, Rae et al., 2019), which showed that gating the residual connections stabilizes transformer training in reinforcement learning — an environment where standard transformers completely fail to learn. The second comes from Shazeer (2020), who showed that replacing ReLU with Gated Linear Unit variants (SwiGLU, GEGLU) in the FFN sublayer improves language model quality at matched parameter and compute budgets.

The common thread is **multiplicative gating**: instead of additive combination (residuals) or pointwise activation (ReLU), these methods use element-wise products where one factor is a sigmoid or similar function that learns to selectively pass or suppress information. This is the same principle that made LSTMs trainable and Highway Networks deep. We are applying it to the transformer block.

We will derive every gating variant from scratch, trace the forward pass with concrete numbers, and verify parameter counts numerically.

---

## The Running Example

We use a single token's hidden state as it flows through one transformer layer. Fix the model parameters from the series:

- $d_\text{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$, $L = 12$ layers, fp16

For numerical derivations, we work with a tiny $d = 4$ hidden state to make every computation tractable by hand:

$$
x = \begin{pmatrix} 1.0 \\ -0.5 \\ 0.2 \\ 0.8 \end{pmatrix}
$$

This vector $x$ represents the hidden state of a single token entering a transformer sub-block (either the attention sub-block or the FFN sub-block). We will trace how different gating mechanisms transform it.

We also fix a submodule output — the result of the attention or FFN computation on $x$:

$$
y = \begin{pmatrix} 0.3 \\ 0.7 \\ -0.4 \\ 0.1 \end{pmatrix}
$$

So $x$ is the residual stream and $y = f(x)$ is the submodule output. The question this entire blog asks is: how should we combine $x$ and $y$ to produce the updated hidden state?

---

## 1. The Standard Residual Connection

### 1.1 The formula

In a standard transformer, the residual connection combines $x$ and $y$ by simple addition:

$$
\text{output} = x + y
$$

This is the **identity shortcut** introduced by He et al. (2016a) for ResNets. The gradient flows through the addition unchanged — $\frac{\partial (x + y)}{\partial x} = I$ — which prevents vanishing gradients in deep networks.

### 1.2 Numerical example

$$
\text{output} = \begin{pmatrix} 1.0 \\ -0.5 \\ 0.2 \\ 0.8 \end{pmatrix} + \begin{pmatrix} 0.3 \\ 0.7 \\ -0.4 \\ 0.1 \end{pmatrix} = \begin{pmatrix} 1.3 \\ 0.2 \\ -0.2 \\ 0.9 \end{pmatrix}
$$

Every dimension of $y$ is added with equal weight of 1. There is no mechanism for the network to say "dimension 2 of the submodule output is useful but dimension 3 is noise — keep the first, suppress the second." The addition is unconditional.

### 1.3 Why this becomes a problem

For supervised learning on large, well-curated datasets, the standard residual connection works well. But in reinforcement learning, two problems emerge:

**Training instability.** RL gradients are inherently noisy — the reward signal is sparse, delayed, and highly variable across episodes. The submodule output $y$ can be large and poorly directed early in training. Adding it unconditionally to the residual stream can destabilize the hidden state, causing policy collapse or divergent losses.

**No selective filtering.** When a transformer is used as a memory for an RL agent, different layers contribute information of varying quality. Lower layers may have learned useful local features while upper layers are still producing random outputs. The standard residual forces the agent to accept all contributions equally.

Parisotto et al. (2019) found that the canonical Transformer-XL (TrXL), trained with V-MPO on the DMLab-30 multitask RL benchmark, achieved a mean human-normalized score of only $5.0 \pm 0.2$ — essentially random. The LSTM baseline achieved $99.3 \pm 1.0$. The transformer completely failed to learn.

---

## 2. Identity Map Reordering (Pre-Norm)

### 2.1 The canonical transformer block

The original Transformer (Vaswani et al., 2017) applies layer normalization after the residual connection. For the attention sub-block, the computation is:

$$
\bar{Y}^{(l)} = \text{MultiHeadAttention}(E^{(l-1)})
$$

$$
\hat{Y}^{(l)} = E^{(l-1)} + \bar{Y}^{(l)}
$$

$$
Y^{(l)} = \text{LayerNorm}(\hat{Y}^{(l)})
$$

This is **post-norm**: normalize after the residual addition. The residual path from input to output passes through $L$ layer normalization operations, one at each layer. Each LayerNorm is a nonlinear function (it divides by the standard deviation), so the path from the first layer's input to the last layer's output is a composition of $L$ nonlinear transformations. There is no clean identity map.

### 2.2 The reordering

**Identity map reordering**, described by He et al. (2016b) for ResNets and adopted by Radford et al. (2019) and Baevski and Auli (2019) for transformers, moves the layer normalization to the input of each sub-block:

$$
\bar{Y}^{(l)} = \text{MultiHeadAttention}(\text{LayerNorm}(E^{(l-1)}))
$$

$$
Y^{(l)} = E^{(l-1)} + \bar{Y}^{(l)}
$$

This is **pre-norm**: normalize before the submodule, not after. Now the residual connection is truly an identity map — no nonlinear transformations lie on the skip path. The gradient flows from layer $L$ back to layer 0 through pure additions.

### 2.3 Why this matters for stability

Consider the gradient of the loss $\mathcal{L}$ with respect to the input at layer $l$. In the pre-norm formulation, unrolling the residual connections gives:

$$
E^{(L)} = E^{(l)} + \sum_{i=l}^{L-1} f_i(\text{LayerNorm}(E^{(i)}))
$$

Taking the derivative by the **chain rule of calculus**:

$$
\frac{\partial \mathcal{L}}{\partial E^{(l)}} = \frac{\partial \mathcal{L}}{\partial E^{(L)}} \left( I + \sum_{i=l}^{L-1} \frac{\partial f_i}{\partial E^{(l)}} \right)
$$

The $I$ here is the **identity matrix** — the $d_\text{model} \times d_\text{model}$ matrix with ones on the diagonal and zeros everywhere else:

$$
I = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix}
$$

Its defining property: for any vector $v$, $Iv = v$. It is the matrix that does nothing — the identity function in matrix form. It appears here because the derivative of the addition $E^{(l)} + (\text{something})$ with respect to $E^{(l)}$ is exactly $I$ — each dimension of the input passes through to the output with a derivative of 1, and no dimension affects any other dimension.

This $I$ term is the reason the gradient cannot vanish. Even if all the $\frac{\partial f_i}{\partial E^{(l)}}$ terms are small (which they are at initialization, when the submodules produce near-zero outputs), the gradient is at least $\frac{\partial \mathcal{L}}{\partial E^{(L)}} \cdot I = \frac{\partial \mathcal{L}}{\partial E^{(L)}}$ — it passes through unchanged. In post-norm, the LayerNorm operations on the skip path multiply additional **Jacobian** factors (the matrix of all partial derivatives of a vector-valued function) that can shrink or rotate the gradient, breaking this clean pass-through.

### 2.4 The effect on initialization

The pre-norm layout has a second, more subtle benefit for RL. At initialization, the submodule outputs $f_i$ are close to zero (random weights produce near-zero outputs in expectation). So $E^{(L)} \approx E^{(0)}$ — the output of the transformer is approximately the input embedding. This means the agent starts with a near-Markovian policy: it acts based on the current observation, ignoring history. This is a good starting point for RL, because reactive behaviors (responding to what is on screen right now) need to be learned before memory-dependent behaviors (remembering what happened 100 steps ago).

### 2.5 Numerical verification: TrXL-I results

The paper calls the pre-norm Transformer-XL "TrXL-I" (for Identity map reordering). On DMLab-30:

| Model | Mean Human Norm. Score |
|---|---|
| TrXL (post-norm) | $5.0 \pm 0.2$ |
| **TrXL-I (pre-norm)** | $107.0 \pm 1.2$ |
| LSTM | $99.3 \pm 1.0$ |

Pre-norm alone transforms the transformer from a complete failure ($5.0$, essentially random) to superhuman performance ($107.0$, above the human baseline of $100$). This is a $21\times$ improvement from a single architectural change that does not add a single parameter.

### 2.6 Interpretation

Identity map reordering is not a new idea — it was known in the ResNet literature (He et al., 2016b) and adopted by GPT-2 (Radford et al., 2019). But its effect in RL is dramatic. The reason is that RL's noisy gradients amplify the problems of post-norm: the gradient signal is already weak and variable, and passing it through $L$ nonlinear LayerNorm operations on the skip path can destroy it entirely. Pre-norm removes this amplification.

But pre-norm is not enough. While TrXL-I vastly outperforms TrXL, it is still less stable than the LSTM across hyperparameter settings. The next step is to replace the residual connection itself with a learned gate.

---

## 3. Gating Layers

### 3.1 The general idea

A **gating layer** replaces the residual connection $\text{output} = x + y$ with a function $g(x, y)$ that uses a learned, element-wise multiplicative mechanism to control the flow of information. The gate is typically a sigmoid function $\sigma(\cdot)$ applied to a linear transformation of the inputs, producing values in $[0, 1]$ for each dimension independently.

The key insight, borrowed from LSTMs (Hochreiter and Schmidhuber, 1997), is that multiplicative interactions give the network fine-grained, per-dimension control over information flow. A gate value of 0.9 in dimension $k$ means "let 90% of this signal through." A gate value of 0.1 means "suppress this dimension." The network learns these gate values from data.

### 3.2 The GTrXL block

The final Gated Transformer-XL (GTrXL) block combines identity map reordering with gating layers. For the attention sub-block:

$$
\bar{Y}^{(l)} = \text{RelativeMultiHeadAttention}(\text{LayerNorm}([M^{(l-1)}, E^{(l-1)}]))
$$

$$
Y^{(l)} = g^{(l)}_\text{MHA}(E^{(l-1)}, \text{ReLU}(\bar{Y}^{(l)}))
$$

For the FFN sub-block:

$$
\bar{E}^{(l)} = f^{(l)}(\text{LayerNorm}(Y^{(l)}))
$$

$$
E^{(l)} = g^{(l)}_\text{MLP}(Y^{(l)}, \text{ReLU}(\bar{E}^{(l)}))
$$

Two things changed compared to the standard block. First, the layer normalization is applied to the input (pre-norm). Second, the residual addition $x + y$ is replaced by the gating function $g(x, y)$.

Note the ReLU activation applied to the submodule output before gating. This is because the identity map reordering creates a path where two consecutive linear layers (the submodule output projection and the gating layer's linear transformation) could collapse into a single linear layer. The ReLU breaks this degeneracy.

### 3.3 Five gating variants

The paper ablates five different gating functions, each with increasing expressivity. We derive each one from scratch, trace the computation with our running example ($x$ is the residual stream, $y$ is the submodule output), and count parameters.

For the numerical examples, we need a weight matrix. Fix a small gating weight matrix for $d = 4$:

$$
W_g^{(l)} = \begin{pmatrix} 0.1 & -0.2 & 0.3 & 0.0 \\ 0.0 & 0.4 & -0.1 & 0.2 \\ -0.3 & 0.1 & 0.2 & 0.1 \\ 0.2 & 0.0 & -0.2 & 0.3 \end{pmatrix}
$$

and a bias vector $b_g^{(l)} = (1.0, 1.0, 1.0, 1.0)^\top$ (this large positive bias is the "gated identity initialization" — we will explain why in Section 4).

---

## 3.4 Variant 1: Input Gating

### Definition

The **input gate** applies a sigmoid modulation to the residual stream $x$, then adds the submodule output $y$:

$$
g^{(l)}(x, y) = \sigma(W_g^{(l)} x) \odot x + y
$$

where $\sigma$ is the **logistic sigmoid function** $\sigma(z) = \frac{1}{1 + e^{-z}}$ (derived in the [math prerequisites for RL](/blog/math-prerequisites-for-rl)) and $\odot$ is the **Hadamard product** (element-wise multiplication). Given two vectors $a, b \in \mathbb{R}^d$, the Hadamard product is:

$$
(a \odot b)_i = a_i \cdot b_i, \quad i = 1, \ldots, d
$$

Each dimension is multiplied independently — there is no interaction between dimensions. This is fundamentally different from the dot product $a^\top b = \sum_i a_i b_i$, which collapses $d$ dimensions into a single scalar. The Hadamard product preserves dimensionality: the input is two $d$-vectors, the output is one $d$-vector. It is the operation that makes per-dimension gating possible — each gate value in $[0, 1]$ scales its own dimension independently.

This variant is similar to the short-cut-only gating of He et al. (2016b).

The gate $\sigma(W_g^{(l)} x)$ decides, for each dimension, how much of the residual stream to keep. When the gate is 1 (fully open), $x$ passes through unchanged and $y$ is added — recovering the standard residual. When the gate is 0 (fully closed), $x$ is zeroed out and only $y$ remains.

### Numerical example

First, compute $W_g^{(l)} x$:

$$
W_g^{(l)} x = \begin{pmatrix} 0.1(1.0) + (-0.2)(-0.5) + 0.3(0.2) + 0.0(0.8) \\ 0.0(1.0) + 0.4(-0.5) + (-0.1)(0.2) + 0.2(0.8) \\ (-0.3)(1.0) + 0.1(-0.5) + 0.2(0.2) + 0.1(0.8) \\ 0.2(1.0) + 0.0(-0.5) + (-0.2)(0.2) + 0.3(0.8) \end{pmatrix}
$$

$$
= \begin{pmatrix} 0.1 + 0.1 + 0.06 + 0 \\ 0 - 0.2 - 0.02 + 0.16 \\ -0.3 - 0.05 + 0.04 + 0.08 \\ 0.2 + 0 - 0.04 + 0.24 \end{pmatrix} = \begin{pmatrix} 0.26 \\ -0.06 \\ -0.23 \\ 0.40 \end{pmatrix}
$$

Note: the input gate variant in the paper has no bias term. But we include the computation for the sigmoid. Apply $\sigma$ element-wise (using $\sigma(z) = \frac{1}{1 + e^{-z}}$):

$$
\sigma(0.26) = \frac{1}{1 + e^{-0.26}} = \frac{1}{1 + 0.771} = \frac{1}{1.771} \approx 0.565
$$

$$
\sigma(-0.06) = \frac{1}{1 + e^{0.06}} = \frac{1}{1 + 1.062} = \frac{1}{2.062} \approx 0.485
$$

$$
\sigma(-0.23) = \frac{1}{1 + e^{0.23}} = \frac{1}{1 + 1.259} = \frac{1}{2.259} \approx 0.443
$$

$$
\sigma(0.40) = \frac{1}{1 + e^{-0.40}} = \frac{1}{1 + 0.670} = \frac{1}{1.670} \approx 0.599
$$

Now apply the Hadamard product with $x$ and add $y$:

$$
g(x, y) = \begin{pmatrix} 0.565 \\ 0.485 \\ 0.443 \\ 0.599 \end{pmatrix} \odot \begin{pmatrix} 1.0 \\ -0.5 \\ 0.2 \\ 0.8 \end{pmatrix} + \begin{pmatrix} 0.3 \\ 0.7 \\ -0.4 \\ 0.1 \end{pmatrix} = \begin{pmatrix} 0.565 \\ -0.243 \\ 0.089 \\ 0.479 \end{pmatrix} + \begin{pmatrix} 0.3 \\ 0.7 \\ -0.4 \\ 0.1 \end{pmatrix} = \begin{pmatrix} 0.865 \\ 0.457 \\ -0.311 \\ 0.579 \end{pmatrix}
$$

Compare to the standard residual output of $(1.3, 0.2, -0.2, 0.9)^\top$. The gate has scaled down the residual stream contribution — dimension 3, for example, kept only 44.3% of $x_3$ instead of the full value, while dimension 1 kept 56.5%.

### Parameters

Input gating adds one $d_\text{model} \times d_\text{model}$ weight matrix per gate. Each transformer layer has two gates (one for MHA, one for MLP):

$$
\text{Params per layer} = 2 \times d_\text{model}^2 = 2 \times 512^2 = 524{,}288
$$

Across $L = 12$ layers:

$$
\text{Total gating params} = 12 \times 524{,}288 = 6{,}291{,}456 \approx 6.3\text{M}
$$

---

## 3.5 Variant 2: Output Gating

### Definition

The **output gate** applies a sigmoid modulation to the submodule output $y$ instead of the residual stream:

$$
g^{(l)}(x, y) = x + \sigma(W_g^{(l)} x - b_g^{(l)}) \odot y
$$

The gate controls how much of the submodule's contribution to let through. When the gate is 0, $g(x, y) = x$ — the submodule is completely ignored and the residual stream passes through unchanged. When the gate is 1, $g(x, y) = x + y$ — the standard residual connection is recovered.

The minus sign on the bias $b_g^{(l)}$ is a convention: with $b_g^{(l)} > 0$, the sigmoid input is shifted negative, biasing the gate toward 0 (closed). This implements a conservative initialization where new layers start by doing nothing.

### Numerical example

Compute $W_g^{(l)} x - b_g^{(l)}$:

$$
W_g^{(l)} x - b_g^{(l)} = \begin{pmatrix} 0.26 \\ -0.06 \\ -0.23 \\ 0.40 \end{pmatrix} - \begin{pmatrix} 1.0 \\ 1.0 \\ 1.0 \\ 1.0 \end{pmatrix} = \begin{pmatrix} -0.74 \\ -1.06 \\ -1.23 \\ -0.60 \end{pmatrix}
$$

Apply $\sigma$:

$$
\sigma(-0.74) \approx 0.323, \quad \sigma(-1.06) \approx 0.257, \quad \sigma(-1.23) \approx 0.226, \quad \sigma(-0.60) \approx 0.354
$$

The gate values are all well below 0.5 — the positive bias pushes the gate toward closed. Now compute the output:

$$
g(x, y) = \begin{pmatrix} 1.0 \\ -0.5 \\ 0.2 \\ 0.8 \end{pmatrix} + \begin{pmatrix} 0.323 \\ 0.257 \\ 0.226 \\ 0.354 \end{pmatrix} \odot \begin{pmatrix} 0.3 \\ 0.7 \\ -0.4 \\ 0.1 \end{pmatrix} = \begin{pmatrix} 1.0 \\ -0.5 \\ 0.2 \\ 0.8 \end{pmatrix} + \begin{pmatrix} 0.097 \\ 0.180 \\ -0.090 \\ 0.035 \end{pmatrix} = \begin{pmatrix} 1.097 \\ -0.320 \\ 0.110 \\ 0.835 \end{pmatrix}
$$

The output is much closer to $x$ than the standard residual $(1.3, 0.2, -0.2, 0.9)^\top$. The gate is letting through only 22–35% of each dimension of $y$. This is the conservative initialization at work: early in training, the layer barely modifies the residual stream.

### Parameters

Same as input gating: one $d_\text{model} \times d_\text{model}$ weight matrix plus one $d_\text{model}$ bias vector per gate. The bias is negligible:

$$
\text{Params per layer} \approx 2 \times (d_\text{model}^2 + d_\text{model}) = 2 \times (262{,}144 + 512) = 525{,}312
$$

---

## 3.6 Variant 3: Highway

### Definition

The **Highway connection** (Srivastava et al., 2015) modulates both streams with a single gate. When the gate opens for $y$, it closes for $x$, and vice versa:

$$
g^{(l)}(x, y) = \sigma(W_g^{(l)} x + b_g^{(l)}) \odot x + (1 - \sigma(W_g^{(l)} x + b_g^{(l)})) \odot y
$$

Let $s = \sigma(W_g^{(l)} x + b_g^{(l)})$. Then:

$$
\boxed{g^{(l)}(x, y) = s \odot x + (1 - s) \odot y}
$$

This is a **convex combination**. In general, a convex combination of two values $a$ and $b$ with weight $\lambda \in [0, 1]$ is:

$$
\lambda \, a + (1 - \lambda) \, b
$$

The defining property: the weights are non-negative and sum to 1 ($\lambda + (1 - \lambda) = 1$). This guarantees that the result always lies "between" $a$ and $b$ — for scalars, literally on the line segment from $b$ to $a$; for vectors, in the convex hull of the two endpoints. No matter what $s$ is, the output cannot exceed both $x$ and $y$ in any dimension, nor fall below both. It is a constrained interpolation.

Here, the weight $s$ is determined per-dimension by the gate. When $s_i = 1$, dimension $i$ of the output is $x_i$ (skip the submodule entirely). When $s_i = 0$, it is $y_i$ (use only the submodule). When $s_i = 0.5$, it is the midpoint $\frac{x_i + y_i}{2}$.

This is exactly the gating mechanism of Highway Networks, which were the first architectures to successfully train networks with hundreds of layers — predating ResNets.

### Numerical example

Compute $s = \sigma(W_g^{(l)} x + b_g^{(l)})$:

$$
W_g^{(l)} x + b_g^{(l)} = \begin{pmatrix} 0.26 \\ -0.06 \\ -0.23 \\ 0.40 \end{pmatrix} + \begin{pmatrix} 1.0 \\ 1.0 \\ 1.0 \\ 1.0 \end{pmatrix} = \begin{pmatrix} 1.26 \\ 0.94 \\ 0.77 \\ 1.40 \end{pmatrix}
$$

$$
s = \sigma\begin{pmatrix} 1.26 \\ 0.94 \\ 0.77 \\ 1.40 \end{pmatrix} \approx \begin{pmatrix} 0.779 \\ 0.719 \\ 0.684 \\ 0.802 \end{pmatrix}
$$

The positive bias pushes $s$ toward 1, so the gate favors keeping the residual stream $x$.

$$
g(x, y) = \begin{pmatrix} 0.779 \\ 0.719 \\ 0.684 \\ 0.802 \end{pmatrix} \odot \begin{pmatrix} 1.0 \\ -0.5 \\ 0.2 \\ 0.8 \end{pmatrix} + \begin{pmatrix} 0.221 \\ 0.281 \\ 0.316 \\ 0.198 \end{pmatrix} \odot \begin{pmatrix} 0.3 \\ 0.7 \\ -0.4 \\ 0.1 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0.779 \\ -0.360 \\ 0.137 \\ 0.642 \end{pmatrix} + \begin{pmatrix} 0.066 \\ 0.197 \\ -0.126 \\ 0.020 \end{pmatrix} = \begin{pmatrix} 0.845 \\ -0.163 \\ 0.011 \\ 0.662 \end{pmatrix}
$$

### Numerical check: convex combination

For dimension 1: $s_1 = 0.779$, so the output should satisfy $g_1 = 0.779 \times 1.0 + 0.221 \times 0.3 = 0.779 + 0.066 = 0.845$. $\checkmark$

This verifies the convex combination property: the output for each dimension lies between $x_i$ and $y_i$ (or at one of them).

### The constraint

The Highway gate imposes a hard constraint: the total weight on $x$ and $y$ must sum to 1 in each dimension. If the gate lets more of $x$ through, it must proportionally reduce $y$. This is more structured than the input or output gates, which can independently scale each stream. Whether this constraint helps or hurts depends on the task.

### Parameters

Same as output gating: $d_\text{model}^2 + d_\text{model}$ per gate.

---

## 3.7 Variant 4: Sigmoid-Tanh (SigTanh)

### Definition

The **sigmoid-tanh gate** (Van den Oord et al., 2016) is similar to the output gate but adds a tanh activation on a separate linear projection of $y$:

$$
g^{(l)}(x, y) = x + \sigma(W_g^{(l)} y - b^{(l)}) \odot \tanh(U_g^{(l)} y)
$$

Note that both the sigmoid and the tanh operate on $y$, not $x$. The sigmoid $\sigma(W_g^{(l)} y - b^{(l)})$ controls how much to add, while $\tanh(U_g^{(l)} y)$ is a re-projected and bounded version of $y$. This is the gating mechanism used in WaveNet and PixelCNN.

### Why tanh?

The $\tanh$ function squashes its input to $[-1, 1]$, which serves two purposes. First, it bounds the magnitude of the update — unlike the output gate where $y$ can be arbitrarily large. Second, the separate projection $U_g^{(l)}$ allows the gated update to be in a different subspace than the raw submodule output $y$.

### Parameters

This variant has two $d_\text{model} \times d_\text{model}$ weight matrices per gate ($W_g$ and $U_g$), plus bias vectors:

$$
\text{Params per layer} = 2 \times (2 \times d_\text{model}^2 + d_\text{model}) \approx 2 \times 2 \times 512^2 = 1{,}048{,}576
$$

Double the parameters of the simpler variants.

---

## 3.8 Variant 5: GRU-Type Gating

### Definition

The most expressive variant adapts the **Gated Recurrent Unit** (GRU) (Chung et al., 2014) as a gating function. The GRU is a recurrent architecture that simplifies the LSTM by using two gates instead of three. Here it is applied as a depth-wise (layer-to-layer) gate rather than a time-wise (step-to-step) gate:

$$
r = \sigma(W_r^{(l)} y + U_r^{(l)} x)
$$

$$
z = \sigma(W_z^{(l)} y + U_z^{(l)} x - b_g^{(l)})
$$

$$
\hat{h} = \tanh(W_g^{(l)} y + U_g^{(l)} (r \odot x))
$$

$$
\boxed{g^{(l)}(x, y) = (1 - z) \odot x + z \odot \hat{h}}
$$

Let us trace what each component does:

- $r$ is the **reset gate**: it controls how much of the residual stream $x$ is visible when computing the candidate update $\hat{h}$. When $r \approx 0$, the candidate is computed from $y$ alone, ignoring $x$. When $r \approx 1$, the full $x$ is available.
- $z$ is the **update gate**: it controls the interpolation between $x$ and the candidate $\hat{h}$. This is the same convex combination as the Highway gate, but the "new value" $\hat{h}$ is a more complex function of both $x$ and $y$.
- $\hat{h}$ is the **candidate update**: a tanh-bounded combination of $y$ and the reset-gated $x$.

The final output is a convex combination of the old state $x$ and the candidate $\hat{h}$, weighted by $z$.

### Why this is the most expressive variant

The GRU gate has three matrix-vector products involving $y$ ($W_r y$, $W_z y$, $W_g y$) and three involving $x$ ($U_r x$, $U_z x$, $U_g x$), for a total of six $d_\text{model} \times d_\text{model}$ matrices per gate. It can represent all of the simpler variants as special cases:

- Setting $r = 1$ and making $\hat{h} = y$ recovers the Highway gate (with $z$ as the Highway's gate)
- Setting $z$ to a fixed small value recovers something close to the output gate
- The tanh on $\hat{h}$ bounds the update, like the SigTanh variant

### Parameters per gate

$$
6 \times d_\text{model}^2 + \text{biases} = 6 \times 512^2 = 1{,}572{,}864
$$

Per layer (two gates):

$$
2 \times 1{,}572{,}864 = 3{,}145{,}728
$$

Across 12 layers:

$$
12 \times 3{,}145{,}728 = 37{,}748{,}736 \approx 37.7\text{M}
$$

### Parameter count comparison

| Gating variant | Matrices per gate | Params per layer | Total (12 layers) |
|---|---|---|---|
| Input | 1 | 524K | 6.3M |
| Output | 1 | 525K | 6.3M |
| Highway | 1 | 525K | 6.3M |
| SigTanh | 2 | 1.05M | 12.6M |
| **GRU** | **6** | **3.15M** | **37.7M** |

For a baseline TrXL with approximately 28.6M parameters (12 layers, $d_\text{model} = 256$, 8 heads, $d_k = 64$), the GRU gating adds 37.7M parameters — more than the base model. The paper addresses this by testing a "Thin GTrXL" variant with halved embedding dimension, which we discuss in Section 5.

---

## 4. Gated Identity Initialization

### 4.1 The motivation

We have argued that pre-norm (identity map reordering) helps because the initial transformer acts like an identity function — the randomly initialized submodules contribute near-zero outputs, so $E^{(L)} \approx E^{(0)}$. But the gating variants introduce new parameters ($W_g$, $b_g$) that, if randomly initialized, will produce gate values near $\sigma(0) = 0.5$. This means the gate is "half open" from the start, which partially disrupts the identity property.

**Gated identity initialization** explicitly sets the bias $b_g^{(l)}$ to a positive value so that the gate starts near the identity function. The specific value depends on the gating variant.

### 4.2 How it works for each variant

**Output gate**: $g(x, y) = x + \sigma(W_g x - b_g) \odot y$. Setting $b_g > 0$ makes $\sigma(W_g x - b_g) \approx \sigma(-b_g) \approx 0$ at initialization (since $W_g x \approx 0$ for random $W_g$). So $g(x, y) \approx x + 0 \cdot y = x$. The gate starts closed: the submodule output is suppressed.

**Highway gate**: $g(x, y) = s \odot x + (1 - s) \odot y$ where $s = \sigma(W_g x + b_g)$. Setting $b_g > 0$ makes $s \approx \sigma(b_g) \approx 1$. So $g(x, y) \approx 1 \cdot x + 0 \cdot y = x$. Same effect: identity.

**GRU gate**: $g(x, y) = (1 - z) \odot x + z \odot \hat{h}$. Setting $b_g > 0$ in $z = \sigma(W_z y + U_z x - b_g)$ makes $z \approx 0$. So $g(x, y) \approx x$. Identity again.

### 4.3 Numerical verification

For the output gate with $b_g = 2$ (the value used in the paper for GRU gating), at initialization where $W_g x \approx 0$:

$$
\sigma(-b_g) = \sigma(-2) = \frac{1}{1 + e^2} = \frac{1}{1 + 7.389} = \frac{1}{8.389} \approx 0.119
$$

So each dimension of $y$ is scaled by approximately 0.119 — the submodule's contribution is reduced to about 12% of its value. For $b_g = 1$:

$$
\sigma(-1) = \frac{1}{1 + e^1} = \frac{1}{1 + 2.718} = \frac{1}{3.718} \approx 0.269
$$

About 27% passes through. The paper uses $b_g = 2$ for GRU gating and $b_g = 1$ for other variants.

### 4.4 The effect on learning speed

The paper ablates the gated identity initialization on the Memory Maze task using the GRU-gated GTrXL. With $b_g = 2$, the model reaches human-level performance ($\sim 8$ reward) with 10 out of 10 hyperparameter settings by 4B environment steps. Without the bias ($b_g = 0$), only 2 out of 10 settings reach human level, and the rest plateau below 4 reward.

The mechanism is clear: without the identity bias, the randomly initialized gates produce gate values near 0.5 from the start. This means the untrained submodule outputs immediately corrupt the residual stream with noise. With the bias, the gates start nearly closed, so the network begins as an approximately Markovian policy and gradually opens the gates as the submodules learn useful transformations.

---

## 5. The Full GTrXL Results

### 5.1 DMLab-30 performance

The paper evaluates all gating variants on the DMLab-30 multitask RL suite. All transformer variants use 12 layers, $d_\text{model} = 256$, 8 heads, $d_k = 64$, and memory size 512:

| Model | Mean Human Norm. | 100-capped |
|---|---|---|
| LSTM (3-layer) | $99.3 \pm 1.0$ | $84.0 \pm 0.4$ |
| TrXL (post-norm) | $5.0 \pm 0.2$ | $5.0 \pm 0.2$ |
| TrXL-I (pre-norm) | $107.0 \pm 1.2$ | $87.4 \pm 0.3$ |
| GTrXL (Input) | $51.2 \pm 13.2$ | $47.6 \pm 12.1$ |
| GTrXL (Output) | $112.8 \pm 0.8$ | $87.8 \pm 0.3$ |
| GTrXL (Highway) | $90.9 \pm 12.9$ | $75.2 \pm 10.4$ |
| GTrXL (SigTanh) | $101.0 \pm 1.3$ | $83.9 \pm 0.7$ |
| **GTrXL (GRU)** | **$117.6 \pm 0.3$** | **$89.1 \pm 0.2$** |
| MERLIN@100B | 115.2 | 89.4 |

Several observations:

**GRU gating is the clear winner.** It achieves $117.6$ mean human-normalized score, beating the LSTM ($99.3$) by 18 points and exceeding even MERLIN ($115.2$), an external memory architecture that was trained for $10\times$ more environment steps (100B vs 10B).

**Input gating fails.** At $51.2 \pm 13.2$, it performs worse than TrXL-I without any gating ($107.0$). This is because input gating modulates the residual stream before adding the submodule output, which disrupts the identity path. The gate suppresses parts of $x$ that may be important, and the raw $y$ is added without any filtering.

**Output gating and Highway have opposite stability profiles.** Output gating is strong ($112.8$) with low variance ($\pm 0.8$). Highway gating has a comparable best case but much higher variance ($\pm 12.9$), indicating sensitivity to hyperparameters.

**Standard error matters.** The GRU variant's standard error of $\pm 0.3$ is the smallest of all models. This means it is not just the highest-performing but also the most robust across different hyperparameter settings and random seeds.

### 5.2 Parameter-controlled comparison

The GRU gating adds substantial parameters ($64.4$M total vs $28.6$M for TrXL). To verify that the improvement is not simply from added capacity, the paper tests a "Thin GTrXL (GRU)" with halved embedding dimension ($d_\text{model} = 128$, 4 heads), giving $22.4$M total parameters — fewer than the baseline TrXL.

| Model | Params | Mean Human Norm. |
|---|---|---|
| TrXL | 28.6M | $5.0 \pm 0.2$ |
| TrXL-I | 28.6M | $107.0 \pm 1.2$ |
| **Thin GTrXL (GRU)** | **22.4M** | **$111.5 \pm 0.6$** |
| GTrXL (Output) | 34.9M | $112.8 \pm 0.8$ |
| GTrXL (GRU) | 66.4M | $117.6 \pm 0.3$ |

The Thin GTrXL achieves $111.5$ with 22.4M parameters — fewer parameters than any other transformer variant, yet it matches the best-performing GTrXL (Output) at $112.8$ and beats every non-GRU gating variant. This confirms that the GRU's advantage comes from the gating mechanism itself, not from parameter count.

### 5.3 Divergence rates

The paper tracks how often each model's training loss diverges to infinity across 25 random hyperparameter settings on the Memory Maze task:

| Model | % Diverged |
|---|---|
| LSTM | 0% |
| TrXL | 0% |
| TrXL-I | 16% |
| GTrXL (GRU) | **0%** |
| GTrXL (Output) | 12% |

The GRU-gated GTrXL never diverges — matching the LSTM's stability — while TrXL-I diverges 16% of the time. The GRU gate provides both higher performance and greater stability.

### 5.4 Scaling with memory horizon

On the Numpad task, which requires memorizing sequences of increasing length, the LSTM's performance degrades sharply as the pad size increases from 2 to 4. The GTrXL (GRU) maintains strong performance at all sizes and "almost instantly solves the environment" at pad sizes 2 and 3, demonstrating superior memory capacity.

---

## 6. GLU: Gating the Feed-Forward Network

We now turn to the second paper: "GLU Variants Improve Transformer" (Shazeer, 2020). Where GTrXL applied gating to the residual connections (the wrapper around submodules), GLU applies gating inside the FFN submodule itself — replacing the activation function.

### 6.1 The standard FFN

The standard Transformer FFN (Vaswani et al., 2017) for a single token's hidden state $x \in \mathbb{R}^{d_\text{model}}$ is:

$$
\text{FFN}(x) = \max(0, \, x W_1 + b_1) \, W_2 + b_2
$$

where $W_1 \in \mathbb{R}^{d_\text{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_\text{model}}$, and $d_{ff}$ is the FFN hidden dimension. Typically $d_{ff} = 4 \, d_\text{model}$.

The activation is **ReLU**: $\max(0, z)$. It passes positive values unchanged and zeros out negative values. There is no learned control over which dimensions are active — the decision is made purely by the sign of the pre-activation.

Following T5 (Raffel et al., 2019), we use a bias-free version:

$$
\text{FFN}_\text{ReLU}(x, W_1, W_2) = \max(x W_1, 0) \, W_2
$$

### 6.2 Parameters in the standard FFN

Two weight matrices:

$$
\text{Params} = d_\text{model} \times d_{ff} + d_{ff} \times d_\text{model} = 2 \, d_\text{model} \, d_{ff}
$$

With $d_\text{model} = 512$ and $d_{ff} = 4 \times 512 = 2{,}048$:

$$
\text{Params} = 2 \times 512 \times 2{,}048 = 2{,}097{,}152 \approx 2.1\text{M per layer}
$$

### 6.3 Other activations: GELU and Swish

Before introducing gating, two other activation functions were proposed as ReLU replacements:

**GELU** (Gaussian Error Linear Unit, Hendrycks and Gimpel, 2016):

$$
\text{GELU}(z) = z \, \Phi(z)
$$

where $\Phi(z)$ is the standard Gaussian CDF. This can be seen as a smooth approximation to ReLU that weights each value by its probability of being positive under a Gaussian distribution.

**Swish** (Ramachandran et al., 2017):

$$
\text{Swish}_\beta(z) = z \, \sigma(\beta z)
$$

where $\sigma$ is the logistic sigmoid and $\beta$ is a parameter (typically $\beta = 1$). Swish is similar to GELU and was found by neural architecture search.

Both replace the hard zero of ReLU with a smooth, non-monotonic function that allows small negative values through. Importantly, both have the form $z \cdot (\text{something})$ — they multiply the input by a function of the input. This is already a form of self-gating, but with a single linear transformation.

---

## 7. The Gated Linear Unit (GLU)

### 7.1 Definition

The **Gated Linear Unit** (Dauphin et al., 2016) is a neural network layer defined as:

$$
\boxed{\text{GLU}(x, W, V) = \sigma(xW) \odot (xV)}
$$

where $W, V \in \mathbb{R}^{d_\text{model} \times d_{ff}}$ are two separate weight matrices, $\sigma$ is the sigmoid function, and $\odot$ is the Hadamard product.

The two linear projections $xW$ and $xV$ compute two different views of the input. The first, $\sigma(xW)$, produces gate values in $[0, 1]$ — it decides, for each dimension of the hidden representation, how much information to let through. The second, $xV$, produces the actual values to be gated.

### 7.2 Why this is fundamentally different from ReLU

In the ReLU FFN, a single linear projection $xW_1$ is computed, and ReLU decides which dimensions to keep based solely on sign. Positive values pass, negative values are zeroed. The gating decision is:

$$
\text{ReLU gate} = \begin{cases} 1 & \text{if } (xW_1)_i > 0 \\ 0 & \text{if } (xW_1)_i \leq 0 \end{cases}
$$

This is a hard, binary, input-independent (for a fixed $W_1$) decision. The network has no way to say "this dimension is positive but I want to scale it down to 30%."

In the GLU, the gating decision is a separate learned function $\sigma(xW)$, which can produce any value in $(0, 1)$:

$$
\text{GLU gate} = \sigma(xW) \in (0, 1)^{d_{ff}}
$$

This is a soft, continuous, learned decision. The gate is computed from a different projection than the value — so the network can learn that certain input patterns should produce high gate values even when the value projection is small, or vice versa.

### 7.3 Numerical example

Use $d = 4$ and $d_{ff} = 3$ for tractability. Fix:

$$
x = \begin{pmatrix} 1.0 & -0.5 & 0.2 & 0.8 \end{pmatrix}
$$

$$
W = \begin{pmatrix} 0.5 & -0.3 & 0.1 \\ 0.2 & 0.4 & -0.2 \\ -0.1 & 0.6 & 0.3 \\ 0.3 & -0.1 & 0.4 \end{pmatrix}, \quad V = \begin{pmatrix} -0.2 & 0.4 & 0.1 \\ 0.3 & -0.1 & 0.5 \\ 0.1 & 0.2 & -0.3 \\ -0.4 & 0.3 & 0.2 \end{pmatrix}
$$

Compute $xW$ (the gate projection):

$$
xW = \begin{pmatrix} 1.0(0.5) + (-0.5)(0.2) + 0.2(-0.1) + 0.8(0.3) \\ 1.0(-0.3) + (-0.5)(0.4) + 0.2(0.6) + 0.8(-0.1) \\ 1.0(0.1) + (-0.5)(-0.2) + 0.2(0.3) + 0.8(0.4) \end{pmatrix}^\top
$$

$$
= \begin{pmatrix} 0.5 - 0.1 - 0.02 + 0.24 \\ -0.3 - 0.2 + 0.12 - 0.08 \\ 0.1 + 0.1 + 0.06 + 0.32 \end{pmatrix}^\top = \begin{pmatrix} 0.62 & -0.46 & 0.58 \end{pmatrix}
$$

Apply sigmoid: $\sigma(xW) = (\sigma(0.62), \sigma(-0.46), \sigma(0.58)) \approx (0.650, 0.387, 0.641)$.

Compute $xV$ (the value projection):

$$
xV = \begin{pmatrix} 1.0(-0.2) + (-0.5)(0.3) + 0.2(0.1) + 0.8(-0.4) \\ 1.0(0.4) + (-0.5)(-0.1) + 0.2(0.2) + 0.8(0.3) \\ 1.0(0.1) + (-0.5)(0.5) + 0.2(-0.3) + 0.8(0.2) \end{pmatrix}^\top
$$

$$
= \begin{pmatrix} -0.2 - 0.15 + 0.02 - 0.32 \\ 0.4 + 0.05 + 0.04 + 0.24 \\ 0.1 - 0.25 - 0.06 + 0.16 \end{pmatrix}^\top = \begin{pmatrix} -0.65 & 0.73 & -0.05 \end{pmatrix}
$$

Apply the Hadamard product:

$$
\text{GLU}(x) = \sigma(xW) \odot xV = \begin{pmatrix} 0.650 \\ 0.387 \\ 0.641 \end{pmatrix} \odot \begin{pmatrix} -0.65 \\ 0.73 \\ -0.05 \end{pmatrix} = \begin{pmatrix} -0.423 \\ 0.283 \\ -0.032 \end{pmatrix}
$$

The gate has scaled each value dimension independently. Dimension 1 had a high gate value (0.650) so the negative value $-0.65$ mostly passes through. Dimension 2 had a lower gate (0.387), reducing the positive value $0.73$ to $0.283$. Dimension 3 was nearly zeroed: even though the gate was fairly open (0.641), the value itself was tiny ($-0.05$).

### 7.4 The Bilinear variant

Dauphin et al. (2016) also suggest dropping the sigmoid entirely, creating the **Bilinear** layer:

$$
\text{Bilinear}(x, W, V) = (xW) \odot (xV)
$$

No activation at all — just the element-wise product of two linear projections. Despite the absence of a nonlinearity, the Hadamard product itself is a nonlinear operation (it is bilinear in the two projections, but nonlinear in $x$). This is an important observation: the gating structure provides nonlinearity even without sigmoid or tanh.

---

## 8. GLU Variants in the Transformer FFN

### 8.1 The FFN with GLU

Replacing ReLU with GLU in the FFN gives:

$$
\text{FFN}_\text{GLU}(x, W, V, W_2) = (\sigma(xW) \odot xV) \, W_2
$$

There are now three weight matrices instead of two: $W \in \mathbb{R}^{d_\text{model} \times d_{ff}}$ (gate projection), $V \in \mathbb{R}^{d_\text{model} \times d_{ff}}$ (value projection), and $W_2 \in \mathbb{R}^{d_{ff} \times d_\text{model}}$ (output projection).

### 8.2 The full family of variants

Shazeer (2020) systematically replaces the sigmoid in GLU with other activation functions:

$$
\text{FFN}_\text{GLU}(x, W, V, W_2) = (\sigma(xW) \odot xV) \, W_2
$$

$$
\text{FFN}_\text{Bilinear}(x, W, V, W_2) = (xW \odot xV) \, W_2
$$

$$
\text{FFN}_\text{ReGLU}(x, W, V, W_2) = (\max(0, xW) \odot xV) \, W_2
$$

$$
\text{FFN}_\text{GEGLU}(x, W, V, W_2) = (\text{GELU}(xW) \odot xV) \, W_2
$$

$$
\text{FFN}_\text{SwiGLU}(x, W, V, W_2) = (\text{Swish}_1(xW) \odot xV) \, W_2
$$

Each variant uses a different activation on the gate branch while keeping the value branch linear. The general pattern is:

$$
\boxed{\text{FFN}_\text{*GLU}(x) = (\text{activation}(xW) \odot xV) \, W_2}
$$

### 8.3 The $\frac{2}{3}$ parameter budget trick

This is the part that makes GLU variants practical. The standard FFN has two matrices totaling $2 \, d_\text{model} \, d_{ff}$ parameters. The GLU variants have three matrices totaling $3 \, d_\text{model} \, d_{ff}$ parameters — a 50% increase.

To match the parameter count and computation of the original FFN, Shazeer reduces the hidden dimension from $d_{ff}$ to $\frac{2}{3} d_{ff}$:

$$
\text{Standard FFN params} = 2 \, d_\text{model} \, d_{ff}
$$

$$
\text{GLU FFN params} = 3 \, d_\text{model} \, d'_{ff}
$$

Setting these equal:

$$
3 \, d_\text{model} \, d'_{ff} = 2 \, d_\text{model} \, d_{ff}
$$

$$
d'_{ff} = \frac{2}{3} \, d_{ff}
$$

### 8.4 Numerical check

With $d_\text{model} = 768$ (T5-base) and $d_{ff} = 3{,}072$:

**Standard FFN:** $2 \times 768 \times 3{,}072 = 4{,}718{,}592$ parameters per layer.

**GLU variant with $d'_{ff} = \frac{2}{3} \times 3{,}072 = 2{,}048$:** $3 \times 768 \times 2{,}048 = 4{,}718{,}592$ parameters per layer. $\checkmark$

The parameter counts match exactly. The GLU variant has three smaller matrices instead of two larger ones, but the total parameter budget and FLOP count are the same.

### 8.5 Numerical check with our running model

For the running model ($d_\text{model} = 512$, $d_{ff} = 2{,}048$):

**Standard FFN:** $2 \times 512 \times 2{,}048 = 2{,}097{,}152$ per layer.

**GLU variant with $d'_{ff} = \frac{2}{3} \times 2{,}048 \approx 1{,}365$:** $3 \times 512 \times 1{,}365 = 2{,}096{,}640$ per layer.

The small difference ($512$) comes from rounding $\frac{2}{3} \times 2{,}048 = 1{,}365.33$ to $1{,}365$. In practice, $d'_{ff}$ is rounded to a multiple of 64 or 128 for hardware efficiency.

---

## 9. Experimental Results: GLU Variants

### 9.1 Pre-training perplexity

Shazeer evaluates all FFN variants using the T5 setup: encoder-decoder transformer, $d_\text{model} = 768$, 12 layers, 12 heads, trained on C4 with the span-filling denoising objective. All GLU variants use $d'_{ff} = 2{,}048$ to match the baseline's $d_{ff} = 3{,}072$.

| FFN Variant | Log-perplexity (65K steps) | Log-perplexity (524K steps) |
|---|---|---|
| FFN$_\text{ReLU}$ (baseline) | 1.997 | 1.677 |
| FFN$_\text{GELU}$ | 1.983 | 1.679 |
| FFN$_\text{Swish}$ | 1.994 | 1.683 |
| FFN$_\text{GLU}$ | 1.982 | 1.663 |
| FFN$_\text{Bilinear}$ | 1.960 | 1.648 |
| **FFN$_\text{GEGLU}$** | **1.942** | **1.633** |
| **FFN$_\text{SwiGLU}$** | **1.944** | **1.636** |
| FFN$_\text{ReGLU}$ | 1.953 | 1.645 |

The two best variants — GEGLU and SwiGLU — achieve log-perplexities of 1.633 and 1.636 respectively, compared to the ReLU baseline's 1.677. This is a significant improvement: a reduction of 0.044 in log-perplexity at matched parameters and compute.

### 9.2 The ranking

At convergence (524K steps), the ranking from best to worst is:

$$
\text{GEGLU} > \text{SwiGLU} > \text{ReGLU} > \text{Bilinear} > \text{GLU} > \text{ReLU} \approx \text{GELU} \approx \text{Swish}
$$

Two patterns emerge:

**Gating helps.** Every GLU variant (bottom five) outperforms every non-gated variant (top three). The worst GLU variant (GLU at 1.663) beats the best non-gated variant (ReLU at 1.677).

**GELU and Swish gates beat sigmoid and ReLU gates.** Among the GLU variants, GEGLU and SwiGLU are the best. The sigmoid-gated GLU (1.663) is worse than the GELU-gated GEGLU (1.633). This is somewhat surprising — the sigmoid produces values in $(0, 1)$, which is the "correct" range for a gate, while GELU and Swish can produce values outside this range. Apparently, the smooth, non-monotonic shape of GELU and Swish is more important than having outputs bounded to $[0, 1]$.

### 9.3 Fine-tuning results

On GLUE:

| FFN Variant | Score Average |
|---|---|
| FFN$_\text{ReLU}$ | 83.80 |
| FFN$_\text{GEGLU}$ | 84.20 |
| FFN$_\text{SwiGLU}$ | 84.36 |
| FFN$_\text{ReGLU}$ | **84.67** |
| FFN$_\text{Bilinear}$ | 83.79 |

On SuperGLUE:

| FFN Variant | Score Average |
|---|---|
| FFN$_\text{ReLU}$ | 72.76 |
| FFN$_\text{GEGLU}$ | 73.96 |
| FFN$_\text{SwiGLU}$ | 73.66 |
| FFN$_\text{Bilinear}$ | 73.81 |
| FFN$_\text{ReGLU}$ | 73.66 |

On SQuAD v1.1:

| FFN Variant | EM | F1 |
|---|---|---|
| FFN$_\text{ReLU}$ | 83.18 | 90.87 |
| FFN$_\text{Bilinear}$ | **83.82** | 91.06 |
| FFN$_\text{GEGLU}$ | 83.55 | 91.12 |
| FFN$_\text{ReGLU}$ | 83.53 | **91.18** |

The results are noisy across tasks, but the overall pattern is consistent: GLU variants match or slightly exceed the ReLU baseline on every downstream benchmark, while also achieving better pre-training perplexity. As Shazeer concludes: "These architectures are simple to implement, and have no apparent computational drawbacks."

### 9.4 Why SwiGLU became standard

Since this paper, SwiGLU has been adopted by LLaMA (Touvron et al., 2023), PaLM (Chowdhery et al., 2022), and most subsequent large language models. The $\frac{2}{3}$ parameter trick makes it a drop-in replacement for the standard FFN, and the consistent improvements in perplexity translate to downstream quality gains at scale. It is now the default FFN activation in modern transformers.

---

## 10. The Unified View: Gating as Multiplicative Control

### 10.1 The common structure

Every gating mechanism we have derived in this blog shares a single structural motif: an element-wise product where one factor acts as a learned controller.

$$
\text{output} = \text{controller}(\cdot) \odot \text{content}(\cdot)
$$

The controller produces values that modulate the content, dimension by dimension. The differences lie in what the controller sees, what activation it uses, and what the content is:

| Mechanism | Controller | Activation | Content |
|---|---|---|---|
| GTrXL Output gate | $W_g x$ | $\sigma$ | $y$ (submodule output) |
| GTrXL Highway gate | $W_g x + b_g$ | $\sigma$ | $x$ and $y$ (convex) |
| GTrXL GRU gate | $W_z y + U_z x - b_g$ | $\sigma$ | $x$ and $\hat{h}$ (convex) |
| GLU | $xW$ | $\sigma$ | $xV$ |
| GEGLU | $xW$ | GELU | $xV$ |
| SwiGLU | $xW$ | Swish | $xV$ |
| ReGLU | $xW$ | ReLU | $xV$ |
| LSTM forget gate | $W_f x_t + U_f h_{t-1}$ | $\sigma$ | $c_{t-1}$ (cell state) |

### 10.2 Where each mechanism acts

The GTrXL gates act on the **residual connections** — the wiring between submodules. They control how submodule outputs enter the residual stream.

The GLU gates act **inside the FFN** — the activation function within the submodule. They control which dimensions of the intermediate representation pass through.

These are orthogonal modifications. A modern transformer can use both: SwiGLU in the FFN (Axis 5, inside the submodule) and potentially gated residuals (Axis 5, around the submodule). They modify different parts of the same axis.

### 10.3 The LSTM connection

This is not coincidence. The LSTM (Hochreiter and Schmidhuber, 1997) was the first architecture to use learned multiplicative gates for controlling information flow. It used three gates (input, forget, output) to regulate a persistent cell state. The GRU (Chung et al., 2014) simplified this to two gates (reset, update).

Highway Networks (Srivastava et al., 2015) took the LSTM's gating mechanism and applied it to feedforward depth — the same idea as GTrXL's Highway variant. GLU (Dauphin et al., 2016) applied gating to convolutional language models. GTrXL and SwiGLU bring these ideas into the transformer, applied to different components.

The progression is: gates for temporal memory (LSTM, 1997) $\to$ gates for network depth (Highway, 2015) $\to$ gates for convolutional channels (GLU, 2016) $\to$ gates for transformer residuals (GTrXL, 2019) $\to$ gates for transformer FFN (SwiGLU, 2020).

### 10.4 Placing gating in the taxonomy

In the taxonomy from the earlier blog, Axis 5 covers "layer-level architecture" — everything about how the attention block is wrapped: normalization, residual connections, FFN design, and block ordering.

Both GTrXL and GLU variants are Axis 5 modifications. They do not change the attention pattern (Axis 3), the KV representation (Axis 2), or the number of heads (Axis 1). The attention computation itself — queries, keys, values, softmax, weighted sum — is completely unchanged. What changes is the infrastructure surrounding it.

---

## Summary

Gating replaces the fixed, unconditional operations in a transformer block — the additive residual connection and the ReLU activation — with learned, multiplicative control mechanisms. The GTrXL paper (Parisotto et al., 2019) showed that two changes to the Transformer-XL, identity map reordering (pre-norm) and GRU-type gating on residual connections, transform the architecture from a complete failure in RL ($5.0$ human-normalized score) to state-of-the-art ($117.6$), exceeding both LSTMs and external memory architectures while matching the LSTM's stability. The GLU Variants paper (Shazeer, 2020) showed that replacing ReLU with gated linear units in the FFN — specifically SwiGLU or GEGLU — improves pre-training perplexity and downstream task quality at matched parameter and compute budgets, using the $\frac{2}{3} d_{ff}$ trick to equalize costs. Both papers apply the same principle: let the network learn, dimension by dimension, how much information to pass through — the same principle that made LSTMs trainable two decades earlier.

---

*Previous: [DeepSeek Sparse Attention: Learned Token Selection from Scratch](/blog/attention-deepseek-sparse)*
