---
title: "Mamba and Mamba-2: Selective State Spaces and Structured State Space Duality"
description: "Building selective state space models from the ground up — why the fixed dynamics of prior SSMs cannot perform content-based reasoning on discrete data, deriving the selection mechanism that makes Δ, B, C input-dependent so the model can focus on or ignore tokens at will, proving that the selective SSM with A = −1 and B = 1 reduces exactly to classical RNN gating through the zero-order hold discretization, the hardware-aware selective scan that fuses discretization, recurrence, and output projection into a single SRAM-resident kernel, the Mamba architecture that merges the SSM and MLP blocks into one homogeneous unit, deriving the matrix transformation form that reveals every SSM is a semiseparable matrix multiplication, the structured state space duality that shows scalar-identity SSMs and 1-semiseparable structured masked attention are dual algorithms for computing the same function, the SSD block decomposition algorithm that combines quadratic attention within chunks and linear recurrence between chunks for optimal hardware efficiency, and the Mamba-2 architecture that uses parallel projections, grouped-value attention head structure, and extra normalization to be 2–8× faster than Mamba while matching Transformer++ quality — all derived step by step with concrete examples."
date: 2026-04-08
tags: [machine-learning, attention, transformers, state-space-models, mamba, mamba-2, selective-ssm, structured-state-space-duality, ssd, efficiency, sequence-modeling]
order: 1
---

The previous blogs in this series modified or replaced the attention mechanism. The Why Replace Attention blog showed that softmax attention is secretly a kernel, and replacing the kernel with $\phi(q)^\top \phi(k)$ turns a Transformer into an RNN with constant-size state. The RetNet blog added exponential decay. The Gated DeltaNet blog added targeted erasure. The Kernel Zoo blog explored different feature maps $\phi$. All of these start from attention and work backward toward efficient recurrence.

This blog takes the opposite path. We start from **state space models** (SSMs) — a class of sequence models rooted in control theory and signal processing — and work forward toward attention. We derive insights from two papers:

1. **Gu and Dao (2023)**, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces": identifies the fundamental limitation of prior SSMs (fixed dynamics cannot reason about content), introduces the **selection mechanism** that makes SSM parameters input-dependent, and proposes a hardware-aware architecture that matches Transformer quality for the first time.

2. **Dao and Gu (2024)**, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality": reveals that SSMs and attention are not separate ideas but two algorithms for computing the same function on **semiseparable matrices**, uses this duality to design a faster algorithm (SSD) that leverages matrix multiplication units, and proposes the **Mamba-2** architecture that is 2–8× faster than Mamba while improving quality.

The first paper says: give SSMs the ability to select. The second paper says: selection makes SSMs equivalent to a form of attention. Together, they close the loop between the two paradigms.

---

## The Running Example

We use a single-channel input sequence of 4 tokens:

$$
x = (x_1, x_2, x_3, x_4) = (1.0, \; 0.5, \; -0.3, \; 0.8)
$$

This represents one channel ($D = 1$) of a sequence after the input projection. For the SSM, we use a state dimension of $N = 1$ (the simplest possible latent state — a single scalar). For the model-scale analysis, we use the same parameters as the series: $d_\text{model} = 512$, $L = 12$ layers.

For the selective SSM and the Mamba-2 duality sections, we extend to a multi-channel example with $D = 4$ and head dimension $P = 2$.

---

## 1. State Space Models

### 1.1 The continuous system

A **state space model** (SSM) is defined by a continuous-time dynamical system that maps an input signal $x(t) \in \mathbb{R}$ to an output signal $y(t) \in \mathbb{R}$ through a latent state $h(t) \in \mathbb{R}^N$:

$$
h'(t) = Ah(t) + Bx(t) \qquad \text{(state equation)}
$$

$$
y(t) = Ch(t) \qquad \text{(output equation)}
$$

where $A \in \mathbb{R}^{N \times N}$ is the **state matrix** that governs the dynamics, $B \in \mathbb{R}^{N \times 1}$ is the **input matrix** that controls how the input enters the state, and $C \in \mathbb{R}^{1 \times N}$ is the **output matrix** that reads from the state. This is the standard **linear time-invariant (LTI)** system from classical control theory (Kalman, 1960).

The name "state space model" comes from the fact that the system is fully described by the trajectory of $h(t)$ through an $N$-dimensional **state space**.

For our running example with $N = 1$: $A = -1$, $B = 1$, $C = 1$. The state equation becomes:

$$
h'(t) = -h(t) + x(t)
$$

This is a **leaky integrator**: the state decays exponentially (the $-h(t)$ term) while accumulating the input (the $x(t)$ term). The decay rate is $|A| = 1$, meaning the state loses about 63% of its value per unit time.

### 1.2 Discretization

Neural networks operate on discrete sequences, not continuous signals. We need to convert the continuous parameters $(\Delta, A, B)$ to discrete parameters $(\bar{A}, \bar{B})$ through a **discretization rule**. The parameter $\Delta \in \mathbb{R}_{>0}$ is the **step size** — it controls the resolution at which the continuous system is sampled.

Mamba uses the **zero-order hold (ZOH)** discretization:

$$
\bar{A} = \exp(\Delta A)
$$

$$
\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B
$$

The ZOH assumes the input $x(t)$ is constant over each interval $[t, t + \Delta)$. Let us derive these formulas.

The continuous state equation $h'(t) = Ah(t) + Bx(t)$ is a **first-order linear ODE**. The general solution (by the **integrating factor method**) is:

$$
h(t + \Delta) = \exp(\Delta A) h(t) + \int_0^{\Delta} \exp((\Delta - \tau) A) \cdot B \cdot x(t + \tau) \, d\tau
$$

Under the ZOH assumption, $x(t + \tau) = x(t)$ for $\tau \in [0, \Delta)$, so the integral simplifies:

$$
\int_0^{\Delta} \exp((\Delta - \tau) A) \, d\tau \cdot B \cdot x(t)
$$

Computing the integral by substitution $u = \Delta - \tau$, $du = -d\tau$:

$$
\int_0^{\Delta} \exp(uA) \, du = A^{-1}(\exp(\Delta A) - I)
$$

This uses the **matrix exponential integral identity**: $\int_0^T \exp(uA) du = A^{-1}(\exp(TA) - I)$, which follows from the fact that $\frac{d}{du}\exp(uA) = A\exp(uA)$ (the **matrix exponential derivative**).

Multiplying by $\Delta B / \Delta$:

$$
\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B
$$

For our running example with $A = -1$, $B = 1$, $\Delta = 1.0$:

$$
\bar{A} = \exp(-1 \cdot 1.0) = \exp(-1) = 0.368
$$

$$
\bar{B} = (-1)^{-1}(\exp(-1) - 1) \cdot 1.0 = (-1)(0.368 - 1) = (-1)(-0.632) = 0.632
$$

### 1.3 The discrete recurrence

With the discretized parameters, the SSM becomes a **linear recurrence**:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t
$$

$$
y_t = C h_t
$$

For our running example with $\bar{A} = 0.368$, $\bar{B} = 0.632$, $C = 1$, $h_0 = 0$:

**Step $t = 1$:** $x_1 = 1.0$.

$$
h_1 = 0.368 \cdot 0 + 0.632 \cdot 1.0 = 0.632
$$

$$
y_1 = 1 \cdot 0.632 = 0.632
$$

**Step $t = 2$:** $x_2 = 0.5$.

$$
h_2 = 0.368 \cdot 0.632 + 0.632 \cdot 0.5 = 0.233 + 0.316 = 0.549
$$

$$
y_2 = 0.549
$$

**Step $t = 3$:** $x_3 = -0.3$.

$$
h_3 = 0.368 \cdot 0.549 + 0.632 \cdot (-0.3) = 0.202 - 0.190 = 0.012
$$

$$
y_3 = 0.012
$$

**Step $t = 4$:** $x_4 = 0.8$.

$$
h_4 = 0.368 \cdot 0.012 + 0.632 \cdot 0.8 = 0.004 + 0.506 = 0.510
$$

$$
y_4 = 0.510
$$

The output is $y = (0.632, 0.549, 0.012, 0.510)$.

### 1.4 The convolutional form

Since the parameters $(\bar{A}, \bar{B}, C)$ are constant across time (the LTI property), the recurrence can be unrolled into a **global convolution**. Let us derive this.

Expanding the recurrence:

$$
h_1 = \bar{B} x_1
$$

$$
h_2 = \bar{A} h_1 + \bar{B} x_2 = \bar{A}\bar{B} x_1 + \bar{B} x_2
$$

$$
h_3 = \bar{A} h_2 + \bar{B} x_3 = \bar{A}^2 \bar{B} x_1 + \bar{A}\bar{B} x_2 + \bar{B} x_3
$$

The pattern: $h_t = \sum_{s=0}^{t-1} \bar{A}^s \bar{B} \, x_{t-s}$.

Multiplying by $C$:

$$
y_t = C h_t = \sum_{s=0}^{t-1} C \bar{A}^s \bar{B} \, x_{t-s}
$$

Define the **SSM convolution kernel** $\bar{K} = (C\bar{B}, \; C\bar{A}\bar{B}, \; C\bar{A}^2\bar{B}, \; \ldots)$. Then:

$$
\boxed{y = x * \bar{K} \qquad \text{(causal convolution)}}
$$

For our running example:

$$
\bar{K}_0 = C\bar{B} = 1 \cdot 0.632 = 0.632
$$

$$
\bar{K}_1 = C\bar{A}\bar{B} = 1 \cdot 0.368 \cdot 0.632 = 0.233
$$

$$
\bar{K}_2 = C\bar{A}^2\bar{B} = 1 \cdot 0.368^2 \cdot 0.632 = 1 \cdot 0.135 \cdot 0.632 = 0.086
$$

$$
\bar{K}_3 = C\bar{A}^3\bar{B} = 1 \cdot 0.368^3 \cdot 0.632 = 1 \cdot 0.050 \cdot 0.632 = 0.031
$$

Numerical check for $y_2$:

$$
y_2 = \bar{K}_0 x_2 + \bar{K}_1 x_1 = 0.632 \cdot 0.5 + 0.233 \cdot 1.0 = 0.316 + 0.233 = 0.549 \quad \checkmark
$$

Numerical check for $y_3$:

$$
y_3 = \bar{K}_0 x_3 + \bar{K}_1 x_2 + \bar{K}_2 x_1 = 0.632 \cdot (-0.3) + 0.233 \cdot 0.5 + 0.086 \cdot 1.0
$$

$$
= -0.190 + 0.117 + 0.086 = 0.013
$$

This matches $y_3 = 0.012$ from the recurrence (the difference is rounding in the kernel coefficients). $\checkmark$

### 1.5 The dual computation modes

This is the crucial property. The SSM has **two equivalent computation modes**:

1. **Recurrent mode** (equation in Section 1.3): processes one token at a time with $O(1)$ memory per step. Total cost: $O(TN)$ for $T$ tokens. Ideal for autoregressive inference.

2. **Convolutional mode** (equation in Section 1.4): processes the entire sequence at once via an FFT-based convolution. Total cost: $O(T \log T)$ for $T$ tokens. Ideal for parallel training.

Prior SSMs (S4, DSS, S4D, S5, H3, Hyena) exploit this duality: train with convolutions, infer with recurrence. The convolutional mode gives training parallelism. The recurrent mode gives constant-time inference per step.

### 1.6 The LTI property — and its limitation

The duality between recurrence and convolution exists **because** the parameters $(\bar{A}, \bar{B}, C)$ are constant across time. This is the **linear time-invariance (LTI)** property. A time-invariant system applies the same transformation regardless of when or where in the sequence a token appears.

LTI is a double-edged sword. It enables convolutions (because a fixed kernel can be applied everywhere). But it prevents **content-based reasoning** — the model cannot decide to focus on one token and ignore another based on what those tokens contain. The dynamics are the same for every input.

---

## 2. Why Selection Matters

### 2.1 The failure mode

Gu and Dao (2023) identify two tasks that reveal the LTI limitation:

The **Selective Copying** task modifies the standard Copying task by randomizing the spacing between tokens that need to be memorized. The standard Copying task has constant spacing, so a fixed convolution kernel can solve it by simply counting positions. The Selective Copying task has random spacing, so the model must look at the content of each token to decide whether to memorize it.

The **Induction Heads** task requires the model to perform associative recall: given a pattern like "Harry ... Harry Potter", the model must predict "Potter" when it sees the second "Harry". This requires recognizing that the current token matches a previously seen token — a content-dependent operation.

LTI SSMs fail on both tasks. From the recurrent view, the $(\bar{A}, \bar{B})$ transitions are constant, so the model cannot selectively focus on or ignore tokens based on their content. From the convolutional view, a fixed convolution kernel is inherently position-aware but not content-aware — it cannot vary the spacing dynamically.

### 2.2 What selection means

The solution is to make the SSM parameters **functions of the input**. Instead of fixed $(\Delta, B, C)$, the selective SSM uses:

$$
B_t = s_B(x_t) = \text{Linear}_N(x_t)
$$

$$
C_t = s_C(x_t) = \text{Linear}_N(x_t)
$$

$$
\Delta_t = \tau_\Delta(\text{Parameter} + s_\Delta(x_t)) = \text{softplus}(\text{Parameter} + \text{Linear}_1(x_t))
$$

where $\text{Linear}_d$ denotes a learned linear projection to dimension $d$, and $\tau_\Delta = \text{softplus}$ ensures $\Delta_t > 0$.

The **softplus** function is $\text{softplus}(x) = \log(1 + \exp(x))$, a smooth approximation to $\text{ReLU}$. It is always positive: $\text{softplus}(x) > 0$ for all $x$.

The matrix $A$ remains fixed (not input-dependent). Gu and Dao hypothesize that making $A$ selective in addition to $\Delta$ would have similar performance, since $\Delta$ already controls $A$ through the discretization $\bar{A}_t = \exp(\Delta_t A)$.

The critical consequence: once $(\Delta, B, C)$ vary with time, the parameters $(\bar{A}_t, \bar{B}_t)$ are no longer constant. The LTI property breaks. The convolution kernel $\bar{K}$ is no longer well-defined (it would need to be different at every position). The convolutional computation mode is lost.

This is the fundamental tradeoff. Selection gives the model content-dependent dynamics. But it removes the fast convolutional training path. The model must be computed recurrently — or with a new algorithm.

### 2.3 Interpretation of $\Delta$

$\Delta$ controls the balance between focusing on the current input $x_t$ and persisting the state $h_{t-1}$. Mechanistically:

- A **large** $\Delta_t$ means $\bar{A}_t = \exp(\Delta_t A) \to 0$ (since $A$ has negative entries) and $\bar{B}_t \to$ large. The state is reset, and the current input $x_t$ is written strongly. The system is "selecting" $x_t$.

- A **small** $\Delta_t$ means $\bar{A}_t \to 1$ and $\bar{B}_t \to 0$. The state is preserved, and the current input is ignored. The system is "skipping" $x_t$.

This is exactly the behavior needed for Selective Copying: the model should produce large $\Delta_t$ for content tokens and small $\Delta_t$ for noise tokens.

### 2.4 Numerical example: selective vs. non-selective

Let us trace the selective SSM for our running example. We keep $A = -1$, $C = 1$, but now $\Delta_t$ varies.

Suppose the selection mechanism produces:

$$
\Delta_1 = 2.0, \quad \Delta_2 = 0.1, \quad \Delta_3 = 0.1, \quad \Delta_4 = 2.0
$$

This says: focus on tokens 1 and 4, ignore tokens 2 and 3.

**Step $t = 1$:** $\Delta_1 = 2.0$.

$$
\bar{A}_1 = \exp(-2.0) = 0.135, \quad \bar{B}_1 = (-1)^{-1}(\exp(-2.0) - 1) \cdot 2.0 = (-1)(-0.865) \cdot 2.0 = 1.729
$$

Wait — let us be more careful with the ZOH formula. For the scalar case with $A = -1$:

$$
\bar{B}_t = (\Delta_t A)^{-1}(\exp(\Delta_t A) - 1) \cdot \Delta_t B = \frac{\exp(-\Delta_t) - 1}{-\Delta_t} \cdot \Delta_t \cdot 1 = 1 - \exp(-\Delta_t)
$$

This uses the simplification: $\frac{\exp(-\Delta_t) - 1}{-\Delta_t} \cdot \Delta_t = -((\exp(-\Delta_t) - 1)) = 1 - \exp(-\Delta_t)$. The $\Delta_t$ in the numerator and denominator cancel (by **algebraic cancellation**).

So:

$$
\bar{A}_t = \exp(-\Delta_t), \qquad \bar{B}_t = 1 - \exp(-\Delta_t)
$$

Note that $\bar{A}_t + \bar{B}_t = 1$. This is a **convex combination** of the previous state and the current input. This identity holds specifically for the scalar case $A = -1$, $B = 1$ with ZOH discretization.

**Step $t = 1$:** $\Delta_1 = 2.0$.

$$
\bar{A}_1 = \exp(-2.0) = 0.135, \quad \bar{B}_1 = 1 - 0.135 = 0.865
$$

$$
h_1 = 0.135 \cdot 0 + 0.865 \cdot 1.0 = 0.865, \quad y_1 = 0.865
$$

Strong focus on $x_1 = 1.0$: the state captures 86.5% of the input.

**Step $t = 2$:** $\Delta_2 = 0.1$.

$$
\bar{A}_2 = \exp(-0.1) = 0.905, \quad \bar{B}_2 = 1 - 0.905 = 0.095
$$

$$
h_2 = 0.905 \cdot 0.865 + 0.095 \cdot 0.5 = 0.783 + 0.048 = 0.831, \quad y_2 = 0.831
$$

The state barely changed: 90.5% of the previous state is retained, and only 9.5% of the new input enters. Token 2 is effectively ignored.

**Step $t = 3$:** $\Delta_3 = 0.1$.

$$
\bar{A}_3 = 0.905, \quad \bar{B}_3 = 0.095
$$

$$
h_3 = 0.905 \cdot 0.831 + 0.095 \cdot (-0.3) = 0.752 - 0.029 = 0.723, \quad y_3 = 0.723
$$

Again, the state is mostly preserved. Token 3 is ignored.

**Step $t = 4$:** $\Delta_4 = 2.0$.

$$
\bar{A}_4 = 0.135, \quad \bar{B}_4 = 0.865
$$

$$
h_4 = 0.135 \cdot 0.723 + 0.865 \cdot 0.8 = 0.098 + 0.692 = 0.790, \quad y_4 = 0.790
$$

Strong focus on $x_4 = 0.8$: the state is mostly overwritten.

Compare the outputs:
- Non-selective (Section 1.3): $y = (0.632, 0.549, 0.012, 0.510)$
- Selective: $y = (0.865, 0.831, 0.723, 0.790)$

The selective model retains the signal from token 1 through tokens 2 and 3 (where $y$ stays near 0.8), then switches to capture token 4. The non-selective model treats all tokens equally, leading to the state collapsing near zero at token 3 (because $x_3 = -0.3$ partially cancels the accumulated positive state).

---

## 3. The Connection to Gating

### 3.1 Theorem 1: selective SSMs are gated RNNs

Gu and Dao (2023) prove that the selective SSM, under specific parameter choices, reduces exactly to a classical gated RNN. This is Theorem 1 of the Mamba paper.

**Theorem 1.** *When $N = 1$, $A = -1$, $B = 1$, $s_\Delta = \text{Linear}(x)$, and $\tau_\Delta = \text{softplus}$, the selective SSM recurrence takes the form:*

$$
g_t = \sigma(\text{Linear}(x_t))
$$

$$
h_t = (1 - g_t) h_{t-1} + g_t x_t
$$

where $\sigma$ is the **sigmoid function** $\sigma(z) = 1 / (1 + \exp(-z))$.

### 3.2 Proof

The proof is in Appendix C of the Mamba paper. Let us re-derive it step by step.

The continuous system with $N = 1$, $A = -1$, $B = 1$ is:

$$
h'(t) = -h(t) + x(t)
$$

The discretization step size is:

$$
\Delta_t = \text{softplus}(\text{Parameter} + \text{Linear}(x_t))
$$

We observe that the Parameter can be absorbed as a bias term in the linear projection. So we write $\Delta_t = \text{softplus}(\text{Linear}(x_t))$ where $\text{Linear}$ includes the bias.

Applying ZOH with $A = -1$:

$$
\bar{A}_t = \exp(\Delta_t \cdot (-1)) = \exp(-\Delta_t) = \frac{1}{\exp(\Delta_t)}
$$

Now we use the identity $\text{softplus}(z) = \log(1 + \exp(z))$, so $\Delta_t = \log(1 + \exp(\text{Linear}(x_t)))$. Exponentiating:

$$
\exp(\Delta_t) = 1 + \exp(\text{Linear}(x_t))
$$

Therefore:

$$
\bar{A}_t = \frac{1}{1 + \exp(\text{Linear}(x_t))} = \sigma(-\text{Linear}(x_t))
$$

By the **sigmoid reflection identity** $\sigma(-z) = 1 - \sigma(z)$:

$$
\bar{A}_t = 1 - \sigma(\text{Linear}(x_t))
$$

Define $g_t = \sigma(\text{Linear}(x_t))$. Then $\bar{A}_t = 1 - g_t$.

For $\bar{B}_t$, we showed in Section 2.4 that $\bar{B}_t = 1 - \exp(-\Delta_t) = 1 - \bar{A}_t = g_t$.

The recurrence becomes:

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t = (1 - g_t) h_{t-1} + g_t x_t
$$

$$
\boxed{h_t = (1 - g_t) h_{t-1} + g_t x_t}
$$

This is exactly the gated recurrence. $\square$

### 3.3 Numerical verification

For $x_1 = 1.0$ with $\text{Linear}(x) = 2x - 1$ (a concrete linear projection):

$$
g_1 = \sigma(2 \cdot 1.0 - 1) = \sigma(1.0) = \frac{1}{1 + \exp(-1)} = \frac{1}{1 + 0.368} = \frac{1}{1.368} = 0.731
$$

$$
h_1 = (1 - 0.731) \cdot 0 + 0.731 \cdot 1.0 = 0.731
$$

Cross-check via $\Delta_1 = \text{softplus}(1.0) = \log(1 + \exp(1.0)) = \log(1 + 2.718) = \log(3.718) = 1.313$:

$$
\bar{A}_1 = \exp(-1.313) = 0.269, \quad \bar{B}_1 = 1 - 0.269 = 0.731
$$

$$
h_1 = 0.269 \cdot 0 + 0.731 \cdot 1.0 = 0.731 \quad \checkmark
$$

For $x_3 = -0.3$:

$$
g_3 = \sigma(2 \cdot (-0.3) - 1) = \sigma(-1.6) = \frac{1}{1 + \exp(1.6)} = \frac{1}{1 + 4.953} = \frac{1}{5.953} = 0.168
$$

The gate is small ($0.168$), so the model retains most of the state and largely ignores $x_3 = -0.3$. This is the selection mechanism in action.

### 3.4 Interpretation

The connection to gating is not merely formal. It means that **discretization of SSMs is the principled foundation of heuristic gating mechanisms.** The gate $g_t = \sigma(\text{Linear}(x_t))$ in an LSTM or GRU was introduced as a heuristic to control information flow. The SSM perspective derives the same gate from first principles: start with a continuous dynamical system, discretize with ZOH, and the gate emerges naturally from the interaction between $\Delta$ and $A$.

This also explains why making $\Delta$ input-dependent is the most important selective parameter. Table 7 of the Mamba paper ablates the three selective parameters ($\Delta$, $B$, $C$). Making $\Delta$ alone selective reduces perplexity from 10.93 (no selection) to 10.15. Making $B$ or $C$ alone selective gives smaller improvements (10.93 → 10.15 for $\Delta$ vs 10.93 → 9.98 for $\Delta + B$ vs 10.93 → 9.81 for all three). $\Delta$ is the most important because it directly controls the gate — it determines whether to read or skip each token.

---

## 4. The Mamba Architecture

### 4.1 The architecture

Prior SSM architectures (H3, Hyena) interleave an SSM layer with an MLP block, following the Transformer's pattern of alternating attention and MLP. Mamba simplifies this by **merging the two blocks into one**.

The Mamba block consists of:

1. **Input projection**: $x \in \mathbb{R}^D \to (x', z) \in \mathbb{R}^{ED} \times \mathbb{R}^{ED}$, where $E$ is the expansion factor (typically $E = 2$).

2. **Short convolution**: a 1D depthwise convolution with kernel size $d$ (typically $d = 4$) applied to $x'$. This provides local context before the SSM.

3. **SSM**: the selective state space model applied to the convolved $x'$. The parameters $(\Delta, B, C)$ are computed from the post-convolution activation.

4. **Gating**: the SSM output is multiplied element-wise by $\sigma(z)$, where $\sigma$ is the **SiLU** (Sigmoid Linear Unit) activation $\sigma(z) = z \cdot \text{sigmoid}(z)$.

5. **Output projection**: the gated result is projected back to $\mathbb{R}^D$.

The SiLU gating makes the Mamba block analogous to a **SwiGLU MLP** (Shazeer, 2020) — the standard MLP used in modern Transformers like PaLM and LLaMA. Compared to the MLP block, Mamba simply adds a convolution and SSM to the main branch.

### 4.2 Parameter count

For each Mamba block with model dimension $D$ and expansion factor $E$:

- Input projections: $2ED^2$ (for $x'$ and $z$)
- Output projection: $ED^2$
- Total from projections: $3ED^2$

With $E = 2$, this is $6D^2$ per block. The SSM parameters ($A$, projections for $\Delta$, $B$, $C$) are much smaller in comparison. Two Mamba blocks (stacked homogeneously) match the $12D^2$ parameters of a Transformer layer (one attention + one MLP):

$$
\text{Transformer layer}: \underbrace{4D^2}_\text{Q,K,V,O projections} + \underbrace{8D^2}_\text{MLP (up + down)} = 12D^2
$$

$$
\text{Two Mamba blocks}: 2 \times 6D^2 = 12D^2
$$

### 4.3 Key results

Mamba achieves several firsts:

1. **Selective Copying**: Mamba solves the task with 99.8% accuracy, compared to 97.0% for S4 (no gate) and 18.3% for S4 without the selection mechanism (Table 1).

2. **Induction Heads**: Mamba extrapolates perfectly to sequences 4000× longer than training length ($2^8 = 256$ training → $2^{20} = 1{,}048{,}576$ test). No other method exceeds 2× extrapolation (Table 2).

3. **Language modeling**: Mamba is the first linear-time model to match the quality of a strong Transformer++ recipe (PaLM/LLaMA-style) on scaling laws from 125M to 1.3B parameters (Figure 4). Mamba-3B matches Transformers at twice the size on downstream tasks.

4. **Inference throughput**: Mamba achieves 4–5× higher generation throughput than a Transformer of similar size, because it does not require a KV cache that grows with sequence length.

---

## 5. The Hardware-Aware Algorithm

### 5.1 The problem

The selective SSM loses the convolutional computation mode. The naive recurrence requires materializing the expanded state $h \in \mathbb{R}^{B \times L \times D \times N}$ in GPU HBM (high-bandwidth memory), which is prohibitively large. For batch size $B = 16$, sequence length $L = 2048$, $D = 2048$, $N = 16$: the state requires $16 \times 2048 \times 2048 \times 16 \times 2 = 2$ GB in fp16.

### 5.2 The solution: kernel fusion

The key insight is that the SSM can be computed entirely in fast SRAM (on-chip memory), without materializing the full state in HBM.

1. Load the SSM parameters $(\Delta, A, B, C)$ from HBM to SRAM. Size: $O(BLD + DN)$.
2. Compute discretization ($\bar{A}_t, \bar{B}_t$) in SRAM.
3. Perform the selective scan (recurrence) in SRAM.
4. Multiply by $C$ and write the output $(B, L, D)$ back to HBM.

The intermediate states of size $(B, L, D, N)$ never leave SRAM. This reduces memory IOs by a factor of $O(N)$ (the state dimension), which in practice gives 20–40× speedup over a naive implementation.

### 5.3 Parallel scan

Despite being sequential in nature, the recurrence can be parallelized with a **parallel associative scan** (Blelloch, 1990). The key observation: the recurrence $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ can be written as an associative binary operation on pairs $(\bar{A}_t, \bar{B}_t x_t)$. The scan computes all prefix products in $O(\log L)$ parallel steps, using $O(L)$ work.

### 5.4 Recomputation

To reduce memory during training, Mamba does not save intermediate states for backpropagation. Instead, it **recomputes** them in the backward pass by reloading the inputs from HBM and re-running the scan in SRAM. This is the same technique as gradient checkpointing in Transformers (e.g., FlashAttention). The result: the selective SSM layer uses the same activation memory as a FlashAttention layer.

---

## 6. SSMs Are Structured Matrices

We now turn to the second paper (Dao and Gu, 2024), which reveals a deep connection between SSMs and attention. The starting point is a simple observation: every SSM can be written as a matrix multiplication.

### 6.1 The matrix transformation form

Recall the SSM recurrence:

$$
h_t = A_t h_{t-1} + B_t x_t, \qquad y_t = C_t^\top h_t
$$

We can unroll this. By induction (starting from $h_0 = 0$):

$$
h_t = \sum_{s=0}^{t} A_{t:s}^\times B_s x_s
$$

where $A_{t:s}^\times = A_t A_{t-1} \cdots A_{s+1}$ denotes the **cumulative product** of the $A$ matrices from time $s+1$ to $t$, with the convention $A_{t:t}^\times = I$.

Multiplying by $C_t^\top$:

$$
y_t = C_t^\top h_t = \sum_{s=0}^{t} C_t^\top A_{t:s}^\times B_s x_s
$$

This is a matrix-vector product $y = Mx$ where:

$$
\boxed{M_{ts} = C_t^\top A_{t:s}^\times B_s \qquad \text{for } t \geq s}
$$

and $M_{ts} = 0$ for $t < s$ (causality). The matrix $M \in \mathbb{R}^{T \times T}$ is lower-triangular.

### 6.2 Semiseparable matrices

Dao and Gu (2024) identify the matrix $M$ as belonging to a well-studied class called **semiseparable matrices**.

**Definition 3.1.** A lower-triangular matrix $M$ is **N-semiseparable** if every submatrix contained in the lower-triangular portion has rank at most N.

This is exactly what the SSM matrix $M$ satisfies. Consider any off-diagonal block $M_{j':j, i':i}$ with $j' > j > i > i'$. By the formula above:

$$
M_{j,i} = C_j^\top A_{j:i}^\times B_i
$$

This can be factored as a product of a column vector ($C_j^\top A_{j:j'}$), a chain of $A$ matrices, and a row vector ($A_{i':i} B_i$). The rank of any such block is at most $N$ (the state dimension).

This is Theorem 3.5 of the paper: **the SSM transformation $y = \text{SSM}(A, B, C)(x)$ is identical to matrix multiplication by an N-semiseparable matrix $M = \text{SSS}(A, B, C)$**.

### 6.3 The scalar case: 1-semiseparable matrices

The most important special case is when $A_t$ is a **scalar times the identity**: $A_t = a_t I$ for some scalar $a_t \in [0, 1]$. Then the cumulative product simplifies:

$$
A_{t:s}^\times = a_t a_{t-1} \cdots a_{s+1} \cdot I = a_{t:s}^\times \cdot I
$$

and the matrix becomes:

$$
M_{ts} = a_{t:s}^\times \cdot (C_t^\top B_s)
$$

This can be decomposed as:

$$
M = L \circ (CB^\top)
$$

where $L_{ts} = a_{t:s}^\times$ is a **1-semiseparable matrix** (also called a 1-SS matrix) and $\circ$ denotes the **Hadamard (element-wise) product**.

The 1-SS matrix $L$ has a very specific structure:

$$
L = \text{1SS}(a) = \begin{pmatrix} 1 \\ a_1 & 1 \\ a_2 a_1 & a_2 & 1 \\ a_3 a_2 a_1 & a_3 a_2 & a_3 & 1 \end{pmatrix}
$$

Each entry is a cumulative product of consecutive $a_t$ values. The diagonal is all 1's. This is the **structured mask** — it replaces the causal mask of standard attention.

### 6.4 Numerical example

Let us trace through a 4-token example with scalar $A$. We use $a = (a_1, a_2, a_3, a_4) = (0.9, 0.8, 0.5, 0.7)$ and:

$$
B = \begin{pmatrix} 1.0 \\ 0.5 \\ -0.3 \\ 0.8 \end{pmatrix}, \quad C = \begin{pmatrix} 0.5 \\ 1.0 \\ 0.7 \\ 0.3 \end{pmatrix}, \quad X = \begin{pmatrix} 1.0 \\ 2.0 \\ 1.5 \\ 0.5 \end{pmatrix}
$$

(Here $B$, $C$, $X$ are all 1-dimensional per token since $N = P = 1$.)

The 1-SS mask $L$:

$$
L = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0.8 & 1 & 0 & 0 \\ 0.8 \cdot 0.5 & 0.5 & 1 & 0 \\ 0.8 \cdot 0.5 \cdot 0.7 & 0.5 \cdot 0.7 & 0.7 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0.8 & 1 & 0 & 0 \\ 0.4 & 0.5 & 1 & 0 \\ 0.28 & 0.35 & 0.7 & 1 \end{pmatrix}
$$

Note: $a_1$ is not used in $L$ because $L_{ts}$ involves the product $a_{s+1} \cdots a_t$, and the first row only has $L_{11} = 1$.

The Gram matrix $G = CB^\top$:

$$
G_{ts} = C_t \cdot B_s = \begin{pmatrix} 0.5 \cdot 1.0 & 0.5 \cdot 0.5 & 0.5 \cdot (-0.3) & 0.5 \cdot 0.8 \\ 1.0 \cdot 1.0 & 1.0 \cdot 0.5 & 1.0 \cdot (-0.3) & 1.0 \cdot 0.8 \\ 0.7 \cdot 1.0 & 0.7 \cdot 0.5 & 0.7 \cdot (-0.3) & 0.7 \cdot 0.8 \\ 0.3 \cdot 1.0 & 0.3 \cdot 0.5 & 0.3 \cdot (-0.3) & 0.3 \cdot 0.8 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0.5 & 0.25 & -0.15 & 0.4 \\ 1.0 & 0.5 & -0.3 & 0.8 \\ 0.7 & 0.35 & -0.21 & 0.56 \\ 0.3 & 0.15 & -0.09 & 0.24 \end{pmatrix}
$$

The full matrix $M = L \circ G$ (element-wise product, lower-triangular):

$$
M = \begin{pmatrix} 0.5 & 0 & 0 & 0 \\ 0.8 & 0.5 & 0 & 0 \\ 0.28 & 0.175 & -0.21 & 0 \\ 0.084 & 0.0525 & -0.063 & 0.24 \end{pmatrix}
$$

Output $Y = MX$:

$$
y_1 = 0.5 \cdot 1.0 = 0.5
$$

$$
y_2 = 0.8 \cdot 1.0 + 0.5 \cdot 2.0 = 0.8 + 1.0 = 1.8
$$

$$
y_3 = 0.28 \cdot 1.0 + 0.175 \cdot 2.0 + (-0.21) \cdot 1.5 = 0.28 + 0.35 - 0.315 = 0.315
$$

$$
y_4 = 0.084 \cdot 1.0 + 0.0525 \cdot 2.0 + (-0.063) \cdot 1.5 + 0.24 \cdot 0.5 = 0.084 + 0.105 - 0.0945 + 0.12 = 0.2145
$$

We can verify $y_2$ via the recurrence. With scalar $A$, the recurrence is:

$$
h_t = a_t h_{t-1} + B_t x_t, \quad y_t = C_t h_t
$$

$$
h_1 = a_1 \cdot 0 + B_1 x_1 = 1.0 \cdot 1.0 = 1.0, \quad y_1 = C_1 h_1 = 0.5 \cdot 1.0 = 0.5 \quad \checkmark
$$

$$
h_2 = a_2 h_1 + B_2 x_2 = 0.8 \cdot 1.0 + 0.5 \cdot 2.0 = 0.8 + 1.0 = 1.8, \quad y_2 = C_2 h_2 = 1.0 \cdot 1.8 = 1.8 \quad \checkmark
$$

The matrix form and recurrence produce the same output.

### 6.5 Interpretation

The matrix $M$ encodes the full input-output map of the SSM. The recurrent mode computes $y = Mx$ by exploiting the sequential structure (each row depends on the previous state). The quadratic mode computes $y = Mx$ by materializing $M$ and doing direct matrix multiplication. These are two algorithms for the same computation — one is $O(TN)$ time, the other is $O(T^2N)$ time but more hardware-friendly because it uses matrix multiplications.

This is the core insight of the Mamba-2 paper: **different methods of computing SSMs can be reframed as different algorithms for multiplying by semiseparable matrices.**

---

## 7. Structured State Space Duality

### 7.1 From SSMs to attention

The matrix form $M = L \circ (CB^\top)$ with $L = \text{1SS}(a)$ looks strikingly similar to **masked attention**:

$$
Y = (L \circ QK^\top) \cdot V
$$

In standard causal attention, $L$ is the lower-triangular matrix of all 1's (the causal mask), $Q$ and $K$ are queries and keys, and $V$ is the value matrix. In the SSM, $L$ is the 1-semiseparable mask of cumulative decay products, $C$ plays the role of queries, $B$ plays the role of keys, and $X$ plays the role of values.

The correspondence is exact:

| SSM | Attention |
|---|---|
| $C$ (output matrix) | $Q$ (queries) |
| $B$ (input matrix) | $K$ (keys) |
| $X$ (input sequence) | $V$ (values) |
| $a_{t:s}^\times$ (cumulative product) | $L_{ts}$ (mask entry) |
| $N$ (state dimension) | $N$ (feature dimension) |

### 7.2 The duality

Dao and Gu (2024) make this precise with **structured state space duality (SSD)**.

**State space models** are usually defined through a recurrence (Definition 2.2 of the paper) and computed with a linear-time algorithm (the scan). **Attention** is usually defined through pairwise comparisons (equation 9) and computed with a quadratic-time algorithm (materializing $QK^\top$).

But both have dual forms:

- An SSM can be computed quadratically by materializing $M = \text{SSS}(A, B, C)$ and multiplying $Y = MX$. This is the **quadratic (attention-like) mode**.

- Attention can be computed linearly by using the cumsum trick from the linear attention framework (Katharopoulos et al., 2020). This is the **linear (recurrent) mode**.

The duality says: **for scalar-identity $A$ matrices, these are the same function computed by different algorithms**. The SSM recurrence is the linear mode. Materializing $M$ is the quadratic mode. They produce identical outputs.

### 7.3 The SSD layer

The **state space dual (SSD)** layer is the specific SSM that the duality applies to. Compared to Mamba's selective SSM (S6), SSD makes two simplifications:

1. $A$ is restricted from diagonal to **scalar times identity**: $A_t = a_t I$. Each $a_t$ is a single scalar shared across all state dimensions.

2. The head dimension $P$ is increased from $P = 1$ (in Mamba) to $P = 64$ or $128$ (matching Transformer conventions).

The first restriction slightly decreases expressivity but enables the quadratic mode. The second compensates by using attention-like multi-head structure.

### 7.4 Why scalar $A$ matters

With diagonal $A_t = \text{diag}(a_t^{(1)}, \ldots, a_t^{(N)})$, each state dimension has an independent decay rate. The matrix $M$ becomes:

$$
M_{ts} = C_t^\top \text{diag}(a_{t:s}^{(1)\times}, \ldots, a_{t:s}^{(N)\times}) B_s
$$

This involves $N$ different 1-SS masks, one per state dimension. The quadratic mode requires materializing all $N$ masks and their Hadamard products, which is expensive.

With scalar $A_t = a_t I$, all state dimensions share the same decay: $a_{t:s}^{(i)\times} = a_{t:s}^\times$ for all $i$. The matrix simplifies to $M = L \circ (CB^\top)$ with a single mask $L$, and the quadratic mode becomes a single masked matrix multiplication — just like attention.

---

## 8. The SSD Algorithm

### 8.1 The idea

The linear (recurrent) mode takes $O(TN)$ time. The quadratic (attention-like) mode takes $O(T^2N)$ time. Neither is optimal in practice:

- The recurrent mode is sequential and cannot exploit matrix multiplication units (tensor cores on GPUs).
- The quadratic mode is parallelizable but scales poorly with sequence length.

The SSD algorithm combines both: **split the sequence into chunks, compute within each chunk quadratically (using matmul), and connect chunks recurrently (using a scan)**. This is a **block decomposition** of the semiseparable matrix $M$.

### 8.2 Block decomposition

Partition the $T$-length sequence into $T/Q$ chunks of size $Q$. The matrix $M$ decomposes into blocks:

$$
M = \begin{pmatrix} M^{(0,0)} & & \\ M^{(1,0)} & M^{(1,1)} & \\ \vdots & & \ddots \end{pmatrix}
$$

The **diagonal blocks** $M^{(j,j)}$ represent intra-chunk interactions. They are small ($Q \times Q$) and can be computed quadratically using matrix multiplication.

The **off-diagonal blocks** $M^{(j,i)}$ for $j > i$ represent inter-chunk interactions. By the semiseparable property, these blocks are **low-rank** (rank at most $N$). They factor as:

$$
M^{(j,i)} = \underbrace{C_\text{block}}_\text{left factor} \cdot \underbrace{A_\text{chain}}_\text{center factor} \cdot \underbrace{B_\text{block}^\top}_\text{right factor}
$$

The center factors are connected by a scalar recurrence (a 1-SS multiplication of length $T/Q$), which is $Q$ times shorter than the original sequence.

### 8.3 The four steps

The SSD algorithm has four steps:

**Step 1: Diagonal blocks (intra-chunk).** For each chunk $j$, compute the output from tokens within the chunk using the quadratic form:

$$
Y_\text{diag}^{(j)} = (L^{(j)} \circ C^{(j)} B^{(j)\top}) X^{(j)}
$$

This is a batched matrix multiplication. Cost: $\text{BMM}(T/Q, Q, Q, P)$.

**Step 2: Right factors (chunk → state).** For each chunk, compute the final state assuming the initial state is zero:

$$
h_\text{local}^{(j)} = B^{(j)\top} \text{decay} \cdot X^{(j)}
$$

This is a matrix multiplication. Cost: $\text{BMM}(T/Q, N, P, Q)$.

**Step 3: Center factors (state → state).** Connect the chunks by propagating states through a scalar SSM scan of length $T/Q$:

$$
h_\text{true}^{(j)} = a_\text{chunk}^{(j)} h_\text{true}^{(j-1)} + h_\text{local}^{(j)}
$$

This is a 1-SS multiplication on $(N, P)$ independent channels. Cost: $O(T/Q \cdot NP)$ — negligible.

**Step 4: Left factors (state → output).** For each chunk, compute the output contribution from prior chunks:

$$
Y_\text{off}^{(j)} = C^{(j)} \text{decay}_\text{out} \cdot h_\text{true}^{(j-1)}
$$

This is a matrix multiplication. Cost: $\text{BMM}(T/Q, Q, P, N)$.

**Final output:** $Y = Y_\text{diag} + Y_\text{off}$.

### 8.4 Complexity

Setting $N = P = Q$ (state dimension = head dimension = chunk length):

- Total FLOPs: $O(TN^2)$ — same as attention but linear in $T$ for the dominant terms.
- Total memory: $O(TN)$ — linear in both sequence length and state size.
- The work is dominated by matrix multiplications on $(N, N)$ matrices.

This is the key advantage over Mamba's selective scan: SSD uses **matrix multiplication as its core primitive**, which tensor cores are optimized for. Mamba's scan is a custom CUDA kernel that cannot leverage these hardware units.

### 8.5 Speed comparison

The SSD algorithm is 2–8× faster than Mamba's fused selective scan (Figure 10 of the Mamba-2 paper). For large state expansion ($N = 256$), SSD is 6× faster. For the default $N = 64$, SSD is 2× faster. SSD is also faster than FlashAttention-2 at sequence lengths beyond 2K and 6× faster at 16K.

---

## 9. The Mamba-2 Architecture

### 9.1 Block design changes

The Mamba-2 block modifies the Mamba block in two ways motivated by the attention connection:

**Parallel parameter projections.** In Mamba, the SSM parameters $(A, B, C)$ are computed from the post-convolution activation $x_c$, which depends on the initial linear projection. The projections are sequential.

In Mamba-2, $(A, X, B, C)$ are all produced from the input $u$ in **parallel** — analogous to how $(Q, K, V)$ are produced in parallel in a Transformer. This slightly reduces parameters and, more importantly, enables tensor parallelism for larger models by reducing the number of synchronization points per block from two to one.

**Extra normalization.** Mamba-2 adds a normalization layer (GroupNorm or RMSNorm) after the gating multiplication and before the output projection. This improves training stability at larger scales and is analogous to the NormFormer architecture (Shleifer, Weston, and Ott, 2021) that adds normalization at the end of MLP and attention blocks.

### 9.2 Multi-head patterns

The state space duality allows transferring multi-head design choices from attention to SSMs.

**Multi-head SSM (MHS) / Multi-head attention (MHA).** The classic pattern: $H$ independent heads, each with its own $(A, B, C, X)$. The state size per head is $N$, and the head dimension is $P = D/H$.

**Multi-input SSM (MIS) / Multi-value attention (MVA).** The original Mamba architecture uses this pattern: $X$ has $H$ heads (one per channel), but $B$ and $C$ are shared across all heads. In attention terms, the keys and queries are shared while the values have independent heads. This is the natural choice from the SSM perspective because $X$ is the main input to the SSM, while $B$ and $C$ are auxiliary parameters.

**Grouped-input SSM (GIS) / Grouped-value attention (GVA).** Analogous to grouped-query attention (GQA), this creates $G$ groups of $B$ and $C$ projections, each shared across $H/G$ input heads. Mamba-2 uses this pattern with $G$ set to be a multiple of the tensor parallelism degree for efficient sharding.

The paper ablates these patterns (Table 5) and finds that the MVA/MIS pattern performs best, matching the choice naturally derived from the SSM perspective.

### 9.3 Kernel feature maps

The SSD framework allows incorporating kernel feature maps from the linear attention literature. In Mamba-2, the feature map $\psi$ is applied to the $B$ and $C$ branches (corresponding to $K$ and $Q$ in attention). By default, $\psi(x) = \text{Swish}(x) = x \cdot \sigma(x)$, following Mamba's SiLU activation.

The paper ablates various kernel approximations (Table 6): cosFormer, Random Feature Attention, and Positive Random Features (Performer). None significantly improve over simple pointwise nonlinearities. This is expected because SSD differs from vanilla linear attention by the inclusion of the 1-semiseparable mask $L$, which already captures positional structure that kernel approximations were designed to provide.

### 9.4 Hybrid architectures

A striking finding from the Mamba-2 paper is that mixing SSD layers with attention layers improves over either alone. Table 2 shows that adding approximately 10% attention layers (6 out of 48 layers in a 350M model) reduces perplexity from 8.60 (pure SSD) to 8.26, with the best configuration using 7 attention layers.

At the 2.7B scale (Table 3), a Mamba-2 + MLP + Attention hybrid (28 SSD + 4 attention + 32 MLP layers) achieves an average downstream accuracy of 60.7%, compared to 60.2% for pure Transformer++ and 60.2% for pure Mamba-2.

The hypothesis: SSM layers function as general sequence-to-sequence mappings that compress context into their recurrent state, while attention layers act as a retrieval mechanism that can refer directly to previous tokens without compression. A small number of attention layers provides an "escape hatch" for tasks that require exact token lookup, while SSD handles the bulk of the computation more efficiently.

---

## 10. Scaling Results

### 10.1 Mamba scaling laws

On the Pile dataset (Figure 4 of the Mamba paper), Mamba matches Transformer++ scaling from 125M to 1.3B parameters at context length 2048. At context length 8192, Mamba further improves relative to Transformers (which are limited by the quadratic cost of longer sequences).

On downstream zero-shot evaluations (Table 3), Mamba at each model size matches baselines at twice the size:

- Mamba-130M matches Pythia-160M
- Mamba-370M matches Pythia-410M
- Mamba-1.4B matches Pythia-2.8B

### 10.2 Mamba-2 scaling laws

Mamba-2 is Pareto-dominant over both Mamba and Transformer++ (Figure 9 of the Mamba-2 paper): it achieves lower perplexity at every FLOP budget from 125M to 1.3B parameters. This is because SSD is both faster (enabling more training tokens per wall-clock hour) and slightly more expressive (due to larger state sizes enabled by the efficient algorithm).

On downstream evaluations at 2.7B scale (Table 1 of the Mamba-2 paper), Mamba-2 matches Mamba's quality while being 2–8× faster to train.

### 10.3 State expansion

One of SSD's most important practical benefits is efficient state expansion. In Mamba, increasing the state dimension $N$ from 16 to 64 provides a significant perplexity improvement (from 9.82 to 8.71 for a 350M model — Table 10 of the Mamba paper) but at the cost of proportionally slower selective scan.

In Mamba-2, the SSD algorithm's speed is nearly independent of $N$ up to $N = 256$ (Figure 10, right panel). This allows Mamba-2 to use much larger state sizes without slowdown, effectively making the capacity-efficiency tradeoff from the Kernel Zoo blog far less severe.

---

## 11. The Full Picture: From Selection to Duality

### 11.1 The progression

The two papers together tell a coherent story:

1. **Prior SSMs** (S4, S5, H3, Hyena) are LTI systems. They process sequences through convolutions during training and recurrence during inference. They are fast but cannot do content-based reasoning.

2. **Mamba** makes the SSM parameters input-dependent (selective). This enables content-based reasoning and matches Transformer quality. But the convolutional mode is lost, and the model relies on a custom scan kernel.

3. **Mamba-2** reveals that selective SSMs with scalar $A$ are equivalent to a form of structured attention. This equivalence — structured state space duality — exposes both a quadratic (attention-like) and a linear (recurrence-like) algorithm. The SSD algorithm combines both by chunking the sequence: quadratic within chunks, linear between chunks. The result is faster than both pure attention and pure recurrence.

### 11.2 Connection to the blog series

The linear attention framework from the Why Replace Attention blog replaces the softmax with a kernel: $Y = (L \circ \phi(Q)\phi(K)^\top) V$, where $L$ is the all-1's causal mask. RetNet replaces $L$ with a decay mask $L_{ts} = \gamma^{t-s}$. The Gated DeltaNet blog modifies the recurrent update rule.

SSD generalizes all of these. The mask $L$ is a 1-semiseparable matrix with **input-dependent** entries $a_t$, not fixed scalars. This means:

- Linear attention is SSD with $a_t = 1$ for all $t$ (no decay, causal mask of 1's).
- RetNet is SSD with $a_t = \gamma$ for all $t$ (constant decay).
- Mamba-2 is SSD with $a_t = \exp(\Delta_t \cdot A_\text{scalar})$ varying per token (input-dependent decay).

The feature map $\phi$ and the mask $L$ are orthogonal design choices. The Kernel Zoo blog explored $\phi$. The SSD framework shows that $L$ — the structured mask — is equally important, and that making $L$ input-dependent is what gives SSMs their selectivity.

---

## Summary

State space models process sequences through a latent state governed by a linear recurrence, with dual convolutional and recurrent computation modes. Prior SSMs kept their parameters fixed (LTI), which enabled convolutions but prevented content-based reasoning. Mamba introduces selection — making $\Delta$, $B$, $C$ functions of the input — which breaks the LTI property but gives the model a learnable gate ($g_t = \sigma(\text{Linear}(x_t))$, equivalent to the ZOH discretization of a leaky integrator) that decides per token whether to read or skip. Mamba-2 then reveals the deeper structure: every SSM is a multiplication by a semiseparable matrix, and when $A$ is scalar, this matrix factors as a 1-semiseparable mask times a Gram matrix — exactly the structure of masked attention with an input-dependent decay mask. The SSD algorithm exploits this duality through block decomposition: quadratic attention within chunks (leveraging matrix multiplication hardware), linear recurrence between chunks (keeping the cost linear in sequence length), producing an architecture that is 2–8× faster than Mamba's custom scan while matching or exceeding Transformer quality at scales up to 2.7B parameters.

---

*Previous: [The Kernel Zoo: Performers, Fast Weight Programmers, and the Capacity-Approximation Tradeoff in Linear Attention](/blog/attention-linear-attention)*
