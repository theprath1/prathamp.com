---
title: "What Attention is Really Doing: Weighted Memory Retrieval from Scratch"
description: "Building attention from first principles — the fixed-length bottleneck that broke RNN encoder–decoders on long sentences, the weighted-sum solution introduced by Bahdanau, Cho, and Bengio (2015), alignment score derivation, softmax normalization, and a complete 3-word numerical walkthrough — all derived step by step"
date: 2026-03-31
tags: ["deep-learning", "attention", "transformers", "nlp", "neural-machine-translation"]
order: 4
---

Before the Transformer, before Q/K/V, before any of the modern terminology, there was a concrete failure mode: sequence-to-sequence models fell apart on long sentences. The fix that Bahdanau, Cho, and Bengio proposed in 2015 is what we now call attention. We will derive it from first principles, starting from the failure and arriving at the mechanism.

We will keep the entire post anchored to one tiny running example and compute the full mechanism by hand.

If you haven't read [Mathematical Prerequisites for the Attention Series](/blog/math-prerequisites-for-attention), start there — we will use tanh, exponentials, and a few basic algebraic identities throughout this post.

---

## 1. Before Attention: The Fixed-Length Bottleneck

### 1.1 The Encoder–Decoder Framework

A **sequence-to-sequence model** maps a source sequence

$$
\mathbf{x} = (x_1, x_2, \ldots, x_{T_x})
$$

to a target sequence

$$
\mathbf{y} = (y_1, y_2, \ldots, y_{T_y})
$$

with potentially different lengths.

Before attention, the standard design had two pieces: an **encoder** that reads the full source sequence and compresses it into one vector $\mathbf{c}$, and a **decoder** that generates each target token from that same vector $\mathbf{c}$. The conditional probability at target step $i$ is written as

$$
p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, \mathbf{c})
$$

where $y_{i-1}$ is the previous target token, $s_i$ is the decoder hidden state at step $i$, and $g$ is a nonlinear function that uses the decoder's current state and source context to assign probabilities to possible next target words.

The important point is not the exact form of $g$ — the important point is that the same $\mathbf{c}$ appears in every conditional.

### 1.2 The Bottleneck

The encoder is a recurrent neural network:

$$
h_t = f(x_t, h_{t-1})
$$

Here $x_t$ is the $t$-th source word, and $h_t$ is the encoder hidden state after reading up to that position. You can think of $h_t$ as the encoder's running summary of the source sequence so far.

The simplest context choice is just the last hidden state:

$$
\mathbf{c} = h_{T_x}
$$

So the whole source sequence must be compressed into one vector before decoding even starts. That means the decoder uses $g(y_0, s_1, \mathbf{c})$, then $g(y_1, s_2, \mathbf{c})$, then $g(y_2, s_3, \mathbf{c})$, and so on. The previous target token changes, the decoder hidden state changes, and the target step changes — but the source summary $\mathbf{c}$ does not. This is the problem. The entire source sentence has to survive inside one fixed-size summary, and the decoder has no way to go back and look at particular source positions later.

### 1.3 Why one vector is too rigid

Suppose the source sentence is long and the decoder is currently generating target word 17. The information needed for word 17 might live near source word 3, but a few steps later, when generating target word 18, the useful information might live near source word 11. With a fixed context vector, the decoder cannot ask for different source information at different times — it gets one precomputed summary and has to reuse it for every target position. This is the **fixed-length context vector bottleneck**. Bahdanau et al. showed empirically that translation quality degrades as source sentence length grows. The model is not failing because recurrent networks are impossible. It is failing because the decoder is forced to read the whole source through one frozen summary.

---

## 2. The Running Example

We will use one tiny source sequence throughout the post so that every derivation stays concrete.

After encoding three source words with a bidirectional RNN, suppose we obtain the scalar annotations

$$
h_1 = 0.2, \qquad h_2 = 0.9, \qquad h_3 = 0.1
$$

These are deliberately one-dimensional so every arithmetic step stays visible.

We also assume the decoder state just before generating the first target word is

$$
s_0 = 0.5
$$

Our goal is to compute the context vector used for generating the first target word.

If we used the old fixed-vector design and took only the last encoder state, then the context would be

$$
\mathbf{c} = h_3 = 0.1
$$

for every target word. Attention replaces that fixed choice with a learned weighted sum.

---

## 3. The Fix: A Position-Dependent Context Vector

Instead of one context vector for the whole sentence, Bahdanau et al. define one context vector per target position:

$$
\boxed{c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j}
$$

The new objects are the weights $\alpha_{ij}$.

An **attention weight** $\alpha_{ij}$ tells us how much target position $i$ should use source position $j$. For example, $\alpha_{2,3}$ means "how much should the second target word attend to the third source word?"

### 3.1 The two properties the weights must satisfy

If the context vector is supposed to behave like a soft selection over source positions, the weights have to satisfy two basic constraints: they must be nonnegative, $\alpha_{ij} \ge 0$ for all $j$, and they must sum to one, $\sum_{j=1}^{T_x} \alpha_{ij} = 1$, for each fixed target position $i$. These two conditions make $c_i$ a **convex combination** of the encoder states.

### 3.2 What a convex combination means here

A **convex combination** is a weighted average where the weights are nonnegative and sum to 1.

In our scalar running example, that means $c_i$ must lie between the smallest and largest source annotations.

The source values are:

$$
0.1,\ 0.2,\ 0.9
$$

So any valid context vector must satisfy

$$
0.1 \le c_i \le 0.9
$$

Why? Because weighted averages cannot leave the interval spanned by the values being averaged.

### 3.3 Numerical checks

If all weight goes to the second source word:

$$
(\alpha_{i1}, \alpha_{i2}, \alpha_{i3}) = (0, 1, 0)
$$

then

$$
c_i = 0 \cdot 0.2 + 1 \cdot 0.9 + 0 \cdot 0.1 = 0.9
$$

If the model splits evenly between the first two source words:

$$
(\alpha_{i1}, \alpha_{i2}, \alpha_{i3}) = \left(\tfrac{1}{2}, \tfrac{1}{2}, 0\right)
$$

then

$$
c_i = \tfrac{1}{2}\cdot 0.2 + \tfrac{1}{2}\cdot 0.9 + 0 \cdot 0.1 = 0.1 + 0.45 = 0.55
$$

If the weights are uniform:

$$
(\alpha_{i1}, \alpha_{i2}, \alpha_{i3}) = \left(\tfrac{1}{3}, \tfrac{1}{3}, \tfrac{1}{3}\right)
$$

then

$$
c_i = \tfrac{1}{3}(0.2 + 0.9 + 0.1) = \tfrac{1}{3}(1.2) = 0.4
$$

All three results lie inside $[0.1, 0.9]$, as they must.

### 3.4 Why the weighted sum must be soft

A natural question is why we do not simply pick one source position and stop there. That would mean using a hard argmax $j^* = \arg\max_j e_{ij}$ and then setting $c_i = h_{j^*}$. The problem is that the argmax is discrete — gradients do not flow cleanly through it during ordinary backpropagation. The weighted sum is differentiable, so the decoder can learn where to look by gradient descent rather than by a separate combinatorial procedure. That is the key engineering reason for soft attention.

---

## 4. Alignment Scores: How the Model Decides Where to Look

To get attention weights, we first need unnormalized alignment scores:

$$
e_{ij} = a(s_{i-1}, h_j)
$$

The function $a$ is the **alignment model**. It measures how compatible the decoder state $s_{i-1}$ is with encoder state $h_j$.

### 4.1 Bahdanau's alignment model

Bahdanau et al. use a one-hidden-layer feedforward network:

$$
e_{ij} = \mathbf{v}_a^\top \tanh\!\left(W_a s_{i-1} + U_a h_j\right)
$$

This looks dense, so let us unpack it. We first project the decoder state with $W_a$ and the encoder state with $U_a$, add those projected vectors, apply $\tanh$, and finally take a dot product with $\mathbf{v}_a$ to produce one scalar score. A useful implementation detail from the paper is that the term $U_a h_j$ depends only on the encoder side, not on the decoder step $i$, so it can be precomputed once for every source position.

### 4.2 The scalar version of the running example

To keep the arithmetic transparent, take the one-dimensional case

$$
W_a = 1,\qquad U_a = 1,\qquad v_a = 1
$$

Then the alignment model simplifies to

$$
e_{ij} = \tanh(s_{i-1} + h_j)
$$

This is not the full expressive model used in practice. It is a toy scalar version that lets us trace every number by hand without hiding the arithmetic.

### 4.3 Numerical check: compute the three scores

For the first target word we use $s_0 = 0.5$.

**Source word 1**

$$
e_{1,1} = \tanh(0.5 + 0.2) = \tanh(0.7)
$$

Use the definition of the **hyperbolic tangent**:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

So

$$
\tanh(0.7)
= \frac{e^{0.7} - e^{-0.7}}{e^{0.7} + e^{-0.7}}
= \frac{2.0138 - 0.4966}{2.0138 + 0.4966}
= \frac{1.5172}{2.5104}
\approx 0.6044
$$

**Source word 2**

$$
e_{1,2} = \tanh(0.5 + 0.9) = \tanh(1.4)
$$

$$
\tanh(1.4)
= \frac{e^{1.4} - e^{-1.4}}{e^{1.4} + e^{-1.4}}
= \frac{4.0552 - 0.2466}{4.0552 + 0.2466}
= \frac{3.8086}{4.3018}
\approx 0.8854
$$

**Source word 3**

$$
e_{1,3} = \tanh(0.5 + 0.1) = \tanh(0.6)
$$

$$
\tanh(0.6)
= \frac{e^{0.6} - e^{-0.6}}{e^{0.6} + e^{-0.6}}
= \frac{1.8221 - 0.5488}{1.8221 + 0.5488}
= \frac{1.2733}{2.3709}
\approx 0.5370
$$

So the score vector is

$$
\boxed{(e_{1,1}, e_{1,2}, e_{1,3}) \approx (0.6044, 0.8854, 0.5370)}
$$

### 4.4 What the ordering means

In our toy scalar setup, $0.8854 > 0.6044 > 0.5370$, so source word 2 receives the highest score. That is consistent with the source values: $h_2 = 0.9$ is the largest annotation, and the scalar function

$$
\tanh(s_0 + h_j)
$$

is monotone increasing in $h_j$ because

$$
\frac{d}{dx}\tanh(x) = \text{sech}^2(x) > 0
$$

This derivative formula is the **derivative identity for hyperbolic tangent**.

So in this toy setup, larger $h_j$ gives larger $e_{1j}$. The ranking is not mysterious. It is coming directly from monotonicity.

---

## 5. From Scores to Weights: Softmax

The alignment scores are arbitrary real numbers. We still need to turn them into proper attention weights.

Bahdanau et al. apply the **softmax function**:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

### 5.1 Why softmax gives valid weights

We need two things: nonnegative weights, and weights that sum to 1. The exponential gives the first immediately, since $\exp(e_{ij}) > 0$ for every real $e_{ij}$, which means $\alpha_{ij} > 0$ for every $j$. Now sum over all $j$:

$$
\sum_{j=1}^{T_x} \alpha_{ij}
= \sum_{j=1}^{T_x} \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

The denominator is the same in every term, so factor it out by the **common-denominator rule**:

$$
= \frac{\sum_{j=1}^{T_x} \exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

The numerator and denominator are the same sum, just with different dummy indices $j$ and $k$, so:

$$
\sum_{j=1}^{T_x} \alpha_{ij} = 1
$$

Exactly what we need.

### 5.2 A useful identity: score differences turn into weight ratios

Take two source positions $a$ and $b$ at the same target step $i$. Their weight ratio is $\frac{\alpha_{ia}}{\alpha_{ib}} = \frac{\exp(e_{ia}) / \sum_k \exp(e_{ik})}{\exp(e_{ib}) / \sum_k \exp(e_{ik})}$. The shared denominator cancels:

$$
\frac{\alpha_{ia}}{\alpha_{ib}} = \frac{\exp(e_{ia})}{\exp(e_{ib})}
$$

Now apply the **exponential quotient identity**:

$$
\frac{e^u}{e^v} = e^{u-v}
$$

to get

$$
\boxed{\frac{\alpha_{ia}}{\alpha_{ib}} = \exp(e_{ia} - e_{ib})}
$$

This tells us something important: softmax cares about score differences, not absolute score levels.

### 5.3 Numerical check: compute the weights

We start from

$$
(e_{1,1}, e_{1,2}, e_{1,3}) \approx (0.6044, 0.8854, 0.5370)
$$

Exponentiate each score:

$$
\exp(0.6044) \approx 1.8302
$$

$$
\exp(0.8854) \approx 2.4235
$$

$$
\exp(0.5370) \approx 1.7108
$$

Add them:

$$
Z_1 = 1.8302 + 2.4235 + 1.7108 = 5.9645
$$

Now divide:

$$
\alpha_{1,1} = \frac{1.8302}{5.9645} \approx 0.3069
$$

$$
\alpha_{1,2} = \frac{2.4235}{5.9645} \approx 0.4063
$$

$$
\alpha_{1,3} = \frac{1.7108}{5.9645} \approx 0.2868
$$

So

$$
\boxed{(\alpha_{1,1}, \alpha_{1,2}, \alpha_{1,3}) \approx (0.3069, 0.4063, 0.2868)}
$$

Verification:

$$
0.3069 + 0.4063 + 0.2868 = 1.0000
$$

### 5.4 Numerical check of the ratio identity

Take the ratio of the two largest attention weights:

$$
\frac{\alpha_{1,2}}{\alpha_{1,1}} = \frac{0.4063}{0.3069} \approx 1.324
$$

Now compute the exponential of the score difference:

$$
\exp(e_{1,2} - e_{1,1}) = \exp(0.8854 - 0.6044) = \exp(0.2810) \approx 1.324
$$

Both sides match.

This identity is useful because it tells us exactly how much a score advantage translates into a weight advantage.

### 5.5 Another useful identity: adding the same constant changes nothing

Softmax has a second property that matters constantly in implementations. If we add the same constant $c$ to every score in one row, the attention weights do not change.

Start from

$$
\tilde{\alpha}_{ij} = \frac{\exp(e_{ij} + c)}{\sum_k \exp(e_{ik} + c)}
$$

Use the **exponential product rule**

$$
e^{u+v} = e^u e^v
$$

in both numerator and denominator:

$$
\tilde{\alpha}_{ij} = \frac{\exp(e_{ij})\exp(c)}{\sum_k \exp(e_{ik})\exp(c)}
$$

The factor $\exp(c)$ is common to every term in the denominator, so factor it out:

$$
\tilde{\alpha}_{ij} = \frac{\exp(e_{ij})\exp(c)}{\exp(c)\sum_k \exp(e_{ik})}
$$

Now cancel the common factor:

$$
\tilde{\alpha}_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})} = \alpha_{ij}
$$

So we have the **softmax shift-invariance identity**:

$$
\boxed{\text{softmax}(e_i + c\mathbf{1}) = \text{softmax}(e_i)}
$$

### 5.6 Numerical check of shift invariance

Take our score vector

$$
(0.6044,\ 0.8854,\ 0.5370)
$$

and add 5 to every entry:

$$
(5.6044,\ 5.8854,\ 5.5370)
$$

Exponentiate:

$$
e^{5.6044} \approx 271.60,\qquad e^{5.8854} \approx 359.58,\qquad e^{5.5370} \approx 253.89
$$

Add them:

$$
Z = 271.60 + 359.58 + 253.89 = 885.07
$$

Normalize:

$$
\left(
\frac{271.60}{885.07},
\frac{359.58}{885.07},
\frac{253.89}{885.07}
\right)
\approx
(0.3069,\ 0.4063,\ 0.2868)
$$

Exactly the same weights as before.

This is why practical implementations subtract the row maximum before exponentiating. It changes nothing mathematically, but it prevents large exponentials from blowing up numerically.

---

## 6. The Context Vector

Now we finally have everything needed for the actual context vector:

$$
c_1 = \sum_{j=1}^{3} \alpha_{1j} h_j
$$

Expand the sum:

$$
c_1 = \alpha_{1,1} h_1 + \alpha_{1,2} h_2 + \alpha_{1,3} h_3
$$

Substitute the numbers:

$$
c_1 = 0.3069 \cdot 0.2 + 0.4063 \cdot 0.9 + 0.2868 \cdot 0.1
$$

### 6.1 Numerical check

Work through the three terms separately:

$$
0.3069 \cdot 0.2 = 0.0614
$$

$$
0.4063 \cdot 0.9 = 0.3657
$$

$$
0.2868 \cdot 0.1 = 0.0287
$$

Add them:

$$
c_1 = 0.0614 + 0.3657 + 0.0287 = 0.4558
$$

So the first context vector is

$$
\boxed{c_1 = 0.4558}
$$

### 6.2 Interpretation

The value $0.4558$ is not equal to any one source annotation, and that is the whole point. The decoder is not copying one source state — it is retrieving a weighted mixture. Source word 2 pulls the context upward because it received the largest weight, while source words 1 and 3 pull it back down because their annotations are smaller. The result sits between the source values:

$$
0.1 \le 0.4558 \le 0.9
$$

as predicted by the convex-combination argument earlier.

### 6.3 Compare with the no-attention baseline

Without attention, we said the simplest old context would be $c = h_3 = 0.1$. With attention, the decoder instead receives $c_1 = 0.4558$. The difference is not a small tweak — it is a different access pattern. In the old model, every target word receives the same source summary. In the attention model, each target word performs its own retrieval from the full source memory bank.

### 6.4 Rewriting the context to see what matters most

Because the weights sum to 1, we can eliminate one of them. Write

$$
\alpha_{1,3} = 1 - \alpha_{1,1} - \alpha_{1,2}
$$

and substitute this into the context formula:

$$
c_1 = \alpha_{1,1} h_1 + \alpha_{1,2} h_2 + (1 - \alpha_{1,1} - \alpha_{1,2}) h_3
$$

Distribute $h_3$ by the **distributive law**:

$$
c_1 = \alpha_{1,1} h_1 + \alpha_{1,2} h_2 + h_3 - \alpha_{1,1} h_3 - \alpha_{1,2} h_3
$$

Group the $\alpha_{1,1}$ and $\alpha_{1,2}$ terms:

$$
c_1 = h_3 + \alpha_{1,1}(h_1 - h_3) + \alpha_{1,2}(h_2 - h_3)
$$

Now substitute our source annotations:

$$
c_1 = 0.1 + \alpha_{1,1}(0.2 - 0.1) + \alpha_{1,2}(0.9 - 0.1)
$$

So

$$
\boxed{c_1 = 0.1 + 0.1\alpha_{1,1} + 0.8\alpha_{1,2}}
$$

This form is easier to read. It tells us immediately which source positions matter most. Increasing $\alpha_{1,1}$ moves the context up only slightly, because $h_1$ sits just $0.1$ above $h_3$. Increasing $\alpha_{1,2}$ moves it much more strongly, because $h_2$ sits $0.8$ above $h_3$.

### 6.5 Numerical check of the rewritten form

Substitute the attention weights:

$$
c_1 = 0.1 + 0.1(0.3069) + 0.8(0.4063)
$$

Compute each term:

$$
0.1(0.3069) = 0.03069
$$

$$
0.8(0.4063) = 0.32504
$$

Add them:

$$
c_1 = 0.1 + 0.03069 + 0.32504 = 0.45573
$$

which matches the earlier result up to rounding:

$$
0.45573 \approx 0.4558
$$

This rewritten form makes the dominant contribution of source word 2 completely explicit.

---

## 7. Why the Mechanism Is Trainable End to End

The weighted-sum formulation is not only expressive. It is differentiable from end to end.

### 7.1 Gradient with respect to the attention weights

Treat the encoder annotations as fixed for a moment. The context vector is $c_i = \sum_j \alpha_{ij} h_j$, so differentiating with respect to one weight $\alpha_{ij}$ gives $\frac{\partial c_i}{\partial \alpha_{ij}} = \frac{\partial}{\partial \alpha_{ij}} \left(\sum_{r} \alpha_{ir} h_r\right)$. By the **linearity of differentiation**, every term with $r \ne j$ differentiates to zero because it does not depend on $\alpha_{ij}$, and only the $r=j$ term remains:

$$
\frac{\partial c_i}{\partial \alpha_{ij}}
= h_j
$$

So the gradient flowing into $\alpha_{ij}$ is directly modulated by the encoder annotation $h_j$.

### 7.2 Numerical check

In our running example,

$$
c_1 = \alpha_{1,1} h_1 + \alpha_{1,2} h_2 + \alpha_{1,3} h_3
$$

so

$$
\frac{\partial c_1}{\partial \alpha_{1,2}} = h_2 = 0.9
$$

and likewise

$$
\frac{\partial c_1}{\partial \alpha_{1,1}} = h_1 = 0.2,
\qquad
\frac{\partial c_1}{\partial \alpha_{1,3}} = h_3 = 0.1
$$

This makes the derivative formula concrete. A small change in $\alpha_{1,2}$ changes the context more strongly than the same-sized change in $\alpha_{1,1}$ or $\alpha_{1,3}$, because source position 2 carries the largest annotation in this toy example.

### 7.3 Why this matters

This means the loss can push on the attention weights smoothly. Those weights are smooth functions of the alignment scores through softmax, and the alignment scores are smooth functions of the encoder and decoder states through the feedforward alignment model. So gradients flow from the loss through $c_i$, then through the attention weights $\alpha_{ij}$, then through the alignment scores $e_{ij}$, and finally into the encoder and decoder states $(s_{i-1}, h_j)$. That is why the whole system can be trained end to end with backpropagation.

---

## 8. Where the Encoder Annotations Come From

So far we have treated $h_1, h_2, h_3$ as given. In the actual model, they come from a bidirectional encoder.

Bahdanau et al. obtain them from a **Bidirectional Recurrent Neural Network**.

### 8.1 Forward and backward states

The forward encoder reads the source left to right, computing $\overrightarrow{h}_t = f(x_t, \overrightarrow{h}_{t-1})$, while the backward encoder reads right to left, computing $\overleftarrow{h}_t = f(x_t, \overleftarrow{h}_{t+1})$. The annotation at source position $j$ is the concatenation of both directions:

$$
h_j = \begin{bmatrix}
\overrightarrow{h}_j \\
\overleftarrow{h}_j
\end{bmatrix}
$$

So each annotation contains both left context and right context. If the source word is "bank," its meaning may depend on both the word before it and the word after it, and a bidirectional annotation can encode that local context before attention ever begins. That is why the encoder states are a good memory bank — each $h_j$ is not a raw word embedding but already a contextual summary centered on source position $j$.

### 8.2 Numerical check

Our scalar running example hid this structure by collapsing each annotation to one number. In an actual bidirectional encoder, each annotation would contain one part from the left-to-right pass and one part from the right-to-left pass.

For example, suppose that at source position 2 the forward encoder produces

$$
\overrightarrow{h}_2 = 0.4
$$

and the backward encoder produces

$$
\overleftarrow{h}_2 = 0.7
$$

Then the bidirectional annotation at that position is

$$
h_2 =
\begin{bmatrix}
0.4 \\
0.7
\end{bmatrix}
$$

The important point is not the specific numbers. The important point is that one source position now carries information from both directions at once. Attention does not read from raw source words. It reads from contextualized source annotations.

---

## 9. The Full Decoder Objective

The decoder does not stop at computing $c_i$. At target step $i$, it computes the attention-based context vector, updates the decoder state, and then predicts the next word.

A standard attention-based decoder step can be written abstractly as

$$
s_i = f(s_{i-1}, y_{i-1}, c_i)
$$

followed by

$$
p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, c_i)
$$

The first equation says that the new decoder state is built from three things: the old decoder state, the previous target token, and the source information retrieved for the current step. The second says that this updated state and retrieved context are then used to predict the next target word. So attention does not replace the decoder. It gives the decoder a better source-dependent input at each step.

The full translation probability factorizes by the **chain rule of probability**:

$$
p(\mathbf{y}\mid\mathbf{x})
= \prod_{i=1}^{T_y} p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x})
$$

Taking logs and using the **logarithm product rule**

$$
\log(ab) = \log a + \log b
$$

gives

$$
\log p(\mathbf{y}\mid\mathbf{x})
= \sum_{i=1}^{T_y} \log p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x})
$$

### 9.1 Numerical check

Suppose a target sentence has two words, and the model assigns

$$
p(y_1 \mid \mathbf{x}) = 0.8,
\qquad
p(y_2 \mid y_1, \mathbf{x}) = 0.6
$$

Then the full sentence probability is

$$
p(\mathbf{y}\mid\mathbf{x}) = 0.8 \cdot 0.6 = 0.48
$$

Taking logs gives

$$
\log p(\mathbf{y}\mid\mathbf{x}) = \log(0.48) \approx -0.73397
$$

Now compute the sum of the two token-level log-probabilities:

$$
\log(0.8) + \log(0.6) \approx -0.22314 + (-0.51083) = -0.73397
$$

The two results match exactly, which is why training can be written as a sum over target positions rather than as one monolithic sentence-level quantity.

This is the training objective the model maximizes. The alignment model receives no direct supervision. It learns useful alignments only because better alignments improve the translation log-likelihood.

---

## 10. The Alignment Matrix and the Memory-Retrieval View

If we stack the attention weights over all target positions, we get the **alignment matrix**:

$$
\Alpha =
\begin{bmatrix}
\alpha_{1,1} & \alpha_{1,2} & \cdots & \alpha_{1,T_x} \\
\alpha_{2,1} & \alpha_{2,2} & \cdots & \alpha_{2,T_x} \\
\vdots & \vdots & \ddots & \vdots \\
\alpha_{T_y,1} & \alpha_{T_y,2} & \cdots & \alpha_{T_y,T_x}
\end{bmatrix}
$$

Each row is a probability distribution over source positions for one target token. This makes the memory interpretation explicit: the encoder annotations $h_j$ are the memory slots, the decoder state $s_{i-1}$ is the query, the scores $e_{ij}$ measure compatibility between that query and each slot, softmax turns those compatibilities into a distribution, and the weighted sum returns the retrieved content.

This is already most of the modern query-key-value story, just without the names. In Bahdanau attention, the query is the decoder state, the key is the encoder annotation, and the value is the encoder annotation as well. The Transformer will keep the same memory-retrieval pattern but separate keys and values into different learned projections.

---

## 11. Common Confusions

### 11.1 Attention is not hard selection

Attention does not pick one source token and ignore the rest. Even when one weight is large, the output is still a weighted average. In our running example, source word 2 has the largest weight at $0.4063$, but it does not get all the mass — the other two source words still contribute.

### 11.2 Attention weights are not explanations by default

The attention weights tell us where the model routed information for this mechanism. They do not automatically tell us why the translation is correct or whether the model "understood" the sentence in a human sense. They are routing coefficients, not magical semantic certificates.

### 11.3 The toy scalar model is less expressive than the real one

This is the part that usually feels wrong on a first pass through a tiny example. In our scalar toy model,

$$
e_{ij} = \tanh(s_{i-1} + h_j)
$$

and $\tanh$ is increasing. So if $h_2 > h_1 > h_3$, then source word 2 will always receive the highest score for any fixed decoder state added equally to all three.

That might make it seem like attention cannot change its ranking across target positions.

That conclusion would be wrong. The full model is

$$
e_{ij} = \mathbf{v}_a^\top \tanh(W_a s_{i-1} + U_a h_j)
$$

where $s_{i-1}$ and $h_j$ are vectors and $W_a$, $U_a$, $\mathbf{v}_a$ are learned. Different decoder states can interact with different encoder annotations in different directions, so the ranking can change from one target position to the next.

To see that change explicitly, leave the scalar toy for a moment and take a tiny 2-dimensional example with two source positions:

$$
h_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix},
\qquad
h_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix},
\qquad
W_a = I,\quad U_a = I,\quad \mathbf{v}_a = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

Now compare two decoder states.

For the first target step, let

$$
s_0 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

Then

$$
e_{1,1}
= \mathbf{v}_a^\top \tanh(s_0 + h_1)
= \begin{bmatrix} 1 & 1 \end{bmatrix}
\tanh\!\left(\begin{bmatrix} 2 \\ 0 \end{bmatrix}\right)
= \begin{bmatrix} 1 & 1 \end{bmatrix}
\begin{bmatrix} 0.9640 \\ 0 \end{bmatrix}
= 0.9640
$$

while

$$
e_{1,2}
= \mathbf{v}_a^\top \tanh(s_0 + h_2)
= \begin{bmatrix} 1 & 1 \end{bmatrix}
\tanh\!\left(\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right)
= \begin{bmatrix} 1 & 1 \end{bmatrix}
\begin{bmatrix} 0.7616 \\ 0.7616 \end{bmatrix}
= 1.5232
$$

So at the first target step, source position 2 receives the higher score.

For the second target step, let

$$
s_1 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

Then the same computation gives

$$
e_{2,1} = 1.5232,
\qquad
e_{2,2} = 0.9640
$$

Now source position 1 receives the higher score. The ranking has flipped.

If we softmax each pair of scores, the first step gives attention weights approximately

$$
(\alpha_{1,1}, \alpha_{1,2}) \approx (0.3637,\, 0.6363)
$$

while the second step gives

$$
(\alpha_{2,1}, \alpha_{2,2}) \approx (0.6363,\, 0.3637)
$$

So the decoder really can look in different places at different target positions. Our scalar example is useful because it makes the arithmetic transparent, while the full vector model is useful because it restores expressive power.

---

## Summary

Attention solves the fixed-length bottleneck by replacing one global source summary with a target-specific weighted sum of encoder annotations. Bahdanau's alignment model produces scores $e_{ij}$, softmax turns them into valid weights $\alpha_{ij}$, and the context vector $c_i = \sum_j \alpha_{ij} h_j$ becomes a differentiable soft retrieval from the source memory bank.

In our running example, the mechanism turns source annotations $(0.2, 0.9, 0.1)$ and decoder state $s_0 = 0.5$ into attention weights $(0.3069, 0.4063, 0.2868)$ and the context value $c_1 = 0.4558$. The next post keeps this retrieval view intact but asks a new question: why use this feedforward alignment function at all, and how do we arrive at queries, keys, and values?

---

*Previous: [Mathematical Prerequisites for the Attention Series](/blog/math-prerequisites-for-attention)*  
*Next: [From Soft Alignment to Queries, Keys, and Values](/blog/attention-q-k-v-from-scratch)*
