---
title: "What Attention is Really Doing: Weighted Memory Retrieval from Scratch"
description: "Building attention from first principles — the fixed-length bottleneck that broke RNN encoder–decoders on long sentences, the weighted-sum solution introduced by Bahdanau, Cho, and Bengio (2015), alignment score derivation, softmax normalization, and a complete 3-word numerical walkthrough — all derived step by step"
date: 2026-03-31
tags: ["deep-learning", "attention", "transformers", "nlp", "neural-machine-translation"]
order: 4
---

Before the Transformer, before Q/K/V, before any of the modern terminology, there was a concrete failure mode: sequence-to-sequence models fell apart on long sentences. The fix that Bahdanau, Cho, and Bengio proposed in 2015 is what we now call attention. We will derive it from first principles, starting from the failure and arriving at the mechanism.

The running example we will use for every derivation in this post: a 3-word source sequence with encoder hidden states $h_1 = 0.2$, $h_2 = 0.9$, $h_3 = 0.1$, and a decoder state $s_0 = 0.5$. We will compute alignment scores, attention weights, and a context vector entirely by hand, step by step.

---

## 1. Before Attention: The Fixed-Length Bottleneck

### 1.1 The Encoder–Decoder Framework

A **sequence-to-sequence** (seq2seq) model maps a source sequence $\mathbf{x} = (x_1, x_2, \ldots, x_{T_x})$ to a target sequence $\mathbf{y} = (y_1, y_2, \ldots, y_{T_y})$ with potentially different lengths. The standard approach before 2015 used an **encoder–decoder framework** with two components:

1. An **encoder** that reads the source sequence and produces a fixed-size context vector $\mathbf{c}$.
2. A **decoder** that generates each target word from $\mathbf{c}$, conditioned on previously generated words.

The decoder models each conditional probability as:

$$
p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, \mathbf{c})
$$

where $s_i$ is the decoder's hidden state at step $i$, and $g$ is a nonlinear function. The same fixed vector $\mathbf{c}$ appears in every call to $g$ — for generating $y_1$, $y_2$, $\ldots$, $y_{T_y}$.

### 1.2 The Bottleneck

The encoder is a recurrent neural network (RNN) that reads $x_1, x_2, \ldots, x_{T_x}$ sequentially:

$$
h_t = f(x_t, h_{t-1})
$$

The context vector is computed from all hidden states, most commonly as just the final hidden state:

$$
\mathbf{c} = h_{T_x}
$$

This is the problem. The entire source sentence — every word, every dependency, every long-range relationship — must be compressed into a single fixed-size vector. The decoder reads from this one vector to generate every target word.

Consider translating a 44-word sentence. Every piece of information about every word must fit inside $\mathbf{c} \in \mathbb{R}^{1000}$. When the decoder generates word 40 of the translation, it has access only to this same compressed summary — it cannot go back to look at specific source words.

Cho et al. (2014b) verified this empirically: the BLEU score of RNN encoder–decoders drops sharply as source sentence length increases. Bahdanau et al. (2015) called this the **fixed-length context vector bottleneck** and set out to remove it.

---

## 2. The Running Example

We will use this concrete example throughout every derivation in this post.

**Source sequence**: 3 words — word$_1$, word$_2$, word$_3$. After encoding through a bidirectional RNN, each word is represented by a scalar hidden state (annotation):

$$
h_1 = 0.2, \quad h_2 = 0.9, \quad h_3 = 0.1
$$

Think of these numbers as capturing how informationally salient each source word is. Word$_2$ ($h_2 = 0.9$) is the most prominent; words 1 and 3 are quieter.

**Decoder initial state**: $s_0 = 0.5$. This represents the decoder's hidden state just before it generates the first target word.

**Task**: Compute the context vector $c_1$ that the decoder should use when generating the first target word $y_1$.

---

## 3. The Fix: A Position-Dependent Context Vector

The key insight of Bahdanau et al. is to replace the single fixed $\mathbf{c}$ with a **distinct context vector $c_i$ for each target position $i$**. Instead of forcing the entire source sentence into one summary, the decoder gets a customized view of the source at each generation step.

We define:

$$
\boxed{c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j}
$$

The context vector $c_i$ is a **weighted sum** of all encoder hidden states. The weight $\alpha_{ij}$ answers: "when generating the $i$-th target word, how much should I attend to the $j$-th source word?"

This is the entire core of attention. Everything else — the alignment model, the softmax — is machinery for computing the weights $\alpha_{ij}$.

### 3.1 Why a Weighted Sum?

There are two key properties we want from the weights $\alpha_{ij}$:

1. **Non-negativity**: $\alpha_{ij} \geq 0$ for all $i, j$.
2. **Normalization**: $\sum_{j=1}^{T_x} \alpha_{ij} = 1$ for each target position $i$.

Together, these make $c_i$ a **convex combination** of the encoder states $h_1, \ldots, h_{T_x}$. Geometrically, $c_i$ lies inside the convex hull of the hidden states. It can equal any one $h_j$ exactly (by setting $\alpha_{ij} = 1$ for that $j$ and 0 for the rest), or blend multiple states.

The alternative — a hard selection, where $\alpha_{ij} = 1$ for exactly one $j$ and 0 elsewhere — would be computationally clean, but it is not differentiable. We cannot compute gradients through a discrete argmax. The soft weighted sum is differentiable everywhere, so gradients flow cleanly from the decoder's loss all the way back through the attention weights into the encoder. This differentiability is the entire reason for using a soft sum rather than a hard selection.

---

## 4. The Alignment Score

To compute the weights $\alpha_{ij}$, we first need unnormalized **alignment scores** — numbers that quantify how well source position $j$ aligns with generating target position $i$.

We define an **alignment model** $a$:

$$
e_{ij} = a(s_{i-1}, h_j)
$$

where $s_{i-1}$ is the decoder hidden state just before generating $y_i$ (not after — we need to know what we are about to generate), and $h_j$ is the $j$-th encoder annotation.

The function $a$ is a learned **compatibility function**: it returns a high score when the decoder state $s_{i-1}$ and the encoder state $h_j$ are aligned in the sense that $h_j$ contains useful information for generating $y_i$.

### 4.1 Bahdanau's Alignment Model

Bahdanau et al. parametrize $a$ as a single-hidden-layer feedforward network:

$$
e_{ij} = \mathbf{v}_a^\top \tanh\!\left(W_a s_{i-1} + U_a h_j\right)
$$

where $W_a \in \mathbb{R}^{n' \times n}$ and $U_a \in \mathbb{R}^{n' \times 2n}$ are weight matrices, and $\mathbf{v}_a \in \mathbb{R}^{n'}$ is a weight vector. The scalar output $e_{ij}$ is the dot product of $\mathbf{v}_a$ with the $\tanh$ nonlinearity applied to the sum of two linear projections.

A critical efficiency observation from the paper: since $U_a h_j$ does not depend on the decoder step $i$, it can be **precomputed once** for all $j$ before decoding starts. This saves $T_y \times T_x$ redundant matrix-vector multiplications.

For our scalar running example ($n = n' = 1$), with all weight scalars set to 1 (i.e., $W_a = U_a = v_a = 1$), the alignment model becomes:

$$
e_{ij} = \tanh\!\left(s_{i-1} + h_j\right)
$$

This is the **hyperbolic tangent function** applied to the sum $s_{i-1} + h_j$. The hyperbolic tangent is defined as:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

and maps any real input to the interval $(-1, +1)$.

### 4.2 Numerical Check: Alignment Scores for Target Word 1

We compute $e_{1,j}$ for $j = 1, 2, 3$, using $s_0 = 0.5$.

**Score for source word 1** ($h_1 = 0.2$):

$$
e_{1,1} = \tanh(s_0 + h_1) = \tanh(0.5 + 0.2) = \tanh(0.7)
$$

Computing $\tanh(0.7)$ from the definition:

$$
\tanh(0.7) = \frac{e^{0.7} - e^{-0.7}}{e^{0.7} + e^{-0.7}} = \frac{2.0138 - 0.4966}{2.0138 + 0.4966} = \frac{1.5172}{2.5104} \approx 0.6044
$$

**Score for source word 2** ($h_2 = 0.9$):

$$
e_{1,2} = \tanh(s_0 + h_2) = \tanh(0.5 + 0.9) = \tanh(1.4)
$$

$$
\tanh(1.4) = \frac{e^{1.4} - e^{-1.4}}{e^{1.4} + e^{-1.4}} = \frac{4.0552 - 0.2466}{4.0552 + 0.2466} = \frac{3.8086}{4.3018} \approx 0.8854
$$

**Score for source word 3** ($h_3 = 0.1$):

$$
e_{1,3} = \tanh(s_0 + h_3) = \tanh(0.5 + 0.1) = \tanh(0.6)
$$

$$
\tanh(0.6) = \frac{e^{0.6} - e^{-0.6}}{e^{0.6} + e^{-0.6}} = \frac{1.8221 - 0.5488}{1.8221 + 0.5488} = \frac{1.2733}{2.3709} \approx 0.5370
$$

Collecting the three scores:

$$
e_{1,1} \approx 0.6044, \quad e_{1,2} \approx 0.8854, \quad e_{1,3} \approx 0.5370
$$

These numbers say: source word 2 ($h_2 = 0.9$, the most prominent source word) aligns most strongly with generating target word 1. Word 1 and word 3 have similar, lower alignment scores.

---

## 5. From Scores to Weights: Softmax

The alignment scores $e_{ij}$ are real numbers in $(-1, +1)$ (because $\tanh$ is bounded). We need to convert them into proper weights: non-negative and summing to 1.

The **softmax function** achieves this. For each target position $i$, we apply softmax over all source positions $j$:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\displaystyle\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

### 5.1 Why Softmax?

We could normalize directly: $\alpha_{ij} = e_{ij} / \sum_k e_{ik}$. But this fails if any score is negative — the weights could be negative, violating non-negativity. Exponentiating first ensures all weights are strictly positive, since $\exp(x) > 0$ for all real $x$.

The softmax also has a sharpening effect: if one score is much larger than the others, the softmax concentrates nearly all weight on that one position. If all scores are equal, the softmax gives uniform weights.

This sharpening connects to the **Boltzmann distribution** from statistical mechanics. If we interpret $e_{ij}$ as a negative energy and $\alpha_{ij}$ as the probability of "occupying state $j$," the softmax is exactly the Boltzmann distribution. At high temperature (scores close together), the distribution is diffuse. At low temperature (one score far larger than the rest), the distribution is sharply peaked.

### 5.2 Numerical Check: Attention Weights for Target Word 1

Starting from $e_{1,1} \approx 0.6044$, $e_{1,2} \approx 0.8854$, $e_{1,3} \approx 0.5370$.

**Exponentiate each score:**

$$
\exp(0.6044) \approx 1.8302
$$
$$
\exp(0.8854) \approx 2.4235
$$
$$
\exp(0.5370) \approx 1.7108
$$

**Compute the normalizing constant:**

$$
Z_1 = 1.8302 + 2.4235 + 1.7108 = 5.9645
$$

**Divide each by $Z_1$:**

$$
\alpha_{1,1} = \frac{1.8302}{5.9645} \approx 0.3069
$$

$$
\alpha_{1,2} = \frac{2.4235}{5.9645} \approx 0.4063
$$

$$
\alpha_{1,3} = \frac{1.7108}{5.9645} \approx 0.2868
$$

**Verification**: $0.3069 + 0.4063 + 0.2868 = 1.0000$. ✓

The attention weights say: when generating target word 1, we place 40.6% of our attention on source word 2, 30.7% on source word 1, and 28.7% on source word 3. No source word is ignored entirely — all three contribute. But source word 2 (the most salient, with $h_2 = 0.9$) dominates.

---

## 6. The Context Vector

With the attention weights computed, the context vector is a weighted sum:

$$
c_1 = \sum_{j=1}^{3} \alpha_{1,j} \cdot h_j = \alpha_{1,1} \cdot h_1 + \alpha_{1,2} \cdot h_2 + \alpha_{1,3} \cdot h_3
$$

This operation — a weighted sum of vectors with weights provided by a separate mechanism — is the **inner product** of the attention weight vector $[\alpha_{1,1},\, \alpha_{1,2},\, \alpha_{1,3}]$ with the hidden state vector $[h_1,\, h_2,\, h_3]$.

### 6.1 Numerical Check

$$
c_1 = 0.3069 \times 0.2 + 0.4063 \times 0.9 + 0.2868 \times 0.1
$$

Working through each term:
- $0.3069 \times 0.2 = 0.0614$
- $0.4063 \times 0.9 = 0.3657$
- $0.2868 \times 0.1 = 0.0287$

$$
\boxed{c_1 = 0.0614 + 0.3657 + 0.0287 = 0.4558}
$$

### 6.2 Comparison: With vs. Without Attention

Without attention, the decoder would use a single fixed context for all target positions. In the simplest case — using only the last encoder hidden state — that would be $c = h_3 = 0.1$. The decoder would generate every target word from the same poor summary.

With attention, the decoder uses $c_1 = 0.4558$ for generating word 1. For generating word 2, the decoder state $s_1$ will differ, producing different alignment scores, different weights $\alpha_{2,j}$, and a different context $c_2$. Each target word gets a customized view of the source sentence.

The value $c_1 = 0.4558$ sits between $h_2 = 0.9$ (the most attended source word) and the others. The 40.6% weight on $h_2$ pulls $c_1$ toward 0.9; the remaining words temper it back toward their lower values.

---

## 7. The Encoder: Annotations from a Bidirectional RNN

We have been treating $h_1, h_2, h_3$ as given. Where do they come from?

Bahdanau et al. use a **bidirectional RNN** (BiRNN) encoder, introduced by Schuster and Paliwal (1997). A unidirectional RNN produces hidden state $h_j$ from only the words $x_1, \ldots, x_j$ — the prefix. But a good annotation $h_j$ should summarize the full context around $x_j$, including what comes after.

The **Bidirectional Recurrent Neural Network** addresses this with two passes:

**Forward pass**: Read $x_1, x_2, \ldots, x_{T_x}$ left to right:
$$
\overrightarrow{h}_t = f(x_t,\, \overrightarrow{h}_{t-1}), \quad \overrightarrow{h}_0 = \mathbf{0}
$$

**Backward pass**: Read $x_{T_x}, x_{T_x-1}, \ldots, x_1$ right to left:
$$
\overleftarrow{h}_t = f(x_t,\, \overleftarrow{h}_{t+1}), \quad \overleftarrow{h}_{T_x+1} = \mathbf{0}
$$

The annotation for word $j$ is the **concatenation** of both hidden states:

$$
h_j = \left[\overrightarrow{h}_j^\top;\; \overleftarrow{h}_j^\top\right]^\top \in \mathbb{R}^{2n}
$$

This makes $h_j$ a function of the entire source sequence, with strong focus on word $j$ — the forward state has just processed $x_j$ from the left, and the backward state has just processed $x_j$ from the right. In our running example, $h_1, h_2, h_3$ are these concatenated annotations, treated as scalars for simplicity.

---

## 8. The Full Decoder

The decoder generates each target word $y_i$ using three things:

1. The attention-based context vector $c_i$ (derived above).
2. The decoder hidden state update: $s_i = f(s_{i-1}, y_{i-1}, c_i)$.
3. The output probability: $p(y_i \mid s_i, y_{i-1}, c_i)$.

The model is trained end-to-end to maximize the **log-likelihood** of the correct translations. By the **chain rule of probability** (the factorization of a joint distribution into a product of conditionals):

$$
\log p(\mathbf{y} \mid \mathbf{x}) = \sum_{i=1}^{T_y} \log p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x})
$$

Gradients flow from the loss back through $g$, through $s_i$, through $c_i$, through $\alpha_{ij}$, through $e_{ij}$, and into both the encoder and the alignment model. Every parameter — encoder, alignment model, decoder — is trained jointly.

The alignment model receives **no explicit supervision**. It discovers useful alignments purely from the translation objective. The model learns to align source and target words because doing so improves translation quality, not because we told it what alignment means.

---

## 9. What the Model Learns: The Alignment Matrix

One of the most striking results in Bahdanau et al. (2015) is qualitative: the model learns linguistically plausible alignments without any explicit alignment supervision.

If we visualize the full attention weight matrix $\alpha_{ij}$ — with target positions on one axis and source positions on the other — we get an **alignment matrix**. Each entry $\alpha_{ij}$ is the weight the decoder placed on source word $j$ when generating target word $i$.

For English-to-French translation, this matrix is nearly diagonal — English and French have similar word order. But for multi-word phrases that translate differently, the alignment is non-monotone. The paper shows "European Economic Area" translating to "zone économique européenne": the noun comes last in French, so the alignment reorders. When generating "zone," the model attends to "Area." When generating "économique," it attends to "Economic." When generating "européenne," it attends to "European."

This reordering was learned automatically from translation data, with no instruction about French grammar.

Another example: "the" in English can translate to "l'," "le," "la," or "les" in French depending on the following noun's gender and number. To translate "the" correctly, the decoder must look forward at the noun. Soft alignment handles this: "the" can simultaneously attend to itself and the following noun, blending their representations into a context vector that contains the information needed to choose the correct article.

---

## 10. Attention as Soft Memory Retrieval

Let us now name what we have built.

The encoder hidden states $h_1, h_2, h_3$ form a **memory bank** — a collection of stored contextual representations, one for each source position. Each $h_j$ is not an isolated word embedding; it encodes the word and its surrounding context (because the BiRNN processes the entire sequence before and after $x_j$).

The decoder current state $s_{i-1}$ acts as a **query**: "what do I need to know right now, given where I am in the generation?"

The alignment model $a(s_{i-1}, h_j)$ computes a **compatibility score** between the query and each memory entry. Softmax converts these scores into a **probability distribution over memory positions**. The context vector $c_i$ is the **expected memory content** under this distribution — a soft retrieval.

This framing — query, key, value — will be made explicit in the Transformer paper (Vaswani et al., 2017). In Bahdanau's model:

- The **query** is $s_{i-1}$ (decoder state).
- The **keys** are $h_1, \ldots, h_{T_x}$ (encoder states, used for scoring).
- The **values** are also $h_1, \ldots, h_{T_x}$ (the same encoder states, used for retrieval).

In Bahdanau's model, keys and values are the same vectors. The Transformer will separate them into independent projections. That separation turns out to matter — we will derive why in the next post.

---

## 11. What Attention is NOT Doing

This point trips people up, so let us be direct.

Attention does **not** select one thing. It computes a soft weighted average. Even if $\alpha_{1,2} = 0.9999$, the context vector is not exactly $h_2$ — it is $0.9999 \cdot h_2 + 0.0001 \cdot (h_1 + h_3)$. Hard selection requires discrete sampling; soft attention never does.

Attention does **not** understand meaning. The weights $\alpha_{ij}$ are determined entirely by the alignment model $a(s_{i-1}, h_j)$, which is a learned function of two vectors. Whether the resulting attention patterns are semantically meaningful depends on what was learned during training, not on any intrinsic property of the mechanism.

Attention does **not** inherently encode position. Bahdanau et al. use a BiRNN encoder, where each $h_j$ encodes position implicitly through the recurrence. The Transformer drops the RNN and must add explicit positional encodings — that story belongs to the next post.

---

## Summary

We started with the fixed-length bottleneck: compressing the entire source into one vector causes performance to degrade sharply on long sentences. The fix, derived step by step:

1. Produce one hidden state $h_j$ per source position (BiRNN encoder).
2. At each decoding step $i$, compute alignment scores $e_{ij} = \mathbf{v}_a^\top \tanh(W_a s_{i-1} + U_a h_j)$ — a learned compatibility function between decoder state and each source annotation.
3. Apply softmax to get attention weights: $\alpha_{ij} = \exp(e_{ij}) / \sum_k \exp(e_{ik})$.
4. Compute a position-specific context: $c_i = \sum_j \alpha_{ij} h_j$.

In our running example: alignment scores $[0.6044,\, 0.8854,\, 0.5370]$, attention weights $[0.3069,\, 0.4063,\, 0.2868]$, context vector $c_1 = 0.4558$.

In the next post, we ask: why use a feedforward network as the compatibility function? What if we used a dot product instead? And why would we want separate query, key, and value vectors — the three matrices that define the Transformer?

---

*Previous: [Mathematical Prerequisites for the Attention Series](/blog/math-prerequisites-for-attention)*  
*Next: [From Soft Alignment to Queries, Keys, and Values](/blog/attention-q-k-v-from-scratch)*
