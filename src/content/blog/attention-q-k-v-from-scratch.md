---
title: "From Soft Alignment to Queries, Keys, and Values: Deriving the Transformer's Attention"
description: "The Q/K/V abstraction derived from first principles — why Bahdanau's feedforward alignment collapses to a dot product, how the Transformer formalizes queries, keys, and values as separate projections, why we scale by √d_k (with a complete variance proof), what multi-head attention adds, and a full 3-token self-attention numerical walkthrough"
date: 2026-03-31
tags: ["deep-learning", "attention", "transformers", "q-k-v", "self-attention"]
order: 3
---

In the previous post, we derived Bahdanau attention from scratch: a feedforward compatibility function $e_{ij} = \mathbf{v}_a^\top \tanh(W_a s_{i-1} + U_a h_j)$, softmax normalization, and a dynamic context vector $c_i = \sum_j \alpha_{ij} h_j$. It worked. The model learned to align source and target words without any explicit supervision.

But the Transformer paper (Vaswani et al., 2017) replaced that feedforward compatibility function with something far simpler: a dot product. In doing so, it unlocked full parallelization, enabled self-attention, and produced the architecture that underpins every major language model today. In this post we derive why, starting from a single question: what is the simplest compatibility function that still works?

---

## 1. The Running Example

We keep one tiny sequence throughout:

$$
\mathbf{x}_1 =
\begin{bmatrix}
1 \\ 0
\end{bmatrix},
\qquad
\mathbf{x}_2 =
\begin{bmatrix}
0 \\ 1
\end{bmatrix},
\qquad
\mathbf{x}_3 =
\begin{bmatrix}
0.5 \\ 0.5
\end{bmatrix}
$$

Think of dimension 1 as encoding "noun-ness" and dimension 2 as encoding "verb-ness." Token 1 is a pure noun, token 2 is a pure verb, and token 3 is equally both.

We will compute the full scaled dot-product self-attention mechanism on these three vectors by hand.

For readability we display the token vectors vertically, but from this point on we will stack them as **rows** of the matrix $X$. So individual queries, keys, and values will also be treated as row vectors, and the scalar score between positions $i$ and $j$ will be written as $q_i k_j^\top$.

---

## 2. Reconsidering the Compatibility Function

In Bahdanau attention, the compatibility score was

$$
e_{ij} = \mathbf{v}_a^\top \tanh(W_a s_{i-1} + U_a h_j)
$$

This score answered a good question: how compatible is decoder state $s_{i-1}$ with encoder annotation $h_j$?

But the Transformer wants something else in addition. It wants the score function to work for every pair of positions in parallel, and it wants the whole score matrix to be produced by matrix multiplication.

### 2.1 What Bahdanau's Alignment Model Does

The additive alignment model mixes the two inputs inside a nonlinearity: $\tanh(W_a s_{i-1} + U_a h_j)$. The query and the key are each projected into a shared hidden space, added, passed through $\tanh$, and then dotted with $\mathbf{v}_a$. That makes the compatibility function expressive, but it also means we cannot precompute a representation of the query alone and a representation of the key alone and then combine them with one large matrix multiply — the interaction happens before the final dot product with $\mathbf{v}_a$. This is not wrong. It is just much less GPU-friendly.

### 2.2 The Dot-Product Compatibility Function

Suppose we want a score function with the form

$$
e_{ij} = q_i k_j^\top
$$

where $q_i$ depends only on token $i$ and $k_j$ depends only on token $j$.

Then the full matrix of all pairwise scores is

$$
E = QK^\top
$$

with

$$
Q =
\begin{bmatrix}
q_1^\top \\
q_2^\top \\
\vdots
\end{bmatrix},
\qquad
K =
\begin{bmatrix}
k_1^\top \\
k_2^\top \\
\vdots
\end{bmatrix}
$$

This is exactly what modern GPUs are built to do: one matrix multiply computes all pairwise similarities.

### 2.3 A bilinear view

If the input token at position $i$ is the row vector $x_i$, then the most general batched linear score looks like

$$
e_{ij} = x_i A x_j^\top
$$

for some learned matrix $A$.

Now factor $A$ as

$$
A = W_Q W_K^\top
$$

Then

$$
e_{ij} = x_i W_Q W_K^\top x_j^\top
$$

Group the terms by **associativity of matrix multiplication**:

$$
e_{ij} = (x_i W_Q)(x_j W_K)^\top
$$

Define

$$
q_i = x_i W_Q,
\qquad
k_j = x_j W_K
$$

and we get

$$
\boxed{e_{ij} = q_i k_j^\top}
$$

That is the Transformer's dot-product compatibility function.

So the dot product is not arbitrary. It is the simplest factorized bilinear score that gives us all pairwise similarities in one matrix multiply.

---

## 3. Queries, Keys, and Values

Once the compatibility function has been factorized, the three projections appear naturally.

### 3.1 Query

A **query** is what the current position is asking for. If token $i$ wants to find relevant information elsewhere in the sequence, its query is $q_i = x_i W_Q$.

### 3.2 Key

A **key** is how a position advertises what kind of information it contains. At position $j$, the key is $k_j = x_j W_K$. The query and key interact only to produce scores.

### 3.3 Value

A **value** is the actual content retrieved once a position is selected. At position $j$, the value is $v_j = x_j W_V$.

### 3.4 Matrix form

Pack the full sequence into a matrix

$$
X \in \mathbb{R}^{n \times d_\text{model}}
$$

Then the three projections are

$$
Q = XW_Q \in \mathbb{R}^{n \times d_k}
$$

$$
K = XW_K \in \mathbb{R}^{n \times d_k}
$$

$$
V = XW_V \in \mathbb{R}^{n \times d_v}
$$

### 3.5 Why keys and values are separated

This is the key conceptual difference from Bahdanau attention. In Bahdanau's model, the encoder states $h_j$ served two roles at once: they were the things scored against, and they were also the things retrieved. In the Transformer, those roles are separated — keys decide how positions are matched, while values decide what content is actually mixed into the output. A token might be easy to find for one reason and useful to retrieve for another.

### 3.6 A concrete check: scores can stay the same while values change

This is the part that usually feels too abstract on first reading, so let us pin it down with the running example.

Keep the query and key projections equal to the identity:

$$
W_Q = W_K = I
$$

so the score matrix is unchanged. But now choose a different value projection:

$$
W_V =
\begin{bmatrix}
1 & 0 \\
1 & 1
\end{bmatrix}
$$

Then the transformed values are:

$$
v_1 = [1,0]W_V = [1,0]
$$

$$
v_2 = [0,1]W_V = [1,1]
$$

$$
v_3 = [0.5,0.5]W_V = [1,0.5]
$$

Notice what happened. The keys used for matching did not change, but the values returned after matching did change. So the attention weights can stay exactly the same while the retrieved content changes. That is the practical meaning of separating keys from values.

---

## 4. Deriving the Full Attention Formula

Once we have scores, the rest of the mechanism follows the same pattern as Bahdanau attention. We compute unnormalized scores, normalize them into weights, and then retrieve a weighted combination of values.

### 4.1 Score matrix

All pairwise scores are

$$
S = QK^\top
$$

Entry $(i,j)$ is

$$
S_{ij} = q_i k_j^\top
$$

### 4.2 Row-wise softmax

For each query position $i$, we apply softmax over all keys: $A_{ij} = \frac{\exp(S_{ij})}{\sum_{r=1}^{n} \exp(S_{ir})}$. This gives a full attention weight matrix $A = \text{softmax}(S)$, where the softmax is applied row by row.

### 4.3 Weighted value retrieval

Now retrieve content by multiplying those weights by the values: $O = AV$. Entry-wise, this says $o_i = \sum_{j=1}^{n} A_{ij} v_j$. This is exactly the same weighted-memory-retrieval structure as Bahdanau attention, except that the weights came from a batched dot product instead of an additive feedforward score.

### 4.4 The scaling factor

The Transformer inserts one additional factor of $\frac{1}{\sqrt{d_k}}$, so the final formula becomes

$$
\boxed{\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}
$$

We now derive why that scaling is needed.

---

## 5. Why Divide by $\sqrt{d_k}$?

This is the part that confuses almost everyone. The scaling is not cosmetic. It is a variance correction.

### 5.1 Variance of a dot product

Assume the components of a query and key are independent standard normal variables, $q_r \sim \mathcal{N}(0,1)$ and $k_r \sim \mathcal{N}(0,1)$ for $r = 1, \ldots, d_k$. We want the variance of $q^\top k = \sum_{r=1}^{d_k} q_r k_r$.

First, the mean of one product term is $\mathbb{E}[q_r k_r] = \mathbb{E}[q_r]\mathbb{E}[k_r] = 0 \cdot 0 = 0$, using independence and the **multiplication rule for expectations of independent variables**. Now the variance of one term:

$$
\text{Var}(q_r k_r) = \mathbb{E}[(q_r k_r)^2] - (\mathbb{E}[q_r k_r])^2
$$

The second term is zero, so

$$
\text{Var}(q_r k_r) = \mathbb{E}[q_r^2]\mathbb{E}[k_r^2]
$$

again by independence. Because each variable has variance 1 and mean 0, we have $\mathbb{E}[q_r^2] = 1$ and $\mathbb{E}[k_r^2] = 1$, so $\text{Var}(q_r k_r) = 1$. Now sum over all $d_k$ terms. Since the terms are independent, apply the **additivity of variance for independent random variables**:

$$
\text{Var}\!\left(\sum_{r=1}^{d_k} q_r k_r\right)
= \sum_{r=1}^{d_k} \text{Var}(q_r k_r)
= \sum_{r=1}^{d_k} 1
= d_k
$$

So the unscaled dot product has variance

$$
\boxed{\text{Var}(q^\top k) = d_k}
$$

and therefore standard deviation

$$
\sqrt{d_k}
$$

### 5.2 Why large variance hurts softmax

If $d_k$ is large, the dot products become large in magnitude. Large positive gaps in the score vector make softmax almost one-hot, and when that happens the softmax saturates and its gradients become tiny. Dividing by $\sqrt{d_k}$ rescales the variance back to 1:

$$
\text{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right)
= \frac{1}{d_k} \text{Var}(q^\top k)
= \frac{1}{d_k} \cdot d_k
= 1
$$

This uses the **variance scaling rule**:

$$
\text{Var}(aX) = a^2 \text{Var}(X)
$$

### 5.3 Numerical check for our running example

In our running example:

$$
d_k = 2
$$

so

$$
\sqrt{d_k} = \sqrt{2} \approx 1.4142
$$

If an unscaled score is 1, the scaled score becomes

$$
\frac{1}{1.4142} \approx 0.7071
$$

If an unscaled score is 0.5, the scaled score becomes

$$
\frac{0.5}{1.4142} \approx 0.3536
$$

The scores are compressed toward zero, which keeps the softmax less extreme.

---

## 6. Full Numerical Walkthrough: Self-Attention on Three Tokens

Now we run the whole mechanism start to finish.

To isolate attention itself, we take the simplest projections:

$$
W_Q = W_K = W_V = I
$$

the $2 \times 2$ identity matrix.

Then

$$
Q = K = V = X
$$

for our running example.

### 6.1 Compute $QK^\top$

Write the input matrix:

$$
X =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0.5 & 0.5
\end{bmatrix}
$$

Then

$$
QK^\top = XX^\top
=
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0.5 & 0.5
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0.5 \\
0 & 1 & 0.5
\end{bmatrix}
$$

By the **definition of matrix multiplication**, each entry is a row-column dot product. Row 1 against the three columns gives $1 \cdot 1 + 0 \cdot 0 = 1$, then $1 \cdot 0 + 0 \cdot 1 = 0$, and then $1 \cdot 0.5 + 0 \cdot 0.5 = 0.5$. Row 2 gives $0 \cdot 1 + 1 \cdot 0 = 0$, then $0 \cdot 0 + 1 \cdot 1 = 1$, and then $0 \cdot 0.5 + 1 \cdot 0.5 = 0.5$. Row 3 gives $0.5 \cdot 1 + 0.5 \cdot 0 = 0.5$, then $0.5 \cdot 0 + 0.5 \cdot 1 = 0.5$, and finally $0.5 \cdot 0.5 + 0.5 \cdot 0.5 = 0.25 + 0.25 = 0.5$.

So

$$
QK^\top =
\begin{bmatrix}
1 & 0 & 0.5 \\
0 & 1 & 0.5 \\
0.5 & 0.5 & 0.5
\end{bmatrix}
$$

### 6.2 Apply the scaling

Divide by $\sqrt{2}$:

$$
\frac{QK^\top}{\sqrt{2}}
=
\begin{bmatrix}
0.7071 & 0 & 0.3536 \\
0 & 0.7071 & 0.3536 \\
0.3536 & 0.3536 & 0.3536
\end{bmatrix}
$$

### 6.3 Softmax row 1

Row 1 is

$$
[0.7071,\ 0,\ 0.3536]
$$

Exponentiate:

$$
e^{0.7071} \approx 2.0281,\qquad
e^0 = 1,\qquad
e^{0.3536} \approx 1.4240
$$

Add them:

$$
Z_1 = 2.0281 + 1 + 1.4240 = 4.4521
$$

Normalize:

$$
A_{1,\cdot} =
\left[
\frac{2.0281}{4.4521},
\frac{1}{4.4521},
\frac{1.4240}{4.4521}
\right]
\approx
[0.4556,\ 0.2247,\ 0.3199]
$$

### 6.4 Softmax rows 2 and 3

Row 2 is

$$
[0,\ 0.7071,\ 0.3536]
$$

which is just a permutation of row 1, so

$$
A_{2,\cdot} \approx [0.2247,\ 0.4556,\ 0.3199]
$$

Row 3 has equal entries:

$$
[0.3536,\ 0.3536,\ 0.3536]
$$

Equal inputs to softmax produce the uniform distribution, so

$$
A_{3,\cdot} =
\left[\frac{1}{3},\ \frac{1}{3},\ \frac{1}{3}\right]
\approx
[0.333,\ 0.333,\ 0.333]
$$

The full attention matrix is therefore

$$
A \approx
\begin{bmatrix}
0.456 & 0.225 & 0.320 \\
0.225 & 0.456 & 0.320 \\
0.333 & 0.333 & 0.333
\end{bmatrix}
$$

### 6.5 Multiply by the values

Since $V = X$,

$$
O = AV
$$

Compute the first output vector:

$$
o_1^{(1)} = 0.456 \cdot 1 + 0.225 \cdot 0 + 0.320 \cdot 0.5 = 0.456 + 0 + 0.160 = 0.616
$$

$$
o_1^{(2)} = 0.456 \cdot 0 + 0.225 \cdot 1 + 0.320 \cdot 0.5 = 0 + 0.225 + 0.160 = 0.385
$$

So

$$
\boxed{o_1 \approx [0.616,\ 0.385]}
$$

By symmetry,

$$
\boxed{o_2 \approx [0.385,\ 0.616]}
$$

For token 3:

$$
o_3^{(1)} = 0.333 \cdot 1 + 0.333 \cdot 0 + 0.333 \cdot 0.5 = 0.333 + 0 + 0.167 = 0.500
$$

$$
o_3^{(2)} = 0.333 \cdot 0 + 0.333 \cdot 1 + 0.333 \cdot 0.5 = 0 + 0.333 + 0.167 = 0.500
$$

So

$$
\boxed{o_3 = [0.500,\ 0.500]}
$$

### 6.6 Interpretation

Token 1 started as a pure noun-like vector $[1,\ 0]$ and after self-attention became $[0.616,\ 0.385]$. It kept mostly noun content but absorbed some verb content from the other tokens. That is the core role of self-attention: contextual mixing through learned, content-dependent weighted averages.

---

## 7. A Short Comparison: Unscaled vs Scaled

It is worth looking at the scaling effect directly on the running example rather than only through the variance proof.

### 7.1 Unscaled row 1

Without the $\sqrt{d_k}$ scaling, row 1 would be

$$
[1,\ 0,\ 0.5]
$$

Exponentiate:

$$
e^1 \approx 2.7183,\qquad e^0 = 1,\qquad e^{0.5} \approx 1.6487
$$

Normalize:

$$
Z = 2.7183 + 1 + 1.6487 = 5.3670
$$

So

$$
\text{softmax}([1,0,0.5]) \approx [0.5065,\ 0.1863,\ 0.3072]
$$

### 7.2 Scaled row 1

With scaling, we got

$$
\text{softmax}([0.7071,0,0.3536]) \approx [0.4556,\ 0.2247,\ 0.3199]
$$

### 7.3 What changed

The scaled version is less sharp. The largest weight dropped from about $0.507$ to $0.456$, and the smaller weights rose correspondingly.

For $d_k = 2$, the effect is mild. For realistic head sizes like $d_k = 64$ or $128$, the effect becomes much more important.

---

## 8. Multi-Head Attention

A single head gives one attention pattern. If we want the model to track several different relations at once, we run several heads in parallel.

### 8.1 Definition

Each head $r$ computes $\text{head}_r = \text{Attention}(QW_r^Q, KW_r^K, VW_r^V)$, and the results are concatenated and projected:

$$
\text{MultiHead}(Q,K,V) = \text{concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

### 8.2 Parameter count

With $h$ heads and $d_k = d_v = d_\text{model}/h$, the attention parameter count is $4 d_\text{model}^2$, as we will re-derive in more detail later in the series. For $d_\text{model} = 512$, that gives $4 \cdot 512^2 = 1{,}048{,}576$ parameters in the attention block. The main point of multi-head attention is not more total parameters — it is more parallel attention subspaces.

### 8.3 Why multiple heads help

A single head produces one attention distribution per token. If token 1 needs to look at token 2 for syntax and token 3 for semantics, one head has to blend those two requirements into one row of weights. Multiple heads let the model keep several attention patterns alive at once — one head can specialize in one relation, another head in another relation, and the output projection $W^O$ can recombine them afterward. This is not a proof that heads always specialize cleanly. It is the representational reason the design exists.

---

## 9. Self-Attention, Cross-Attention, and Masked Attention

### 9.1 Self-attention

In **self-attention**, queries, keys, and values all come from the same sequence: $Q = XW_Q$, $K = XW_K$, $V = XW_V$. This is what we computed above.

### 9.2 Cross-attention

In **cross-attention**, the queries come from one sequence and the keys/values come from another: $Q = X_\text{query} W_Q$, $K = X_\text{memory} W_K$, $V = X_\text{memory} W_V$. This is the Transformer version of Bahdanau-style encoder-decoder attention.

### 9.3 Masked self-attention

In decoder self-attention, a token must not look at future tokens, so we add a **causal mask**:

$$
S_\text{masked} = \frac{QK^\top + M}{\sqrt{d_k}}
$$

where forbidden entries of $M$ are $-\infty$.

### 9.4 Numerical check on row 2

Take token 2 in our running example. Its unmasked scaled scores were

$$
[0,\ 0.7071,\ 0.3536]
$$

If token 2 is not allowed to look at token 3, then we replace the third entry with $-\infty$:

$$
[0,\ 0.7071,\ -\infty]
$$

Exponentiate:

$$
e^0 = 1,\qquad e^{0.7071} \approx 2.0281,\qquad e^{-\infty} = 0
$$

Normalize:

$$
Z = 1 + 2.0281 + 0 = 3.0281
$$

So the masked attention row becomes

$$
\left[
\frac{1}{3.0281},
\frac{2.0281}{3.0281},
0
\right]
\approx
[0.330,\ 0.670,\ 0]
$$

The mask does not merely discourage future attention. It sets the forbidden probability exactly to zero.

---

## 10. Why Self-Attention Changed the Path Length

A bidirectional RNN can eventually move information from token 1 to token 3, but it has to do so through intermediate recurrent steps.

In a three-token sequence, the path from token 1 to token 3 through a left-to-right RNN is:

$$
x_1 \to h_1 \to h_2 \to h_3
$$

That is two recurrent transitions between the endpoints.

In self-attention, token 3 can attend directly to token 1 in one layer because the score

$$
S_{3,1} = q_3 k_1^\top
$$

is computed in the same matrix multiply as every other pair.

So the interaction path length between any two positions inside one self-attention layer is 1.

### 10.1 Numerical check

In our running example, token 3 attends to token 1 with score $0.5$ before scaling and $0.3536$ after scaling. That direct token-3-to-token-1 interaction is present immediately in the score matrix — no recurrence is needed to transmit it across intermediate positions. This shorter path length is one of the reasons self-attention handles long-range interactions so well.

---

## 11. Positional Information

Self-attention has one major omission: the formula itself does not know token order.

### 11.1 Why order is missing

If we permute the rows of $X$, then the rows of $Q$, $K$, and $V$ are permuted in the same way, and the output rows are then permuted in the same way. That means the bare self-attention mechanism is **permutation equivariant**. This is good for set processing, but bad for language, where "dog bites man" and "man bites dog" must not mean the same thing.

### 11.2 Positional encodings

The Transformer solves this by adding a positional vector to each token embedding before attention:

$$
\tilde{x}_t = x_t + \text{PE}(t)
$$

The original paper uses sinusoidal positional encodings:

$$
\text{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d_\text{model}}}\right),
\qquad
\text{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d_\text{model}}}\right)
$$

### 11.3 Why sinusoids are convenient

The sine and cosine features have a useful relative-position property because of the **angle-addition identities**:

$$
\sin(a+b) = \sin a \cos b + \cos a \sin b
$$

$$
\cos(a+b) = \cos a \cos b - \sin a \sin b
$$

These identities mean a fixed offset in position can be represented as a linear transformation of the sinusoidal features.

That is why the model can learn relative offsets even though the encoding is written in absolute positions.

### 11.4 Numerical check for the first sinusoidal pair

Take the first two positional dimensions, where the denominator is 1. Then the encoding pair is simply

$$
(\sin t,\ \cos t)
$$

At positions $t = 0, 1, 2$:

$$
\text{PE}(0) = (\sin 0,\ \cos 0) = (0,\ 1)
$$

$$
\text{PE}(1) = (\sin 1,\ \cos 1) \approx (0.8415,\ 0.5403)
$$

$$
\text{PE}(2) = (\sin 2,\ \cos 2) \approx (0.9093,\ -0.4161)
$$

So even this first frequency pair already gives each position a distinct two-dimensional signature. Higher dimensions add slower oscillations, which let the model represent both fine local offsets and broader global position.

---

## 12. Bahdanau Attention and Transformer Attention as One Story

We can now place both mechanisms inside one unified retrieval template.

### 12.1 Bahdanau attention

In Bahdanau attention, the query is the decoder state $s_{i-1}$, the key is the encoder annotation $h_j$, the value is that same encoder annotation $h_j$, and the score is produced by an additive feedforward compatibility function.

### 12.2 Transformer attention

In Transformer attention, the query is a learned projection of the current token, the key is a learned projection of a candidate memory token, the value is another learned projection of that candidate memory token, and the score is the scaled dot product.

### 12.3 The unified view

Both mechanisms compute

$$
\text{weights} = \text{softmax}(\text{compatibility scores})
$$

and then

$$
\text{output} = \text{weighted average of stored values}
$$

So the Transformer does not abandon attention — it re-expresses the same retrieval idea in a form that is easier to batch, easier to parallelize, and better suited to self-attention over one sequence.

---

## Summary

The Transformer's attention formula comes from one simple design goal: factor the compatibility score so all pairwise interactions can be computed by matrix multiplication. That leads to queries and keys, softmax turns their dot products into weights, values carry the retrieved content, and the $\sqrt{d_k}$ factor rescales the score variance so the softmax does not saturate.

In our three-token example, this produces scaled scores, attention weights, and contextualized outputs entirely by hand. The next post asks what happens when we stop thinking about three tokens and start thinking about 4K, 8K, or 128K tokens, where the same clean formula runs head-first into quadratic cost.

---

*Previous: [What Attention is Really Doing](/blog/attention-what-is-it-really)*  
*Next: [Why Vanilla Attention Breaks at Scale](/blog/attention-why-vanilla-breaks)*
