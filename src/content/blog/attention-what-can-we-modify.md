---
title: "What Can We Actually Modify in Attention? A Taxonomy of Every Variant"
description: "Attention has five independent axes of variation: number of heads, KV representation, attention pattern, storage and caching, and layer-level architecture. Every variant in the literature is a modification of exactly one or two of these axes."
date: 2026-03-31
tags: [machine-learning, attention, transformers, architecture]
order: 1
---

Every attention variant you will ever encounter — Multi-Query Attention, GQA, FlashAttention, Longformer, Linear Attention, MLA, Mamba — is a modification of vanilla multi-head attention. But papers rarely frame it that way. They present the new thing from scratch, and you are left wondering: what actually changed?

This blog answers that question by breaking the vanilla attention formula into its constituent choices. There are exactly five axes on which you can make a different decision. Every variant in the literature sits at a specific point in this five-dimensional space. Once you see the space, every subsequent paper becomes: which axis did this modify, and why?

This is the structural blog in the series. The posts that follow are easier to place once this map is explicit.

---

## The Running Model

We need a concrete model to anchor the numbers throughout. Fix:

- $d_\text{model} = 512$
- $h = 8$ heads
- $d_k = d_v = 64$ (so $h \cdot d_k = 512 = d_\text{model}$)
- $L = 12$ layers
- $N = 512$ tokens (training), $n$ tokens generated so far (inference)
- fp16 throughout (2 bytes per element)

The identity

$$
h \cdot d_k = 8 \cdot 64 = 512 = d_\text{model}
$$

will appear constantly. It is the reason many formulas simplify cleanly.

---

## 1. Vanilla Attention, Written with Every Choice Made Explicit

For one transformer layer, standard multi-head attention is

$$
\text{MHA}(X) = \text{concat}(o^1, \ldots, o^h)\, W_O
$$

where for each head $i \in \{1, \ldots, h\}$:

$$
Q^i = XW_Q^i, \qquad K^i = XW_K^i, \qquad V^i = XW_V^i
$$

$$
A^i = \text{softmax}\!\left(\frac{Q^i {K^i}^\top + M}{\sqrt{d_k}}\right), \qquad o^i = A^i V^i
$$

The formula looks compact. It is actually hiding several independent design decisions.

### 1.1 The five questions hidden inside the formula

When we write the vanilla formula, we have already committed to answers to five questions: how many heads there are, how keys and values are represented, which query-key pairs are allowed to interact, what persistent state is kept across decoding steps, and where the attention block sits inside the larger transformer layer. Those are the five axes we will use in this series. The following table maps each symbol in the formula to the axis it belongs to and the design alternatives that exist.

### 1.2 A symbol-to-axis map

| Symbol | Current choice | What could change |
|---|---|---|
| $h$ | 8 separate heads | Fewer, shared, or grouped heads |
| $W_K^i, W_V^i$ | One per head, full rank | Shared across heads, or low-rank |
| $M$ | Full or causal mask | Local window, global tokens, cross |
| KV persistence | No training cache, full inference KV cache | Cache K, V, or a compressed latent |
| Layer placement | Standard residual + norm + FFN block | Parallel FFN, different norm, different wrapper |

The rest of this post expands each axis carefully.

### 1.3 A generalized attention template

We can expose the same five axes in one slightly more general template:

$$
Q^i = XW_Q^i
$$

$$
K^{g(i)},\ V^{g(i)} = R_\text{KV}(X, z_t)
$$

$$
A^i = \text{softmax}\!\left(\frac{Q^i {K^{g(i)}}^\top + M_\text{pattern}}{\sqrt{d_k}}\right)
$$

$$
o^i = A^i V^{g(i)}
$$

$$
Y = B\!\left(X,\ \text{concat}(o^1, \ldots, o^h),\ z_t\right)
$$

This notation is more general than vanilla MHA, but that is the point. The index $i$ labels query heads, the function $g(i)$ tells us which KV group or shared KV representation head $i$ uses, $R_\text{KV}$ is the rule that builds keys and values, $M_\text{pattern}$ is the mask or bias that determines which interactions are allowed, $z_t$ is any persistent state carried across decoding steps, and $B$ is the larger wrapper around the attention output. Once we write the formula this way, the five axes stop being vague: Axis 1 changes how many query heads exist, Axis 2 changes the KV construction rule and the grouping function $g(i)$, Axis 3 changes the pattern term $M_\text{pattern}$, Axis 4 changes what state survives across time, and Axis 5 changes the block wrapper $B$.

Vanilla MHA is just one point in this larger space: $g(i) = i$ for every head, $R_\text{KV}$ computes one key and one value projection per head from the current sequence, $M_\text{pattern}$ is the standard dense full or causal mask, $z_t$ is empty during training and the usual KV cache during inference, and $B$ is the standard transformer block. We will keep coming back to this template because it lets us describe very different papers in one common language instead of paper-specific vocabulary.

---

## 2. Axis 1: Number of Heads

### 2.1 What a head is

A **head** is one independent attention computation with its own projections $W_Q^i, W_K^i, W_V^i$. If we have $h$ heads, we do not compute one attention map — we compute $h$ of them and concatenate the outputs at the end. In vanilla attention, we choose $h = 8$ heads.

### 2.2 Parameter count

The first thing to notice is that head count sounds like it should control parameter count. In the standard transformer parameterization, it mostly does not. Let us derive that carefully.

Each head has one query matrix $W_Q^i \in \mathbb{R}^{d_\text{model} \times d_k}$, one key matrix $W_K^i \in \mathbb{R}^{d_\text{model} \times d_k}$, and one value matrix $W_V^i \in \mathbb{R}^{d_\text{model} \times d_v}$, with parameter counts $d_\text{model} d_k$, $d_\text{model} d_k$, and $d_\text{model} d_v$ respectively. Since $d_v = d_k$ in our running model, the three together cost $3 d_\text{model} d_k$ per head, and across $h$ heads the total is $3 h d_\text{model} d_k$. The output projection maps the concatenated head outputs back to model width: $W_O \in \mathbb{R}^{(h d_v) \times d_\text{model}}$. Because $h d_v = d_\text{model}$, this has $d_\text{model}^2$ parameters. So the full attention block has $3 h d_\text{model} d_k + d_\text{model}^2$ parameters. Now use $h d_k = d_\text{model}$:

$$
3 h d_\text{model} d_k + d_\text{model}^2
= 3 d_\text{model}(h d_k) + d_\text{model}^2
$$

Substitute $h d_k = d_\text{model}$:

$$
= 3 d_\text{model}^2 + d_\text{model}^2
= 4 d_\text{model}^2
$$

So under the standard choice $d_k = d_\text{model}/h$, the total attention parameter count is

$$
\boxed{4 d_\text{model}^2}
$$

and is independent of the number of heads.

### 2.3 Numerical check

With $d_\text{model} = 512$:

$$
4 d_\text{model}^2 = 4 \cdot 512^2 = 4 \cdot 262{,}144 = 1{,}048{,}576
$$

Exactly as expected.

We can also verify it the long way:

$$
3 h d_\text{model} d_k + d_\text{model}^2
= 3 \cdot 8 \cdot 512 \cdot 64 + 512^2
$$

Compute the first term:

$$
3 \cdot 8 \cdot 512 \cdot 64 = 786{,}432
$$

Add the output projection:

$$
786{,}432 + 262{,}144 = 1{,}048{,}576
$$

Both routes match.

### 2.4 What changing the number of heads actually changes

This is the first place where the literature can be misleading. People often talk about "more heads" as if that obviously means "more parameters." Under the standard choice $d_k = d_\text{model}/h$, that is not really what is happening. If we keep $d_\text{model}$ fixed and change $h$, the total parameter count stays roughly fixed, the width per head changes because $d_k = d_\text{model}/h$, and the number of distinct attention patterns changes. So Axis 1 is not mainly about raw parameter count — it is about how the model slices up representational capacity. Fewer heads mean wider per-head subspaces but fewer independent patterns, while more heads mean narrower per-head subspaces but more separate patterns alive at once.

Most famous efficiency variants do not primarily change Axis 1. They keep the number of query heads fixed and instead change how many **distinct KV heads** exist. That is the next axis.

---

## 3. Axis 2: KV Representation

### 3.1 The vanilla choice

Vanilla MHA gives every query head its own key and value projections:

$$
Q^i = XW_Q^i, \qquad K^i = XW_K^i, \qquad V^i = XW_V^i
$$

for every head $i$.

This means the model stores and manipulates $h$ distinct key streams and $h$ distinct value streams. Each query head gets its own private KV view of the sequence.

### 3.2 KV parameter count

The key and value projections alone cost $h \cdot d_\text{model} d_k + h \cdot d_\text{model} d_v$. Factoring out $h d_\text{model}$ gives $h d_\text{model}(d_k + d_v)$. In our running model, that is $8 \cdot 512 \cdot (64 + 64) = 8 \cdot 512 \cdot 128 = 524{,}288$. So half of the attention parameters live in the KV projections — that is already a clue that this axis matters, because even before inference enters the picture, keys and values are carrying a large fraction of the attention block's parameter budget.

### 3.3 KV cache bytes per token

During autoregressive inference, the key and value vectors are cached for every previous token.

For one token, one head, one layer, the cache stores $d_k + d_v$ elements. In fp16, that is $(d_k + d_v) \cdot 2$ bytes. Across $h$ heads and $L$ layers:

$$
\boxed{\text{KV bytes/token} = L \cdot h \cdot (d_k + d_v) \cdot 2}
$$

Substitute our model:

$$
12 \cdot 8 \cdot (64 + 64) \cdot 2
= 12 \cdot 8 \cdot 128 \cdot 2
= 24{,}576 \text{ bytes}
$$

So vanilla MHA costs

$$
\boxed{24 \text{ KB/token}}
$$

of KV cache. That single number is the quantity later posts will keep attacking.

### 3.4 What can change on this axis

This is where the design space opens up. Once we realize that query heads and KV heads do not have to be counted the same way, MHA stops looking inevitable.

**Multi-Query Attention (MQA)** keeps $h$ query heads but shares one K head and one V head across all of them:

$$
Q^i = XW_Q^i,\qquad K = XW_K,\qquad V = XW_V
$$

The KV cache becomes $L \cdot 1 \cdot (d_k + d_v) \cdot 2$, which is smaller than MHA by a factor of $h/1 = h$. In our running model, that gives $12 \cdot 1 \cdot 128 \cdot 2 = 3{,}072$ bytes, or 3 KB per token.

**Grouped-Query Attention (GQA)** keeps $h$ query heads but uses only $g$ KV heads, where each KV head is shared by $h/g$ queries. The cache cost becomes $\text{KV bytes/token} = L \cdot g \cdot (d_k + d_v) \cdot 2$, giving a reduction factor of $\frac{h}{g}$. With $g = 2$ in our running model, that gives $12 \cdot 2 \cdot 128 \cdot 2 = 6{,}144$ bytes, or 6 KB per token.

**Multi-head Latent Attention (MLA)** takes a different approach entirely: instead of caching per-head K and V directly, it caches a smaller latent representation $c_t \in \mathbb{R}^{d_c}$ and reconstructs K and V from it later. The cache cost becomes $L \cdot d_c \cdot 2$ bytes per token.

The point is simple. Axis 2 directly changes the shape of the stored and computed KV representation, and that is why MQA, GQA, and MLA belong together even though the papers sound very different.

### 3.5 Numerical table

| KV scheme | KV heads / latent | Bytes per token |
|---|---|---|
| MHA | $h = 8$ KV heads | 24,576 |
| GQA-4 | $g = 4$ KV heads | 12,288 |
| GQA-2 | $g = 2$ KV heads | 6,144 |
| MQA | 1 shared KV head | 3,072 |

This axis is the source of MQA, GQA, and MLA. It is also the axis most directly tied to long-context inference cost.

---

## 4. Axis 3: Attention Pattern

### 4.1 The vanilla choice

Vanilla self-attention is dense. Every query attends to every allowed key. The mask $M$ decides what "allowed" means. In encoder self-attention, all pairs are allowed. In decoder self-attention, only past and present positions are allowed. But in both cases, once a position is allowed, it is included. There is no sparsity inside the allowed region.

### 4.2 Counting nonzero attention entries

For full bidirectional attention, each of the $N$ queries attends to all $N$ keys, giving $N^2$ attention scores per head. For causal attention, query position $i$ attends to positions $1, \ldots, i$, so the total number of allowed entries is $1 + 2 + \cdots + N$. By the **triangular number formula**, $1 + 2 + \cdots + N = \frac{N(N+1)}{2}$, so causal attention still has $\Theta(N^2)$ nonzero entries. The factor of $1/2$ helps, but it does not change the scaling class.

### 4.3 Local windows

Suppose each position attends only to a window of width $w$ on each side. Then each query sees at most $2w + 1$ keys, and the total number of attention entries becomes $N(2w + 1)$, which is linear in $N$ when $w$ is fixed.

### 4.4 Numerical check

Take $N = 512$ and $w = 64$.

**Full attention:**

$$
N^2 = 512^2 = 262{,}144
$$

entries per head.

**Local attention:**

$$
N(2w + 1) = 512(129) = 66{,}048
$$

entries per head.

Reduction factor:

$$
\frac{262{,}144}{66{,}048} \approx 3.97
$$

So a width-64 local window already cuts the attention pattern by about $4\times$ at this sequence length.

### 4.5 Variants on this axis

Several well-known methods live on this axis. **Sliding-window attention** restricts each position to attend only locally. **Local + global attention** uses local windows for most positions but lets a few designated tokens attend globally. **Cross-attention** draws queries from one sequence and keys/values from another, making the pattern rectangular ($N_q \times N_k$) instead of square. **Block-sparse attention** makes the mask sparse at the level of blocks rather than individual tokens.

Axis 3 changes which interactions are computed at all, and that is why it is the axis most directly tied to the quadratic wall from the previous post. Change this axis and you are changing the combinatorics of the attention matrix itself.

---

## 5. Axis 4: Storage and Caching

Axis 2 asked what representation keys and values live in. Axis 4 asks a different question: what information do we persist across time, and for how long?

### 5.1 The vanilla choices

Training and inference behave differently here, and that difference is easy to blur if we only stare at the attention formula. During training, we compute attention over the whole sequence in parallel, and there is no persistent cache carried from one token to the next. During autoregressive inference, we cache all previous K and V tensors so the next token can reuse them. So the default inference state is the full KV cache — vanilla attention is not only an equation but also a policy about what survives from one decoding step to the next.

### 5.2 Sliding-window cache

Instead of storing all previous tokens, we can keep only the most recent $w$.

Then the cache cost becomes

$$
L \cdot h \cdot (d_k + d_v) \cdot 2 \cdot w
$$

which no longer grows with total sequence length once the window is full.

### 5.3 Numerical check

Using vanilla MHA with our running model and a window of $w = 512$:

$$
12 \cdot 8 \cdot 128 \cdot 2 \cdot 512
$$

First compute the bytes per token:

$$
12 \cdot 8 \cdot 128 \cdot 2 = 24{,}576
$$

Now multiply by the window size:

$$
24{,}576 \cdot 512 = 12{,}582{,}912 \text{ bytes}
$$

So the full window cache is about

$$
12 \text{ MB}
$$

regardless of how long generation continues after that.

### 5.4 Other variants on this axis

Several other approaches modify this axis. A **quantized cache** keeps the same logical KV tensors but stores them in int8 or int4 instead of fp16. A **paged cache** keeps the same logical KV tensors but manages physical memory in pages so serving systems waste less space. At the far end of this axis, some architectures replace the whole growing history with a **recurrent state** of fixed size, as state-space models do.

This axis is about persistence over time, and it is not the same as Axis 2. GQA shrinks the cache by changing the number of KV heads, while a sliding-window cache shrinks it by changing how much history is retained. Those two changes compose cleanly because they are attacking different objects: one changes the representation, the other changes the retention rule.

---

## 6. Axis 5: Layer-Level Architecture

So far, every axis changed some part of the attention computation itself. Axis 5 is different. It changes how the attention sublayer is embedded inside the larger transformer block.

### 6.1 The vanilla block

The original transformer uses a post-norm block:

$$
x' = \text{LayerNorm}(x + \text{Attention}(x))
$$

$$
x'' = \text{LayerNorm}(x' + \text{FFN}(x'))
$$

Attention is computed first. Then FFN. Normalization happens after each residual addition.

### 6.2 Pre-norm

Modern language models usually move the normalization before the sublayer: $x' = x + \text{Attention}(\text{Norm}(x))$ and $x'' = x' + \text{FFN}(\text{Norm}(x'))$. This leaves a cleaner identity path through the residual stream and usually improves optimization stability.

### 6.3 Parallel attention and FFN

Another choice is to compute the attention and FFN branches from the same normalized input and add them in parallel: $x' = x + \text{Attention}(\text{Norm}(x)) + \text{FFN}(\text{Norm}(x))$. This does not change the attention formula itself — it changes the larger layer wrapper.

### 6.4 LayerNorm vs RMSNorm

Normalization type also lives on Axis 5. **LayerNorm** subtracts the mean and divides by the standard deviation, while **RMSNorm** divides only by the root mean square:

$$
\text{RMSNorm}(x)_i
= \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot \gamma_i
$$

### 6.5 Numerical check

For one token with width $d = 512$, **LayerNorm** needs a mean over 512 elements, a variance over 512 elements, normalization, and then scale and shift. **RMSNorm** skips the mean subtraction and the learned shift, so it removes one whole reduction and one whole elementwise subtraction pass.

The point is not the exact scalar-op count. The point is that Axis 5 changes optimization behavior, signal flow, and hardware efficiency without touching the core attention equation.

---

## 7. How the Axes Compose

Once the axes are separated, composition becomes much easier to reason about.

### 7.1 Example 1: GQA + sliding window

Here we combine an Axis 2 change and an Axis 4 change. Axis 2 reduces the number of KV heads from $h$ to $g$, and Axis 4 keeps only the last $w$ tokens.

The cache becomes

$$
L \cdot g \cdot (d_k + d_v) \cdot 2 \cdot w
$$

In our running model with $g = 2$ and $w = 512$:

$$
12 \cdot 2 \cdot 128 \cdot 2 \cdot 512 = 3{,}145{,}728 \text{ bytes}
$$

which is about

$$
3 \text{ MB}
$$

Compare that to full MHA with an unrestricted history at $128{,}000$ tokens:

$$
24{,}576 \cdot 128{,}000 = 3{,}145{,}728{,}000 \text{ bytes} \approx 3 \text{ GB}
$$

So these two axes together give a $1000\times$ reduction in this simple accounting.

### 7.2 Example 2: Longformer + pre-norm

This time we combine an Axis 3 change and an Axis 5 change. Axis 3 introduces a local/global sparse pattern, and Axis 5 changes the larger layer wrapper.

The pattern change reduces attention entries. The layer change affects optimization and throughput. They mostly act on different parts of the system.

### 7.3 Example 3: What FlashAttention is not

FlashAttention is extremely important, but this taxonomy is useful precisely because it tells us what FlashAttention is not. It does not naturally sit on one of the five axes because it does not change what the model computes. It changes how the same dense attention computation is scheduled on hardware.

That is why it is better thought of as an implementation strategy than as a new point in the architectural design space.

### 7.4 Why this taxonomy helps when reading papers

Suppose a paper claims "8x smaller KV memory," "same dense attention pattern," and "same per-token arithmetic." Even before reading the experiments, we can infer that it is probably an Axis 2 or Axis 4 paper — Axis 2 if the KV representation itself changed, and Axis 4 if the persistence rule or cache duration changed. Now suppose a paper claims "same outputs as vanilla attention," "lower HBM traffic," and "faster kernels." That is probably not an axis change at all — it is much more likely an implementation paper in the FlashAttention family. This is the practical value of the taxonomy: it lets us classify a paper's claim before we get lost in its naming scheme.

---

## 8. A Practical Variant Map

Here is a compact placement table for the most common variants we will meet later:

| Variant | Axis 1 | Axis 2 | Axis 3 | Axis 4 | Axis 5 |
|---|---|---|---|---|---|
| Vanilla MHA | standard heads | one KV per head | dense full/causal | full KV cache at inference | standard residual block |
| MQA | standard heads | one shared KV head | dense | KV cache | unchanged |
| GQA | standard heads | $g < h$ KV heads | dense | KV cache | unchanged |
| MLA | standard heads | latent KV representation | dense | latent cache | unchanged |
| Longformer | standard heads | standard KV | local + global | standard cache | unchanged |
| Sliding-window decoder | standard heads | standard or grouped KV | local causal | bounded cache window | unchanged |
| Pre-norm transformer | standard heads | standard KV | dense | standard cache | pre-norm |
| RMSNorm transformer | standard heads | standard KV | dense | standard cache | RMSNorm |

The table is useful because it answers a paper's main question immediately: if the paper changes Axis 2, expect KV-cache consequences; if it changes Axis 3, expect FLOP and access-pattern consequences; and if it changes Axis 5, expect training-stability or throughput consequences.

### 8.1 A roadmap for the rest of the series

This taxonomy is also the scaffolding of the later attention posts in the series. [attention-kv-bottleneck.md](/home/pratham/prathamp/prathamp.com/src/content/blog/attention-kv-bottleneck.md) motivates Axis 2 and Axis 4 by deriving why KV storage dominates long-context inference. [attention-gqa.md](/home/pratham/prathamp/prathamp.com/src/content/blog/attention-gqa.md) is primarily an Axis 2 post: same dense pattern, fewer distinct KV heads. Sparse or sliding-window attention posts are Axis 3 posts, because they keep the basic retrieval rule but change the allowed interactions. Residual-layout and normalization posts are Axis 5 posts, because they keep the attention mechanism but change the block wrapper.

So the taxonomy is not only a map of the literature. It is also a map of where the series goes next.

---

## 9. Why This Framing Matters

Without a taxonomy, the literature feels like a list of names. With a taxonomy, most papers turn back into concrete questions: do we really need one KV head per query head? Do we really need every position to attend to every other? Do we really need to keep the whole history in fp16? Do we really need this exact residual or normalization layout? Once we ask the questions directly, the variants stop looking mysterious. The payoff is practical — we can predict tradeoffs earlier, combine techniques more safely, and tell whether two "efficient attention" papers are actually solving the same problem. That is the map we will use for the rest of the series.

---

## Summary

Vanilla attention hides five major design axes in one compact formula: the number of heads, the representation used for keys and values, the attention pattern, the persistent storage strategy, and the larger layer architecture around the attention block. Different papers feel different because they modify different axes, but once we separate those axes, most variants become straightforward coordinates in one shared design space.

In this framing, MQA, GQA, and MLA live on the KV-representation axis; sparse and windowed methods live on the attention-pattern axis; cache truncation and quantization live on the storage axis; and pre-norm or RMSNorm live on the layer-architecture axis. The next post zooms into the first big efficiency story that this map explains: why vanilla attention breaks at scale at all.

---

*Previous: [Why Vanilla Attention Breaks at Scale](/blog/attention-why-vanilla-breaks)*  
*Next: [The KV Bottleneck Explained Deeply](/blog/attention-kv-bottleneck)*
