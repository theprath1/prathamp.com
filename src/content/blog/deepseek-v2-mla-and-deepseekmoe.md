---
title: "DeepSeek-V2 from Scratch: Multi-head Latent Attention and DeepSeekMoE"
description: "Building DeepSeek-V2's two core innovations from the ground up — low-rank KV joint compression, decoupled RoPE, matrix absorption, fine-grained expert segmentation, shared experts, and three-level load balancing — all derived step by step with concrete numbers."
date: 2026-04-05
tags: [machine-learning, attention, transformers, mla, kv-cache, mixture-of-experts, deepseek]
order: 2
---

The previous blogs derived how GQA reduces the KV cache by sharing key-value heads across query groups. GQA with $g$ groups cuts the cache by a factor of $h/g$, and MQA ($g = 1$) goes all the way down to a single KV head — but at a quality cost. The fundamental tension was clear: fewer KV heads means less memory, but also less representational capacity.

DeepSeek-V2 ([DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434)) resolves this tension with a completely different approach. Instead of reducing the *number* of KV heads, it compresses keys and values *jointly* into a single low-rank latent vector. The result is an attention mechanism called **Multi-head Latent Attention (MLA)** that uses *less* KV cache than MQA while achieving *stronger* performance than standard MHA. That is not a typo. Less cache *and* better quality.

On the feed-forward side, DeepSeek-V2 replaces standard FFNs with **DeepSeekMoE**, an architecture that segments experts into finer granularity and isolates shared experts from routed ones. Together, MLA and DeepSeekMoE produce a 236B-parameter model that activates only 21B parameters per token, saves 42.5% of training costs compared to the dense DeepSeek 67B, reduces the KV cache by 93.3%, and boosts generation throughput by 5.76x.

We will derive every equation from scratch. The linear algebra tools we need — matrix multiplication, the transpose identity, associativity, and low-rank factorization — are derived in [Mathematical Prerequisites for DeepSeek-V2](/blog/math-prerequisites-for-deepseek-v2).

---

## The Running Model

We continue with the same model from the previous blogs, but we now need to specify MLA-specific dimensions. We keep the base dimensions identical so that all KV cache comparisons are apples-to-apples:

- $d = 512$ (embedding dimension, called $d_\text{model}$ previously)
- $n_h = 8$ attention heads
- $d_h = 64$ (per-head dimension, so $n_h \cdot d_h = 512 = d$)
- $L = 12$ layers
- fp16 throughout (2 bytes per element)

For MLA, we introduce two new dimensions:

- $d_c = 128$ (KV compression dimension — the size of the latent vector)
- $d_c' = 96$ (query compression dimension)
- $d_h^R = 32$ (per-head dimension for decoupled RoPE queries and keys)

These are smaller than the DeepSeek-V2 production values ($d_c = 512$, $d_c' = 1536$, $d_h^R = 64$) but preserve all the ratios and make arithmetic tractable. In particular, $d_c = 128 \ll n_h \cdot d_h = 512$, which is the compression that makes MLA work.

---

## 1. Standard MHA: What We Are Replacing

We established the MHA formulas in previous blogs. Let us restate them in the notation of the DeepSeek-V2 paper so that every equation lines up exactly.

Let $\mathbf{h}_t \in \mathbb{R}^d$ be the hidden state of the $t$-th token at the input to an attention layer. Standard MHA produces queries, keys, and values through three projection matrices:

$$
\mathbf{q}_t = W^Q \mathbf{h}_t
$$

$$
\mathbf{k}_t = W^K \mathbf{h}_t
$$

$$
\mathbf{v}_t = W^V \mathbf{h}_t
$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d_h n_h \times d}$. Since $d_h n_h = 512 = d$ in our model, these are square $512 \times 512$ matrices. The concatenated $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ are each in $\mathbb{R}^{d_h n_h}$ — that is, $\mathbb{R}^{512}$.

We then split each into $n_h = 8$ heads:

$$
[\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; \ldots; \mathbf{q}_{t,n_h}] = \mathbf{q}_t
$$

$$
[\mathbf{k}_{t,1}; \mathbf{k}_{t,2}; \ldots; \mathbf{k}_{t,n_h}] = \mathbf{k}_t
$$

$$
[\mathbf{v}_{t,1}; \mathbf{v}_{t,2}; \ldots; \mathbf{v}_{t,n_h}] = \mathbf{v}_t
$$

where $\mathbf{q}_{t,i}, \mathbf{k}_{t,i}, \mathbf{v}_{t,i} \in \mathbb{R}^{d_h}$ — each is a 64-dimensional vector.

The output of head $i$ at position $t$ is:

$$
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h}}\right) \mathbf{v}_{j,i}
$$

The final output concatenates all heads and projects back:

$$
\mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \ldots; \mathbf{o}_{t,n_h}]
$$

where $W^O \in \mathbb{R}^{d \times d_h n_h}$.

### KV cache cost of MHA

During autoregressive generation, we must cache $\mathbf{k}_t$ and $\mathbf{v}_t$ for every previous token at every layer. For a single token at a single layer, we store:

$$
\text{KV elements per token per layer (MHA)} = 2 \cdot n_h \cdot d_h = 2 \times 8 \times 64 = 1024
$$

### Numerical check

Across all $L = 12$ layers, the total number of elements cached per token is $1024 \times 12 = 12{,}288$. In fp16, this is $12{,}288 \times 2 = 24{,}576$ bytes $= 24$ KB per token. This matches the number we derived in the KV bottleneck blog.

---

## 2. The Core Idea: Low-Rank Key-Value Joint Compression

Here is the key insight of MLA. In standard MHA, the keys and values are each $n_h \cdot d_h = 512$ dimensional vectors. We cache both, so we store $2 \times 512 = 1024$ elements per token per layer. But keys and values are both linear projections of the same hidden state $\mathbf{h}_t$. They share the same source of information.

MLA exploits this by first compressing $\mathbf{h}_t$ down to a much smaller **latent vector** $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$, and then recovering keys and values from this compressed representation via up-projection matrices. The latent vector is what we cache — not the full keys and values.

### 2.1 The Down-Projection

We define a **down-projection matrix** $W^{DKV} \in \mathbb{R}^{d_c \times d}$ that compresses the hidden state:

$$
\boxed{\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t}
$$

In our running model, $W^{DKV} \in \mathbb{R}^{128 \times 512}$. This takes the 512-dimensional hidden state and compresses it to a 128-dimensional latent vector. The ratio is $128/512 = 1/4$ — a 4x compression.

### 2.2 The Up-Projections

From this latent vector, we recover full-dimensional keys and values through two **up-projection matrices**:

$$
\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}
$$

$$
\mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}
$$

where $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$. In our model, these are $512 \times 128$ matrices. The superscript $C$ stands for "content" — we will soon add a separate "RoPE" component to the keys.

Let us be precise about what happens dimensionally. $\mathbf{c}_t^{KV} \in \mathbb{R}^{128}$, and $W^{UK} \in \mathbb{R}^{512 \times 128}$, so $\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV} \in \mathbb{R}^{512}$. This 512-dimensional vector is then split into $n_h = 8$ heads of dimension $d_h = 64$ each, exactly like standard MHA:

$$
[\mathbf{k}_{t,1}^C; \mathbf{k}_{t,2}^C; \ldots; \mathbf{k}_{t,n_h}^C] = \mathbf{k}_t^C
$$

$$
[\mathbf{v}_{t,1}^C; \mathbf{v}_{t,2}^C; \ldots; \mathbf{v}_{t,n_h}^C] = \mathbf{v}_t^C
$$

### KV cache cost of MLA (first attempt)

Here is the crucial difference. During inference, we do **not** cache $\mathbf{k}_t^C$ or $\mathbf{v}_t^C$. We only cache the latent vector $\mathbf{c}_t^{KV}$. When we need keys and values for a previous token, we reconstruct them by applying $W^{UK}$ and $W^{UV}$ to the cached latent.

The cache cost per token per layer is now:

$$
\text{KV elements per token per layer (MLA, first attempt)} = d_c = 128
$$

Compare this to MHA's $2 \cdot n_h \cdot d_h = 1024$. The ratio is $128/1024 = 1/8$. We have reduced the KV cache by 8x.

### Numerical check

Across $L = 12$ layers: $128 \times 12 = 1{,}536$ elements per token. In fp16: $1{,}536 \times 2 = 3{,}072$ bytes $= 3$ KB per token. Compare to MHA's 24 KB per token. The ratio is $3/24 = 1/8$. Correct.

### 2.3 Why This Is Not Just Low-Rank Factorization

This is the part that is easy to gloss over. One might think: "You replaced $W^K$ with $W^{UK} W^{DKV}$ and $W^V$ with $W^{UV} W^{DKV}$. That is just a low-rank factorization of the weight matrices." And that is true — mathematically, $\mathbf{k}_t^C = W^{UK} W^{DKV} \mathbf{h}_t$ is identical to using a single rank-$d_c$ key projection. The same holds for values.

But the point is not about the weight matrices. The point is about what we *cache*. In standard MHA, we cache the outputs $\mathbf{k}_t$ and $\mathbf{v}_t$ — two separate 512-dimensional vectors. In MLA, we cache a single 128-dimensional latent vector $\mathbf{c}_t^{KV}$ from which *both* keys and values can be recovered. The keys and values share a compressed representation. This joint compression is what makes the cache reduction so dramatic: instead of caching $2 \times 512 = 1024$ elements (keys and values separately), we cache $128$ elements (one shared latent).

---

## 3. The RoPE Incompatibility Problem

We cannot simply plug the compressed keys into the attention formula and call it done. There is a fundamental incompatibility with **Rotary Position Embedding (RoPE)**.

### 3.1 What RoPE Does

RoPE encodes position information by applying a position-dependent rotation matrix to queries and keys before computing attention scores. For a token at position $t$, the RoPE operation transforms a query or key vector $\mathbf{x}$ into $\text{RoPE}(\mathbf{x}, t)$, where the rotation depends on $t$. The key property is that the dot product $\text{RoPE}(\mathbf{q}, t)^T \text{RoPE}(\mathbf{k}, j)$ depends only on the *relative* position $t - j$, which is what gives Transformers their relative position awareness.

### 3.2 Why RoPE Breaks Low-Rank KV Compression

In standard MHA with RoPE, the attention computation is:

$$
\mathbf{q}_{t,i}^T \mathbf{k}_{j,i} \longrightarrow \text{RoPE}(\mathbf{q}_{t,i}, t)^T \text{RoPE}(\mathbf{k}_{j,i}, j)
$$

Now consider what happens with MLA. The key is $\mathbf{k}_{t,i}^C = W_i^{UK} \mathbf{c}_t^{KV}$, where $W_i^{UK}$ is the slice of $W^{UK}$ corresponding to head $i$. If we apply RoPE to this key:

$$
\text{RoPE}(W_i^{UK} \mathbf{c}_t^{KV}, t)
$$

This is a *position-dependent* transformation of $W_i^{UK} \mathbf{c}_t^{KV}$. We cannot separate the position dependence from the content dependence. Specifically, we cannot precompute this and cache just $\mathbf{c}_t^{KV}$, because the RoPE rotation is applied *after* the up-projection $W_i^{UK}$ — so we would need to cache the rotated, up-projected key, which is 512-dimensional. That defeats the entire purpose of low-rank compression.

The mathematical issue is precise: RoPE is a rotation, and matrix multiplication does not commute with rotation in general. That is, $\text{RoPE}(W \mathbf{c}, t) \neq W \cdot \text{RoPE}(\mathbf{c}, t)$ because $W$ and the RoPE rotation matrix do not commute (**non-commutativity of matrix multiplication**). So we cannot "push" RoPE inside the up-projection and apply it to the latent vector instead.

---

## 4. The Solution: Decoupled RoPE

The DeepSeek-V2 solution is to decouple the position-dependent part from the content-dependent part entirely. We create *additional* query and key vectors specifically for carrying RoPE, separate from the content-based queries and keys.

### 4.1 Decoupled RoPE Queries

We produce extra "RoPE query" heads $\mathbf{q}_{t,i}^R \in \mathbb{R}^{d_h^R}$ for each attention head. These come from a separate projection. But MLA also compresses the queries (to save activation memory during training, not KV cache). Let us derive the full query pathway.

First, we compress the hidden state for queries:

$$
\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t
$$

where $W^{DQ} \in \mathbb{R}^{d_c' \times d}$. In our model, $W^{DQ} \in \mathbb{R}^{96 \times 512}$. The compressed query latent $\mathbf{c}_t^Q \in \mathbb{R}^{96}$.

From this, we produce the content queries and the RoPE queries:

$$
\mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q
$$

$$
\mathbf{q}_t^R = \text{RoPE}(W^{QR} \mathbf{c}_t^Q)
$$

where $W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c'}$ is the query up-projection (size $512 \times 96$ in our model), and $W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c'}$ produces the RoPE queries (size $256 \times 96$, since $d_h^R \cdot n_h = 32 \times 8 = 256$).

The content queries split into heads: $[\mathbf{q}_{t,1}^C; \ldots; \mathbf{q}_{t,n_h}^C] = \mathbf{q}_t^C$, with each $\mathbf{q}_{t,i}^C \in \mathbb{R}^{d_h} = \mathbb{R}^{64}$.

The RoPE queries split into heads: $[\mathbf{q}_{t,1}^R; \ldots; \mathbf{q}_{t,n_h}^R] = \mathbf{q}_t^R$, with each $\mathbf{q}_{t,i}^R \in \mathbb{R}^{d_h^R} = \mathbb{R}^{32}$.

### 4.2 Decoupled RoPE Keys

For the key side, we produce a *shared* RoPE key $\mathbf{k}_t^R \in \mathbb{R}^{d_h^R}$ (a single vector, not per-head):

$$
\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t)
$$

where $W^{KR} \in \mathbb{R}^{d_h^R \times d}$ — a $32 \times 512$ matrix in our model.

The key insight: $\mathbf{k}_t^R$ is shared across all heads. This is a single 32-dimensional vector that carries position information for all 8 heads. This is efficient because position information is inherently the same across heads — token $t$ is at position $t$ regardless of which head is looking at it.

### 4.3 Assembling the Full Query and Key

For each head $i$, the full query is the concatenation of the content part and the RoPE part:

$$
\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R]
$$

This gives $\mathbf{q}_{t,i} \in \mathbb{R}^{d_h + d_h^R} = \mathbb{R}^{64 + 32} = \mathbb{R}^{96}$.

Similarly, for each head $i$, the full key concatenates the content part (head-specific) with the shared RoPE part:

$$
\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R]
$$

This also gives $\mathbf{k}_{t,i} \in \mathbb{R}^{d_h + d_h^R} = \mathbb{R}^{96}$.

### 4.4 The Attention Computation

The attention score between query at position $t$ and key at position $j$ for head $i$ is:

$$
\mathbf{q}_{t,i}^T \mathbf{k}_{j,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R]^T [\mathbf{k}_{j,i}^C; \mathbf{k}_j^R]
$$

We expand this dot product. By the definition of the dot product of concatenated vectors, this equals the sum of the dot products of the corresponding parts:

$$
\mathbf{q}_{t,i}^T \mathbf{k}_{j,i} = (\mathbf{q}_{t,i}^C)^T \mathbf{k}_{j,i}^C + (\mathbf{q}_{t,i}^R)^T \mathbf{k}_j^R
$$

The first term $(\mathbf{q}_{t,i}^C)^T \mathbf{k}_{j,i}^C$ captures content-based similarity with no position information. The second term $(\mathbf{q}_{t,i}^R)^T \mathbf{k}_j^R$ captures relative position information via RoPE. The two concerns are cleanly separated.

The full attention output for head $i$ is:

$$
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}}\right) \mathbf{v}_{j,i}^C
$$

Note that the scaling factor is $\sqrt{d_h + d_h^R} = \sqrt{64 + 32} = \sqrt{96}$, since the query-key dot product now operates in the concatenated $(d_h + d_h^R)$-dimensional space.

The final output is:

$$
\mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \ldots; \mathbf{o}_{t,n_h}]
$$

### 4.5 What Gets Cached

During inference, we cache two things per token per layer:

1. The KV latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c} = \mathbb{R}^{128}$
2. The decoupled RoPE key $\mathbf{k}_t^R \in \mathbb{R}^{d_h^R} = \mathbb{R}^{32}$

The RoPE key must be cached because it has already been rotated by the position-dependent RoPE matrix — we cannot reconstruct it from $\mathbf{c}_t^{KV}$.

$$
\boxed{\text{KV cache elements per token per layer (MLA)} = d_c + d_h^R = 128 + 32 = 160}
$$

### Numerical check

Across $L = 12$ layers: $160 \times 12 = 1{,}920$ elements per token. In fp16: $1{,}920 \times 2 = 3{,}840$ bytes $= 3.75$ KB per token.

Compare to MHA's 24 KB per token. The ratio is $3.75/24 = 0.15625$, or about a 6.4x reduction. Compare to MQA's cache of $2 \cdot d_h \cdot L = 2 \times 64 \times 12 = 1{,}536$ elements $= 3$ KB per token. MLA at 3.75 KB is slightly larger than MQA's 3 KB — but MLA achieves *stronger* performance than full MHA, while MQA degrades quality significantly.

### Comparison with GQA

Let us verify the paper's claim that MLA's cache is equivalent to GQA with approximately 2.25 groups. In our model, GQA with $g$ groups caches $2 \cdot g \cdot d_h \cdot L$ elements per token. We want $2 \cdot g \cdot d_h = d_c + d_h^R$:

$$
2 \cdot g \cdot 64 = 128 + 32 = 160
$$

$$
g = \frac{160}{128} = 1.25
$$

So in our model, MLA's cache is equivalent to GQA with 1.25 groups. In the production DeepSeek-V2 model, $d_c = 512$, $d_h^R = 64$, and $d_h = 128$, giving $g = (512 + 64)/(2 \times 128) = 576/256 = 2.25$ groups. This matches the paper's claim exactly.

---

## 5. The Matrix Absorption Trick

We have established that MLA caches only $\mathbf{c}_t^{KV}$ and $\mathbf{k}_t^R$. But there is a computational concern: to compute attention, we need the full keys $\mathbf{k}_{t,i}^C = W_i^{UK} \mathbf{c}_t^{KV}$ and full values $\mathbf{v}_{t,i}^C = W_i^{UV} \mathbf{c}_t^{KV}$. For every previous token $j$ in the context, we would need to apply these up-projection matrices to the cached latent $\mathbf{c}_j^{KV}$. This would add $O(T)$ matrix-vector multiplications per layer per generated token, which is expensive.

The solution is to absorb the up-projection matrices into the query and output projections. This is possible because of the **associative law of matrix multiplication**: $(AB)C = A(BC)$.

### 5.1 Absorbing $W^{UK}$ Into $W^{UQ}$

Consider the content part of the attention score for head $i$:

$$
(\mathbf{q}_{t,i}^C)^T \mathbf{k}_{j,i}^C = (\mathbf{q}_{t,i}^C)^T W_i^{UK} \mathbf{c}_j^{KV}
$$

Here $\mathbf{q}_{t,i}^C \in \mathbb{R}^{d_h}$, $W_i^{UK} \in \mathbb{R}^{d_h \times d_c}$, and $\mathbf{c}_j^{KV} \in \mathbb{R}^{d_c}$. We can rewrite this as:

$$
(\mathbf{q}_{t,i}^C)^T W_i^{UK} \mathbf{c}_j^{KV} = \left((W_i^{UK})^T \mathbf{q}_{t,i}^C\right)^T \mathbf{c}_j^{KV}
$$

We used the **transpose identity**: $\mathbf{a}^T B \mathbf{c} = (B^T \mathbf{a})^T \mathbf{c}$ for any vectors $\mathbf{a}, \mathbf{c}$ and matrix $B$ of compatible dimensions.

Define $\tilde{\mathbf{q}}_{t,i}^C = (W_i^{UK})^T \mathbf{q}_{t,i}^C \in \mathbb{R}^{d_c}$. Then:

$$
(\mathbf{q}_{t,i}^C)^T \mathbf{k}_{j,i}^C = (\tilde{\mathbf{q}}_{t,i}^C)^T \mathbf{c}_j^{KV}
$$

This is a dot product directly between the transformed query and the cached latent vector. We never need to compute $\mathbf{k}_{j,i}^C$ at all. The up-projection $W_i^{UK}$ has been absorbed into the query side.

In practice, we can precompute a combined matrix $\tilde{W}_i^Q = (W_i^{UK})^T W_i^{UQ}$ that maps directly from the query latent to the $d_c$-dimensional space, so $\tilde{\mathbf{q}}_{t,i}^C = \tilde{W}_i^Q \mathbf{c}_t^Q$. By the **associative law of matrix multiplication**, $(W_i^{UK})^T (W_i^{UQ} \mathbf{c}_t^Q) = ((W_i^{UK})^T W_i^{UQ}) \mathbf{c}_t^Q$, so this is mathematically identical.

### Numerical check

$\tilde{W}_i^Q = (W_i^{UK})^T W_i^{UQ}$ has dimensions $(d_c \times d_h)(d_h \times d_c'/n_h)$... let us be more careful. Actually, $W_i^{UQ}$ is the slice of $W^{UQ}$ for head $i$, so $W_i^{UQ} \in \mathbb{R}^{d_h \times d_c'}$. And $W_i^{UK} \in \mathbb{R}^{d_h \times d_c}$. So $(W_i^{UK})^T \in \mathbb{R}^{d_c \times d_h}$. The product $(W_i^{UK})^T W_i^{UQ} \in \mathbb{R}^{d_c \times d_c'}= \mathbb{R}^{128 \times 96}$. The resulting query $\tilde{\mathbf{q}}_{t,i}^C \in \mathbb{R}^{d_c} = \mathbb{R}^{128}$. This matches the dimension of $\mathbf{c}_j^{KV}$, so the dot product $(\tilde{\mathbf{q}}_{t,i}^C)^T \mathbf{c}_j^{KV}$ is well-defined and produces a scalar. Correct.

### 5.2 Absorbing $W^{UV}$ Into $W^O$

Similarly, the output of head $i$ involves:

$$
\mathbf{o}_{t,i} = \sum_j \alpha_{t,j} \mathbf{v}_{j,i}^C = \sum_j \alpha_{t,j} W_i^{UV} \mathbf{c}_j^{KV} = W_i^{UV} \sum_j \alpha_{t,j} \mathbf{c}_j^{KV}
$$

where $\alpha_{t,j}$ are the attention weights (scalars). We pulled $W_i^{UV}$ out of the sum by **linearity of matrix multiplication**: $\sum_j \alpha_j (W \mathbf{x}_j) = W \sum_j \alpha_j \mathbf{x}_j$.

Now the output projection for head $i$ applies $W_i^O \in \mathbb{R}^{d \times d_h}$:

$$
W_i^O \mathbf{o}_{t,i} = W_i^O W_i^{UV} \sum_j \alpha_{t,j} \mathbf{c}_j^{KV}
$$

By the **associative law**, we precompute $\tilde{W}_i^O = W_i^O W_i^{UV} \in \mathbb{R}^{d \times d_c}$ (that is, $512 \times 128$), and then:

$$
W_i^O \mathbf{o}_{t,i} = \tilde{W}_i^O \sum_j \alpha_{t,j} \mathbf{c}_j^{KV}
$$

Again, we never compute the full values $\mathbf{v}_{j,i}^C$. The up-projection $W_i^{UV}$ has been absorbed into the output projection.

### 5.3 What This Means

After absorption, the attention computation for the content part works directly in the $d_c$-dimensional latent space:

1. Compute $\tilde{\mathbf{q}}_{t,i}^C \in \mathbb{R}^{d_c}$ from the current token's query latent (one matrix-vector multiply per head)
2. For each cached position $j$, compute the attention score as $(\tilde{\mathbf{q}}_{t,i}^C)^T \mathbf{c}_j^{KV}$ (a dot product in $d_c = 128$ dimensions)
3. Compute the weighted sum $\sum_j \alpha_{t,j} \mathbf{c}_j^{KV} \in \mathbb{R}^{d_c}$ (weighted sum of latent vectors)
4. Apply $\tilde{W}_i^O$ to the result (one matrix-vector multiply per head)

We never up-project the keys or values for any previous token. The entire computation stays in the compressed latent space. This is the computational efficiency of MLA: not only is the cache smaller, but the per-token attention arithmetic is cheaper.

---

## 6. The Complete MLA Algorithm

Let us assemble all the pieces into the full computation. For the current token $t$:

**Query pathway:**

$$
\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \quad (\mathbb{R}^{d_c'})
$$

$$
[\mathbf{q}_{t,1}^C; \ldots; \mathbf{q}_{t,n_h}^C] = \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q \quad (\mathbb{R}^{d_h n_h})
$$

$$
[\mathbf{q}_{t,1}^R; \ldots; \mathbf{q}_{t,n_h}^R] = \mathbf{q}_t^R = \text{RoPE}(W^{QR} \mathbf{c}_t^Q) \quad (\mathbb{R}^{d_h^R n_h})
$$

$$
\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R] \quad (\mathbb{R}^{d_h + d_h^R})
$$

**KV pathway:**

$$
\boxed{\mathbf{c}_t^{KV}} = W^{DKV} \mathbf{h}_t \quad (\mathbb{R}^{d_c}) \quad \leftarrow \text{cached}
$$

$$
[\mathbf{k}_{t,1}^C; \ldots; \mathbf{k}_{t,n_h}^C] = \mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV} \quad (\mathbb{R}^{d_h n_h})
$$

$$
\boxed{\mathbf{k}_t^R} = \text{RoPE}(W^{KR} \mathbf{h}_t) \quad (\mathbb{R}^{d_h^R}) \quad \leftarrow \text{cached}
$$

$$
\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R] \quad (\mathbb{R}^{d_h + d_h^R})
$$

$$
[\mathbf{v}_{t,1}^C; \ldots; \mathbf{v}_{t,n_h}^C] = \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV} \quad (\mathbb{R}^{d_h n_h})
$$

**Attention:**

$$
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}}\right) \mathbf{v}_{j,i}^C
$$

$$
\mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \ldots; \mathbf{o}_{t,n_h}]
$$

The boxed vectors are the only quantities cached per token per layer. Everything else is either computed on-the-fly for the current token or absorbed into the query/output matrices.

---

## 7. KV Cache Comparison: MHA vs. GQA vs. MQA vs. MLA

Let us now derive the KV cache formula for every attention mechanism and compare them in a single table. For our running model ($n_h = 8$, $d_h = 64$, $d_c = 128$, $d_h^R = 32$, $L = 12$):

| Mechanism | Elements per token per layer | Total elements ($\times L$) | Bytes (fp16) | Capability |
|-----------|------------------------------|----------------------------|-------------|------------|
| MHA | $2 n_h d_h = 1024$ | $12{,}288$ | 24 KB | Strong |
| GQA ($g = 2$) | $2 g d_h = 256$ | $3{,}072$ | 6 KB | Moderate |
| MQA ($g = 1$) | $2 d_h = 128$ | $1{,}536$ | 3 KB | Weak |
| MLA | $d_c + d_h^R = 160$ | $1{,}920$ | 3.75 KB | Stronger |

### Numerical check

Let us verify the MLA entry. Per layer: $d_c + d_h^R = 128 + 32 = 160$. Across 12 layers: $160 \times 12 = 1{,}920$. In fp16: $1{,}920 \times 2 = 3{,}840$ bytes $= 3.75$ KB. The cache sits between MQA (3 KB) and GQA with 2 groups (6 KB), yet delivers stronger performance than full MHA (24 KB). This is the headline result.

For the production DeepSeek-V2 model ($n_h = 128$, $d_h = 128$, $d_c = 512$, $d_h^R = 64$, $L = 60$):

$$
\text{MHA: } 2 \times 128 \times 128 \times 60 = 1{,}966{,}080 \text{ elements}
$$

$$
\text{MLA: } (512 + 64) \times 60 = 34{,}560 \text{ elements}
$$

The ratio is $34{,}560 / 1{,}966{,}080 = 0.0176$, or a **56.9x** reduction. The paper reports a 93.3% reduction in KV cache, which corresponds to a ratio of $1 - 0.933 = 0.067$. The difference is because the paper compares to the actual DeepSeek 67B model (which has different dimensions), not to a hypothetical MHA version of DeepSeek-V2.

---

## 8. DeepSeekMoE: Fine-Grained Expert Specialization

We now turn to the second architectural innovation: the feed-forward network. In a standard Transformer, each layer has a dense FFN applied to every token. In DeepSeek-V2, all FFNs except the first layer's are replaced with **DeepSeekMoE** layers ([Dai et al., 2024](https://arxiv.org/abs/2401.06066)).

DeepSeekMoE has two key ideas that distinguish it from conventional MoE architectures like GShard:

1. **Fine-grained expert segmentation**: Instead of a few large experts, use many small experts. This allows finer specialization — each expert can focus on a narrower subset of the input space.

2. **Shared expert isolation**: Designate some experts as "shared" — they process *every* token regardless of routing decisions. The remaining experts are "routed" by a gating mechanism. This prevents redundant knowledge from being duplicated across multiple routed experts.

### 8.1 The FFN Output

Let $\mathbf{u}_t \in \mathbb{R}^d$ be the FFN input for the $t$-th token (the output of the attention sublayer after the residual connection and normalization). The DeepSeekMoE output is:

$$
\boxed{\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \, \text{FFN}_i^{(r)}(\mathbf{u}_t)}
$$

where:
- $N_s$ is the number of **shared experts** (always active)
- $N_r$ is the number of **routed experts** (selectively activated)
- $\text{FFN}_i^{(s)}(\cdot)$ is the $i$-th shared expert (a standard FFN)
- $\text{FFN}_i^{(r)}(\cdot)$ is the $i$-th routed expert (a standard FFN)
- $g_{i,t}$ is the gate value for routed expert $i$ on token $t$
- The leading $\mathbf{u}_t$ is the residual connection

In DeepSeek-V2: $N_s = 2$ shared experts and $N_r = 160$ routed experts, with $K_r = 6$ routed experts activated per token.

For our running example, let us use $N_s = 1$ shared expert, $N_r = 8$ routed experts, and $K_r = 2$ activated per token. This keeps the arithmetic simple while preserving the architecture's structure.

### 8.2 The Routing Mechanism

The gate values $g_{i,t}$ determine which routed experts are activated and with what weight. They are computed in three steps.

**Step 1: Token-to-expert affinity scores.** For each routed expert $i$, we compute a raw affinity score:

$$
s_{i,t} = \text{Softmax}_i(\mathbf{u}_t^T \mathbf{e}_i)
$$

where $\mathbf{e}_i \in \mathbb{R}^d$ is the **centroid** of the $i$-th routed expert — a learned parameter vector. The softmax is taken over all $N_r$ experts, so $\sum_{i=1}^{N_r} s_{i,t} = 1$. Each $s_{i,t}$ can be interpreted as "the probability that expert $i$ is the right expert for token $t$."

### Numerical check

In our running example, suppose token $t$ has hidden state $\mathbf{u}_t$ and the dot products with the 8 expert centroids are $[2.0, 1.5, 0.3, -0.1, 0.8, -0.5, 1.0, 0.2]$. After softmax:

$$
s_{i,t} = \frac{e^{z_i}}{\sum_{j=1}^{8} e^{z_j}} \quad \text{where } z_i = \mathbf{u}_t^T \mathbf{e}_i
$$

Computing: $e^{2.0} = 7.389$, $e^{1.5} = 4.482$, $e^{0.3} = 1.350$, $e^{-0.1} = 0.905$, $e^{0.8} = 2.226$, $e^{-0.5} = 0.607$, $e^{1.0} = 2.718$, $e^{0.2} = 1.221$. Sum $= 20.897$.

$$
s_{1,t} = 7.389/20.897 = 0.354, \quad s_{2,t} = 4.482/20.897 = 0.214
$$

$$
s_{3,t} = 0.065, \quad s_{4,t} = 0.043, \quad s_{5,t} = 0.107
$$

$$
s_{6,t} = 0.029, \quad s_{7,t} = 0.130, \quad s_{8,t} = 0.058
$$

Sum check: $0.354 + 0.214 + 0.065 + 0.043 + 0.107 + 0.029 + 0.130 + 0.058 = 1.000$. Correct.

**Step 2: Top-K selection.** We select the $K_r$ experts with the highest affinity scores:

$$
\text{TopK}(\{s_{j,t} \mid 1 \leq j \leq N_r\}, K_r)
$$

In our example with $K_r = 2$: the top-2 experts are expert 1 ($s_{1,t} = 0.354$) and expert 2 ($s_{2,t} = 0.214$).

**Step 3: Gate values.** The gate value for expert $i$ is:

$$
g_{i,t} = \begin{cases} s_{i,t} & \text{if } s_{i,t} \in \text{TopK}(\{s_{j,t}\}, K_r) \\ 0 & \text{otherwise} \end{cases}
$$

So in our example: $g_{1,t} = 0.354$, $g_{2,t} = 0.214$, and $g_{i,t} = 0$ for $i = 3, \ldots, 8$.

### Numerical check of the FFN output

The FFN output for token $t$ is:

$$
\mathbf{h}_t' = \mathbf{u}_t + \text{FFN}_1^{(s)}(\mathbf{u}_t) + 0.354 \cdot \text{FFN}_1^{(r)}(\mathbf{u}_t) + 0.214 \cdot \text{FFN}_2^{(r)}(\mathbf{u}_t)
$$

Only 3 FFNs are evaluated: 1 shared expert and 2 routed experts. The remaining 6 routed experts are not computed at all. This is the computational savings of MoE: we have $N_s + N_r = 9$ experts in total, but only evaluate $N_s + K_r = 3$ per token.

In the production model: $N_s + N_r = 2 + 160 = 162$ experts per layer, but only $N_s + K_r = 2 + 6 = 8$ are evaluated per token. That is $8/162 \approx 4.9\%$ of the experts — a 20x sparsity ratio.

---

## 9. Why Fine-Grained Segmentation Helps

This is a design choice that deserves explanation. Why use 160 small routed experts instead of, say, 16 large ones? The answer lies in **combinatorial specialization**.

With $N_r = 16$ large experts and $K_r = 2$, the number of possible expert combinations per token is:

$$
\binom{16}{2} = \frac{16!}{2! \cdot 14!} = \frac{16 \times 15}{2} = 120
$$

This is the **binomial coefficient** formula $\binom{n}{k} = \frac{n!}{k!(n-k)!}$.

With $N_r = 160$ fine-grained experts and $K_r = 6$:

$$
\binom{160}{6} = \frac{160 \times 159 \times 158 \times 157 \times 156 \times 155}{6!} = \frac{160 \times 159 \times 158 \times 157 \times 156 \times 155}{720}
$$

Let us compute this step by step. Numerator: $160 \times 159 = 25{,}440$. Then $25{,}440 \times 158 = 4{,}019{,}520$. Then $4{,}019{,}520 \times 157 = 631{,}064{,}640$. Then $631{,}064{,}640 \times 156 = 98{,}446{,}083{,}840$. Then $98{,}446{,}083{,}840 \times 155 = 15{,}259{,}142{,}995{,}200$.

Dividing by $720$: $15{,}259{,}142{,}995{,}200 / 720 = 21{,}193{,}254{,}160$.

So $\binom{160}{6} \approx 21.2$ billion possible expert combinations. Compare to 120 with the coarse-grained design. The fine-grained design has $21.2 \times 10^9 / 120 \approx 1.77 \times 10^8$ times more possible specialization patterns. Each token can be served by a highly specific combination of micro-experts, enabling much finer-grained knowledge representation.

The total number of parameters is roughly the same in both designs (you can match total expert parameters by making each fine-grained expert proportionally smaller), but the combinatorial expressiveness is vastly greater.

---

## 10. Shared Expert Isolation

The second key idea in DeepSeekMoE is **shared expert isolation**. In a standard MoE without shared experts, common knowledge (e.g., basic language patterns, universal syntactic rules) must be learned independently by multiple routed experts, because any token might be routed to any subset of experts. This wastes capacity — multiple experts end up storing redundant copies of the same common knowledge.

Shared experts solve this by providing a dedicated pathway for common knowledge. Every token passes through all $N_s$ shared experts, so common patterns need only be learned once. The routed experts are then free to specialize in less common, more specific patterns.

Mathematically, the decomposition is clean:

$$
\mathbf{h}_t' = \mathbf{u}_t + \underbrace{\sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t)}_{\text{common knowledge}} + \underbrace{\sum_{i=1}^{N_r} g_{i,t} \, \text{FFN}_i^{(r)}(\mathbf{u}_t)}_{\text{specialized knowledge}}
$$

The residual $\mathbf{u}_t$ preserves the input, the shared experts add common transformations, and the routed experts add token-specific refinements. This three-way decomposition is analogous to a principal component decomposition: the shared experts capture the high-variance common modes, and the routed experts capture the low-variance specialized modes.

---

## 11. Load Balancing: Three Auxiliary Losses

A persistent problem with MoE models is **routing collapse**: the router learns to send most tokens to a small number of "popular" experts, leaving many experts undertrained and underutilized. DeepSeek-V2 addresses this with three auxiliary losses, each targeting a different level of the deployment hierarchy.

### 11.1 Expert-Level Balance Loss

The expert-level balance loss encourages each individual expert to receive an approximately equal share of tokens. It is defined as:

$$
\mathcal{L}_\text{ExpBal} = \alpha_1 \sum_{i=1}^{N_r} f_i P_i
$$

where $\alpha_1$ is a hyperparameter (set to 0.003 in DeepSeek-V2), and $f_i$ and $P_i$ capture the actual and intended load on expert $i$ across a sequence of $T$ tokens:

$$
f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ selects Expert } i)
$$

$$
P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}
$$

Here $f_i$ is the **fraction of tokens routed to expert** $i$, normalized so that perfectly balanced routing gives $f_i = 1$ for all $i$. The indicator function $\mathbb{1}(\cdot)$ equals 1 when the condition is true and 0 otherwise. $P_i$ is the **mean gate probability** of expert $i$ across all tokens.

### Numerical check

In our running example ($N_r = 8$, $K_r = 2$), suppose over $T = 4$ tokens, the routing decisions are:

| Token | Selected experts | $s_{1,t}$ | $s_{2,t}$ | $s_{3,t}$ | $s_{4,t}$ | $s_{5,t}$ | $s_{6,t}$ | $s_{7,t}$ | $s_{8,t}$ |
|-------|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1 | 1, 2 | 0.35 | 0.21 | 0.07 | 0.04 | 0.11 | 0.03 | 0.13 | 0.06 |
| 2 | 1, 7 | 0.30 | 0.10 | 0.05 | 0.08 | 0.12 | 0.04 | 0.25 | 0.06 |
| 3 | 1, 5 | 0.28 | 0.15 | 0.09 | 0.06 | 0.22 | 0.03 | 0.11 | 0.06 |
| 4 | 2, 3 | 0.10 | 0.32 | 0.26 | 0.05 | 0.08 | 0.07 | 0.07 | 0.05 |

Expert 1 is selected by tokens 1, 2, 3 (3 times). Expert 2 is selected by tokens 1, 4 (2 times). Expert 3 is selected by token 4 (1 time). Expert 5 is selected by token 3 (1 time). Expert 7 is selected by token 2 (1 time). Experts 4, 6, 8 are never selected.

Computing $f_i$: $f_1 = \frac{8}{2 \times 4} \times 3 = 1 \times 3 = 3$. $f_2 = 1 \times 2 = 2$. $f_3 = 1 \times 1 = 1$. $f_5 = 1$. $f_7 = 1$. $f_4 = f_6 = f_8 = 0$.

Computing $P_i$: $P_1 = (0.35 + 0.30 + 0.28 + 0.10)/4 = 1.03/4 = 0.2575$. $P_2 = (0.21 + 0.10 + 0.15 + 0.32)/4 = 0.78/4 = 0.195$. And so on for each expert.

The loss $\mathcal{L}_\text{ExpBal} = \alpha_1 \sum_i f_i P_i$ penalizes experts that have both high actual load ($f_i$) and high routing probability ($P_i$). If expert 1 is receiving too many tokens ($f_1 = 3$, three times the balanced level) and also has high probability ($P_1 = 0.2575$), the product $f_1 P_1 = 3 \times 0.2575 = 0.7725$ contributes a large term to the loss, pushing the router to spread tokens more evenly.

### Why the product $f_i P_i$?

This is the same formulation used in Switch Transformer (Fedus et al., 2021) and GShard (Lepikhin et al., 2021). The key insight is that $f_i$ involves a discrete selection (the indicator function $\mathbb{1}$), which is not differentiable. But $P_i$ involves the softmax probabilities $s_{i,t}$, which *are* differentiable. The product $f_i P_i$ creates a differentiable loss that the router can optimize via gradient descent: the gradient flows through $P_i$ while $f_i$ acts as a coefficient that amplifies the gradient for overloaded experts.

### 11.2 Device-Level Balance Loss

When experts are distributed across multiple devices (GPUs), imbalanced routing causes some devices to become bottlenecks. The **device-level balance loss** encourages balanced computation across devices.

Partition the $N_r$ routed experts into $D$ groups $\{\mathcal{E}_1, \mathcal{E}_2, \ldots, \mathcal{E}_D\}$, one per device. The loss is:

$$
\mathcal{L}_\text{DevBal} = \alpha_2 \sum_{i=1}^{D} f_i' P_i'
$$

where:

$$
f_i' = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j, \quad P_i' = \sum_{j \in \mathcal{E}_i} P_j
$$

$f_i'$ is the average load fraction across experts on device $i$, and $P_i'$ is the total routing probability mass directed to device $i$.

### 11.3 Communication Balance Loss

Even if devices have balanced *computation*, they may have unbalanced *communication*: some devices receive tokens from many other devices while others receive few. The **communication balance loss** addresses this:

$$
\mathcal{L}_\text{CommBal} = \alpha_3 \sum_{i=1}^{D} f_i'' P_i''
$$

where:

$$
f_i'' = \frac{D}{MT} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ is sent to Device } i)
$$

$$
P_i'' = \sum_{j \in \mathcal{E}_i} P_j
$$

Here $M$ is the maximum number of devices each token can be sent to (from the device-limited routing constraint). The three hyperparameters are set to $\alpha_1 = 0.003$, $\alpha_2 = 0.05$, and $\alpha_3 = 0.02$ in DeepSeek-V2.

---

## 12. Device-Limited Routing

Standard top-K routing has a communication problem when experts are spread across devices. If a token's top-$K_r$ experts happen to live on $K_r$ different devices, that token must be sent to $K_r$ devices — generating $K_r$ cross-device communications. With fine-grained expert segmentation ($N_r = 160$, $K_r = 6$), the worst case means each token communicates with 6 different devices.

DeepSeek-V2 introduces **device-limited routing** to cap this. For each token:

1. Compute affinity scores for all $N_r$ experts
2. Identify the top $M$ *devices* (by total affinity of their hosted experts)
3. Select the top-$K_r$ experts *only from experts on these $M$ devices*

In DeepSeek-V2, $M = 3$ and $D = 8$. So each token communicates with at most 3 out of 8 devices, regardless of how many experts are activated. The paper reports that $M \geq 3$ achieves performance comparable to unrestricted routing.

---

## 13. Token-Dropping Strategy

Balance losses encourage but do not guarantee perfect balance. To handle residual imbalance during training, DeepSeek-V2 uses a **token-dropping strategy**:

1. Compute the average computational budget per device (capacity factor 1.0)
2. On each device, drop the tokens with the lowest affinity scores until the device's load does not exceed its budget
3. Ensure that no more than approximately 10% of tokens in any training sequence are dropped

This is applied only during training. During inference, no tokens are dropped — the model processes all tokens, accepting any load imbalance.

---

## 14. Putting It All Together: The DeepSeek-V2 Transformer Block

A single DeepSeek-V2 Transformer block processes a token $\mathbf{h}_t$ through two sublayers:

**Sublayer 1 — Multi-head Latent Attention (MLA):**

$$
\mathbf{h}_t' = \mathbf{h}_t + \text{MLA}(\text{RMSNorm}(\mathbf{h}_t))
$$

**Sublayer 2 — DeepSeekMoE (or dense FFN for layer 1):**

$$
\mathbf{h}_t'' = \mathbf{h}_t' + \text{DeepSeekMoE}(\text{RMSNorm}(\mathbf{h}_t'))
$$

The model has $L = 60$ such blocks stacked. The first block uses a dense FFN instead of MoE. All other blocks use MoE with $N_s = 2$ shared experts and $N_r = 160$ routed experts.

### Parameter count

The full DeepSeek-V2 model has 236B total parameters. Per token, only 21B are activated: the MLA parameters (which are used for every token) plus the $N_s + K_r = 8$ active expert FFNs per layer (out of 162 total). The ratio is $21/236 \approx 8.9\%$ — over 91% of parameters are inactive for any given token.

---

## Summary

DeepSeek-V2 contributes two architectural innovations that are independent and complementary.

**Multi-head Latent Attention** replaces the standard practice of caching separate key and value vectors with a single low-rank latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$, from which keys and values are recovered via learned up-projections. Position information is carried by a separate decoupled RoPE key $\mathbf{k}_t^R$ that avoids the incompatibility between RoPE and low-rank compression. During inference, the up-projection matrices are absorbed into the query and output projections via the associative law of matrix multiplication, so keys and values are never explicitly computed for cached tokens. The result: a KV cache equivalent to GQA with only 2.25 groups, but with performance stronger than full MHA.

**DeepSeekMoE** replaces dense FFNs with a mixture of fine-grained experts (160 small routed experts plus 2 shared experts), of which only 8 are evaluated per token. Fine-grained segmentation enables $\binom{160}{6} \approx 21.2$ billion possible expert combinations per token, vastly more than coarse-grained alternatives. Shared expert isolation prevents common knowledge from being redundantly stored across routed experts. Three levels of auxiliary losses (expert, device, communication) and device-limited routing ensure balanced, efficient training across multi-device setups.
