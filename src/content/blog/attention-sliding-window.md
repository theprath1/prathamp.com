---
title: "Sliding Window Attention: From Local Windows to Global Context"
description: "Borrowing from convolutions — how a sliding local window keeps attention linear in sequence length, recovers a global receptive field by stacking layers, and reaches across the whole document through a small set of task-driven global tokens."
date: 2026-04-06
tags: [machine-learning, attention, transformers, sliding-window, longformer, efficiency, sparse-attention]
order: 1
---

The previous blog derived sparse factorized attention: replace the dense $n^2$ pattern with two structured sparse patterns of size $O(\sqrt{n})$ each, reducing total cost from $O(n^2)$ to $O(n\sqrt{n})$. Strided and fixed factorizations preserved full reachability through two-hop paths, and the Sparse Transformer achieved state-of-the-art results on images, text, and audio.

But the Sparse Transformer has limitations. Its factorized patterns are rigid — strided attention assumes periodic structure, fixed attention designates relay positions by location. Neither adapts to the task at hand. And $O(n\sqrt{n})$, while better than $O(n^2)$, still grows faster than linear.

Longformer (Beltagy, Peters, and Cohan, 2020) takes a different, simpler approach on the same axis. It combines a sliding window for local context with task-motivated global attention on a few designated tokens. The sliding window costs $O(n \times w)$ where $w$ is the fixed window size — linear in $n$. The global attention adds $O(n \times g)$ where $g$ is the number of global tokens — also linear. The total is $O(n)$, strictly linear, with no growing exponents.

Even more importantly, Longformer is designed as a drop-in replacement for existing pretrained models. You can take a RoBERTa checkpoint, swap its full attention for Longformer's sliding window attention, and continue pretraining. This makes it the first efficient attention mechanism that cleanly integrates with the pretrain-finetune paradigm that dominates modern NLP.

We will derive the entire mechanism from scratch, trace every connectivity set by hand, and verify every count numerically.

---

## The Running Example

We use a sequence of $n = 16$ tokens throughout the entire post, now interpreted as a short document rather than an image:

$$
\underbrace{x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_{10}, x_{11}, x_{12}, x_{13}, x_{14}, x_{15}}_{n = 16 \text{ tokens}}
$$

We fix the following parameters:

- Window size: $w = 4$ (each token attends to $w/2 = 2$ tokens on each side)
- Dilation: $d = 2$ (for dilated heads)
- Global tokens: $g = 2$ (positions 0 and 8 are designated global)

For cost comparisons, the same model as always:

- $d_\text{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$, $L = 12$ layers, fp16

We define the **half-window radius** as $r = w/2$. With $w = 4$, we have $r = 2$. This $r$ is the number of positions each token can see on each side.

---

## 1. Sliding Window Attention

### 1.1 The connectivity set

In **sliding window attention**, each token attends only to tokens within a fixed distance $r = w/2$ on each side. The connectivity set for position $i$ in the bidirectional (encoder) case is:

$$
S_i = \{j : |i - j| \leq r\} = \{j : \max(0,\, i - r) \leq j \leq \min(n - 1,\, i + r)\}
$$

This is the set of all positions within distance $r$ of position $i$, clipped to the valid range $[0, n-1]$.

### 1.2 Tracing for our running example

With $r = 2$, $n = 16$:

| Position $i$ | $S_i$ | $\|S_i\|$ |
|---|---|---|
| 0 | $\{0, 1, 2\}$ | 3 |
| 1 | $\{0, 1, 2, 3\}$ | 4 |
| 2 | $\{0, 1, 2, 3, 4\}$ | 5 |
| 3 | $\{1, 2, 3, 4, 5\}$ | 5 |
| 4 | $\{2, 3, 4, 5, 6\}$ | 5 |
| $\vdots$ | $\vdots$ | $\vdots$ |
| 13 | $\{11, 12, 13, 14, 15\}$ | 5 |
| 14 | $\{12, 13, 14, 15\}$ | 4 |
| 15 | $\{13, 14, 15\}$ | 3 |

Positions 0 and 15 are at the boundary and see only $r + 1 = 3$ tokens. Position 1 and 14 see $2r = 4$ tokens. All interior positions (2 through 13) see $2r + 1 = 5$ tokens.

### 1.3 Counting entries

The total number of attention entries across all positions is:

$$
\sum_{i=0}^{n-1} |S_i|
$$

We split this into three groups: left boundary, interior, and right boundary.

**Left boundary** (positions $0, 1, \ldots, r-1$): Position $i$ has $|S_i| = i + r + 1$ (it can look $i$ positions to the left and $r$ to the right, plus itself). So:

$$
\sum_{i=0}^{r-1} (i + r + 1) = \sum_{i=0}^{r-1} i + r(r) + r = \frac{r(r-1)}{2} + r^2 + r = \frac{r(r-1)}{2} + r(r+1)
$$

Simplify by factoring out $r$:

$$
= r\left[\frac{r-1}{2} + r + 1\right] = r\left[\frac{r - 1 + 2r + 2}{2}\right] = r\left[\frac{3r + 1}{2}\right] = \frac{r(3r+1)}{2}
$$

**Interior** (positions $r, r+1, \ldots, n-1-r$): Each has $|S_i| = 2r + 1$. There are $n - 2r$ such positions:

$$
(n - 2r)(2r + 1)
$$

**Right boundary** (positions $n-r, \ldots, n-1$): By symmetry with the left boundary:

$$
\frac{r(3r+1)}{2}
$$

**Total:**

$$
\text{Total}_\text{window} = 2 \cdot \frac{r(3r+1)}{2} + (n - 2r)(2r+1) = r(3r+1) + (n-2r)(2r+1)
$$

Expand the second term by the **distributive law**:

$$
= 3r^2 + r + (2r+1)n - 2r(2r+1)
$$

$$
= 3r^2 + r + (2r+1)n - 4r^2 - 2r
$$

$$
= (2r+1)n - r^2 - r
$$

$$
= (2r+1)n - r(r+1)
$$

Factor differently for clarity:

$$
\boxed{\text{Total}_\text{window} = (2r+1)n - r(r+1) = (w+1)n - \frac{w}{2}\left(\frac{w}{2}+1\right)}
$$

### 1.4 Numerical check

Substitute $r = 2$, $n = 16$:

$$
\text{Total}_\text{window} = (2 \cdot 2 + 1) \cdot 16 - 2 \cdot (2 + 1) = 5 \cdot 16 - 6 = 80 - 6 = 74
$$

Verify by direct summation from the table:

$$
3 + 4 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 4 + 3
$$

Interior: $12 \times 5 = 60$. Boundaries: $3 + 4 + 4 + 3 = 14$. Total: $60 + 14 = 74$. $\checkmark$

### 1.5 Comparison to full attention and sparse factorization

For $n = 16$:

| Pattern | Total entries | Formula |
|---|---|---|
| Full bidirectional | $n^2 = 256$ | $O(n^2)$ |
| Full causal | $n(n+1)/2 = 136$ | $O(n^2)$ |
| Sparse factorized (Blog 9) | 110 | $O(n\sqrt{n})$ |
| **Sliding window** ($w = 4$) | **74** | $O(n \cdot w)$ |

At $n = 16$, the sliding window already computes fewer entries than sparse factorization. But the real difference appears at scale.

### 1.6 Scaling comparison

For large $n$, the approximate entries per pattern are:

$$
\text{Full} \approx n^2, \qquad \text{Sparse} \approx \frac{3}{2} n^{3/2}, \qquad \text{Window} \approx (2r+1) n = (w+1)n
$$

| $n$ | Full $n^2$ | Sparse $\frac{3}{2}n^{3/2}$ | Window $(w{+}1)n$, $w{=}512$ | Window reduction vs full |
|---|---|---|---|---|
| 1,024 | 1,048,576 | 49,152 | 525,312 | $2\times$ |
| 4,096 | 16,777,216 | 393,216 | 2,101,248 | $8\times$ |
| 16,384 | 268,435,456 | 3,145,728 | 8,404,992 | $32\times$ |
| 65,536 | 4,294,967,296 | 25,165,824 | 33,619,968 | $128\times$ |

At moderate sequence lengths ($n \approx 4{,}096$), sparse factorization is more aggressive because $n^{3/2}$ grows slower than $n \cdot w$ when $w$ is large. But at very long sequences, both are far better than full attention. The crucial difference is that the sliding window's cost is a constant factor times $n$ — it does not grow with $n$ at all once $w$ is fixed. Sparse factorization's cost still grows as $n^{1/2}$.

### 1.7 Interpretation

Sliding window attention makes a strong bet: local context is what matters most. This is empirically well-supported. Kovaleva et al. (2019) showed that BERT's attention heads overwhelmingly attend to nearby tokens — the vast majority of attention weight falls within a local window. The sliding window formalizes this observation into an architectural constraint.

But local context alone is not enough. Token 0 cannot reach token 15 in a single layer. The receptive field is limited to $w + 1 = 5$ positions per layer. We need a mechanism for long-range information flow.

---

## 2. Receptive Field Growth Through Stacking

### 2.1 The key insight: layers compound the window

This is where the sliding window reveals a deep structural connection to convolutional neural networks.

Consider what happens when we stack multiple sliding window attention layers. In layer 1, token $i$ receives information from tokens within distance $r$. In layer 2, each of those tokens has already aggregated information from its own window of radius $r$. So token $i$ in layer 2 indirectly has access to information from tokens within distance $2r$ of its original position.

### 2.2 Deriving the receptive field

Let $\text{RF}_\ell$ denote the receptive field radius after $\ell$ layers. The receptive field is the set of original input positions that can influence a given token's representation at layer $\ell$.

**Layer 1:** Token $i$ directly attends to positions in $[i - r, \, i + r]$.

$$
\text{RF}_1 = r
$$

**Layer 2:** Token $i$ attends to tokens in $[i - r, \, i + r]$, each of which carries information from their own window of radius $r$. The leftmost token $i - r$ carries information from as far left as $(i - r) - r = i - 2r$. The rightmost token $i + r$ carries information from as far right as $(i + r) + r = i + 2r$.

$$
\text{RF}_2 = 2r
$$

**Layer $\ell$:** By the same argument applied inductively:

$$
\boxed{\text{RF}_\ell = \ell \cdot r = \ell \cdot \frac{w}{2}}
$$

The total number of original input positions that can influence token $i$ after $\ell$ layers is $2 \cdot \text{RF}_\ell + 1 = \ell \cdot w + 1$.

### 2.3 Numerical check

With $\ell = L = 12$ layers and $w = 4$ ($r = 2$):

$$
\text{RF}_{12} = 12 \times 2 = 24
$$

The total reachable width is $2 \times 24 + 1 = 49$ positions. Since our sequence has only $n = 16$ tokens, and $49 > 16$, the top layer of a 12-layer model can access the entire sequence through its stacked windows.

For a practical model with $w = 512$ and $L = 12$:

$$
\text{RF}_{12} = 12 \times 256 = 3{,}072
$$

Total reachable width: $2 \times 3{,}072 + 1 = 6{,}145$. For a token far from the boundaries, this is enough to cover a sequence of up to 6,145 positions through local information propagation alone. Boundary tokens see less because the receptive field is clipped by the start or end of the sequence.

### 2.4 The CNN analogy

This is exactly how convolutional neural networks build global representations from local filters. A CNN with kernel size $k$ and $L$ layers has a receptive field of approximately $L \times k$. Each layer sees a small local neighborhood, but stacking many layers gives the top layer a view of the entire input.

Sliding window attention is the attention equivalent: each layer computes a local attention pattern (the "kernel"), and the receptive field grows linearly with depth. Wu et al. (2019) made this connection explicit, showing that stacked local attention layers and deep CNNs have similar representational properties.

The key difference from a CNN is that within each window, the attention weights are data-dependent — they are computed via the softmax of query-key dot products, not learned as fixed filter weights. So the model has the inductive bias of locality (from the window) combined with the flexibility of content-based retrieval (from attention).

### 2.5 When the receptive field is not enough

The receptive field of $\ell \cdot w + 1$ grows linearly with depth, but this growth relies on information propagating one window at a time through the network. For a 4,096-token document with $w = 512$ and $L = 12$, an interior token's receptive field of 6,145 positions is wide enough to cover the full document. But information from the far end still propagates one window at a time: traversing the full 4,096-token span requires about $\lceil 4{,}096 / 256 \rceil = 16$ overlapping-window transfers, and the representation at each hop is lossy.

For tasks that require direct global access — classification from a [CLS] token, comparing a question to an answer span far away in the document — the receptive field argument is not sufficient. We need some tokens to have direct access to the entire sequence. That is the role of global attention, which we introduce in Section 4.

But first, we can improve the receptive field itself without changing the window size.

---

## 3. Dilated Sliding Window

### 3.1 Definition

A **dilated sliding window** introduces gaps of size $d$ (the **dilation rate**) between the attended positions. Instead of attending to consecutive positions within the window, each token attends to positions spaced $d$ apart.

The connectivity set for position $i$ with half-window radius $r$ and dilation $d$ is:

$$
S_i^{(d)} = \{j : j = i + k \cdot d, \; k \in \{-r, -r+1, \ldots, r\}, \; 0 \leq j \leq n - 1\}
$$

This is analogous to **dilated convolutions** (van den Oord et al., 2016), where the kernel samples inputs at regular intervals rather than contiguously.

### 3.2 Tracing for our running example

With $r = 2$, $d = 2$, $n = 16$: each token attends to positions $\{i - 4, i - 2, i, i + 2, i + 4\}$ (clipped to valid range).

| Position $i$ | $S_i^{(2)}$ | $\|S_i^{(2)}\|$ |
|---|---|---|
| 0 | $\{0, 2, 4\}$ | 3 |
| 1 | $\{1, 3, 5\}$ | 3 |
| 2 | $\{0, 2, 4, 6\}$ | 4 |
| 3 | $\{1, 3, 5, 7\}$ | 4 |
| 4 | $\{0, 2, 4, 6, 8\}$ | 5 |
| 5 | $\{1, 3, 5, 7, 9\}$ | 5 |
| 6 | $\{2, 4, 6, 8, 10\}$ | 5 |
| 7 | $\{3, 5, 7, 9, 11\}$ | 5 |
| 8 | $\{4, 6, 8, 10, 12\}$ | 5 |
| 9 | $\{5, 7, 9, 11, 13\}$ | 5 |
| 10 | $\{6, 8, 10, 12, 14\}$ | 5 |
| 11 | $\{7, 9, 11, 13, 15\}$ | 5 |
| 12 | $\{8, 10, 12, 14\}$ | 4 |
| 13 | $\{9, 11, 13, 15\}$ | 4 |
| 14 | $\{10, 12, 14\}$ | 3 |
| 15 | $\{11, 13, 15\}$ | 3 |

### 3.3 Total entries

The total number of entries is identical to the non-dilated case in terms of count — each position still attends to at most $2r + 1$ positions. The entries have the same boundary effects. So:

$$
\text{Total}_{\text{dilated}} = (2r + 1)n - r(r+1) = 74
$$

The same 74 entries as the non-dilated window. The dilation does not change the number of computations — it changes which positions are attended to.

### 3.4 Receptive field with dilation

The reach of each layer increases by a factor of $d$. With dilation $d$ and half-window $r$, each token reaches positions up to $r \cdot d$ away on each side.

After $\ell$ layers, each using dilation $d$:

$$
\boxed{\text{RF}_\ell^{(d)} = \ell \cdot r \cdot d = \ell \cdot \frac{w \cdot d}{2}}
$$

### 3.5 Numerical check

With $\ell = 12$, $r = 2$, $d = 2$:

$$
\text{RF}_{12}^{(2)} = 12 \times 2 \times 2 = 48
$$

Total reachable width: $2 \times 48 + 1 = 97$ positions. Compare to the non-dilated receptive field of 49 — dilation doubles the reach for the same compute cost.

### 3.6 Multi-head mixing: some heads dilated, others not

This is a practical design choice that the paper emphasizes. In multi-head attention, different heads can use different dilation rates. Some heads use $d = 1$ (no dilation) to capture fine-grained local context, while others use $d > 1$ to reach distant positions.

For example, with $h = 8$ heads: heads 1–6 use $d = 1$ (local), heads 7–8 use $d = 2$ (dilated). This gives the model both detailed local information and broader context, without increasing total compute.

### 3.7 Varying window size across layers

The paper found that varying $w$ across layers improves performance. Specifically, using small windows in lower layers and increasing window sizes in higher layers works best. The intuition is:

- **Lower layers** learn local features (part-of-speech, local syntax) — a small window suffices
- **Higher layers** learn document-level features (topic, long-range coreference) — a larger window helps

The ablation in Table 4 of the paper confirms this: increasing $w$ from bottom to top gives 1.21 BPC on text8, while decreasing $w$ from bottom to top gives 1.24 BPC, and using a fixed average $w$ gives 1.23 BPC. The ordering matters.

For the paper's best character-level language model: the bottom layers use $w = 32$ and the top layers use $w = 512$, with the window doubling every few layers across 5 training phases.

### 3.8 Interpretation

Dilation is a free lunch in terms of compute: same number of attention entries, larger receptive field. The price is that dilated attention skips intermediate positions — token $i$ with $d = 2$ sees positions $\{i-4, i-2, i, i+2, i+4\}$ but not $\{i-3, i-1, i+1, i+3\}$. The skipped positions' information must be filled in by non-dilated heads or by multi-layer propagation. This is why the multi-head mixing strategy works: different heads cover different gaps.

---

## 4. Global Attention

### 4.1 Motivation: some tokens need to see everything

The sliding window — even with dilation and multi-layer stacking — builds global context gradually. But certain NLP tasks require specific tokens to have direct, single-layer access to the entire sequence:

- **Classification**: The [CLS] token must aggregate information from every position to make a prediction.
- **Question answering**: Question tokens must be compared to every position in the document to find the answer span.
- **Summarization**: Encoder tokens at the beginning of the document may need to attend to the conclusion.

For these tasks, purely local attention is not flexible enough. The model needs a way to inject global connectivity for a small number of task-relevant tokens.

### 4.2 The global attention mechanism

**Global attention** designates a small set of $g$ tokens as "global." The attention pattern for global tokens is:

1. A global token attends to **all** tokens in the sequence (not just its local window)
2. **All** tokens in the sequence attend to the global token (not just tokens within its window)

This is a symmetric definition: the global token both reads from and is read by the entire sequence.

### 4.3 Formalizing the connectivity sets

Let $G \subset \{0, 1, \ldots, n-1\}$ be the set of global positions, with $|G| = g$. The connectivity set for each position becomes:

**For a global position** $i \in G$:

$$
S_i = \{0, 1, \ldots, n - 1\}
$$

The global token attends to all $n$ positions.

**For a local position** $i \notin G$:

$$
S_i = \{j : |i - j| \leq r\} \cup G
$$

The local token attends to its window plus all global tokens.

### 4.4 Tracing for our running example

With $n = 16$, $r = 2$, $G = \{0, 8\}$ (two global tokens):

| Position $i$ | Type | $S_i$ | $\|S_i\|$ |
|---|---|---|---|
| **0** | global | $\{0, 1, 2, \ldots, 15\}$ | 16 |
| 1 | local | $\{0, 1, 2, 3\} \cup \{0, 8\} = \{0, 1, 2, 3, 8\}$ | 5 |
| 2 | local | $\{0, 1, 2, 3, 4\} \cup \{0, 8\} = \{0, 1, 2, 3, 4, 8\}$ | 6 |
| 3 | local | $\{1, 2, 3, 4, 5\} \cup \{0, 8\} = \{0, 1, 2, 3, 4, 5, 8\}$ | 7 |
| 4 | local | $\{2, 3, 4, 5, 6\} \cup \{0, 8\} = \{0, 2, 3, 4, 5, 6, 8\}$ | 7 |
| 5 | local | $\{3, 4, 5, 6, 7\} \cup \{0, 8\} = \{0, 3, 4, 5, 6, 7, 8\}$ | 7 |
| 6 | local | $\{4, 5, 6, 7, 8\} \cup \{0, 8\} = \{0, 4, 5, 6, 7, 8\}$ | 6 |
| 7 | local | $\{5, 6, 7, 8, 9\} \cup \{0, 8\} = \{0, 5, 6, 7, 8, 9\}$ | 6 |
| **8** | global | $\{0, 1, 2, \ldots, 15\}$ | 16 |
| 9 | local | $\{7, 8, 9, 10, 11\} \cup \{0, 8\} = \{0, 7, 8, 9, 10, 11\}$ | 6 |
| 10 | local | $\{8, 9, 10, 11, 12\} \cup \{0, 8\} = \{0, 8, 9, 10, 11, 12\}$ | 6 |
| 11 | local | $\{9, 10, 11, 12, 13\} \cup \{0, 8\} = \{0, 8, 9, 10, 11, 12, 13\}$ | 7 |
| 12 | local | $\{10, 11, 12, 13, 14\} \cup \{0, 8\} = \{0, 8, 10, 11, 12, 13, 14\}$ | 7 |
| 13 | local | $\{11, 12, 13, 14, 15\} \cup \{0, 8\} = \{0, 8, 11, 12, 13, 14, 15\}$ | 7 |
| 14 | local | $\{12, 13, 14, 15\} \cup \{0, 8\} = \{0, 8, 12, 13, 14, 15\}$ | 6 |
| 15 | local | $\{13, 14, 15\} \cup \{0, 8\} = \{0, 8, 13, 14, 15\}$ | 5 |

Note that when a global token is already in the local window (e.g., position 1's window includes position 0, and position 7's window includes position 8), the union does not add a new entry.

### 4.5 Counting entries with global attention

The total entries split into three parts:

**Global tokens:** Each of the $g$ global tokens attends to all $n$ positions:

$$
g \cdot n
$$

**Local tokens attending to their window:** The $n - g$ local tokens each attend to their local window. We already derived this: approximately $(2r + 1)(n - g)$ for interior tokens, with boundary corrections.

**Local tokens attending to global tokens:** Each local token adds at most $g$ entries for the global tokens. But some global tokens may already fall within the local window, so the exact count depends on the positions of the global tokens. In the worst case (all global tokens outside all local windows), this adds $(n - g) \cdot g$ entries.

The total is:

$$
\text{Total}_\text{global+window} = g \cdot n + \text{window entries for local tokens} + \text{extra global entries}
$$

For a clean upper bound:

$$
\text{Total}_\text{global+window} \leq g \cdot n + (n - g)(2r + 1 + g)
$$

### 4.6 Numerical check

From the table, summing all $|S_i|$ directly:

$$
16 + 5 + 6 + 7 + 7 + 7 + 6 + 6 + 16 + 6 + 6 + 7 + 7 + 7 + 6 + 5 = 126
$$

Compare to:
- Pure sliding window (no global): 74 entries
- Full bidirectional: 256 entries
- Our global + window: 126 entries

The 126 entries represent a $1.7\times$ increase over the pure sliding window (from 74 to 126) due to the 2 global tokens, but still a $2\times$ reduction from full attention (256). As $n$ grows, the global attention adds only $O(g \cdot n)$, which is linear since $g$ is a small constant.

### 4.7 Complexity

For $n$ much larger than $w$ and $g$:

$$
\boxed{\text{Total}_\text{global+window} = O(n \cdot w + n \cdot g) = O(n \cdot (w + g)) = O(n)}
$$

Since both $w$ and $g$ are fixed constants independent of $n$, the total complexity is **linear in sequence length**.

### 4.8 Which tokens get global attention?

This is task-specific, and the paper makes this an explicit design choice:

- **Classification**: Global attention on the [CLS] token only ($g = 1$)
- **Question answering**: Global attention on all question tokens ($g$ = number of question tokens)
- **Summarization** (LED): Global attention on the first token of the encoder ($g = 1$)
- **Language modeling**: No global attention needed (purely autoregressive, left-to-right windowed attention with dilation)

The flexibility to choose different global tokens for different tasks, without retraining the model, is a practical advantage over approaches like the Sparse Transformer's fixed relay positions.

### 4.9 Interpretation

Global attention plays the same role as the Sparse Transformer's summary positions from Blog 9, but with two crucial differences. First, the global tokens are chosen by the task, not by a fixed positional rule. Second, the global attention is symmetric — the global token attends to all positions and all positions attend to the global token. In the Sparse Transformer's fixed pattern, the summary positions only gathered information from their block; they did not explicitly broadcast back to all positions.

---

## 5. Separate Projections for Global Attention

### 5.1 The problem with shared projections

In standard multi-head attention, all queries, keys, and values are computed from the same projections $W_Q$, $W_K$, $W_V$. But in Longformer, local tokens and global tokens play fundamentally different roles. A local token is doing local context gathering — it only needs to attend to nearby positions. A global token is doing sequence-level aggregation — it needs to compare itself to every position in the document.

These two roles benefit from different learned projections.

### 5.2 The two-projection design

Longformer uses two sets of projection matrices:

- $Q_s, K_s, V_s$: The **sliding window projections**, used for computing local attention scores
- $Q_g, K_g, V_g$: The **global projections**, used for computing attention scores involving global tokens

When a global token computes its attention over the entire sequence, it uses $Q_g$ for its query and $K_g$, $V_g$ for the keys and values of all positions. When local tokens attend to the global token, they use $Q_s$ for their queries but $K_g$, $V_g$ for the global token's key and value.

### 5.3 Why this matters

The paper shows through ablation (Table 10, WikiHop) that removing the separate linear projections for global attention drops accuracy from 73.8 to 72.2, a loss of 1.6 points. Removing both the separate projections and global attention entirely drops it to 65.5, a loss of 8.3 points. So the separate projections account for about $1.6 / 8.3 \approx 19\%$ of the total benefit of global attention in that ablation.

The separate projections are initialized from the values of $Q_s$, $K_s$, $V_s$ so that at the start of finetuning, the global attention mechanism behaves identically to the sliding window attention. During finetuning, the global projections specialize: $Q_g$ learns to produce queries that are effective for whole-sequence retrieval, while $K_g$ and $V_g$ learn to present information in a way that is useful for global aggregation.

### 5.4 Parameter count

The separate projections add one full set of $Q$, $K$, $V$ matrices. Using the parameter count we derived in the taxonomy blog, one set of QKV projections costs $3 d_\text{model}^2$ parameters per layer. In our running model:

$$
3 \times 512^2 = 786{,}432 \text{ parameters per layer}
$$

Across $L = 12$ layers:

$$
12 \times 786{,}432 = 9{,}437{,}184 \approx 9.4\text{M extra parameters}
$$

For a RoBERTa-base model with about 125M parameters, this is a 7.5% increase — modest for the capability it provides.

---

## 6. The Attention Matrix as a Banded Matrix

### 6.1 What the sliding window produces

This is the part that connects the mathematical pattern to implementation. In full attention, the score matrix $S = QK^\top / \sqrt{d_k}$ is dense: every entry is computed. In sliding window attention, the score matrix is **banded**: only entries within distance $r$ of the diagonal are nonzero. Entries outside the band are not merely masked to $-\infty$ — they are never computed at all.

A **banded matrix** is a matrix where all nonzero entries lie within a fixed number of diagonals from the main diagonal. For a window of radius $r$, the band has width $2r + 1$: the main diagonal plus $r$ diagonals above and $r$ diagonals below.

### 6.2 Visualizing the band structure

For $n = 16$ and $r = 2$, the attention mask looks like (marking computed entries with $\bullet$ and skipped entries with $\cdot$):

$$
\begin{pmatrix}
\bullet & \bullet & \bullet & \cdot & \cdot & \cdot & \cdots \\
\bullet & \bullet & \bullet & \bullet & \cdot & \cdot & \cdots \\
\bullet & \bullet & \bullet & \bullet & \bullet & \cdot & \cdots \\
\cdot & \bullet & \bullet & \bullet & \bullet & \bullet & \cdots \\
\cdot & \cdot & \bullet & \bullet & \bullet & \bullet & \cdots \\
\vdots & & & & & \ddots &
\end{pmatrix}
$$

This is the band. Standard matrix multiplication computes all $n^2$ entries. Banded matrix multiplication computes only the $\approx (2r+1) \cdot n$ entries within the band.

### 6.3 Implementation challenge

The core computational challenge is that standard deep learning libraries (PyTorch, TensorFlow) do not natively support banded matrix multiplication. The operation $QK^\top$ produces a dense $n \times n$ matrix, and there is no built-in way to say "only compute the band."

The paper describes three implementation strategies with different trade-offs:

**Longformer-loop:** Compute each diagonal of the banded matrix separately in a loop. This is memory-efficient (only stores nonzero values) but extremely slow because loop iterations cannot be parallelized on GPUs.

**Longformer-chunks:** Split $Q$ and $K$ into overlapping blocks of size $w$ with overlap $w/2$. Multiply each block pair using standard dense matrix multiplication, then mask out the entries outside the band. This is fast (uses one large batched matrix multiply) but uses $2\times$ the memory of a perfect implementation because some entries are computed and then discarded.

**Longformer-cuda:** A custom CUDA kernel implemented using TVM (Chen et al., 2018) that computes exactly the banded entries. This is both fast and memory-efficient, and also supports dilation. The paper uses this implementation for the autoregressive language modeling experiments.

### 6.4 Memory scaling

The key advantage of all three implementations over full attention is memory. Full attention materializes an $n \times n$ matrix, consuming $O(n^2)$ memory. The sliding window implementations store at most $O(n \times w)$ values. Since $w$ is fixed:

$$
\text{Memory}_\text{window} = O(n)
$$

This is the result shown in Figure 1 of the paper: Longformer's memory scales linearly, while full self-attention's memory grows quadratically and exceeds GPU capacity around $n = 8{,}000$ tokens on a single GPU.

---

## 7. Autoregressive Language Modeling

### 7.1 Attention pattern for autoregressive models

For autoregressive (left-to-right) language modeling, the attention must be causal: position $i$ can only attend to positions $j \leq i$. The sliding window becomes one-sided:

$$
S_i^\text{causal} = \{j : \max(0, i - w) \leq j \leq i\}
$$

The paper uses dilated sliding window attention for the language modeling experiments, with varying dilation and window sizes across layers.

### 7.2 Staged training procedure

The paper adopts a staged training procedure where the window size and sequence length grow together across 5 phases:

| Phase | Sequence length | Window size | Learning rate |
|---|---|---|---|
| 1 | 2,048 | 32 → varies | 2.5e-4 |
| 2 | 4,096 | 64 → varies | 1.25e-4 |
| 3 | 8,192 | 128 → varies | 6.25e-5 |
| 4 | 16,384 | 256 → varies | 3.125e-5 |
| 5 | 23,040 | 512 → varies | 1.5625e-5 |

In each phase, the sequence length doubles (approximately) and the learning rate halves. The window size also increases, with the bottom layer starting small and the top layer receiving the largest window. This progressive schedule is motivated by the observation that the model needs many gradient updates to learn local context before it can benefit from longer context.

### 7.3 Results on character-level language modeling

The paper evaluates on text8 and enwik8, both standard benchmarks containing 100M characters from Wikipedia.

**Small models** ($\sim$41M parameters, 12 layers):

| Model | text8 BPC | enwik8 BPC |
|---|---|---|
| T12 (Al-Rfou et al., 2018) | 1.18 | 1.11 |
| Adaptive Span (2019) | 1.11 | 1.02 |
| BP-Transformer (2019) | 1.11 | 1.02 |
| **Longformer** | **1.10** | **1.00** |

Longformer achieves new state-of-the-art on both datasets with the small model configuration: **1.10 BPC on text8** and **1.00 BPC on enwik8**.

**Large models** ($\sim$102M parameters, 30 layers):

| Model | #Param | enwik8 BPC |
|---|---|---|
| Transformer-XL (18 layers) | 88M | 1.03 |
| Sparse Transformer | $\approx$100M | 0.99 |
| Transformer-XL (24 layers) | 277M | 0.99 |
| Adaptive Span | 209M | 0.98 |
| Compressive Transformer | 277M | 0.97 |
| **Longformer** | **102M** | **0.99** |

The large Longformer matches the Sparse Transformer's 0.99 BPC and outperforms the comparable Transformer-XL (18 layers, 88M parameters) at 1.03 BPC. Models that achieve 0.97–0.98 BPC use more than twice the parameters.

### 7.4 Ablation: window arrangement matters

Table 4 of the paper shows that the arrangement of window sizes across layers is significant:

| Configuration | Dev BPC |
|---|---|
| Decreasing $w$ (512 → 32, top to bottom) | 1.24 |
| Fixed $w = 230$ (average) | 1.23 |
| **Increasing $w$ (32 → 512, bottom to top)** | **1.21** |

And adding dilation on 2 heads further improves to **1.20 BPC**. The principle is clear: lower layers need local detail, upper layers need broader context.

---

## 8. From Language Model to Pretrained Encoder

### 8.1 The pretrain-finetune gap

Every prior efficient attention method (Sparse Transformer, Adaptive Span, Compressive Transformer) was evaluated primarily on autoregressive language modeling. But the dominant paradigm in NLP is pretrain-finetune: pretrain a bidirectional model (like BERT or RoBERTa) with masked language modeling (MLM), then finetune on downstream tasks. None of the prior methods addressed this paradigm.

Longformer bridges this gap. The paper shows that you can take a pretrained RoBERTa checkpoint, replace its full self-attention with Longformer's sliding window attention, and continue pretraining with MLM — all without changing the model architecture beyond the attention pattern.

### 8.2 The position embedding problem

RoBERTa uses learned absolute position embeddings with a maximum position of 512. To support longer documents (up to 4,096 tokens), new position embeddings are needed for positions 513–4,095.

The paper's solution is simple and effective: copy the first 512 position embeddings repeatedly to fill positions 0–4,095. Specifically, position $i$ is initialized with the embedding from position $i \bmod 512$.

### 8.3 Why copying works

This initialization works because BERT's attention heads exhibit strong local patterns — most attention weight falls on the previous token, the next token, or the token itself. The position embedding encodes relative position within a period of 512. Since the sliding window is much smaller than 512 ($w = 512$ in the pretrained model), local attention patterns are perfectly preserved by the copied embeddings. The only artifacts appear at the period boundaries (positions 511, 512, 1023, 1024, etc.), where the copied pattern breaks.

Table 5 of the paper confirms this. Starting from the RoBERTa-base BPC of 1.846:

| Configuration | MLM BPC |
|---|---|
| Random position embeddings | 10.299 |
| Copied position embeddings (no training) | 1.957 |
| Copied + 2K gradient updates | 1.753 |
| Copied + 65K gradient updates | 1.705 |

Random initialization destroys the model (10.299 BPC — essentially random). Copied embeddings start at 1.957, close to the RoBERTa baseline of 1.846, and improve with continued pretraining. After 65K gradient updates, the model reaches 1.705, substantially better than the short-context RoBERTa baseline.

### 8.4 Frozen-weight experiment

The paper also tries freezing all RoBERTa weights and only training the new position embeddings. This achieves 1.850 BPC — almost exactly matching the RoBERTa baseline of 1.846. This confirms two things: the sliding window attention is fully compatible with the pretrained weights, and the remaining gap (1.850 vs 1.705) comes from the model learning to use longer context, not from fixing broken short-context behavior.

---

## 9. Downstream Tasks

### 9.1 Experimental setup

The paper evaluates Longformer on six downstream tasks spanning question answering, coreference resolution, and document classification:

| Task | Dataset | Avg. length | 95th percentile |
|---|---|---|---|
| QA | WikiHop | 1,535 | 3,627 |
| QA | TriviaQA | 6,589 | 17,126 |
| QA | HotpotQA | 1,316 | 1,889 |
| Coreference | OntoNotes | 506 | 1,147 |
| Classification | IMDB | 300 | 705 |
| Classification | Hyperpartisan | 705 | 1,975 |

All of these except IMDB have contexts that frequently exceed BERT's 512-token limit. The baseline is RoBERTa-base, which must either truncate or chunk long documents.

### 9.2 Longformer-base results

| Task | RoBERTa-base | Longformer-base | $\Delta$ |
|---|---|---|---|
| WikiHop (F1) | 72.4 | **75.0** | +2.6 |
| TriviaQA (F1) | 74.3 | **75.2** | +0.9 |
| HotpotQA (joint F1) | 63.5 | **64.4** | +0.9 |
| OntoNotes (avg F1) | 78.4 | **78.6** | +0.2 |
| IMDB (accuracy) | 95.3 | **95.7** | +0.4 |
| Hyperpartisan (F1) | 87.4 | **94.8** | +7.4 |

Longformer consistently outperforms RoBERTa-base on every task. The gains are largest on tasks with the longest contexts: WikiHop (+2.6), Hyperpartisan (+7.4, which has relatively long documents at 705 average length), and TriviaQA (+0.9, with 95th percentile at 17K tokens). The gains are smallest on OntoNotes (+0.2) and IMDB (+0.4), where most documents fit within 512 tokens.

### 9.3 Longformer-large results

On the QA tasks where long context matters most:

| Task | Previous SOTA | Longformer-large |
|---|---|---|
| WikiHop (F1) | 78.3 | **81.9** |
| TriviaQA (F1) | 73.3 | **77.3** |
| HotpotQA (joint F1) | **74.2** | 73.2 |

Longformer-large sets new state-of-the-art on WikiHop (+3.6 points) and TriviaQA (+4.0 points). On HotpotQA, it places second; the models that outperform it use graph neural networks, which encode an inductive bias specific to multi-hop reasoning.

### 9.4 WikiHop ablation: what matters

Table 10 of the paper provides a detailed ablation on WikiHop that isolates each component's contribution:

| Configuration | Accuracy | $\Delta$ |
|---|---|---|
| Full Longformer (seqlen 4,096) | **73.8** | — |
| RoBERTa-base (seqlen 512) | 72.4 | -1.4 |
| Longformer, seqlen 512, $n^2$ attention | 71.7 | -2.1 |
| Longformer, seqlen 2,048 | 73.1 | -0.7 |
| No MLM pretraining | 73.2 | -0.6 |
| No separate global projections | 72.2 | -1.6 |
| No global attention at all | 65.5 | **-8.3** |

The most important finding: removing global attention entirely causes a catastrophic 8.3-point drop. This confirms that global attention is not optional — it is essential for tasks that require comparing distant parts of the document. The local sliding window builds contextual representations, but global attention is what allows the model to reason across the full sequence.

The second finding: the separate linear projections for global attention ($Q_g$, $K_g$, $V_g$) matter. Removing them costs 1.6 points, confirming that global and local attention benefit from specialized projection matrices.

---

## 10. Longformer-Encoder-Decoder (LED)

### 10.1 Extending to sequence-to-sequence

The paper introduces LED, a variant that applies Longformer's efficient attention to the encoder of an encoder-decoder Transformer. The encoder uses local+global attention (linear in input length), while the decoder uses full cross-attention to the encoded sequence and full self-attention over previously decoded tokens.

LED is initialized from BART (Lewis et al., 2020) — the same approach used for the encoder-only model with RoBERTa. Position embeddings are extended to 16K tokens by repeatedly copying BART's 1K position embeddings.

### 10.2 Results on arXiv summarization

The arXiv summarization dataset (Cohan et al., 2018) contains scientific papers with long inputs (90th percentile: 14.5K tokens), making it an ideal test for LED.

| Model | Seqlen | R-1 | R-2 | R-L |
|---|---|---|---|---|
| Discourse-aware (2018) | — | 35.80 | 11.05 | 31.80 |
| Pegasus (2020) | — | 44.21 | 16.95 | 38.83 |
| BigBird (2020) | 4,096 | 46.63 | 19.02 | 41.77 |
| **LED-large** | **4,096** | **44.40** | **17.94** | **39.76** |
| **LED-large** | **16,384** | **46.63** | **19.62** | **41.83** |

At 16K tokens, LED-large achieves ROUGE scores that slightly outperform BigBird (which uses 4K tokens) — despite LED having no task-specific pretraining. It is initialized from BART with no additional pretraining, demonstrating that the efficient attention pattern alone is sufficient to process long documents effectively.

Figure 3 of the paper shows that ROUGE scores improve monotonically as the input length increases from 1K to 16K tokens, confirming that the model genuinely benefits from seeing more of the document.

---

## 11. The Unified View: Three Axis 3 Approaches

We have now seen three approaches to sparse attention, all living on Axis 3 of the taxonomy from the earlier blog. Let us place them in a single framework.

### 11.1 The common abstraction

Every approach defines a connectivity set $S_i$ for each position $i$. The attention output is identical in all cases:

$$
\text{output}_i = \sum_{j \in S_i} \alpha_{ij} \, v_j, \qquad \alpha_{ij} = \frac{\exp(q_i k_j^\top / \sqrt{d_k})}{\sum_{j' \in S_i} \exp(q_i k_{j'}^\top / \sqrt{d_k})}
$$

The only difference is the rule for constructing $S_i$.

### 11.2 Comparison table

| Property | Full attention | Sparse Transformer | Longformer |
|---|---|---|---|
| $\|S_i\|$ per token | $n$ | $\sim 2\sqrt{n}$ | $w + g$ |
| Total entries | $n^2$ | $O(n\sqrt{n})$ | $O(n(w+g))$ |
| Scales as | Quadratic | Superlinear | **Linear** |
| Reachability | 1 hop | 2 hops | $L \cdot w / 2$ hops (window) or 1 hop (global) |
| Task adaptation | None needed | None | Global token selection |
| Pretrain compatibility | Native | Train from scratch | **Drop-in replacement** |
| Data assumption | None | Periodic (strided) or none (fixed) | None (local + global) |

### 11.3 Interpretation

These three approaches represent a progression. Full attention is the starting point: maximum flexibility, quadratic cost. The Sparse Transformer was the first to show that structured sparsity can match or beat full attention quality — a surprising result that challenged the assumption that models need dense connectivity. Longformer builds on this insight with a simpler, more practical design: local windows handle most of the work, global tokens handle the rest, and the whole thing drops into existing pretrained models.

The trajectory is clear: from $O(n^2)$ to $O(n\sqrt{n})$ to $O(n)$, with each step trading a small amount of per-layer connectivity for a large reduction in cost. The key realization is that full pairwise connectivity was always overkill — models need local detail and occasional global access, and that combination is achievable at linear cost.

---

## Summary

Longformer replaces full self-attention with a combination of sliding window attention and task-motivated global attention, reducing complexity from $O(n^2)$ to $O(n)$. The sliding window gives each token access to $w$ neighboring positions, and stacking $L$ layers grows the receptive field to $L \times w$ positions — the attention analogue of a deep CNN. Dilated sliding windows further extend this reach by a factor of $d$ without additional cost. Global attention designates a small number of task-specific tokens (like [CLS] for classification or question tokens for QA) that attend to and are attended by every position, providing direct long-range connectivity where the task requires it. Separate projection matrices $Q_g, K_g, V_g$ for global attention let the model specialize its global and local computations. The mechanism drops into existing pretrained models — replacing RoBERTa's full attention with Longformer's windowed attention and copying position embeddings to cover longer sequences — enabling continued pretraining on long documents. The result: state-of-the-art character-level language modeling (1.00 BPC on enwik8), consistent improvements over RoBERTa on document-level NLP tasks (up to +7.4 F1 on Hyperpartisan), new state-of-the-art on WikiHop and TriviaQA, and competitive summarization with LED at 16K-token inputs.

---

*Previous: [Why Full Attention Is Wasteful: Sparse Factorization from Scratch](/blog/attention-sparse-factorization)*
