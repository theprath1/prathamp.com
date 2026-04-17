---
title: "Why Full Attention Is Wasteful: Sparse Factorization from Scratch"
description: "Most learned attention weights are near zero, so why pay for all $n^2$ pairs? Building sparse factorized attention from scratch — strided and fixed patterns that keep every token reachable in two hops while cutting the cost to $O(n\\sqrt{n})$."
date: 2026-04-06
tags: [machine-learning, attention, transformers, sparse-attention, efficiency, sparse-transformers]
order: 2
---

The previous blogs attacked the KV cache — Axis 2 in our taxonomy. GQA reduced the number of KV heads. MLA compressed keys and values into a low-rank latent. Both left the attention pattern itself untouched: every query still attends to every allowed key.

This blog attacks Axis 3. The question is simple: does every token really need to attend to every other token? The Sparse Transformer paper (Child, Gray, Radford, and Sutskever, 2019) answers with an empirical observation and a mathematical construction. The observation: when you train a full-attention model and visualize the learned weights, most of them are near zero. The construction: replace the single dense attention pattern with two sparse patterns that, composed across layers, still let any token reach any other token — but with $O(n\sqrt{n})$ total interactions instead of $O(n^2)$.

We will derive everything from scratch, trace every connectivity set by hand, and verify every count numerically.

---

## The Running Example

We will use a tiny "image" throughout the entire post: a $4 \times 4$ grid of 16 pixels, flattened into a one-dimensional sequence of $n = 16$ tokens in raster order (left to right, top to bottom).

$$
\begin{array}{|c|c|c|c|}
\hline
0 & 1 & 2 & 3 \\
\hline
4 & 5 & 6 & 7 \\
\hline
8 & 9 & 10 & 11 \\
\hline
12 & 13 & 14 & 15 \\
\hline
\end{array}
$$

The positions are zero-indexed: token 0 is the top-left pixel, token 15 is the bottom-right pixel. The model generates this image autoregressively — it predicts each pixel conditioned on all previous pixels.

We set the stride to $l = 4 = \sqrt{16} = \sqrt{n}$. This is the key hyperparameter of the Sparse Transformer: the stride is always chosen close to $\sqrt{n}$, because that is the value that balances the two sparse heads and minimizes total work.

For cost comparisons, we keep the same model from the entire series:

- $d_\text{model} = 512$
- $h = 8$ heads
- $d_k = d_v = 64$
- $L = 12$ layers
- fp16 (2 bytes per element)

---

## 1. The Empirical Observation: Most Attention Is Wasted

Before building any theory, Child et al. did something direct: they trained a standard 128-layer full-attention Transformer on CIFAR-10 images (each image is a sequence of 3,072 bytes) and visualized what the learned attention weights actually look like.

The result is striking. Across most layers, the attention matrices are overwhelmingly sparse. The model learns to concentrate its attention on a small number of positions and effectively ignores the rest. Four distinct patterns emerge:

**Pattern (a): Local structure.** Many early layers learn to attend only to nearby positions — the attention pattern looks like a band around the diagonal. This resembles a convolution: each pixel mostly cares about its immediate neighbors.

**Pattern (b): Row and column structure.** Some layers split attention into two complementary pieces — one attending along the row dimension and the other along the column dimension. The network has independently discovered a factorized version of 2D attention.

**Pattern (c): Global data-dependent attention.** A few layers exhibit global, data-dependent patterns where specific positions attend to far-away positions based on content. These are the layers that genuinely need long-range access.

**Pattern (d): High sparsity.** In deeper layers (layers 64–128), most attention weights are extremely sparse, with positions activating rarely and only for specific input patterns.

### 1.1 What this means

The full attention matrix has $n^2$ entries per head. For CIFAR-10 images with $n = 3{,}072$, that is $3{,}072^2 = 9{,}437{,}184$ entries per head. But the learned patterns show that the vast majority of these entries carry near-zero weight. The model is paying $O(n^2)$ compute and memory to produce an attention matrix that is effectively sparse.

This is the core motivation: if the model learns sparse patterns anyway, we can impose structured sparsity from the start, skip the computation of entries that would have been near zero, and potentially even improve optimization by providing a useful inductive bias.

### 1.2 Numerical check

In our running example with $n = 16$ and causal attention:

$$
\text{Full causal entries per head} = \frac{n(n+1)}{2} = \frac{16 \cdot 17}{2} = 136
$$

We used the **triangular number formula** $\sum_{i=0}^{n-1} (i+1) = \frac{n(n+1)}{2}$. Each of these 136 entries requires a dot product of dimension $d_k = 64$, a softmax contribution, and a value-weighted sum. Most of them, based on the empirical evidence, contribute negligibly to the output. We will now build patterns that compute far fewer entries while preserving the model's ability to route information between any two positions.

---

## 2. Formalizing the Attention Pattern

### 2.1 The connectivity set

The key abstraction is the **connectivity set**. For each output position $i$, the connectivity set $S_i$ is the set of input positions that token $i$ is allowed to attend to. The output at position $i$ is then a weighted sum over only the positions in $S_i$:

$$
\text{Attend}(X, S) = \left(a(\mathbf{x}_i, S_i)\right)_{i \in \{0, \ldots, n-1\}}
$$

where

$$
a(\mathbf{x}_i, S_i) = \text{softmax}\!\left(\frac{(W_q \mathbf{x}_i) K_{S_i}^\top}{\sqrt{d}}\right) V_{S_i}
$$

Here $K_{S_i}$ and $V_{S_i}$ are the key and value matrices formed by stacking only the rows corresponding to positions in $S_i$.

### 2.2 Full causal attention as a connectivity set

In full causal self-attention, the connectivity set is simply all previous positions including the current one:

$$
S_i = \{j : j \leq i\}
$$

For our running example with $n = 16$:

- $S_0 = \{0\}$ — 1 entry
- $S_1 = \{0, 1\}$ — 2 entries
- $S_2 = \{0, 1, 2\}$ — 3 entries
- $\vdots$
- $S_{15} = \{0, 1, 2, \ldots, 15\}$ — 16 entries

Total entries across all positions:

$$
\sum_{i=0}^{15}(i + 1) = 1 + 2 + 3 + \cdots + 16 = \frac{16 \cdot 17}{2} = 136
$$

We used the **triangular number formula** again: $\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$.

### 2.3 The size of the connectivity set determines cost

This is the part that is easy to gloss over but drives everything. The compute cost of attention is not determined by $n$ alone — it is determined by $\sum_{i} |S_i|$, the total number of entries across all connectivity sets. Each entry in $S_i$ requires one dot product between a query and a key (costing $2d_k$ FLOPs), one contribution to the softmax, and one weighted value addition.

For full causal attention, $\sum_i |S_i| = \frac{n(n+1)}{2} \approx \frac{n^2}{2}$. The question is: can we design connectivity sets where $\sum_i |S_i|$ grows much slower than $n^2$, while still allowing information to flow between any two positions?

### 2.4 Interpretation

The connectivity set formulation makes the design space precise. Every attention variant we have seen — dense, local, strided, fixed — is just a different rule for constructing $S_i$. The formula for the attention output is identical in every case. Only the set changes.

---

## 3. Factorized Self-Attention: The Core Idea

### 3.1 The factorization principle

**Factorized self-attention** replaces one dense connectivity set with $p$ separate sparse connectivity sets, one per attention head. The $m$-th head uses its own subset:

$$
A_i^{(m)} \subset \{j : j \leq i\}
$$

and the model uses $S_i = A_i^{(m)}$ in different heads (or different layers). The key constraint is that each individual set $A_i^{(m)}$ is small — specifically, $|A_i^{(m)}| \propto n^{1/p}$ — but their composition across $p$ steps of attention recovers full connectivity.

### 3.2 What "full connectivity through composition" means

This is the part that confuses almost everyone on first reading. A single sparse head does not let token $i$ attend to all previous tokens. So how can the model route information from an arbitrary source position $j$ to an arbitrary target position $i$?

The answer is multi-step routing. If we have $p = 2$ factorized heads applied in alternating layers, then for any pair $(j, i)$ with $j \leq i$, there must exist an intermediate position $k$ such that:

$$
j \in A_k^{(1)} \quad \text{and} \quad k \in A_i^{(2)}
$$

In words: Head 1 in one layer moves information from $j$ to $k$. Head 2 in the next layer moves information from $k$ to $i$. Through two hops, $j$ reaches $i$.

### 3.3 The validity criterion

The paper formalizes this as follows. For every pair $j \leq i$, we require that $i$ can attend to $j$ through a path of length at most $p + 1$:

$$
j \in A_a^{(1)}, \quad a \in A_b^{(2)}, \quad b \in A_c^{(3)}, \quad \ldots, \quad \text{ending at } i
$$

If this criterion holds, then information can propagate from any input position to any output position in a constant number of attention steps — the same number $p$ of factorized heads.

### 3.4 Why $n^{1/p}$ is the magic number

Suppose we use $p = 2$ heads. If each head's connectivity set has size proportional to $\sqrt{n}$, then the total number of entries per head is roughly $n \cdot \sqrt{n} = n^{3/2}$. Across both heads, the total work is $2 n^{3/2} = O(n\sqrt{n})$.

Compare this to full attention at $O(n^2)$. The ratio is:

$$
\frac{n^2}{n^{3/2}} = n^{1/2} = \sqrt{n}
$$

So factorized attention saves a factor of $\sqrt{n}$ in compute. For $n = 16{,}384$, that is a factor of $\sqrt{16{,}384} = 128$.

More generally, with $p$ heads each of size $O(n^{1/p})$, the total cost is $O(p \cdot n^{1+1/p})$. The reduction factor from $n^2$ to $n^{1+1/p}$ is $n^{1-1/p}$. As $p$ grows, this approaches $n$ — but $p = 2$ already captures most of the benefit while keeping the architecture simple.

### 3.5 Numerical check

For our running example with $n = 16$ and $p = 2$:

$$
\sqrt{n} = \sqrt{16} = 4
$$

Each head's connectivity set should have roughly $\sqrt{n} = 4$ entries per position (for positions deep enough in the sequence). The total entries per head should be roughly:

$$
n \cdot \sqrt{n} = 16 \cdot 4 = 64
$$

Compare to full causal: 136. The reduction factor is $\frac{136}{64} \approx 2.1$. At $n = 16$, the savings are modest — sparse attention shines at long sequences. We will compute exact counts for both patterns in the next two sections.

---

## 4. Strided Attention: The First Factorization

The **strided attention** pattern is designed for data with periodic structure — images, where pixels in the same column are separated by exactly one row width, or music, where beats recur at regular intervals.

### 4.1 Head 1: Local window

The first head attends to the $l$ most recent positions (plus itself):

$$
A_i^{(1)} = \{t, t+1, \ldots, i\} \quad \text{where } t = \max(0, i - l)
$$

In our running example with $l = 4$:

| Position $i$ | $t = \max(0, i - 4)$ | $A_i^{(1)}$ | $\|A_i^{(1)}\|$ |
|---|---|---|---|
| 0 | 0 | $\{0\}$ | 1 |
| 1 | 0 | $\{0, 1\}$ | 2 |
| 2 | 0 | $\{0, 1, 2\}$ | 3 |
| 3 | 0 | $\{0, 1, 2, 3\}$ | 4 |
| 4 | 0 | $\{0, 1, 2, 3, 4\}$ | 5 |
| 5 | 1 | $\{1, 2, 3, 4, 5\}$ | 5 |
| 6 | 2 | $\{2, 3, 4, 5, 6\}$ | 5 |
| 7 | 3 | $\{3, 4, 5, 6, 7\}$ | 5 |
| 8 | 4 | $\{4, 5, 6, 7, 8\}$ | 5 |
| 9 | 5 | $\{5, 6, 7, 8, 9\}$ | 5 |
| 10 | 6 | $\{6, 7, 8, 9, 10\}$ | 5 |
| 11 | 7 | $\{7, 8, 9, 10, 11\}$ | 5 |
| 12 | 8 | $\{8, 9, 10, 11, 12\}$ | 5 |
| 13 | 9 | $\{9, 10, 11, 12, 13\}$ | 5 |
| 14 | 10 | $\{10, 11, 12, 13, 14\}$ | 5 |
| 15 | 11 | $\{11, 12, 13, 14, 15\}$ | 5 |

### 4.2 Counting entries for Head 1

The size of each connectivity set is:

$$
|A_i^{(1)}| = \min(i + 1, \, l + 1)
$$

For $i < l$, the size is $i + 1$ (we cannot look back further than the start). For $i \geq l$, the size is $l + 1$.

The total number of entries across all positions is:

$$
\sum_{i=0}^{n-1} |A_i^{(1)}| = \sum_{i=0}^{l-1} (i+1) + \sum_{i=l}^{n-1} (l+1)
$$

The first sum is $\frac{l(l+1)}{2}$ by the **triangular number formula**. The second sum has $(n - l)$ terms, each equal to $(l+1)$, so it equals $(n-l)(l+1)$.

$$
\text{Total}_{\text{Head 1}} = \frac{l(l+1)}{2} + (n - l)(l+1)
$$

Factor out $(l+1)$ by the **distributive law**:

$$
= (l+1)\left[\frac{l}{2} + (n - l)\right]
= (l+1)\left[n - \frac{l}{2}\right]
$$

### 4.3 Numerical check for Head 1

Substitute $l = 4$, $n = 16$:

$$
\text{Total}_{\text{Head 1}} = (4 + 1)\left[16 - \frac{4}{2}\right] = 5 \times 14 = 70
$$

Let us verify by summing the table directly:

$$
1 + 2 + 3 + 4 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 = 10 + 12 \times 5 = 10 + 60 = 70 \checkmark
$$

Both routes give 70.

### 4.4 Interpretation of Head 1

Head 1 gives each token a local view: it can see its immediate neighborhood of $l$ previous positions. For our $4 \times 4$ image, this means each pixel sees the preceding pixels on the same row. Token 7 (row 1, column 3) attends to tokens 3 through 7 — the last pixel of row 0 and the entire current row up to itself. This is essentially a 1D convolution-like receptive field.

But Head 1 alone cannot see beyond $l$ positions. Token 15 has no way to access information from token 0 through Head 1 alone — the local window does not stretch that far. That is why we need a second head.

---

## 5. Strided Attention: Head 2

### 5.1 The strided connectivity set

The second head attends to every $l$-th position (counting backwards from the current position):

$$
A_i^{(2)} = \{j : j \leq i \text{ and } (i - j) \bmod l = 0\}
$$

In words: position $i$ attends to itself, to position $i - l$, to position $i - 2l$, and so on, all the way back to the start.

### 5.2 Tracing Head 2 for our running example

With $l = 4$, $n = 16$:

| Position $i$ | $A_i^{(2)}$ | $\|A_i^{(2)}\|$ |
|---|---|---|
| 0 | $\{0\}$ | 1 |
| 1 | $\{1\}$ | 1 |
| 2 | $\{2\}$ | 1 |
| 3 | $\{3\}$ | 1 |
| 4 | $\{0, 4\}$ | 2 |
| 5 | $\{1, 5\}$ | 2 |
| 6 | $\{2, 6\}$ | 2 |
| 7 | $\{3, 7\}$ | 2 |
| 8 | $\{0, 4, 8\}$ | 3 |
| 9 | $\{1, 5, 9\}$ | 3 |
| 10 | $\{2, 6, 10\}$ | 3 |
| 11 | $\{3, 7, 11\}$ | 3 |
| 12 | $\{0, 4, 8, 12\}$ | 4 |
| 13 | $\{1, 5, 9, 13\}$ | 4 |
| 14 | $\{2, 6, 10, 14\}$ | 4 |
| 15 | $\{3, 7, 11, 15\}$ | 4 |

### 5.3 What Head 2 sees in the image

Look at token 14 (row 3, column 2). Its strided set is $\{2, 6, 10, 14\}$. In the $4 \times 4$ grid, these are positions $(0, 2)$, $(1, 2)$, $(2, 2)$, $(3, 2)$ — the entire column 2. Strided attention with stride $l = 4$ (the row width) naturally attends along columns of the image. This is why strided attention is natural for images: the stride matches the spatial structure.

### 5.4 Counting entries for Head 2

The size of each strided set is:

$$
|A_i^{(2)}| = \left\lfloor \frac{i}{l} \right\rfloor + 1
$$

This is because position $i$ can reach positions $i, i - l, i - 2l, \ldots$ back to the smallest non-negative value in that arithmetic sequence, and there are $\lfloor i/l \rfloor + 1$ such values.

The total entries across all positions:

$$
\sum_{i=0}^{n-1} \left(\left\lfloor \frac{i}{l} \right\rfloor + 1\right)
$$

Group positions by their block $b = \lfloor i/l \rfloor$. For block $b$, there are $l$ positions (positions $bl, bl+1, \ldots, bl + l - 1$), each contributing $b + 1$ entries. With $n/l$ blocks total:

$$
\text{Total}_{\text{Head 2}} = \sum_{b=0}^{n/l - 1} l \cdot (b + 1) = l \sum_{b=0}^{n/l - 1} (b + 1) = l \cdot \frac{(n/l)(n/l + 1)}{2}
$$

We used the **triangular number formula** once more: $\sum_{k=1}^{m} k = \frac{m(m+1)}{2}$ with $m = n/l$.

### 5.5 Numerical check for Head 2

Substitute $l = 4$, $n = 16$, so $n/l = 4$:

$$
\text{Total}_{\text{Head 2}} = 4 \cdot \frac{4 \cdot 5}{2} = 4 \cdot 10 = 40
$$

Verify by summing the table:

$$
1 + 1 + 1 + 1 + 2 + 2 + 2 + 2 + 3 + 3 + 3 + 3 + 4 + 4 + 4 + 4 = 4 + 8 + 12 + 16 = 40 \checkmark
$$

---

## 6. Total Cost of Strided Factorization

### 6.1 Combined entry count

The total number of attention entries computed across both heads is:

$$
\text{Total}_{\text{sparse}} = \text{Total}_{\text{Head 1}} + \text{Total}_{\text{Head 2}}
$$

$$
= (l+1)\!\left(n - \frac{l}{2}\right) + l \cdot \frac{(n/l)(n/l + 1)}{2}
$$

### 6.2 Numerical check

For $l = 4$, $n = 16$:

$$
\text{Total}_{\text{sparse}} = 70 + 40 = 110
$$

Full causal attention has 136 entries. The reduction factor is:

$$
\frac{136}{110} \approx 1.24
$$

At $n = 16$, sparse attention saves only about 19% of the entries. This is not impressive — and that is entirely expected. Sparse attention is designed for long sequences. Let us see what happens as $n$ grows.

### 6.3 Asymptotic analysis

For large $n$ with $l = \sqrt{n}$:

**Head 1:**

$$
(l+1)\!\left(n - \frac{l}{2}\right) \approx l \cdot n = \sqrt{n} \cdot n = n^{3/2}
$$

**Head 2:**

$$
l \cdot \frac{(n/l)(n/l + 1)}{2} \approx l \cdot \frac{(n/l)^2}{2} = \frac{n^2}{2l} = \frac{n^2}{2\sqrt{n}} = \frac{n^{3/2}}{2}
$$

**Total sparse:**

$$
\text{Total}_{\text{sparse}} \approx n^{3/2} + \frac{n^{3/2}}{2} = \frac{3}{2}\,n^{3/2}
$$

**Full causal:**

$$
\text{Total}_{\text{full}} = \frac{n(n+1)}{2} \approx \frac{n^2}{2}
$$

**Reduction factor:**

$$
\frac{\text{Total}_{\text{full}}}{\text{Total}_{\text{sparse}}} \approx \frac{n^2/2}{3n^{3/2}/2} = \frac{n^2}{3\,n^{3/2}} = \frac{\sqrt{n}}{3}
$$

### 6.4 Scaling table

| $n$ | $l = \lfloor\sqrt{n}\rfloor$ | Full causal | Sparse total | Reduction |
|---|---|---|---|---|
| 16 | 4 | 136 | 110 | $1.2\times$ |
| 256 | 16 | 32,896 | 6,392 | $5.1\times$ |
| 1,024 | 32 | 524,800 | 50,160 | $10.5\times$ |
| 4,096 | 64 | 8,390,656 | 397,280 | $21.1\times$ |
| 16,384 | 128 | 134,225,920 | 3,162,048 | $42.4\times$ |

The savings grow as $\sqrt{n}/3$. At $n = 16{,}384$ (the sequence length the paper uses for training dense attention with recomputation), sparse attention computes roughly $42\times$ fewer entries.

### 6.5 Interpretation

The boxed result is:

$$
\boxed{\text{Sparse factorized attention costs } O(n\sqrt{n}) \text{ instead of } O(n^2)}
$$

The reduction factor grows with $\sqrt{n}$, which means sparse attention becomes more and more valuable as sequences get longer. At short sequences there is barely any benefit. At long sequences the savings are enormous. This is exactly the regime the paper targets — images, audio, and text at thousands to tens of thousands of tokens.

---

## 7. The Path-Length Argument: Why Nothing Is Lost

### 7.1 The concern

We just showed that each sparse head computes far fewer entries than full attention. The obvious concern is: have we lost something? Can token 15 still access information from token 0?

In full causal attention, token 15 attends directly to token 0 — the information travels in one hop. In strided sparse attention, token 15's Head 1 only sees $\{11, 12, 13, 14, 15\}$ and Head 2 only sees $\{3, 7, 11, 15\}$. Neither head can reach token 0 directly.

### 7.2 The two-hop path

But consider what happens across multiple layers. In the first layer, Head 2 moves information from token 0 to token 4 (because $0 \in A_4^{(2)} = \{0, 4\}$). In the second layer, Head 1 moves information from token 4 to token 8 (because $4 \in A_8^{(1)} = \{4, 5, 6, 7, 8\}$). And in a third layer, Head 1 moves information from token 8 to token 12 (because $8 \in A_{12}^{(1)} = \{8, 9, 10, 11, 12\}$). Finally, Head 1 moves information from token 12 to token 15 (because $12 \in A_{15}^{(1)} = \{11, 12, 13, 14, 15\}$).

Wait — that was four hops. Can we do better?

### 7.3 The optimal two-hop path

Yes. For any positions $j$ and $i$ with $j < i$, there exists a two-hop path (three positions including the start). Here is how:

**Step 1.** Find an intermediate position $k$ that lies in the same residue class as $i$ modulo $l$ and also falls in the interval $[j, j + l]$. Equivalently, we want $j \in A_k^{(1)}$ (Head 1 can reach $j$ from $k$) and $k \in A_i^{(2)}$ (Head 2 can reach $k$ from $i$).

Let us trace this for $j = 0$ and $i = 15$:

We need an intermediate $k$ such that $0 \in A_k^{(1)}$ and $k \in A_{15}^{(2)}$.

- $A_{15}^{(2)} = \{3, 7, 11, 15\}$. So $k$ must be one of $\{3, 7, 11, 15\}$.
- $A_3^{(1)} = \{0, 1, 2, 3\}$. So $0 \in A_3^{(1)}$. Yes.

The path is: $0 \xrightarrow{\text{Head 1}} 3 \xrightarrow{\text{Head 2}} 15$. Two hops.

### 7.4 Verification: token 0 to token 15 in two hops

**Hop 1 (Head 1, some layer $r$):** Token 3 attends to token 0 because $0 \in A_3^{(1)} = \{0, 1, 2, 3\}$. After this layer, token 3's representation contains information from token 0.

**Hop 2 (Head 2, layer $r + 1$):** Token 15 attends to token 3 because $3 \in A_{15}^{(2)} = \{3, 7, 11, 15\}$. After this layer, token 15's representation contains information from token 3, which already contains information from token 0.

The information has traveled from position 0 to position 15 in exactly two attention steps.

### 7.5 The general existence proof

For any $j < i$, we need to find $k$ such that:

1. $k \in A_i^{(2)}$, meaning $(i - k) \bmod l = 0$, meaning $k \bmod l = i \bmod l$
2. $j \in A_k^{(1)}$, meaning $k - l \leq j \leq k$, meaning $j \leq k \leq j + l$

Condition 1 says $k$ must be congruent to $i$ modulo $l$. Condition 2 says $k$ must be within $l$ of $j$. Since the integers congruent to $i \bmod l$ are spaced exactly $l$ apart, there is always at least one such integer in the interval $[j, j + l]$ — because the interval has length $l$ and the spacing is $l$.

More precisely: the integers congruent to $i \bmod l$ in $[j, j + l]$ are given by $k = l \cdot \lceil (j - (i \bmod l)) / l \rceil + (i \bmod l)$, where $\lceil \cdot \rceil$ is the **ceiling function**. Since the interval $[j, j+l]$ has length exactly $l$ and the congruent integers are spaced $l$ apart, at least one must fall in this interval. This is an application of the **pigeonhole principle**: in any interval of length $l$, there is at least one representative from each residue class modulo $l$.

We also need $k \leq i$ (causality). Since $j < i$ and $k \leq j + l$, this is satisfied as long as $j + l \leq i + l$, which is always true. More carefully, we need $k \leq i$. If $j \leq i - l$, then $k \leq j + l \leq i$, so $k \leq i$ is guaranteed. If $j > i - l$, then $j \in A_i^{(1)}$ directly, and no two-hop path is even needed — token $i$ already sees token $j$ through Head 1.

### 7.6 Interpretation

$$
\boxed{\text{Any token can reach any previous token in at most 2 attention steps}}
$$

Full attention does it in 1 step. Strided factorized attention does it in at most 2 steps. The cost of the extra hop is an extra layer of depth — and since Transformers already stack many layers, this is a mild architectural requirement. In exchange, we reduce per-layer cost from $O(n^2)$ to $O(n\sqrt{n})$.

The trade-off is clean: constant-factor more depth for a polynomial reduction in per-layer cost.

---

## 8. Fixed Attention: The Second Factorization

Strided attention works well when the data has periodic spatial structure (images, music). But for data without a natural grid — text, for instance — the stride does not align with any meaningful structure. A token at position $i$ does not have a special relationship with position $i - l$ just because they happen to be $l$ apart in the sequence.

For such data, the paper proposes **fixed attention**.

### 8.1 Head 1: Block-local attention

Divide the sequence into non-overlapping blocks of $l$ consecutive positions. Head 1 attends only within the current block:

$$
A_i^{(1)} = \left\{j : j \leq i \text{ and } \left\lfloor \frac{j}{l} \right\rfloor = \left\lfloor \frac{i}{l} \right\rfloor \right\}
$$

For our running example with $l = 4$:

- Block 0 (positions 0–3): $A_0^{(1)} = \{0\}$, $A_1^{(1)} = \{0, 1\}$, $A_2^{(1)} = \{0, 1, 2\}$, $A_3^{(1)} = \{0, 1, 2, 3\}$
- Block 1 (positions 4–7): $A_4^{(1)} = \{4\}$, $A_5^{(1)} = \{4, 5\}$, $A_6^{(1)} = \{4, 5, 6\}$, $A_7^{(1)} = \{4, 5, 6, 7\}$
- Block 2 (positions 8–11): same pattern, starting from 8
- Block 3 (positions 12–15): same pattern, starting from 12

Total entries for Head 1:

$$
4 \times \frac{l(l+1)}{2} = \frac{n}{l} \cdot \frac{l(l+1)}{2} = \frac{n(l+1)}{2}
$$

For $n = 16$, $l = 4$: $\frac{16 \times 5}{2} = 40$.

### 8.2 Head 2: Attending to summary positions

Head 2 attends to a fixed set of **summary positions** — the last $c$ positions of every previous block — where $c$ is a small hyperparameter (typically $c \in \{8, 16, 32\}$ for practical models, but we use $c = 1$ for our toy example).

The summary positions within each block are the positions $j$ satisfying $j \bmod l \geq l - c$. With $c = 1$, the summary positions are those where $j \bmod l = l - 1$. In our example with $l = 4$, the summary positions are $\{3, 7, 11, 15\}$ — the last position of each block.

Head 2's connectivity set is all summary positions up to and including position $i$:

$$
A_i^{(2)} = \{j : j \leq i \text{ and } j \bmod l \geq l - c\}
$$

For $c = 1$, $l = 4$:

| Position $i$ | Summary positions $\leq i$ | $\|A_i^{(2)}\|$ |
|---|---|---|
| 0–2 | $\emptyset$ | 0 |
| 3 | $\{3\}$ | 1 |
| 4–6 | $\{3\}$ | 1 |
| 7 | $\{3, 7\}$ | 2 |
| 8–10 | $\{3, 7\}$ | 2 |
| 11 | $\{3, 7, 11\}$ | 3 |
| 12–14 | $\{3, 7, 11\}$ | 3 |
| 15 | $\{3, 7, 11, 15\}$ | 4 |

Total entries for Head 2:

$$
0 + 0 + 0 + 1 + 1 + 1 + 1 + 2 + 2 + 2 + 2 + 3 + 3 + 3 + 3 + 4 = 28
$$

### 8.3 Total cost of fixed attention

$$
\text{Total}_{\text{fixed}} = 40 + 28 = 68
$$

Compare to full causal (136): reduction factor $\frac{136}{68} = 2.0$.

Compare to strided (110): fixed attention computes even fewer entries in this example because the summary positions in Head 2 are more compressed than the strided positions.

### 8.4 How information routes through fixed attention

Token 15 wants information from token 0. Can it reach it?

- Head 1 at some layer: token 3 attends to token 0 (both in block 0, and $0 \in A_3^{(1)} = \{0, 1, 2, 3\}$). Now token 3 carries information from token 0.
- Head 2 at the next layer: token 15 attends to token 3 (because $3$ is a summary position and $3 \in A_{15}^{(2)} = \{3, 7, 11, 15\}$). Now token 15 has information from token 0.

Two hops, same as strided attention.

### 8.5 The role of summary positions

The summary positions act as information bottlenecks — relay stations that aggregate local information from their block and broadcast it globally. Every token within a block can pass its information to the block's summary position through Head 1. Every token in later blocks can read from all previous summary positions through Head 2. This is a two-phase communication pattern: gather locally, then broadcast globally.

### 8.6 Why $c > 1$ helps

The paper notes that $c = 1$ is too restrictive: a single summary position per block creates a severe information bottleneck. With $c = 1$, the entire block's worth of information must be compressed into one position's representation before it can be transmitted to future blocks. In practice, $c \in \{8, 16, 32\}$ works well for typical values of $l \in \{128, 256\}$. Using multiple summary positions per block increases the bandwidth of the relay. The cost increases by a factor of $c$ for Head 2, but since $c \ll l$, this is a modest overhead.

Additionally, the paper found that when using multiple heads, having different heads attend to distinct sub-blocks of size $c$ within the summary region (rather than all attending to the same sub-block) was preferable. This gives each head a different "view" of the summary, increasing diversity.

---

## 9. Strided vs Fixed: When to Use Which

The two factorizations are not interchangeable. They make different structural assumptions.

### 9.1 Strided attention assumes periodic structure

Strided attention with stride $l$ connects positions that are $l$ apart. For images with row width $l$, this means attending along columns — a natural choice because vertical neighbors carry correlated information. For music sampled at a fixed rate, stride attention connects positions separated by one beat period.

But for text, positions $i$ and $i - l$ have no inherent relationship. The word 128 tokens ago is not more relevant than the word 127 tokens ago just because 128 happens to be the stride. The paper found that strided attention "failed to do well" on Enwik8 (a text dataset), while fixed attention "were able to recover and surpass the performance of dense attention."

### 9.2 Fixed attention makes no structural assumption

Fixed attention designates specific positions as relays regardless of content. The relay positions are determined by their location within blocks, not by the data. This is less elegant but more robust: it works for any data type because it does not assume any spatial structure.

### 9.3 Experimental comparison

On CIFAR-10 (images, sequence length 3,072):

| Model | Bits per byte | Time per iteration |
|---|---|---|
| Dense Attention | 2.82 | 0.54 |
| Sparse Transformer (Fixed) | 2.85 | 0.47 |
| Sparse Transformer (Strided) | **2.80** | **0.38** |

Strided attention achieves the lowest error and fastest time — the periodic structure of images aligns perfectly with the stride.

On Enwik8 (text, sequence length 12,288):

| Model | Bits per byte | Time per iteration |
|---|---|---|
| Dense Attention | 1.00 | 1.31 |
| Sparse Transformer (Fixed) | **0.99** | 0.55 |
| Sparse Transformer (Strided) | 1.13 | **0.35** |

Fixed attention matches or beats dense attention on text. Strided attention is faster but loses quality — the stride does not match any structure in natural language.

### 9.4 Interpretation

The results are clear. Sparse patterns are not just faster — on both datasets, at least one sparse pattern also achieves lower error than dense attention. The paper speculates this may point to "a useful inductive bias in the patterns we learned or an underlying optimization issue with full attention." Dense attention gives the model more freedom, but that freedom may make optimization harder.

---

## 10. Three Ways to Use Factorized Heads

The paper describes three strategies for incorporating the factorized patterns into a multi-head attention block.

### 10.1 Interleaved heads

The simplest approach: alternate which pattern each residual block uses. If $r$ is the block index and $p = 2$ is the number of patterns, block $r$ uses pattern $A^{(r \bmod p)}$. So odd layers use Head 1 (local), even layers use Head 2 (strided or fixed), and they alternate throughout the network.

$$
\text{attention}(X) = W_p \cdot \text{attend}(X, A^{(r \bmod p)})
$$

### 10.2 Merged heads

A single head attends to the union of all factorized patterns:

$$
\text{attention}(X) = W_p \cdot \text{attend}\!\left(X, \bigcup_{m=1}^{p} A^{(m)}\right)
$$

This is slightly more computationally intensive because the union is larger than either individual set, but only by a constant factor. The advantage is that a single layer gets both local and global information simultaneously.

### 10.3 Multi-head factorized attention

Use standard multi-head attention, but each head uses one of the factorized patterns. With $n_h$ heads, each head $i$ uses pattern $A^{(i)}$:

$$
\text{attention}(X) = W_p \left(\text{attend}(X, A)^{(i)}\right)_{i \in \{1, \ldots, n_h\}}
$$

The key detail: the weight matrices inside each head are reduced by a factor of $1/n_h$, keeping total parameters invariant across different numbers of heads. This is exactly the same parameter-invariance result we derived in the taxonomy blog — $4d_\text{model}^2$ total attention parameters regardless of head count.

---

## 11. The Sparse Transformer Architecture

The paper does not just change the attention pattern. It also modifies the residual block structure and memory management to enable training very deep networks (up to hundreds of layers) on very long sequences.

### 11.1 Pre-activation residual block

The standard Transformer uses a post-norm residual block. The Sparse Transformer uses a **pre-activation residual block** (following He et al., 2016):

$$
H_0 = \text{embed}(X, W_e)
$$

$$
H_k = H_{k-1} + \text{resblock}(H_{k-1})
$$

$$
y = \text{softmax}(\text{norm}(H_N) W_\text{out})
$$

where the residual block has two sub-layers:

$$
a(H) = \text{dropout}(\text{attention}(\text{norm}(H)))
$$

$$
b(H) = \text{dropout}(\text{ff}(\text{norm}(H + a(H))))
$$

$$
\text{resblock}(H) = a(H) + b(H)
$$

The norm function is **Layer Normalization** (Ba, Kiros, and Hinton, 2016). The feed-forward function is $\text{ff}(x) = W_2 f(W_1 x + b_1) + b_2$, where $f$ is the **Gaussian Error Linear Unit** (GELU, Hendrycks and Gimpel, 2016):

$$
\text{GELU}(x) = x \odot \sigma(1.702 \cdot x)
$$

where $\sigma$ is the sigmoid function and $\odot$ is elementwise multiplication.

### 11.2 Why pre-activation helps

The key property of the pre-activation layout is that $H_N$ is a sum of $N$ residual contributions:

$$
H_N = H_0 + \sum_{k=0}^{N-1} \text{resblock}(H_k)
$$

Each function block receives a gradient directly from the output layer — there is no chain of normalizations or nonlinearities between $H_N$ and any individual $\text{resblock}(H_k)$. This is the same gradient-highway property that made ResNets trainable at hundreds of layers, which we derived in the residual connection blogs earlier in the series.

### 11.3 Initialization for depth

Training a network with hundreds of layers requires careful initialization. The paper scales the weight matrices $W_2$ (in the FFN) and $W_p$ (in the attention output projection) by $\frac{1}{\sqrt{2N}}$, where $N$ is the total number of residual blocks.

The reasoning is as follows. Each residual addition adds a contribution to the running sum $H_k$. If each contribution has variance $\sigma^2$, then after $N$ additions, the total variance is approximately $N\sigma^2$ by the **Bienayme formula** (additivity of variances for independent random variables). To keep the output variance constant regardless of depth, we need $N \sigma^2 = 1$, so $\sigma = 1/\sqrt{N}$. The factor of $\sqrt{2}$ accounts for the fact that each residual block has two sub-layers (attention and FFN), giving $2N$ total contributions:

$$
\boxed{\text{Scale output weights by } \frac{1}{\sqrt{2N}}}
$$

### 11.4 Numerical check

For a 128-layer Sparse Transformer ($N = 128$):

$$
\frac{1}{\sqrt{2 \times 128}} = \frac{1}{\sqrt{256}} = \frac{1}{16} = 0.0625
$$

Each output weight matrix starts with values roughly $16\times$ smaller than standard initialization. Without this scaling, the residual sum would grow as $\sqrt{2N} = 16$ in standard deviation by the 128th layer, causing instability.

---

## 12. Saving Memory: Gradient Checkpointing

### 12.1 The memory problem

In standard backpropagation, all intermediate activations from the forward pass must be stored so they can be reused during the backward pass. For attention, this includes the $n \times n$ score matrix and attention weight matrix for every head in every layer.

We derived in the "Why Vanilla Breaks" blog that these matrices alone cost $4n^2$ bytes per head per layer in fp16. For a 128-layer model at $n = 16{,}384$:

$$
128 \times h \times 4 \times 16{,}384^2 \text{ bytes}
$$

Even with $h = 2$ heads (the CIFAR-10 configuration), this is enormous.

### 12.2 The recomputation solution

**Gradient checkpointing** (also called **activation recomputation**, Chen et al., 2016) trades compute for memory: instead of storing intermediate activations, recompute them during the backward pass. For self-attention, this means we do not store the $n \times n$ attention matrices. During the forward pass, we compute the attention output but discard $S$ and $P$. During the backward pass, when we need $S$ and $P$ for gradient computation, we recompute them from the stored $Q$, $K$, $V$ inputs.

### 12.3 Memory savings

Discarding $S$ and $P$ saves $2n^2$ elements per head per layer (both the score matrix and the attention probability matrix). The cost: we must run the $QK^\top$ computation and softmax twice — once in the forward pass and once in the backward pass. The attention FLOPs roughly double, but the memory drops from $O(n^2)$ per layer to $O(n \cdot d)$ per layer (only $Q$, $K$, $V$ need to be stored).

For our running model at $n = 16{,}384$, $h = 2$, $L = 128$:

**Without recomputation:**

$$
128 \times 2 \times 2 \times 16{,}384^2 \times 2 = 128 \times 2 \times 2 \times 268{,}435{,}456 \times 2
$$

$$
= 274{,}877{,}906{,}944 \text{ bytes} \approx 256 \text{ GB}
$$

This is for the attention matrices alone and far exceeds any single GPU's memory.

**With recomputation:** We only store $Q$, $K$, $V$ per layer. Each has shape $(n, d_k)$, costing $n \cdot d_k \cdot 2$ bytes:

$$
128 \times 2 \times 3 \times 16{,}384 \times 64 \times 2 = 128 \times 2 \times 3 \times 2{,}097{,}152
$$

$$
= 1{,}610{,}612{,}736 \text{ bytes} \approx 1.5 \text{ GB}
$$

The memory reduction is:

$$
\frac{256 \text{ GB}}{1.5 \text{ GB}} \approx 170\times
$$

### 12.4 Why this is particularly effective for attention

Gradient checkpointing is a general technique, but it is disproportionately effective for self-attention layers. The reason is the gap between the activation size ($O(n^2)$ for the attention matrices) and the recomputation cost (re-doing $QK^\top$ and softmax, which is fast on modern GPUs). The attention matrices are the single largest activations in the entire network, but they can be cheaply recomputed from the much smaller $Q$, $K$, $V$ tensors. The paper notes that with recomputation, they "are able to train dense attention networks with hundreds of layers on sequence lengths of 16,384, which would be infeasible on modern hardware otherwise."

---

## 13. Efficient Block-Sparse Kernels

### 13.1 Why naive sparse attention is slow

There is a gap between theoretical FLOP savings and practical speedup. Naive implementations of sparse attention — where we simply skip the masked entries — produce irregular memory access patterns that GPUs handle poorly. GPUs are designed for large, contiguous matrix operations. Scattered reads and writes to random positions in an $n \times n$ matrix thrash the memory hierarchy.

### 13.2 Block-sparse computation

The Sparse Transformer's attention patterns are not arbitrary — they have block structure. The local window in Head 1 corresponds to contiguous blocks of $l$ positions. The strided pattern in Head 2 can be computed by transposing the sequence (regrouping positions by their residue modulo $l$) and then computing a local window. The fixed pattern's summary positions can be aggregated and computed in blocks.

The paper implements custom GPU kernels that:

1. Slice sub-blocks from $Q$, $K$, $V$ corresponding to the connectivity pattern
2. Compute the attention within each block using standard dense matrix operations
3. Fuse the softmax into the same kernel to avoid extra memory reads
4. Use registers to avoid loading input data more than once

The result is that the theoretical $O(n\sqrt{n})$ speedup translates to practical wall-clock improvements, as shown in the experimental timing results.

### 13.3 The upper triangle optimization

In causal attention, the score matrix $S$ is lower-triangular (positions cannot attend to future positions). Standard implementations compute the full matrix and then mask the upper triangle to $-\infty$ before softmax. The Sparse Transformer's kernels never compute the upper triangle at all, which directly halves the number of operations compared to a compute-then-mask approach.

---

## 14. Experimental Results

### 14.1 CIFAR-10: Images

The paper trains strided Sparse Transformers on CIFAR-10 images represented as sequences of 3,072 bytes (32 $\times$ 32 pixels $\times$ 3 channels). The models use 2 heads, 128 layers, $d = 256$, half-size feedforward networks and query-key projections.

The best model achieves **2.80 bits per byte** (equivalently, bits per dim), compared to 2.85 for the previous state-of-the-art (PixelSNAIL, Chen et al., 2017). Strided attention reaches this lower error in the shortest training time, and also surpasses the dense attention baseline of 2.82 bits per byte.

The fact that a sparse model beats a dense model is noteworthy. Dense attention has strictly more representational capacity (it can express any pattern that sparse attention can, plus more). Yet the sparse model trains to a better loss. This suggests that the imposed sparsity acts as a beneficial inductive bias — it constrains the model to learn the kinds of structured patterns (local + columnar) that images actually contain, making optimization easier.

### 14.2 Enwik8: Text

On the Enwik8 dataset (the first $10^8$ bytes of Wikipedia), the paper trains 30-layer fixed Sparse Transformers with 8 heads, $d = 512$, a stride of 128, $c = 32$, and merged factorized attention heads. The context length is 12,288 tokens — substantially longer than the 3,584-token context used by Transformer-XL.

The best model achieves **0.99 bits per byte** ($0.992 \pm 0.001$ over 3 seeds), matching the 0.99 achieved by Transformer-XL 277M (Dai et al., 2018) — a model with more than double the parameters — and surpassing the 1.03 of Transformer-XL 88M.

The paper also evaluates with increasing minimum context lengths during test time and finds monotonic improvement up to 12,160 out of 12,288 tokens. This suggests the model is genuinely incorporating long-range dependencies, not just memorizing local patterns.

### 14.3 ImageNet 64$\times$64: Large-scale images

For ImageNet 64$\times$64 (sequence length 12,288 = 64 $\times$ 64 $\times$ 3), the paper trains a 48-layer strided Sparse Transformer with 16 attention heads and $d = 512$, totaling 152 million parameters. Training takes 7 days on 64 V100 GPUs.

The model achieves **3.44 bits per dim** (3.437 across 1 run), compared to the previous best of 3.52 (Menick and Kalchbrenner, 2018). The generated images show global coherence and long-range structure despite the model operating purely on raw pixels without any multi-scale or hierarchical architecture.

### 14.4 Classical music: Very long sequences

To test the limits of sequence length, the paper trains on classical music encoded as $\mu$-law audio at 12 kHz. At a sequence length of 65,536 (about 5 seconds of audio), a 152M-parameter strided Sparse Transformer achieves **1.97 bits per byte**. The generated samples demonstrate global coherence over the sampled period.

The paper also shows that Sparse Transformers can, in principle, handle sequences of over one million timesteps — though model capacity must shrink to fit within GPU memory. At sequence length 1,048,576, the model has only 3M parameters and achieves 2.99 bits per byte. The quality degrades with reduced model capacity, but the fact that self-attention can scale to million-length sequences at all is a qualitative milestone.

### 14.5 The capacity–length trade-off

A key practical finding: increasing sequence length by a factor of 4 requires reducing model capacity by approximately $4\sqrt{4} = 8$. This comes from the memory scaling — even with sparse attention at $O(n\sqrt{n})$, the activations still grow with $n$, and the model must shrink to fit. Table 4 in the paper makes this concrete:

| Sequence length | Parameters | Bits per byte |
|---|---|---|
| 65,536 | 152M | 1.97 |
| 262,144 | 25M | 2.17 |
| 1,048,576 | 3M | 2.99 |

Each $4\times$ increase in sequence length forces roughly an $8\times$ reduction in model size, and quality degrades accordingly.

---

## 15. Where Sparse Factorization Sits in the Taxonomy

Recall the five-axis taxonomy from the earlier blog:

| Axis | What it controls | What Sparse Transformer changes |
|---|---|---|
| 1. Number of heads | How many independent attention computations | Unchanged (2–16 heads) |
| 2. KV representation | How keys/values are stored and shared | Unchanged (standard per-head KV) |
| **3. Attention pattern** | **Which query-key pairs interact** | **Sparse factorized patterns** |
| 4. Storage and caching | What persists across time | Gradient checkpointing (training) |
| 5. Layer architecture | Block wrapper around attention | Pre-activation residual block |

The Sparse Transformer is primarily an Axis 3 paper. It changes which interactions are computed, replacing the dense $n^2$ pattern with structured sparse patterns of size $O(n\sqrt{n})$. But it also touches Axis 5 (pre-activation residual blocks for deep training) and introduces a training-time memory optimization (gradient checkpointing) that is related to Axis 4.

This is different from GQA and MLA, which are Axis 2 papers — they keep the dense attention pattern but change the KV representation. Sparse attention and KV compression are orthogonal and can be composed: a model could use both GQA (fewer KV heads) and sparse attention (fewer query-key interactions) simultaneously.

---

## Summary

Full attention computes $n^2$ pairwise interactions, but empirical visualization shows that trained models learn sparse patterns — most attention weights are near zero. The Sparse Transformer exploits this by replacing the dense $n \times n$ connectivity with two factorized patterns, each of size $O(\sqrt{n})$ per position, reducing total cost from $O(n^2)$ to $O(n\sqrt{n})$. Strided attention pairs a local window with column-wise access (natural for images and audio), while fixed attention pairs block-local windows with designated summary positions (robust for text). The path-length argument guarantees that any token can still reach any other token in exactly two attention hops, preserving the Transformer's ability to model arbitrary dependencies. Combined with pre-activation residual blocks, $1/\sqrt{2N}$ weight scaling, and gradient checkpointing, these changes enable training on sequences of tens of thousands of tokens with hundreds of layers — achieving state-of-the-art density modeling on images, text, and audio while running significantly faster than full attention.

---

*Previous: [Mathematical Prerequisites for Sparse and Sliding Window Attention](/blog/math-prerequisites-for-sparse-attention)*
*Next: [Sliding Window Attention: From Local Windows to Global Context](/blog/attention-sliding-window)*
