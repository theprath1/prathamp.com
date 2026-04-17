---
title: "Why Replace Attention? The Softmax Bottleneck and the Path to Linear Time"
description: "Why softmax is the bottleneck that every attention variant leaves untouched — and how rewriting attention as a kernel exposes an associativity trick that collapses the O(n²) cost to O(n) and turns the transformer into an RNN."
date: 2026-04-07
tags: [machine-learning, attention, transformers, linear-attention, kernels, efficiency, rnn]
order: 3
---

Every change made to attention so far has touched the periphery — which keys participate, how they are stored, what wraps the block — but left the formula

$$
o_i = \sum_{j} \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i^\top k_{j'} / \sqrt{d_k})} \, v_j
$$

untouched. The softmax stayed.

This blog asks the next question: what if the softmax is the problem? The exponential and the denominator force every query to see every key before producing an output. That is what makes attention quadratic. Drop the softmax and the algebra changes — multiplication becomes associative, and a different grouping turns $O(n^2)$ into $O(n)$ with constant memory per token. The transformer becomes an RNN.

The core paper for this blog is Katharopoulos et al. (2020), "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention," supplemented by the taxonomy from the Efficient Transformers survey (Tay et al., 2022).

---

## The Running Example

We continue with the same model parameters from the series:

- $d_\text{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$, $L = 12$ layers, fp16

For cost analysis, we use $n = 16$ tokens (same sequence from the Sparse Factorization, Sliding Window, and DeepSeek Sparse Attention blogs) and scale up to $n = 128{,}000$ to show how costs grow.

For the kernel and associativity derivations, we need to trace every matrix entry by hand. We use a tiny example:

- $n = 4$ tokens, $d_k = d_v = 2$, single head

with concrete query, key, and value matrices:

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}, \quad K = \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \\ 0 & 2 \end{pmatrix}, \quad V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}
$$

Each row is one token. $Q$ has 4 rows (one per query), $K$ has 4 rows (one per key), $V$ has 4 rows (one per value). All matrices are $4 \times 2$.

---

## 1. The Cost Ledger After Twelve Blogs

### 1.1 What we modified, axis by axis

Every blog in the series made a change to exactly one axis of the taxonomy from the Taxonomy blog. Here is the complete map:

| Blog | Topic | Axis | What changed |
|---|---|---|---|
| 1–3 | Attention basics | — | Setup and motivation |
| 4 | Taxonomy | — | The five-axis framework |
| 5 | Residuals | 5 | Skip connections |
| 6 | MQA / KV Bottleneck | 2 | Shared KV across heads |
| 7 | GQA | 2 | Grouped KV sharing |
| 8 | MLA | 2 | Low-rank KV compression |
| 9 | Sparse Transformer | 3 | Fixed factorized patterns |
| 10 | Sliding Window | 3 | Local windows + global tokens |
| 11 | DeepSeek Sparse | 3 | Learned token selection |
| 12 | Gated Attention | 5 | Gated residuals and SwiGLU FFN |

Axes 2, 3, and 5 have been thoroughly explored. Axis 1 (number of heads) is implicitly handled by GQA. Axis 4 (KV cache and storage) was touched by MLA's compressed latent.

### 1.2 What remains unchanged

Despite all these modifications, every blog preserved two things:

1. **The softmax normalization.** Every variant computes $\exp(q_i^\top k_j / \sqrt{d_k})$ for the query-key pairs that participate, then normalizes by the sum of exponentials. This forces the model to evaluate all participating pairs before producing any output.

2. **The per-query dependence on all keys.** Even in the sparse variants, the attention output for query $i$ depends on the full set of selected keys through the softmax denominator. You cannot compute the output for token $i$ without knowing the scores for all tokens in its connectivity set.

These two properties have a direct consequence: the attention operation cannot be decomposed into independent per-token computations. Every query must "see" its full key set before producing an output. This is the fundamental difference between attention and recurrence.

---

## 2. The Cost at Scale

### 2.1 Training: FLOPs per attention layer

For one head of standard causal attention on $n$ tokens with head dimension $d_k$, the dominant operations are:

**Step 1: Compute $QK^\top$.** This is an $(n \times d_k) \times (d_k \times n)$ matrix multiplication, producing an $n \times n$ matrix. The number of multiply-add operations is:

$$
n \times n \times d_k = n^2 d_k
$$

**Step 2: Apply softmax.** This requires exponentiation and normalization over each row — $O(n)$ per row, $O(n^2)$ total. Dominated by the matrix multiply.

**Step 3: Compute $AV$.** This is an $(n \times n) \times (n \times d_v)$ matrix multiplication, producing an $n \times d_v$ matrix:

$$
n \times d_v \times n = n^2 d_v
$$

With $d_k = d_v$, the total per-head cost is:

$$
\boxed{\text{FLOPs per head} = 2n^2 d_k}
$$

Across $h$ heads:

$$
\text{FLOPs per layer} = h \times 2n^2 d_k = 2n^2 (h \, d_k) = 2n^2 d_\text{model}
$$

The last step uses $h \, d_k = d_\text{model}$, the identity that has appeared throughout this series.

### 2.2 Numerical check

With $n = 16$, $d_\text{model} = 512$:

$$
2 \times 16^2 \times 512 = 2 \times 256 \times 512 = 262{,}144 \text{ FLOPs per layer}
$$

With $n = 128{,}000$:

$$
2 \times 128{,}000^2 \times 512 = 2 \times 1.6384 \times 10^{10} \times 512 = 1.678 \times 10^{13} \text{ FLOPs per layer}
$$

That is 16.78 trillion FLOPs per layer for a single forward pass on 128K tokens. Across $L = 12$ layers:

$$
12 \times 1.678 \times 10^{13} = 2.01 \times 10^{14} \text{ FLOPs}
$$

This is just the attention — the FFN adds a comparable amount ($2 \times d_\text{model} \times d_{ff} \times n = 2 \times 512 \times 2048 \times 128{,}000 \approx 2.68 \times 10^{11}$ per layer, much smaller than the attention cost at this sequence length). At $n = 128{,}000$, attention dominates.

### 2.3 The crossover point

The FFN cost per layer is $2 \, d_\text{model} \, d_{ff} \, n$, which is linear in $n$. The attention cost is $2 \, n^2 \, d_\text{model}$, which is quadratic in $n$. Setting them equal:

$$
2 \, n^2 \, d_\text{model} = 2 \, d_\text{model} \, d_{ff} \, n
$$

The $2 \, d_\text{model}$ cancels from both sides:

$$
n^2 = d_{ff} \, n
$$

Divide both sides by $n$ (valid since $n > 0$):

$$
\boxed{n_\text{crossover} = d_{ff}}
$$

With $d_{ff} = 2048$, the crossover is at $n = 2{,}048$ tokens. For sequences shorter than 2,048 tokens, the FFN costs more than attention. For sequences longer than 2,048 tokens, attention dominates — and it dominates more with every additional token.

### 2.4 Numerical check

At $n = 2{,}048$:

$$
\text{Attention: } 2 \times 2{,}048^2 \times 512 = 2 \times 4{,}194{,}304 \times 512 = 4.295 \times 10^9
$$

$$
\text{FFN: } 2 \times 512 \times 2{,}048 \times 2{,}048 = 4.295 \times 10^9 \quad \checkmark
$$

They match exactly at the crossover point.

At $n = 128{,}000$: attention is $1.678 \times 10^{13}$, FFN is $2.68 \times 10^{11}$. The ratio is $1.678 \times 10^{13} / 2.68 \times 10^{11} \approx 62.6\times$. Attention costs 63 times more than the FFN.

### 2.5 What sparse methods achieve

Our sparse methods from the Sparse Factorization, Sliding Window, and DeepSeek Sparse Attention blogs reduce the attention cost:

| Method | Attention FLOPs per layer | At $n = 128{,}000$ |
|---|---|---|
| Full | $2n^2 d_\text{model}$ | $1.678 \times 10^{13}$ |
| Sparse Transformer ($O(n\sqrt{n})$) | $2n^{3/2} d_\text{model} \cdot c$ | $\sim 4.7 \times 10^{10}$ |
| Sliding window ($w = 512$) | $2nw \, d_\text{model}$ | $\sim 6.71 \times 10^{10}$ |
| DSA ($k = 2{,}048$) | $2nk \, d_\text{model}$ | $\sim 2.68 \times 10^{11}$ |

These are enormous improvements. But notice: even the cheapest method (Sparse Transformer) still has a cost that grows super-linearly with $n$. And all three methods still require the softmax over the selected pairs — the $O(|S_t|)$ normalization per query token. The cost is reduced but the mechanism is the same.

### 2.6 Interpretation

Sparse methods do not change the nature of the computation. They reduce how many pairs participate, but each participating pair still requires the full softmax pipeline: exponentiate, sum, normalize, multiply by values. The question this blog asks is: can we change the mechanism itself to eliminate the need for pairwise computation entirely?

---

## 3. The KV Cache Wall

### 3.1 Memory during autoregressive generation

During training, we pay the quadratic cost once for the full sequence. During inference (autoregressive generation), the cost is paid incrementally: each new token computes attention against all previous tokens.

For token $t$ (the $t$-th generated token), the attention computation requires loading all previous key and value vectors from the **KV cache**. Each token stores:

$$
\text{Bytes per token} = 2 \times h \times d_k \times 2 = 4 h \, d_k = 4 \, d_\text{model} \text{ bytes}
$$

The factor of 2 at the front accounts for both keys and values. The factor of 2 at the end is for fp16 (2 bytes per element). We simplify using $h \, d_k = d_\text{model}$.

Across $L$ layers:

$$
\text{Bytes per token, all layers} = 4 \, d_\text{model} \, L
$$

### 3.2 Numerical check

With $d_\text{model} = 512$ and $L = 12$:

$$
4 \times 512 \times 12 = 24{,}576 \text{ bytes} = 24 \text{ KB per token}
$$

After generating $n$ tokens, the KV cache occupies:

$$
\text{Total KV cache} = 4 \, d_\text{model} \, L \, n
$$

At $n = 128{,}000$:

$$
4 \times 512 \times 12 \times 128{,}000 = 3.15 \times 10^9 \text{ bytes} \approx 3.15 \text{ GB}
$$

For a model with $d_\text{model} = 512$, this is manageable. But real models are much larger. For a model with $d_\text{model} = 8{,}192$ (comparable to GPT-4 class models) and $L = 80$ layers:

$$
4 \times 8{,}192 \times 80 \times 128{,}000 = 3.36 \times 10^{11} \text{ bytes} \approx 336 \text{ GB}
$$

That is 336 GB just for the KV cache — more than the memory of most GPUs. This is the **KV cache wall**: the point where the memory required to store past keys and values exceeds available GPU memory.

### 3.3 The per-token generation cost

When generating token $t$, the attention computation for one head involves:

1. Compute $q_t^\top k_j$ for all $j \in \{1, \ldots, t\}$: $t \times d_k$ multiply-adds
2. Apply softmax over $t$ scores: $O(t)$ operations
3. Compute weighted sum $\sum_j \alpha_j v_j$: $t \times d_v$ multiply-adds

Total per head: $2t \, d_k$. Across all heads and layers:

$$
\text{FLOPs for token } t = 2t \, d_k \times h \times L = 2t \, d_\text{model} \, L
$$

The cost to generate token $t$ grows linearly with $t$ — each new token is slower than the last.

### 3.4 Numerical check

Generating the 128,000th token:

$$
2 \times 128{,}000 \times 512 \times 12 = 1.57 \times 10^9 \text{ FLOPs}
$$

Generating the 1st token:

$$
2 \times 1 \times 512 \times 12 = 12{,}288 \text{ FLOPs}
$$

The last token costs $128{,}000\times$ more than the first. In wall-clock time, this means generation slows down as the sequence gets longer — each successive token takes more time to produce.

### 3.5 The total generation cost

To generate all $n$ tokens:

$$
\text{Total FLOPs} = \sum_{t=1}^{n} 2t \, d_\text{model} \, L = 2 \, d_\text{model} \, L \sum_{t=1}^{n} t = 2 \, d_\text{model} \, L \cdot \frac{n(n+1)}{2} = d_\text{model} \, L \, n(n+1)
$$

The sum $\sum_{t=1}^n t = \frac{n(n+1)}{2}$ is **Gauss's summation formula**. For large $n$:

$$
\boxed{\text{Total generation FLOPs} \approx d_\text{model} \, L \, n^2}
$$

Quadratic in $n$, again.

### 3.6 What an RNN gives you

Contrast this with a recurrent neural network. An RNN maintains a fixed-size hidden state $h_t \in \mathbb{R}^{d}$ and updates it at each step:

$$
h_t = f(h_{t-1}, x_t)
$$

The cost to generate each token is constant — $O(d^2)$ for the matrix-vector multiplication in $f$, regardless of position $t$. The total cost for $n$ tokens is $O(n d^2)$ — linear in $n$. The memory is $O(d)$ — constant, regardless of how many tokens have been generated.

| Property | Attention | RNN |
|---|---|---|
| State size at step $t$ | $O(t \cdot d)$ — grows | $O(d^2)$ — constant |
| Cost per token at step $t$ | $O(t \cdot d)$ — grows | $O(d^2)$ — constant |
| Total cost for $n$ tokens | $O(n^2 d)$ — quadratic | $O(n d^2)$ — linear |
| Can access token 1 from token $n$? | Yes, directly | Only through state |

The last row is the tradeoff. Attention pays a growing cost because it maintains direct access to every past token. An RNN pays a constant cost because it compresses all past information into a fixed-size state — but that compression is lossy.

The question is: can we get the best of both worlds? Linear cost like an RNN, but with the expressivity of attention?

The answer begins with examining why the softmax makes the cost quadratic.

---

## 4. The Softmax Bottleneck

### 4.1 The attention output for one query

Let us write the attention output for a single query token $i$ in full generality. For a single head:

$$
o_i = \frac{\sum_{j=1}^{n} \exp\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) v_j}{\sum_{j=1}^{n} \exp\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)}
$$

This is a weighted average of the value vectors $v_j$, where the weight on $v_j$ is proportional to $\exp(q_i^\top k_j / \sqrt{d_k})$.

Let us define $\text{sim}(q, k)$ as the **similarity function** between a query and a key:

$$
\text{sim}(q, k) = \exp\!\left(\frac{q^\top k}{\sqrt{d_k}}\right)
$$

Then the attention output becomes:

$$
\boxed{o_i = \frac{\sum_{j=1}^{n} \text{sim}(q_i, k_j) \, v_j}{\sum_{j=1}^{n} \text{sim}(q_i, k_j)}}
$$

This is the form we will work with for the rest of this blog. The specific choice of similarity function determines everything.

### 4.2 Why softmax forces pairwise computation

This is the part that is easy to gloss over but is the crux of the entire blog.

With the exponential similarity $\text{sim}(q, k) = \exp(q^\top k / \sqrt{d_k})$, the weight for the pair $(i, j)$ depends on the specific combination of $q_i$ and $k_j$. The exponential of a dot product cannot be factored:

$$
\exp(q_i^\top k_j / \sqrt{d_k}) \neq f(q_i) \cdot g(k_j)
$$

for any scalar functions $f$ and $g$. The reason is that the dot product $q_i^\top k_j = \sum_m q_{i,m} \, k_{j,m}$ mixes the components of $q_i$ and $k_j$ inside the exponential, and $\exp(a + b) = \exp(a) \exp(b)$ only factors when the argument is a sum — but the sum is over the $d_k$ dimensions, not over the queries and keys.

To be precise: $\exp(q_i^\top k_j / \sqrt{d_k}) = \exp\!\left(\sum_m q_{i,m} k_{j,m} / \sqrt{d_k}\right) = \prod_m \exp(q_{i,m} k_{j,m} / \sqrt{d_k})$. Each factor in the product involves both $q_{i,m}$ and $k_{j,m}$ multiplicatively inside the exponential. There is no way to separate this into "a function of $q_i$ only" times "a function of $k_j$ only" using a finite number of terms.

This means we must compute $\text{sim}(q_i, k_j)$ for every pair $(i, j)$ separately. There are $n^2$ such pairs. No algebraic rearrangement can avoid this.

### 4.3 Numerical illustration

Let us verify this with our running example. Take query $q_1 = (1, 0)$ and compute its similarity with all 4 keys (using $\sqrt{d_k} = \sqrt{2}$):

$$
\text{sim}(q_1, k_1) = \exp\!\left(\frac{(1)(0) + (0)(1)}{\sqrt{2}}\right) = \exp(0) = 1.000
$$

$$
\text{sim}(q_1, k_2) = \exp\!\left(\frac{(1)(1) + (0)(0)}{\sqrt{2}}\right) = \exp(0.707) = 2.028
$$

$$
\text{sim}(q_1, k_3) = \exp\!\left(\frac{(1)(1) + (0)(1)}{\sqrt{2}}\right) = \exp(0.707) = 2.028
$$

$$
\text{sim}(q_1, k_4) = \exp\!\left(\frac{(1)(0) + (0)(2)}{\sqrt{2}}\right) = \exp(0) = 1.000
$$

Each similarity score is different and depends on both $q_1$ and the specific key. We had to compute 4 exponentials — one per key. For all 4 queries, we would need $4 \times 4 = 16$ exponentials. In general: $n^2$.

### 4.4 The denominator forces full evaluation

Even if we only cared about one value in the output $o_i$, we would still need all $n$ similarity scores for query $i$ because of the denominator $\sum_{j=1}^n \text{sim}(q_i, k_j)$. The denominator is a sum over all keys — you cannot know the correct normalization without evaluating every key.

This is the fundamental constraint. The softmax denominator couples all keys together for each query. It prevents streaming or incremental computation of the output.

### 4.5 What if we could factor the similarity?

Suppose instead we had a similarity function that could be written as:

$$
\text{sim}(q, k) = \phi(q)^\top \phi(k)
$$

for some **feature map** $\phi : \mathbb{R}^{d_k} \to \mathbb{R}^{d_\phi}$ that maps each query or key independently into a new space of dimension $d_\phi$. Then:

$$
o_i = \frac{\sum_{j=1}^{n} \phi(q_i)^\top \phi(k_j) \, v_j}{\sum_{j=1}^{n} \phi(q_i)^\top \phi(k_j)}
$$

Now $\phi(q_i)^\top \phi(k_j)$ is a scalar (a dot product in the feature space), so $\phi(q_i)^\top \phi(k_j) \, v_j$ is a scalar times a vector. The sum $\sum_j \phi(q_i)^\top \phi(k_j) \, v_j$ involves $\phi(q_i)$ interacting with the $\phi(k_j)$ vectors, and crucially, we can rearrange this sum. This rearrangement is the key to everything that follows.

---

## 5. Attention as a Kernel Function

### 5.1 The kernel interpretation

The formulation $\text{sim}(q, k) = \phi(q)^\top \phi(k)$ is a **kernel function** from the theory of reproducing kernel Hilbert spaces. A **kernel** $\kappa(x, y)$ is any function that can be written as an inner product in some (possibly high-dimensional) feature space:

$$
\kappa(x, y) = \langle \phi(x), \phi(y) \rangle
$$

The function $\phi$ is called the **feature map**. It transforms inputs from the original space into a feature space where inner products correspond to similarities.

This is the connection to kernel methods in machine learning — the same mathematical framework behind support vector machines and Gaussian processes. The survey by Tay et al. (2022) categorizes Linear Transformers (Katharopoulos et al., 2020) and Performers (Choromanski et al., 2020) under the "Low Rank / Kernels" class of efficient transformers, precisely because they exploit this kernel structure.

### 5.2 Can softmax attention be written as a kernel?

The softmax similarity $\text{sim}(q, k) = \exp(q^\top k / \sqrt{d_k})$ is indeed a valid kernel. It can be written as an inner product in a feature space — but that feature space is infinite-dimensional. By the **Taylor expansion of the exponential function**:

$$
\exp(q^\top k / \sqrt{d_k}) = \sum_{m=0}^{\infty} \frac{(q^\top k / \sqrt{d_k})^m}{m!}
$$

Each term $(q^\top k)^m$ can be expanded as a sum of products of components of $q$ and $k$, which corresponds to a feature map that includes all monomials of degree $m$ in the components of $q$ (and similarly for $k$). The full feature map includes monomials of all degrees — an infinite-dimensional vector.

So the softmax kernel has a feature map $\phi$, but $\phi(q)$ is an infinite-dimensional vector. We cannot compute $\phi(q)^\top \phi(k)$ by first computing $\phi(q)$ and $\phi(k)$ separately — the vectors have infinitely many entries. We are forced to compute the kernel value $\exp(q^\top k / \sqrt{d_k})$ directly, which brings us back to pairwise computation.

### 5.3 The key idea: use a different kernel

Katharopoulos et al. (2020) propose a simple solution: replace the softmax kernel with a kernel that has a finite-dimensional feature map. Instead of $\text{sim}(q, k) = \exp(q^\top k / \sqrt{d_k})$, use:

$$
\text{sim}(q, k) = \phi(q)^\top \phi(k)
$$

where $\phi : \mathbb{R}^{d_k} \to \mathbb{R}^{d_\phi}$ is a finite-dimensional feature map. Their specific choice is:

$$
\boxed{\phi(x) = \text{elu}(x) + 1}
$$

where **elu** is the **exponential linear unit** (Clevert et al., 2015):

$$
\text{elu}(x) = \begin{cases} x & \text{if } x > 0 \\ e^x - 1 & \text{if } x \leq 0 \end{cases}
$$

Applied element-wise to a vector $x \in \mathbb{R}^{d_k}$, this gives $\phi(x) \in \mathbb{R}^{d_k}$ — the feature map has the same dimension as the input ($d_\phi = d_k$). The $+1$ ensures that all components of $\phi(x)$ are non-negative, which guarantees that the similarity $\phi(q)^\top \phi(k) \geq 0$ — a necessary property since attention weights must be non-negative.

### 5.4 Why non-negativity matters

In softmax attention, $\exp(q^\top k / \sqrt{d_k}) > 0$ always — the exponential function is strictly positive. This guarantees positive attention weights, which means the output is a proper weighted average of the value vectors.

If we replace the similarity with $\phi(q)^\top \phi(k)$ and $\phi$ maps to non-negative outputs, then the dot product of two non-negative vectors is non-negative: $\phi(q)^\top \phi(k) = \sum_m \phi(q)_m \phi(k)_m \geq 0$ since every term is a product of non-negative numbers. This preserves the weighted-average interpretation.

### 5.5 Numerical example: computing the feature map

Let us apply $\phi(x) = \text{elu}(x) + 1$ to the queries and keys from our running example.

**Queries:**

$$
\phi(q_1) = \phi\!\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} \text{elu}(1) + 1 \\ \text{elu}(0) + 1 \end{pmatrix} = \begin{pmatrix} 1 + 1 \\ 0 + 1 \end{pmatrix} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}
$$

For $x > 0$, $\text{elu}(x) = x$, so $\phi(x) = x + 1$. For $x = 0$, $\text{elu}(0) = 0$, so $\phi(0) = 0 + 1 = 1$.

$$
\phi(q_2) = \phi\!\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
$$

$$
\phi(q_3) = \phi\!\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 2 \\ 2 \end{pmatrix}
$$

$$
\phi(q_4) = \phi\!\begin{pmatrix} 2 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}
$$

**Keys:**

$$
\phi(k_1) = \phi\!\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
$$

$$
\phi(k_2) = \phi\!\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}
$$

$$
\phi(k_3) = \phi\!\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 2 \\ 2 \end{pmatrix}
$$

$$
\phi(k_4) = \phi\!\begin{pmatrix} 0 \\ 2 \end{pmatrix} = \begin{pmatrix} 1 \\ 3 \end{pmatrix}
$$

### 5.6 Numerical check: kernel similarity vs dot product

Let us verify that $\phi(q_i)^\top \phi(k_j)$ gives a reasonable similarity measure. Compare with the softmax similarity from Section 4.3 for $q_1$:

| Pair | $\phi(q_1)^\top \phi(k_j)$ | $\exp(q_1^\top k_j / \sqrt{2})$ |
|---|---|---|
| $(1, 1)$ | $(2)(1) + (1)(2) = 4$ | $1.000$ |
| $(1, 2)$ | $(2)(2) + (1)(1) = 5$ | $2.028$ |
| $(1, 3)$ | $(2)(2) + (1)(2) = 6$ | $2.028$ |
| $(1, 4)$ | $(2)(1) + (1)(3) = 5$ | $1.000$ |

The rankings differ: the kernel similarity ranks key 3 highest (score 6) while softmax gives keys 2 and 3 equal scores (both 2.028). The two similarity functions are not identical — they define different attention distributions. The question is whether the kernel version, despite being different, can still produce useful representations. Katharopoulos et al. (2020) show empirically that it can, with competitive performance on language modeling and speech recognition tasks.

### 5.7 The generalized attention formula

With the kernel similarity, the attention output for query $i$ becomes:

$$
o_i = \frac{\sum_{j=1}^{n} \phi(q_i)^\top \phi(k_j) \, v_j}{\sum_{j=1}^{n} \phi(q_i)^\top \phi(k_j)}
$$

This looks identical to the softmax version, just with a different similarity function. The cost appears to be the same: $n^2$ dot products in the feature space. But there is a crucial algebraic difference that we have not yet exploited.

---

## 6. The Associativity Trick

### 6.1 The rearrangement

This is the core mathematical insight of the entire blog. It is a single algebraic step, but it changes the complexity from quadratic to linear.

Consider the numerator for query $i$:

$$
\text{num}_i = \sum_{j=1}^{n} \phi(q_i)^\top \phi(k_j) \, v_j
$$

The term $\phi(q_i)^\top \phi(k_j)$ is a scalar (a dot product of two $d_\phi$-dimensional vectors). We can write it as:

$$
\phi(q_i)^\top \phi(k_j) = \sum_{m=1}^{d_\phi} \phi(q_i)_m \, \phi(k_j)_m
$$

Substituting into the numerator:

$$
\text{num}_i = \sum_{j=1}^{n} \left(\sum_{m=1}^{d_\phi} \phi(q_i)_m \, \phi(k_j)_m\right) v_j
$$

Now we swap the order of summation. The sum over $j$ and the sum over $m$ are both finite, so we can exchange them by **Fubini's theorem** (interchanging finite sums):

$$
\text{num}_i = \sum_{m=1}^{d_\phi} \phi(q_i)_m \left(\sum_{j=1}^{n} \phi(k_j)_m \, v_j\right)
$$

The inner sum $\sum_{j=1}^n \phi(k_j)_m \, v_j$ does not depend on $i$ at all — it is a fixed vector that combines all keys and values, independent of which query we are computing.

### 6.2 The matrix form

To see this more cleanly, let us write the numerator in matrix notation. The value $v_j$ is a row vector in $\mathbb{R}^{d_v}$, so $\phi(k_j) \, v_j^\top$ is an outer product: a $d_\phi \times d_v$ matrix. Define:

$$
\boxed{S = \sum_{j=1}^{n} \phi(k_j) \, v_j^\top \in \mathbb{R}^{d_\phi \times d_v}}
$$

Then the numerator becomes:

$$
\text{num}_i = \phi(q_i)^\top S \in \mathbb{R}^{1 \times d_v}
$$

This is a single matrix-vector product: the $d_\phi$-dimensional vector $\phi(q_i)$ multiplied by the $d_\phi \times d_v$ matrix $S$.

Similarly, the denominator:

$$
\text{den}_i = \sum_{j=1}^{n} \phi(q_i)^\top \phi(k_j) = \phi(q_i)^\top \left(\sum_{j=1}^{n} \phi(k_j)\right)
$$

Define:

$$
\boxed{z = \sum_{j=1}^{n} \phi(k_j) \in \mathbb{R}^{d_\phi}}
$$

Then:

$$
\text{den}_i = \phi(q_i)^\top z
$$

And the full attention output is:

$$
\boxed{o_i = \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}}
$$

### 6.3 Why this changes the complexity

This is the moment the complexity drops.

**The standard way (softmax attention):** Compute the $n \times n$ matrix $A$ where $A_{ij} = \text{sim}(q_i, k_j)$, then multiply $A \times V$. The bottleneck is the $n \times n$ matrix: $O(n^2 d_k)$ to compute $QK^\top$, then $O(n^2 d_v)$ to multiply by $V$. Total: $O(n^2 d_k)$.

**The new way (linear attention):** First compute $S = \sum_j \phi(k_j) v_j^\top$ and $z = \sum_j \phi(k_j)$. Then for each query $i$, compute $\phi(q_i)^\top S$ and $\phi(q_i)^\top z$.

The costs:

- Computing $S$: sum $n$ outer products, each $d_\phi \times d_v$. Cost: $O(n \, d_\phi \, d_v)$.
- Computing $z$: sum $n$ vectors of dimension $d_\phi$. Cost: $O(n \, d_\phi)$.
- Computing all $n$ outputs: for each query, one matrix-vector product $\phi(q_i)^\top S$ of cost $O(d_\phi \, d_v)$, and one dot product $\phi(q_i)^\top z$ of cost $O(d_\phi)$. Total: $O(n \, d_\phi \, d_v)$.

Grand total:

$$
\boxed{O(n \, d_\phi \, d_v)}
$$

With $d_\phi = d_k$ (as in the elu+1 feature map), this is $O(n \, d_k \, d_v) = O(n \, d_k^2)$. Compare to the standard $O(n^2 \, d_k)$.

When is the new way cheaper? When $n \, d_k^2 < n^2 \, d_k$, which simplifies to $d_k < n$. Since $d_k = 64$ and $n$ can be $128{,}000$, this condition is overwhelmingly satisfied for long sequences.

### 6.4 What happened algebraically

The trick is a change of **association** in matrix multiplication. The standard attention computes:

$$
\text{Standard: } (\phi(Q) \, \phi(K)^\top) \, V
$$

The parentheses indicate that we first multiply $\phi(Q) \times \phi(K)^\top$ to get the $n \times n$ attention matrix, then multiply by $V$.

The linear attention computes:

$$
\text{Linear: } \phi(Q) \, (\phi(K)^\top V)
$$

We first multiply $\phi(K)^\top \times V$, which is a $d_\phi \times n$ times $n \times d_v$ product, giving a $d_\phi \times d_v$ matrix — this is our $S$. Then we multiply each row of $\phi(Q)$ by $S$.

Matrix multiplication is **associative**: $(AB)C = A(BC)$ for any compatible matrices $A$, $B$, $C$. This is a fundamental property of matrix multiplication. The two computations produce exactly the same result. But the intermediate matrix has different size:

- Standard: intermediate is $n \times n$ (the attention matrix)
- Linear: intermediate is $d_\phi \times d_v$ (the state matrix $S$)

Since $d_\phi, d_v \ll n$ for long sequences, the linear version avoids ever materializing the $n \times n$ matrix.

### 6.5 Numerical verification

Let us verify both computations produce the same result for our running example (unmasked attention, $n = 4$, $d_k = d_v = 2$).

**Step 1: Compute $\phi(Q)$ and $\phi(K)$.**

From Section 5.5:

$$
\phi(Q) = \begin{pmatrix} 2 & 1 \\ 1 & 2 \\ 2 & 2 \\ 3 & 1 \end{pmatrix}, \quad \phi(K) = \begin{pmatrix} 1 & 2 \\ 2 & 1 \\ 2 & 2 \\ 1 & 3 \end{pmatrix}
$$

**Step 2 (Standard way): Compute $\phi(Q) \, \phi(K)^\top$ — the $4 \times 4$ similarity matrix.**

$$
\phi(Q) \, \phi(K)^\top = \begin{pmatrix} 2 & 1 \\ 1 & 2 \\ 2 & 2 \\ 3 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 & 2 & 1 \\ 2 & 1 & 2 & 3 \end{pmatrix}
$$

Row 1: $(2 \cdot 1 + 1 \cdot 2, \; 2 \cdot 2 + 1 \cdot 1, \; 2 \cdot 2 + 1 \cdot 2, \; 2 \cdot 1 + 1 \cdot 3) = (4, 5, 6, 5)$

Row 2: $(1 \cdot 1 + 2 \cdot 2, \; 1 \cdot 2 + 2 \cdot 1, \; 1 \cdot 2 + 2 \cdot 2, \; 1 \cdot 1 + 2 \cdot 3) = (5, 4, 6, 7)$

Row 3: $(2 \cdot 1 + 2 \cdot 2, \; 2 \cdot 2 + 2 \cdot 1, \; 2 \cdot 2 + 2 \cdot 2, \; 2 \cdot 1 + 2 \cdot 3) = (6, 6, 8, 8)$

Row 4: $(3 \cdot 1 + 1 \cdot 2, \; 3 \cdot 2 + 1 \cdot 1, \; 3 \cdot 2 + 1 \cdot 2, \; 3 \cdot 1 + 1 \cdot 3) = (5, 7, 8, 6)$

$$
\phi(Q) \, \phi(K)^\top = \begin{pmatrix} 4 & 5 & 6 & 5 \\ 5 & 4 & 6 & 7 \\ 6 & 6 & 8 & 8 \\ 5 & 7 & 8 & 6 \end{pmatrix}
$$

Now normalize each row (divide by row sum) and multiply by $V$:

Row 1 sum: $4 + 5 + 6 + 5 = 20$. Normalized: $(0.20, 0.25, 0.30, 0.25)$.

$$
o_1 = 0.20 \begin{pmatrix} 1 \\ 0 \end{pmatrix} + 0.25 \begin{pmatrix} 0 \\ 1 \end{pmatrix} + 0.30 \begin{pmatrix} 1 \\ 1 \end{pmatrix} + 0.25 \begin{pmatrix} 2 \\ 0 \end{pmatrix} = \begin{pmatrix} 0.20 + 0 + 0.30 + 0.50 \\ 0 + 0.25 + 0.30 + 0 \end{pmatrix} = \begin{pmatrix} 1.00 \\ 0.55 \end{pmatrix}
$$

**Step 3 (Linear way): Compute $S = \phi(K)^\top V$ — the $2 \times 2$ state matrix.**

$$
S = \phi(K)^\top V = \begin{pmatrix} 1 & 2 & 2 & 1 \\ 2 & 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{pmatrix}
$$

Row 1 of $S$: $(1 \cdot 1 + 2 \cdot 0 + 2 \cdot 1 + 1 \cdot 2, \; 1 \cdot 0 + 2 \cdot 1 + 2 \cdot 1 + 1 \cdot 0) = (5, 4)$

Row 2 of $S$: $(2 \cdot 1 + 1 \cdot 0 + 2 \cdot 1 + 3 \cdot 2, \; 2 \cdot 0 + 1 \cdot 1 + 2 \cdot 1 + 3 \cdot 0) = (10, 3)$

$$
S = \begin{pmatrix} 5 & 4 \\ 10 & 3 \end{pmatrix}
$$

Also compute $z = \sum_j \phi(k_j)$:

$$
z = \begin{pmatrix} 1 \\ 2 \end{pmatrix} + \begin{pmatrix} 2 \\ 1 \end{pmatrix} + \begin{pmatrix} 2 \\ 2 \end{pmatrix} + \begin{pmatrix} 1 \\ 3 \end{pmatrix} = \begin{pmatrix} 6 \\ 8 \end{pmatrix}
$$

Now compute $o_1$:

$$
\phi(q_1)^\top S = \begin{pmatrix} 2 & 1 \end{pmatrix} \begin{pmatrix} 5 & 4 \\ 10 & 3 \end{pmatrix} = \begin{pmatrix} 2 \cdot 5 + 1 \cdot 10, & 2 \cdot 4 + 1 \cdot 3 \end{pmatrix} = \begin{pmatrix} 20 & 11 \end{pmatrix}
$$

$$
\phi(q_1)^\top z = \begin{pmatrix} 2 & 1 \end{pmatrix} \begin{pmatrix} 6 \\ 8 \end{pmatrix} = 2 \cdot 6 + 1 \cdot 8 = 20
$$

$$
o_1 = \frac{(20, 11)}{20} = (1.00, 0.55)
$$

Both methods give $o_1 = (1.00, 0.55)$. $\checkmark$

The standard way computed a $4 \times 4$ intermediate matrix. The linear way computed a $2 \times 2$ intermediate matrix $S$. Both produced the same output. But the $2 \times 2$ matrix is fixed-size — it does not grow with $n$.

---

## 7. The Recurrent Form

### 7.1 Causal masking

Everything in Section 6 assumed unmasked (bidirectional) attention — each query attends to all keys. For autoregressive (causal) language modeling, query $i$ can only attend to keys $j \leq i$. The attention output becomes:

$$
o_i = \frac{\sum_{j=1}^{i} \phi(q_i)^\top \phi(k_j) \, v_j}{\sum_{j=1}^{i} \phi(q_i)^\top \phi(k_j)}
$$

Applying the same associativity trick, define the causal versions of the state matrix and normalizer:

$$
S_i = \sum_{j=1}^{i} \phi(k_j) \, v_j^\top, \qquad z_i = \sum_{j=1}^{i} \phi(k_j)
$$

Then:

$$
o_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
$$

### 7.2 The recurrence

This is where the connection to RNNs becomes explicit. The causal sums $S_i$ and $z_i$ satisfy:

$$
\boxed{S_i = S_{i-1} + \phi(k_i) \, v_i^\top}
$$

$$
\boxed{z_i = z_{i-1} + \phi(k_i)}
$$

with initial conditions $S_0 = 0$ (the zero matrix) and $z_0 = 0$ (the zero vector).

At each step $i$, we:

1. Compute $\phi(k_i)$ and $v_i$ from the current token's hidden state
2. Update $S_i$ by adding the outer product $\phi(k_i) v_i^\top$ (a rank-1 update)
3. Update $z_i$ by adding $\phi(k_i)$
4. Compute the output $o_i = \phi(q_i)^\top S_i \,/\, \phi(q_i)^\top z_i$

This is a recurrent neural network. The **hidden state** is the pair $(S_i, z_i)$. The update rule is additive — each new token contributes an outer product to $S$ and a vector to $z$. The output is a function of the current query and the accumulated state.

### 7.3 Tracing the recurrence

Let us trace the recurrence for our running example with causal masking.

**Step $i = 1$:** Token 1 can only attend to itself.

$$
S_1 = S_0 + \phi(k_1) v_1^\top = 0 + \begin{pmatrix} 1 \\ 2 \end{pmatrix} \begin{pmatrix} 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix}
$$

$$
z_1 = z_0 + \phi(k_1) = 0 + \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
$$

$$
o_1 = \frac{\phi(q_1)^\top S_1}{\phi(q_1)^\top z_1} = \frac{(2, 1) \begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix}}{(2, 1) \begin{pmatrix} 1 \\ 2 \end{pmatrix}} = \frac{(2 \cdot 1 + 1 \cdot 2, \; 2 \cdot 0 + 1 \cdot 0)}{2 \cdot 1 + 1 \cdot 2} = \frac{(4, 0)}{4} = (1.00, 0.00)
$$

Numerical check: with causal masking, query 1 only attends to key 1. The attention weight is 1 (only one key), so $o_1 = v_1 = (1, 0)$. $\checkmark$

**Step $i = 2$:** Token 2 attends to tokens 1 and 2.

$$
S_2 = S_1 + \phi(k_2) v_2^\top = \begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix} + \begin{pmatrix} 2 \\ 1 \end{pmatrix} \begin{pmatrix} 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix} + \begin{pmatrix} 0 & 2 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}
$$

$$
z_2 = z_1 + \phi(k_2) = \begin{pmatrix} 1 \\ 2 \end{pmatrix} + \begin{pmatrix} 2 \\ 1 \end{pmatrix} = \begin{pmatrix} 3 \\ 3 \end{pmatrix}
$$

$$
o_2 = \frac{\phi(q_2)^\top S_2}{\phi(q_2)^\top z_2} = \frac{(1, 2) \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}}{(1, 2) \begin{pmatrix} 3 \\ 3 \end{pmatrix}} = \frac{(1 + 4, \; 2 + 2)}{3 + 6} = \frac{(5, 4)}{9}
$$

$$
o_2 = (0.556, \; 0.444)
$$

Numerical check: query 2 attends to keys 1 and 2 with similarities $\phi(q_2)^\top \phi(k_1) = (1)(1) + (2)(2) = 5$ and $\phi(q_2)^\top \phi(k_2) = (1)(2) + (2)(1) = 4$. Total: $5 + 4 = 9$. Weights: $5/9 \approx 0.556$ and $4/9 \approx 0.444$.

$$
o_2 = \frac{5}{9} \begin{pmatrix} 1 \\ 0 \end{pmatrix} + \frac{4}{9} \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 0.556 \\ 0.444 \end{pmatrix} \quad \checkmark
$$

**Step $i = 3$:** Token 3 attends to tokens 1, 2, and 3.

$$
S_3 = S_2 + \phi(k_3) v_3^\top = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} + \begin{pmatrix} 2 \\ 2 \end{pmatrix} \begin{pmatrix} 1 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} + \begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix} = \begin{pmatrix} 3 & 4 \\ 4 & 3 \end{pmatrix}
$$

$$
z_3 = z_2 + \phi(k_3) = \begin{pmatrix} 3 \\ 3 \end{pmatrix} + \begin{pmatrix} 2 \\ 2 \end{pmatrix} = \begin{pmatrix} 5 \\ 5 \end{pmatrix}
$$

$$
o_3 = \frac{(2, 2) \begin{pmatrix} 3 & 4 \\ 4 & 3 \end{pmatrix}}{(2, 2) \begin{pmatrix} 5 \\ 5 \end{pmatrix}} = \frac{(6 + 8, \; 8 + 6)}{10 + 10} = \frac{(14, 14)}{20} = (0.700, 0.700)
$$

**Step $i = 4$:** Token 4 attends to all tokens 1 through 4.

$$
S_4 = S_3 + \phi(k_4) v_4^\top = \begin{pmatrix} 3 & 4 \\ 4 & 3 \end{pmatrix} + \begin{pmatrix} 1 \\ 3 \end{pmatrix} \begin{pmatrix} 2 & 0 \end{pmatrix} = \begin{pmatrix} 3 & 4 \\ 4 & 3 \end{pmatrix} + \begin{pmatrix} 2 & 0 \\ 6 & 0 \end{pmatrix} = \begin{pmatrix} 5 & 4 \\ 10 & 3 \end{pmatrix}
$$

$$
z_4 = z_3 + \phi(k_4) = \begin{pmatrix} 5 \\ 5 \end{pmatrix} + \begin{pmatrix} 1 \\ 3 \end{pmatrix} = \begin{pmatrix} 6 \\ 8 \end{pmatrix}
$$

$$
o_4 = \frac{(3, 1) \begin{pmatrix} 5 & 4 \\ 10 & 3 \end{pmatrix}}{(3, 1) \begin{pmatrix} 6 \\ 8 \end{pmatrix}} = \frac{(15 + 10, \; 12 + 3)}{18 + 8} = \frac{(25, 15)}{26} = (0.962, 0.577)
$$

Notice: $S_4$ is exactly the same $S$ matrix we computed in Section 6.5 for unmasked attention! This must be the case — when the causal sum reaches the last token, it includes all tokens, so $S_n = S$.

### 7.4 Memory and compute per step

At each step of the recurrence:

**State size:** $S_i$ is $d_\phi \times d_v$ and $z_i$ is $d_\phi$. Total: $d_\phi d_v + d_\phi = d_\phi(d_v + 1)$.

With $d_\phi = d_k = 64$ and $d_v = 64$: $64 \times 65 = 4{,}160$ elements per head. Across $h = 8$ heads and $L = 12$ layers, in fp16:

$$
8 \times 12 \times 4{,}160 \times 2 = 798{,}720 \text{ bytes} \approx 0.8 \text{ MB}
$$

Compare to the KV cache for $n = 128{,}000$ tokens: $4 \times 512 \times 12 \times 128{,}000 \approx 3.15$ GB. The recurrent state is $3{,}940\times$ smaller.

**Compute per step:** Computing $\phi(k_i) v_i^\top$ is a $d_\phi \times d_v$ outer product: $d_\phi d_v$ multiplications. Computing $\phi(q_i)^\top S_i$ is a $1 \times d_\phi$ times $d_\phi \times d_v$ product: $d_\phi d_v$ multiplications. Total per head: $O(d_\phi d_v) = O(d_k^2)$.

This is constant — it does not depend on $i$ or on how many tokens have been generated. Every token costs the same, regardless of position.

### 7.5 The complete cost comparison

| Property | Softmax attention | Linear attention (recurrent) |
|---|---|---|
| State size per head | $2 \times t \times d_k$ (KV cache, grows with $t$) | $d_k \times d_v + d_k$ (constant) |
| Compute per token (generation) | $O(t \, d_k)$ (grows with $t$) | $O(d_k^2)$ (constant) |
| Total cost for $n$ tokens (generation) | $O(n^2 d_k)$ | $O(n \, d_k^2)$ |
| Training (full sequence) | $O(n^2 d_k)$ | $O(n \, d_k^2)$ with scan; $O(n^2 d_k)$ naive |

### 7.6 Numerical cost comparison

For generation of $n = 128{,}000$ tokens, per head ($d_k = 64$):

**Softmax attention:** $n^2 d_k / 2 = 128{,}000^2 \times 64 / 2 \approx 5.24 \times 10^{11}$ multiply-adds.

**Linear attention:** $n \times d_k^2 = 128{,}000 \times 64^2 = 5.24 \times 10^8$ multiply-adds.

Ratio: $5.24 \times 10^{11} / 5.24 \times 10^8 = 1{,}000\times$. The linear version is three orders of magnitude cheaper.

This matches the claim of Katharopoulos et al. (2020): "The model has been shown to improve inference speeds up to three orders of magnitude without much loss in predictive performance."

### 7.7 The training parallelism issue

This is the part that confuses almost everyone when they first encounter linear attention.

During inference (generation), the recurrent form is strictly better: constant memory, constant compute per token. But during training, there is a catch.

Training processes the full sequence at once (teacher forcing). The softmax attention computes all $n$ outputs in parallel using matrix multiplication — $QK^\top$ and then $AV$. These are large, dense matrix multiplications that GPUs execute extremely efficiently.

The recurrent form computes outputs sequentially: $S_1 \to o_1$, then $S_2 \to o_2$, then $S_3 \to o_3$, and so on. Each step depends on the previous state $S_{i-1}$. This is a sequential dependency — you cannot compute $o_3$ until $S_2$ is done. On a GPU with thousands of parallel cores, this sequential dependency means most cores sit idle.

The asymptotic complexity is better ($O(n d_k^2)$ vs $O(n^2 d_k)$), but the wall-clock time can be worse because of low GPU utilization. This is a crucial practical distinction noted by the Efficient Transformers survey (Tay et al., 2022): "running unidirectional (causal) implementation of kernel-based attention on an autoregressive task can be several times slower than vanilla Transformer during parallelized training due to the need to do a left to right pass (i.e., scan operation) in similar spirit to Recurrent neural networks."

The resolution is the **parallel scan algorithm** (Blelloch, 1990), which computes prefix sums in $O(\log n)$ parallel steps instead of $n$ sequential steps. Modern implementations (like those in Mamba and RWKV) use optimized scan operations to achieve competitive training throughput.

---

## 8. What We Gain and What We Lose

### 8.1 What we gain

**Constant memory during generation.** The recurrent state $(S, z)$ has size $O(d_k^2)$ per head per layer, independent of sequence length. For our running model: 0.8 MB total vs 3.15 GB for the KV cache at 128K tokens. This eliminates the KV cache wall entirely.

**Constant compute per token during generation.** Each new token requires $O(d_k^2)$ operations per head, regardless of how many tokens preceded it. The 128,000th token is no more expensive than the 1st.

**Linear total cost.** Generating $n$ tokens costs $O(n d_k^2)$, linear in $n$. This enables generation at sequence lengths that are impractical with softmax attention.

### 8.2 What we lose

**Exact softmax attention.** The elu+1 kernel is not equivalent to the softmax kernel. The attention distributions are different (as we saw in Section 5.6), and this can affect model quality. Empirically, Katharopoulos et al. (2020) report competitive but not identical performance to softmax transformers on language modeling and speech recognition.

**Information capacity of the state.** The recurrent state $S \in \mathbb{R}^{d_k \times d_v}$ must compress all information from all past tokens into a fixed-size matrix. With $d_k = d_v = 64$, this is $64 \times 64 = 4{,}096$ numbers. Compare to the KV cache, which stores $2 \times d_k = 128$ numbers per past token — 128,000 tokens would require $128 \times 128{,}000 = 16{,}384{,}000$ numbers per head. The recurrent state compresses this by a factor of $4{,}000\times$.

This compression is necessarily lossy. If token 50,000 contained critical information that token 128,000 needs, the softmax attention can retrieve it directly from the KV cache. The recurrent state can only access what survived 78,000 rank-1 updates to the state matrix $S$.

**Training parallelism (naive implementation).** As discussed in Section 7.7, the sequential nature of the recurrence reduces GPU utilization during training. Modern implementations mitigate this with parallel scan, but the engineering complexity is higher.

### 8.3 The landscape after linear attention

The kernel-based linear attention of Katharopoulos et al. (2020) was one of the first rigorous demonstrations that the attention mechanism could be fundamentally replaced. It showed that the mathematical structure of attention — the softmax, the pairwise comparisons, the $n \times n$ matrix — is not sacred. What matters is the function computed: a weighted combination of values based on query-key relevance. The softmax is one way to define relevance. The kernel feature map is another. And once you have the feature map, the associativity of matrix multiplication gives you linear cost for free.

This insight opened the door to a family of architectures that blur the line between transformers and RNNs: RetNet (Sun et al., 2023), RWKV (Peng et al., 2023), Mamba (Gu and Dao, 2023), and others. Each proposes a different way to define the recurrent state and the update rule, with the shared goal of constant-time, constant-memory token generation without sacrificing the quality that made transformers dominant.

The Efficient Transformers survey (Tay et al., 2022) presciently noted: "It is then a question of whether that new xformer will still be a Transformer." The kernel view of attention suggests the answer: the boundary between transformers and RNNs was always algebraic, not architectural.

---

## Summary

After twelve blogs of modifying attention — reducing KV heads, compressing representations, sparsifying patterns, gating residuals and activations — the softmax normalization remained untouched, and with it the quadratic cost of computing pairwise query-key similarities. This blog derived the exact costs that persist (the $O(n^2 d_\text{model})$ attention FLOPs that dominate beyond $n = d_{ff}$ tokens, the linearly growing KV cache that reaches hundreds of gigabytes at long contexts, the per-token generation cost that makes the last token $n$ times more expensive than the first), identified the softmax as the root cause (its infinite-dimensional feature map prevents factoring the similarity into independent query and key functions), replaced it with a finite-dimensional kernel ($\phi(x) = \text{elu}(x) + 1$), applied the associativity of matrix multiplication to change the computation order from $(\phi(Q)\phi(K)^\top)V$ to $\phi(Q)(\phi(K)^\top V)$, and showed that the result is an RNN with state $(S_i, z_i)$ that achieves constant memory ($d_k \times d_v$ per head), constant compute per token ($O(d_k^2)$), and linear total generation cost ($O(n d_k^2)$) — three orders of magnitude faster than softmax attention at 128K tokens, at the price of compressing all past information into a fixed-size state matrix.

---

*Previous: [Gated Attention: Replacing Residuals and ReLU with Learned Gates](/blog/attention-gated-attention)*

*Next: [Hybrid Architectures: RetNet and the Three Computation Paradigms](/blog/attention-retnet)*
