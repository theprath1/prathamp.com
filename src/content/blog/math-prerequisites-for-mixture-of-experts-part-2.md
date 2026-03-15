---
title: "Mathematical Prerequisites for Mixture of Experts — Part 2"
description: "Building the math foundations you need for sparse MoEs and the Switch Transformer — softplus, top-k masking, mean and variance, coefficient of variation, indicator functions, argmax, differentiability, and the dot-product loss — all derived step by step with one consistent example."
date: 2026-03-14
tags: [machine-learning, mixture-of-experts, mathematics, probability, transformers]
order: 1
draft: false
---

In [Part 2](/blog/mixture-of-experts-part-2) of the Mixture of Experts series, we scale MoEs from 4 experts to thousands, introduce sparse gating, and derive the Switch Transformer. The mathematics is different from Part 1 — instead of Gaussian likelihoods and the EM algorithm, we need tools for sparsity, load balancing, and differentiability. By the end of this post, you will have every mathematical tool required to follow Part 2 from the first equation to the last.

We assume you have read the [prerequisites for Part 1](/blog/math-prerequisites-for-mixture-of-experts), where we built softmax, Gaussian densities, Bayes' theorem, and the mixture log-likelihood. We will not re-derive those here. Instead, we build six new tools — each one earning its place by being directly used in Part 2.

---

## The Running Example

We have 4 experts and a batch of 8 tokens. Each token is routed to some subset of the experts. This is the same setup used throughout Part 2.

For concreteness, suppose the gating network produces the following raw scores (logits) for one specific token:

$$
s = [2.1, \; 0.5, \; 3.4, \; 1.2]
$$

These are the scores for experts 1, 2, 3, and 4 respectively. Expert 3 has the highest score (3.4), expert 1 has the second highest (2.1). We will use these four numbers for every concept in this post.

For batch-level concepts, we will use 8 tokens routed across the 4 experts as follows:

| Token | Routed to expert |
|-------|-----------------|
| 1     | 1               |
| 2     | 1               |
| 3     | 2               |
| 4     | 3               |
| 5     | 1               |
| 6     | 3               |
| 7     | 2               |
| 8     | 4               |

Expert 1 receives 3 tokens, experts 2 and 3 receive 2 each, and expert 4 receives 1. We will use these counts for every batch-level computation.

---

## 1. The Softplus Function

In Part 2, Shazeer et al. add tunable noise to the gating logits. The noise magnitude must be positive — you cannot have a negative standard deviation. The **softplus function** guarantees positivity:

$$
\boxed{\text{Softplus}(z) = \ln(1 + e^z)}
$$

Let us verify that this is always positive. Since $e^z > 0$ for all $z$, we have $1 + e^z > 1$, so $\ln(1 + e^z) > \ln(1) = 0$. The output is strictly positive for every input — exactly what we need for a noise scale.

**Why not just use $e^z$?** The exponential $e^z$ is also always positive, but it grows explosively: $e^{10} = 22{,}026$. The softplus grows much more gently. For large positive $z$:

$$
\text{Softplus}(z) = \ln(1 + e^z) \approx \ln(e^z) = z
$$

because $e^z$ dominates the 1. For large negative $z$:

$$
\text{Softplus}(z) = \ln(1 + e^z) \approx \ln(1) = 0
$$

because $e^z \approx 0$. So softplus behaves like $z$ when $z$ is large and positive, and like $0$ when $z$ is large and negative. It is a smooth, always-positive version of $\max(0, z)$ — the **ReLU function**. This is why the softplus is sometimes called a "smooth ReLU."

### Numerical check

Let us compute Softplus for several values:

$$
\text{Softplus}(-2) = \ln(1 + e^{-2}) = \ln(1 + 0.135) = \ln(1.135) = 0.127
$$

$$
\text{Softplus}(0) = \ln(1 + e^{0}) = \ln(1 + 1) = \ln(2) = 0.693
$$

$$
\text{Softplus}(2) = \ln(1 + e^{2}) = \ln(1 + 7.389) = \ln(8.389) = 2.127
$$

$$
\text{Softplus}(5) = \ln(1 + e^{5}) = \ln(1 + 148.4) = \ln(149.4) = 5.007
$$

All outputs are positive. At $z = 5$, the output ($5.007$) is nearly equal to $z$ ($5$) — confirming the large-$z$ approximation $\text{Softplus}(z) \approx z$. At $z = -2$, the output ($0.127$) is close to zero — confirming the negative-$z$ approximation $\text{Softplus}(z) \approx 0$.

In Part 2, the noise term in Shazeer et al.'s gating is $\text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$. The Softplus ensures the noise scale is always positive regardless of what the learned weights $W_{\text{noise}}$ produce.

---

## 2. Top-K Selection and the $-\infty$ Masking Trick

Standard softmax (from the Part 1 prerequisites) converts all $n$ scores into $n$ nonzero probabilities. In Part 2, we want **sparse** gating: only $k$ experts should receive nonzero probability, and the rest should be exactly zero. The **top-k masking trick** achieves this in two steps.

**Step 1: Identify the top $k$ scores.** Given a vector of scores $s = [s_1, s_2, \ldots, s_n]$, find the $k$ largest values.

**Step 2: Mask the rest with $-\infty$.** Define a new vector:

$$
\boxed{\text{KeepTopK}(s, k)_i = \begin{cases} s_i & \text{if } s_i \text{ is among the top } k \text{ values} \\ -\infty & \text{otherwise} \end{cases}}
$$

Then apply softmax to the result.

**Why $-\infty$ works.** Recall that softmax computes $p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$. When $s_i = -\infty$:

$$
e^{-\infty} = \lim_{z \to -\infty} e^z = 0
$$

So the numerator for the masked entries is exactly zero, making $p_i = 0$. The denominator only accumulates contributions from the $k$ non-masked entries. The result is a probability vector with exactly $k$ nonzero entries that sum to 1.

This is different from simply setting $p_i = 0$ after softmax. If we applied softmax first and then zeroed out entries, the remaining entries would no longer sum to 1. By masking *before* softmax, we ensure the nonzero entries sum to 1 automatically — softmax handles the normalisation.

### Numerical check

Using our running example $s = [2.1, \; 0.5, \; 3.4, \; 1.2]$ with $k = 2$:

The top-2 scores are $s_3 = 3.4$ and $s_1 = 2.1$. After masking:

$$
\text{KeepTopK}(s, 2) = [2.1, \; -\infty, \; 3.4, \; -\infty]
$$

Applying softmax to the masked vector:

$$
e^{2.1} = 8.166, \quad e^{-\infty} = 0, \quad e^{3.4} = 29.964, \quad e^{-\infty} = 0
$$

$$
\text{Sum} = 8.166 + 0 + 29.964 + 0 = 38.130
$$

$$
p_1 = \frac{8.166}{38.130} = 0.214, \quad p_2 = 0, \quad p_3 = \frac{29.964}{38.130} = 0.786, \quad p_4 = 0
$$

Check: $0.214 + 0 + 0.786 + 0 = 1.000$. Exactly 2 nonzero entries, summing to 1.

Compare this to full softmax (no masking): $e^{0.5} = 1.649$, $e^{1.2} = 3.320$, sum $= 8.166 + 1.649 + 29.964 + 3.320 = 43.099$. Then $p_1 = 0.189$, $p_2 = 0.038$, $p_3 = 0.695$, $p_4 = 0.077$. All four entries are nonzero — we would have to compute all four expert outputs. With top-2 masking, we only compute 2 expert outputs. With $n = 4096$ experts and $k = 2$, we would compute 2 instead of 4096 — the entire point of sparse gating.

---

## 3. Mean, Variance, and Standard Deviation

Part 2 uses the **coefficient of variation** to measure how unbalanced the experts are. This requires three building blocks: mean, variance, and standard deviation. We derive all three using the expert importance values from our running example.

### 3.1 Mean

The **mean** (or **average**) of $n$ values is their sum divided by $n$:

$$
\boxed{\text{Mean}(v) = \bar{v} = \frac{1}{n} \sum_{i=1}^{n} v_i}
$$

The mean answers: if we spread the total evenly across all entries, how much would each get?

### Numerical check

From our routing table, the number of tokens each expert receives is $[3, 2, 2, 1]$. In Part 2, this is called the **importance** vector. Its mean:

$$
\bar{v} = \frac{3 + 2 + 2 + 1}{4} = \frac{8}{4} = 2.0
$$

Each expert would receive 2 tokens on average. Expert 1 receives more than average (3), expert 4 receives less (1).

### 3.2 Variance

The **variance** measures how spread out the values are from their mean. It is the average squared deviation:

$$
\boxed{\text{Var}(v) = \frac{1}{n} \sum_{i=1}^{n} (v_i - \bar{v})^2}
$$

Each term $(v_i - \bar{v})^2$ measures how far value $i$ is from the mean, squared. Squaring ensures deviations above and below the mean both contribute positively — the same reasoning as squared error from the Part 1 prerequisites.

### Numerical check

With $v = [3, 2, 2, 1]$ and $\bar{v} = 2.0$:

$$
\text{Var} = \frac{1}{4}\left[(3-2)^2 + (2-2)^2 + (2-2)^2 + (1-2)^2\right]
$$

$$
= \frac{1}{4}[1 + 0 + 0 + 1] = \frac{2}{4} = 0.5
$$

The variance is 0.5. If all experts received exactly 2 tokens, every deviation would be zero and the variance would be 0. The farther the values are from the mean, the larger the variance.

### 3.3 Standard deviation

The **standard deviation** is the square root of the variance:

$$
\boxed{\text{Std}(v) = \sqrt{\text{Var}(v)}}
$$

Why take the square root? Variance is measured in squared units — if the values are token counts, the variance has units of "tokens squared." The standard deviation brings us back to the original units (tokens), making it directly comparable to the mean.

### Numerical check

$$
\text{Std} = \sqrt{0.5} = 0.707
$$

The standard deviation is 0.707 tokens. Roughly speaking, expert loads deviate from the mean by about 0.7 tokens on average.

---

## 4. Coefficient of Variation

Part 2 uses the **coefficient of variation** (CV) as a load balancing penalty. The CV is the standard deviation divided by the mean:

$$
\boxed{\text{CV}(v) = \frac{\text{Std}(v)}{\text{Mean}(v)}}
$$

Why divide by the mean? Because the standard deviation alone does not tell us whether the spread is "large" or "small" relative to the values themselves. A standard deviation of 10 is large when the mean is 20 (values are all over the place), but small when the mean is 10{,}000 (values are tightly clustered relative to their size). The CV normalises the spread by the scale of the data.

**The key property.** The CV equals zero if and only if all values are equal. When all values are equal, the standard deviation is zero (no spread), so $\text{CV} = 0 / \bar{v} = 0$. Any departure from uniformity makes the CV positive. This is exactly what we want for a load balancing penalty — it should be zero when experts are perfectly balanced and positive when they are not.

Shazeer et al. use $\text{CV}^2$ (the square of the coefficient of variation) as the importance loss:

$$
L_{\text{importance}} = w \cdot \text{CV}(\text{Importance})^2
$$

Squaring the CV serves two purposes. First, it keeps the loss non-negative (though CV is already non-negative, the square emphasises large imbalances). Second, the squared CV is smoother — its derivative at $\text{CV} = 0$ is zero, so it does not create a gradient discontinuity when experts are perfectly balanced.

### Numerical check

Using our values $v = [3, 2, 2, 1]$, $\text{Std} = 0.707$, $\text{Mean} = 2.0$:

$$
\text{CV} = \frac{0.707}{2.0} = 0.354
$$

$$
\text{CV}^2 = 0.354^2 = 0.125
$$

Now suppose all experts received equal load: $v = [2, 2, 2, 2]$. Then $\text{Std} = 0$, $\text{Mean} = 2.0$, $\text{CV} = 0/2 = 0$, and $\text{CV}^2 = 0$. The loss vanishes — no penalty for perfect balance.

For a more extreme imbalance, $v = [5, 1, 1, 1]$: $\text{Mean} = 2.0$, $\text{Var} = \frac{1}{4}[9 + 1 + 1 + 1] = 3.0$, $\text{Std} = 1.732$, $\text{CV} = 0.866$, $\text{CV}^2 = 0.750$. The penalty grew from 0.125 to 0.750 — a 6x increase, reflecting the much worse imbalance.

---

## 5. The Indicator Function and Argmax

Part 2 uses two closely related tools to describe hard routing decisions: the **indicator function** and the **argmax**.

### 5.1 Argmax

The **argmax** of a vector returns the *index* of the largest element, not the element itself:

$$
\boxed{\arg\max_j \; v_j = \text{the index } j \text{ for which } v_j \text{ is largest}}
$$

The distinction between max and argmax is important. The **max** answers "what is the largest value?" The **argmax** answers "which entry has the largest value?"

### Numerical check

For $s = [2.1, \; 0.5, \; 3.4, \; 1.2]$:

$$
\max_j \; s_j = 3.4 \quad \text{(the value)}
$$

$$
\arg\max_j \; s_j = 3 \quad \text{(the index)}
$$

In Part 2, the Switch Transformer routes each token to the single expert with the highest gate logit: $i = \arg\max_j \; (W_r \cdot x)_j$. This says: compute the router scores for all experts, then pick the expert whose score is largest. The argmax gives us the expert *number*, not the score.

### 5.2 The indicator function

The **indicator function** $\mathbb{1}\{\text{condition}\}$ equals 1 when the condition is true and 0 when it is false:

$$
\boxed{\mathbb{1}\{A\} = \begin{cases} 1 & \text{if } A \text{ is true} \\ 0 & \text{if } A \text{ is false} \end{cases}}
$$

That is the entire definition — nothing more.

### Numerical check

Suppose token 1 is routed to expert $i = \arg\max_j \; s_j = 3$. Then:

$$
\mathbb{1}\{\arg\max \; p(x_1) = 1\} = 0 \quad \text{(token 1 was not routed to expert 1)}
$$

$$
\mathbb{1}\{\arg\max \; p(x_1) = 3\} = 1 \quad \text{(token 1 was routed to expert 3)}
$$

### 5.3 Counting with indicators

The indicator function lets us express counts as sums. The number of tokens routed to expert $i$ in a batch $\mathcal{B}$ of $T$ tokens is:

$$
\text{count}_i = \sum_{x \in \mathcal{B}} \mathbb{1}\{\arg\max \; p(x) = i\}
$$

Each term is either 0 or 1, and summing them counts how many tokens were assigned to expert $i$.

### Numerical check

From our routing table (tokens assigned to experts $[1, 1, 2, 3, 1, 3, 2, 4]$), the count for expert 1 is:

$$
\text{count}_1 = 1 + 1 + 0 + 0 + 1 + 0 + 0 + 0 = 3
$$

The fraction of tokens dispatched to expert 1 is:

$$
f_1 = \frac{\text{count}_1}{T} = \frac{3}{8} = 0.375
$$

This is exactly the $f_i$ quantity from Fedus et al.'s load balancing loss in Part 2.

---

## 6. Differentiable vs. Non-Differentiable Functions

The load balancing loss in Part 2 rests on a subtle but critical distinction: some functions have gradients and some do not. Understanding this distinction is essential for seeing why the loss is designed the way it is.

### 6.1 What differentiable means

A function $f(x)$ is **differentiable** at a point $x_0$ if it has a well-defined slope (derivative) at that point. Visually, this means the function has no jumps, corners, or discontinuities at $x_0$ — you could draw a single, unique tangent line to the curve.

The **derivative** $\frac{df}{dx}$ measures how much $f$ changes when we nudge $x$ by a tiny amount. If $f$ is differentiable, this change is smooth and predictable.

The softmax function $p_i(x) = \frac{e^{(W_r \cdot x)_i}}{\sum_j e^{(W_r \cdot x)_j}}$ is differentiable everywhere. If we nudge the router weights $W_r$ by a tiny amount, the softmax probabilities $p_i$ change smoothly and predictably. Gradients flow through softmax without any issues.

### 6.2 What non-differentiable means

A function is **non-differentiable** at a point where it has a jump or a corner — the slope is not well-defined because the function changes abruptly.

The argmax function is non-differentiable. Consider a simple case with two scores: $s = [s_1, s_2]$. The argmax is:

$$
\arg\max(s_1, s_2) = \begin{cases} 1 & \text{if } s_1 > s_2 \\ 2 & \text{if } s_2 > s_1 \end{cases}
$$

As $s_1$ increases from below $s_2$ to above $s_2$, the argmax jumps from 2 to 1 — an instantaneous switch with no gradual transition. There is no meaningful "slope" at the switching point $s_1 = s_2$.

The indicator function $\mathbb{1}\{\arg\max \; p(x) = i\}$ inherits this problem. It is either 0 or 1 with no values in between, and jumps discontinuously. No gradient can flow through it.

### 6.3 Why this matters for load balancing

In Part 2, Fedus et al. define two quantities:

$$
f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbb{1}\{\arg\max \; p(x) = i\}
$$

$$
P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x)
$$

The first quantity $f_i$ counts hard routing decisions — it uses the indicator function and argmax, both of which are **non-differentiable**. We cannot compute $\frac{\partial f_i}{\partial W_r}$ because the indicator has no meaningful derivative.

The second quantity $P_i$ averages soft router probabilities — it uses the softmax output $p_i(x)$, which is **differentiable**. We can compute $\frac{\partial P_i}{\partial W_r}$ and use it to update the router weights via gradient descent.

The load balancing loss $\sum_i f_i \cdot P_i$ multiplies these two quantities together. When we differentiate this product with respect to $W_r$, by the **product rule** of calculus:

$$
\frac{\partial}{\partial W_r}(f_i \cdot P_i) = f_i \cdot \frac{\partial P_i}{\partial W_r} + P_i \cdot \frac{\partial f_i}{\partial W_r}
$$

The second term vanishes because $\frac{\partial f_i}{\partial W_r}$ does not exist (or is zero almost everywhere). The gradient flows entirely through the differentiable term $P_i$:

$$
\frac{\partial}{\partial W_r}(f_i \cdot P_i) = f_i \cdot \frac{\partial P_i}{\partial W_r}
$$

This is the entire design insight: $f_i$ acts as a **fixed coefficient** that scales the gradient of $P_i$. When expert $i$ receives too many tokens (high $f_i$), the gradient of $P_i$ is amplified, pushing the router to reduce the probability assigned to expert $i$. The non-differentiable counting function $f_i$ provides the signal about *what* is wrong, while the differentiable probability $P_i$ provides the pathway for *fixing* it.

### Numerical check

Using our routing ($f = [0.375, 0.250, 0.250, 0.125]$) and suppose $P = [0.35, 0.25, 0.28, 0.12]$:

The gradient contribution from expert 1: $f_1 \cdot \frac{\partial P_1}{\partial W_r} = 0.375 \cdot \frac{\partial P_1}{\partial W_r}$.

The gradient contribution from expert 4: $f_4 \cdot \frac{\partial P_4}{\partial W_r} = 0.125 \cdot \frac{\partial P_4}{\partial W_r}$.

Expert 1's gradient is scaled by 0.375 while expert 4's is scaled by 0.125 — a 3x ratio. The router receives a stronger push to reduce probability for expert 1 (the overloaded one) than for expert 4 (the underloaded one). This is how the loss encourages balance.

---

## 7. The Dot Product as a Balancing Loss

The load balancing loss from Fedus et al. is built on the **dot product** (also called **inner product** or **scalar product**) of two vectors. The dot product multiplies corresponding entries and sums:

$$
\boxed{\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \cdot b_i}
$$

The dot product takes two vectors of the same length and produces a single number.

### 7.1 What the dot product measures

The dot product is large when both vectors have large values in the *same* positions. If $a_i$ is large whenever $b_i$ is large, the products $a_i b_i$ are all large and their sum is large. If the large values of $\mathbf{a}$ and $\mathbf{b}$ occur in different positions, the products are smaller.

This makes the dot product a measure of **alignment** between two vectors. In the context of load balancing, $\mathbf{a} = f$ (fraction of tokens per expert) and $\mathbf{b} = P$ (fraction of probability per expert). The dot product $\sum_i f_i P_i$ is large when experts that receive many tokens also receive high probability — exactly the imbalance we want to penalise.

### 7.2 The dot product under uniform distribution

Under perfect balance, every expert receives the same fraction of tokens and the same fraction of probability. With $N$ experts:

$$
f_i = \frac{1}{N}, \quad P_i = \frac{1}{N} \quad \text{for all } i
$$

The dot product becomes:

$$
\sum_{i=1}^{N} f_i \cdot P_i = \sum_{i=1}^{N} \frac{1}{N} \cdot \frac{1}{N} = N \cdot \frac{1}{N^2} = \frac{1}{N}
$$

This is the minimum possible value of the dot product, given the constraints $\sum_i f_i = 1$ and $\sum_i P_i = 1$. This is a consequence of the **Cauchy-Schwarz inequality**: for non-negative vectors with fixed sums, the dot product is minimised when both vectors are uniform.

Any deviation from uniformity increases the dot product. This is why Fedus et al. use $N \cdot \sum_i f_i P_i$ as the loss — the factor of $N$ normalises so that the uniform-case loss is $N \cdot \frac{1}{N} = 1$, independent of the number of experts.

### Numerical check

With $f = [0.375, 0.250, 0.250, 0.125]$ and $P = [0.35, 0.25, 0.28, 0.12]$:

$$
\sum_i f_i \cdot P_i = 0.375 \times 0.35 + 0.250 \times 0.25 + 0.250 \times 0.28 + 0.125 \times 0.12
$$

$$
= 0.131 + 0.063 + 0.070 + 0.015 = 0.279
$$

The loss (with $\alpha = 0.01$ and $N = 4$):

$$
\text{loss} = 0.01 \times 4 \times 0.279 = 0.01116
$$

Under perfect uniformity ($f = P = [0.25, 0.25, 0.25, 0.25]$):

$$
\sum_i f_i \cdot P_i = 4 \times 0.25 \times 0.25 = 0.250
$$

$$
\text{loss} = 0.01 \times 4 \times 0.250 = 0.01000
$$

The actual loss ($0.01116$) exceeds the uniform-case loss ($0.01000$), penalising the imbalance. The difference ($0.00116$) creates a gradient that pushes the router toward more uniform probability allocation.

---

## 8. Expert Capacity and Integer Arithmetic

Part 2 introduces a fixed buffer size for each expert called the **expert capacity**. The formula involves integer arithmetic that is worth making precise.

$$
\boxed{\text{expert capacity} = \left\lfloor \frac{T}{N} \times C \right\rfloor}
$$

where $T$ is the number of tokens, $N$ is the number of experts, and $C$ is the **capacity factor** — a hyperparameter that controls how much buffer space each expert gets. The $\lfloor \cdot \rfloor$ notation means the **floor function**: round down to the nearest integer, since we cannot process a fractional token.

The ratio $\frac{T}{N}$ is the number of tokens each expert *would* receive under perfect balance. The capacity factor $C$ scales this up to provide buffer room for imbalanced routing.

### Why this matters

If the router sends more tokens to an expert than its capacity allows, the excess tokens are **dropped** — they skip the expert entirely and pass through the residual connection unchanged. Setting the capacity too low means many tokens are dropped and never processed by any expert. Setting it too high wastes memory on empty buffer slots that are never filled.

### Numerical check

With $T = 8$ tokens and $N = 4$ experts:

**Capacity factor $C = 1.0$:**

$$
\text{expert capacity} = \frac{8}{4} \times 1.0 = 2
$$

Each expert can process at most 2 tokens. In our routing, expert 1 receives 3 tokens but can only process 2 — one token is dropped. Total buffer across experts: $4 \times 2 = 8$ slots for 8 tokens. No wasted space, but risk of dropping.

**Capacity factor $C = 1.5$:**

$$
\text{expert capacity} = \frac{8}{4} \times 1.5 = 3
$$

Each expert can now handle up to 3 tokens. Expert 1 receives 3 tokens and processes all of them — no dropping. Total buffer: $4 \times 3 = 12$ slots for 8 tokens. Four slots are wasted (padding).

**Capacity factor $C = 0.5$:**

$$
\text{expert capacity} = \frac{8}{4} \times 0.5 = 1
$$

Each expert processes at most 1 token. Expert 1 drops 2 of its 3 tokens, experts 2 and 3 each drop 1 of their 2 tokens. Total dropped: 4 out of 8 tokens — half the batch is unprocessed. This is far too aggressive.

The tension is clear: larger $C$ reduces dropping but wastes memory. Fedus et al. found that $C = 1.0$ to $C = 1.25$ works best — the load balancing loss keeps the routing balanced enough that very little capacity buffer is needed.

---

## Summary

We have built six tools for Part 2. The softplus function guarantees positive noise scales by smoothly approximating $\max(0, z)$. Top-k masking with $-\infty$ forces softmax to produce exactly $k$ nonzero probabilities, enabling sparse gating where only $k$ out of $n$ experts are computed. Mean, variance, standard deviation, and the coefficient of variation measure how unbalanced expert loads are — the CV equals zero at perfect balance and increases with any deviation, making its square a natural load balancing penalty. The indicator function and argmax describe hard routing decisions: the argmax picks the best expert, the indicator counts how many tokens go where. The distinction between differentiable functions (softmax probabilities) and non-differentiable functions (argmax, indicators) explains why the load balancing loss multiplies $f_i$ by $P_i$ — gradients flow through the differentiable $P_i$ while the non-differentiable $f_i$ acts as a fixed scaling factor. The dot product $\sum_i f_i P_i$ measures alignment between routing counts and routing probabilities, reaching its minimum at uniform balance. And expert capacity arithmetic sets the buffer size per expert, trading dropped tokens against wasted memory.

With these tools in hand, we are ready for [Part 2](/blog/mixture-of-experts-part-2), where we derive sparse gating, the Switch Transformer, and load balancing losses from scratch.
