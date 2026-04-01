---
title: "Mathematical Prerequisites for the Attention Series"
description: "Building the math foundations for the attention series — the tanh function, dot products as similarity measures, the standard normal distribution, variance, independence of random variables, and why the variance of a dot product equals the vector dimension — all derived step by step with one consistent 2-dimensional example."
date: 2026-03-31
tags: [attention, transformers, mathematics, deep-learning]
order: 5
---

Before diving into the attention series, we need four tools that have not appeared in any earlier prerequisite post: the tanh function, the dot product, the standard normal distribution, and variance. The most important is the last. In [From Soft Alignment to Queries, Keys, and Values](/blog/attention-q-k-v-from-scratch), we derive why the attention score matrix must be divided by $\sqrt{d_k}$ before the softmax. The entire argument rests on one result: when each component of $\mathbf{q}$ and $\mathbf{k}$ is independently drawn from a standard normal distribution, the variance of their dot product equals $d_k$. Everything in this post builds toward that one identity.

Expected value and linearity of expectation were fully derived in [Mathematical Prerequisites for Reinforcement Learning](/blog/math-prerequisites-for-rl). We will use both without re-deriving them.

---

## The Running Example

Two 2-dimensional vectors:

$$
\mathbf{q} = (q_1,\, q_2) = (1.2,\; {-0.4}), \qquad \mathbf{k} = (k_1,\, k_2) = (0.6,\; 0.9)
$$

These four numbers — $1.2$, $-0.4$, $0.6$, $0.9$ — came from drawing each component independently at random. We will use them in every concrete calculation in this post. In the attention blogs, the vectors will be 64-dimensional rather than 2-dimensional, but the structure is identical.

---

## 1. The tanh Function

In [What Attention is Really Doing](/blog/attention-what-is-it-really), the first step is computing an alignment score between a decoder state $s$ and an encoder annotation $h$. The alignment model used by Bahdanau et al. applies the **hyperbolic tangent** to the sum of two projected vectors. We need to know what tanh is, what values it can take, and how to compute it.

The **hyperbolic tangent** is defined as:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

The numerator $e^x - e^{-x}$ is positive when $x > 0$, zero when $x = 0$, and negative when $x < 0$. The denominator $e^x + e^{-x}$ is always strictly positive. The ratio therefore carries the sign of the numerator.

**Range.** The denominator always exceeds the numerator in absolute value: $e^x + e^{-x} > |e^x - e^{-x}|$ for all real $x$, because the larger of $e^x$ and $e^{-x}$ always appears on both sides. Dividing, we get $|\tanh(x)| < 1$. So $\tanh(x) \in (-1, 1)$ for every real input.

**Boundary behaviour.** As $x \to +\infty$, $e^{-x} \to 0$, so

$$
\tanh(x) \to \frac{e^x}{e^x} = 1
$$

As $x \to -\infty$, $e^x \to 0$, so

$$
\tanh(x) \to \frac{-e^{-x}}{e^{-x}} = -1
$$

At $x = 0$: $\tanh(0) = (1 - 1)/(1 + 1) = 0$. The function passes through the origin, approaches $+1$ from below as $x$ grows large, and approaches $-1$ from above as $x$ grows large and negative.

### Numerical check

We compute the alignment score between $q_1$ and $k_1$ from our running example: $\tanh(q_1 + k_1) = \tanh(1.2 + 0.6) = \tanh(1.8)$.

$$
e^{1.8} \approx 6.0496, \qquad e^{-1.8} \approx 0.1653
$$

$$
\tanh(1.8) = \frac{6.0496 - 0.1653}{6.0496 + 0.1653} = \frac{5.8843}{6.2149} \approx 0.9468
$$

The result lies in $(-1, 1)$ as required. In the blog [What Attention is Really Doing](/blog/attention-what-is-it-really), alignment scores for source words $h_1$, $h_2$, $h_3$ are computed exactly this way — each one is a $\tanh$ of a sum of two values, producing a score between $-1$ and $+1$ before softmax normalisation.

---

## 2. The Dot Product

In [From Soft Alignment to Queries, Keys, and Values](/blog/attention-q-k-v-from-scratch), the tanh-based alignment model is replaced by a simpler function: the dot product. Every attention score $S[i,j]$ is the dot product of query vector $i$ with key vector $j$. We need a precise definition.

The **dot product** (also called the **inner product**) of two $d$-dimensional vectors $\mathbf{q} = (q_1, \ldots, q_d)$ and $\mathbf{k} = (k_1, \ldots, k_d)$ is the sum of their component-wise products:

$$
\boxed{\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d} q_i k_i}
$$

Each pair $(q_i, k_i)$ contributes one scalar $q_i k_i$ to the total. The output is a single number, not a vector.

**What does it measure?** When $\mathbf{q}$ and $\mathbf{k}$ point in the same direction, their components tend to share the same sign, each product $q_i k_i$ is positive, and the sum is a large positive number. When they point in opposite directions, the products are negative and the sum is a large negative number. When they are perpendicular, positive and negative contributions cancel and the sum is near zero. The dot product is therefore a measure of directional similarity — large and positive when vectors align, near zero when perpendicular, large and negative when opposite.

For unit-length vectors, $\mathbf{q} \cdot \mathbf{k} = \|\mathbf{q}\|\|\mathbf{k}\|\cos\theta = \cos\theta$, where $\theta$ is the angle between them. This is the **cosine similarity**. For general vectors the magnitude matters too, but the directional interpretation remains.

### Numerical check

$$
\mathbf{q} \cdot \mathbf{k} = q_1 k_1 + q_2 k_2 = 1.2 \times 0.6 + (-0.4) \times 0.9 = 0.72 - 0.36 = 0.36
$$

The first component pair contributes $+0.72$ (same sign, so positive contribution), and the second contributes $-0.36$ (opposite signs, so negative contribution). The net dot product is $0.36$.

In Part 2, this computation is done for every pair of query and key vectors simultaneously via the matrix product $QK^\top$. For our 2-dimensional, 3-token example in that blog, this produces a $3 \times 3$ matrix of alignment scores — each entry computed as we just did.

---

## 3. The Standard Normal Distribution

The $\sqrt{d_k}$ scaling in attention is justified by a probabilistic argument. We need to model the typical size of each vector component. The assumption used in Part 2 is that each component of $\mathbf{q}$ and $\mathbf{k}$ is drawn independently from the **standard normal distribution**, written $\mathcal{N}(0, 1)$.

A **random variable** $X$ distributed as $\mathcal{N}(0, 1)$ has two defining properties:

1. **Mean zero**: $\mathbb{E}[X] = 0$. The distribution is symmetric around zero — $X$ is equally likely to be positive or negative.
2. **Variance one**: $\text{Var}(X) = 1$. The typical distance from zero is about 1.

We define variance precisely in the next section. For now, treat these as the defining numbers.

**One key consequence.** For $X \sim \mathcal{N}(0, 1)$, the expected value of $X^2$ is:

$$
\mathbb{E}[X^2] = \text{Var}(X) + \left(\mathbb{E}[X]\right)^2 = 1 + 0^2 = 1
$$

This uses the computational formula for variance, which we derive in Section 4. The result — $\mathbb{E}[X^2] = 1$ — will be our workhorse in Sections 5 and 6.

**Why this assumption?** Before training, the linear projections that produce $\mathbf{q}$ and $\mathbf{k}$ are initialised with small random weights, and inputs are typically normalised. In this regime, each output component behaves approximately like a draw from $\mathcal{N}(0, 1)$. The analysis is clean and the conclusion is exact under this assumption; in practice, it holds approximately.

### Numerical check

Our running example has $q_1 = 1.2$, $q_2 = -0.4$, $k_1 = 0.6$, $k_2 = 0.9$ — four draws from $\mathcal{N}(0,1)$. Their sample mean is $(1.2 - 0.4 + 0.6 + 0.9)/4 = 2.3/4 = 0.575$, not exactly zero. That is expected: with only 4 samples, the sample mean will not match the theoretical mean of zero. The **law of large numbers** guarantees convergence to zero as the sample count grows.

---

## 4. Variance

We need a precise measure of how spread out a random variable is around its mean. **Variance** answers: on average, how far does $X$ land from $\mathbb{E}[X]$?

If $\mu = \mathbb{E}[X]$, a natural measure of spread is the average squared deviation from the mean:

$$
\text{Var}(X) = \mathbb{E}\!\left[(X - \mu)^2\right]
$$

We square the deviation so that positive and negative deviations do not cancel each other out.

**Computational formula.** We expand $(X - \mu)^2$ using the **binomial expansion** $(a - b)^2 = a^2 - 2ab + b^2$:

$$
\text{Var}(X) = \mathbb{E}[X^2 - 2\mu X + \mu^2]
$$

By **linearity of expectation** (derived in the RL prerequisites), expectation distributes over the sum:

$$
= \mathbb{E}[X^2] - 2\mu\,\mathbb{E}[X] + \mu^2
$$

Since $\mu = \mathbb{E}[X]$, the last two terms combine: $-2\mu\,\mathbb{E}[X] + \mu^2 = -2\mu^2 + \mu^2 = -\mu^2$. Therefore:

$$
\boxed{\text{Var}(X) = \mathbb{E}[X^2] - \left(\mathbb{E}[X]\right)^2}
$$

This is the **computational formula for variance**. It separates the second moment $\mathbb{E}[X^2]$ from the squared first moment $(\mathbb{E}[X])^2$.

### Numerical check for $X \sim \mathcal{N}(0, 1)$

For $X \sim \mathcal{N}(0, 1)$: $\mathbb{E}[X] = 0$ and $\text{Var}(X) = 1$ (by definition). Plugging into the formula:

$$
1 = \mathbb{E}[X^2] - 0^2 \implies \mathbb{E}[X^2] = 1
$$

This confirms the fact stated in Section 3: the expected square of a standard normal variable is exactly 1. We will use this in Sections 5 and 6.

### Interpretation

Variance is not the typical deviation — it is the typical *squared* deviation. The typical deviation is the **standard deviation** $\text{Std}(X) = \sqrt{\text{Var}(X)}$, which has the same units as $X$. For $X \sim \mathcal{N}(0,1)$: $\text{Std}(X) = 1$. Values of $X$ typically fall within 1 unit of zero.

---

## 5. Independence and the Expected Value of a Product

Two random variables $X$ and $Y$ are **independent** if knowing the value of one gives no information about the other. For our vectors, the components $q_1, q_2, k_1, k_2$ are all independent of each other because each was drawn by a separate random process.

For independent random variables, the expected value of their product factors:

$$
\boxed{\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y] \quad \text{(for independent } X, Y\text{)}}
$$

This is the **multiplication rule for independent expectations**.

**Derivation.** For discrete random variables, expectation is a weighted sum over all outcomes. For a joint pair $(X, Y)$:

$$
\mathbb{E}[XY] = \sum_{x}\sum_{y} xy \cdot P(X = x,\, Y = y)
$$

Independence means $P(X = x, Y = y) = P(X = x) \cdot P(Y = y)$. Substituting:

$$
= \sum_{x}\sum_{y} xy \cdot P(X = x) \cdot P(Y = y)
$$

Since the sums are over independent indices, we factor the double sum using the **distributive law**:

$$
= \left(\sum_{x} x \cdot P(X = x)\right)\!\left(\sum_{y} y \cdot P(Y = y)\right) = \mathbb{E}[X]\,\mathbb{E}[Y]
$$

### Consequence for $q_i, k_i \sim \mathcal{N}(0,1)$ independently

Since $\mathbb{E}[q_i] = 0$ and $k_i$ is independent of $q_i$:

$$
\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0 \times 0 = 0
$$

Each component product $q_i k_i$ has mean zero. The dot product $\mathbf{q} \cdot \mathbf{k} = \sum_i q_i k_i$ therefore has mean zero too, by linearity of expectation. On average across many random draws, the dot product is centred at zero.

### Numerical check

Our values: $q_1 k_1 = 1.2 \times 0.6 = 0.72$ and $q_2 k_2 = (-0.4) \times 0.9 = -0.36$. These are two particular values of the random variable $q_i k_i$. Their sample mean is $(0.72 - 0.36)/2 = 0.18$, not zero. With only two samples, this is expected. The formula tells us that with many such pairs, the average converges to zero.

---

## 6. Variance of a Product of Two Independent Standard Normals

We now compute $\text{Var}(q_i k_i)$ when $q_i, k_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$.

We apply the computational formula from Section 4:

$$
\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - \left(\mathbb{E}[q_i k_i]\right)^2
$$

From Section 5, $\mathbb{E}[q_i k_i] = 0$. The second term vanishes:

$$
\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] = \mathbb{E}[q_i^2\, k_i^2]
$$

Since $q_i$ and $k_i$ are independent, $q_i^2$ and $k_i^2$ are also independent — any function of independent variables remains independent. Applying the **multiplication rule for independent expectations**:

$$
\mathbb{E}[q_i^2\, k_i^2] = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2]
$$

From Section 4, $\mathbb{E}[X^2] = 1$ for $X \sim \mathcal{N}(0,1)$. Therefore:

$$
\boxed{\text{Var}(q_i k_i) = 1 \times 1 = 1}
$$

Each component product has variance exactly 1.

### Numerical check

The two component products in our running example are $q_1 k_1 = 0.72$ and $q_2 k_2 = -0.36$. Their sample mean is $0.18$ and their sample variance is:

$$
\frac{(0.72 - 0.18)^2 + (-0.36 - 0.18)^2}{2} = \frac{0.54^2 + (-0.54)^2}{2} = \frac{0.2916 + 0.2916}{2} = 0.2916
$$

The theoretical value is 1. With only 2 samples, the sample variance is unreliable — we need many draws for convergence. The derivation above gives the exact theoretical value.

---

## 7. Additivity of Variance for Independent Variables

When two independent random variables are added, their variances add. This is the **additivity of variance** (also known as the **Bienaymé formula**):

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) \quad \text{(for independent } X, Y\text{)}
$$

**Derivation.** Let $\mu_X = \mathbb{E}[X]$ and $\mu_Y = \mathbb{E}[Y]$. By definition:

$$
\text{Var}(X + Y) = \mathbb{E}\!\left[(X + Y - \mu_X - \mu_Y)^2\right]
$$

Write $X + Y - \mu_X - \mu_Y = (X - \mu_X) + (Y - \mu_Y)$ and expand the square using $(a + b)^2 = a^2 + 2ab + b^2$:

$$
= \mathbb{E}\!\left[(X - \mu_X)^2 + 2(X - \mu_X)(Y - \mu_Y) + (Y - \mu_Y)^2\right]
$$

By linearity of expectation:

$$
= \underbrace{\mathbb{E}[(X - \mu_X)^2]}_{\text{Var}(X)} + 2\,\mathbb{E}[(X - \mu_X)(Y - \mu_Y)] + \underbrace{\mathbb{E}[(Y - \mu_Y)^2]}_{\text{Var}(Y)}
$$

For the middle term: since $X$ and $Y$ are independent, so are $(X - \mu_X)$ and $(Y - \mu_Y)$ — subtracting a constant does not affect independence. By the multiplication rule:

$$
\mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[X - \mu_X]\,\mathbb{E}[Y - \mu_Y] = 0 \times 0 = 0
$$

since $\mathbb{E}[X - \mu_X] = \mathbb{E}[X] - \mu_X = 0$. The cross term vanishes exactly, leaving:

$$
\boxed{\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)}
$$

By induction — applying the formula repeatedly to each new summand — this extends to any finite number of independent terms:

$$
\text{Var}\!\left(\sum_{i=1}^{d} Z_i\right) = \sum_{i=1}^{d} \text{Var}(Z_i)
$$

### Numerical check with $d = 2$

In our running example, $\mathbf{q} \cdot \mathbf{k} = q_1 k_1 + q_2 k_2$. Both terms $q_1 k_1$ and $q_2 k_2$ are independent (they involve separate draws of $q$ and $k$ components). Additivity gives:

$$
\text{Var}(q_1 k_1 + q_2 k_2) = \text{Var}(q_1 k_1) + \text{Var}(q_2 k_2) = 1 + 1 = 2
$$

We can verify the formula directly: $\text{Std}(\mathbf{q}\cdot\mathbf{k}) = \sqrt{2} \approx 1.414$. Our observed dot product $0.36$ is about $0.36/1.414 \approx 0.25$ standard deviations from zero — a perfectly ordinary draw.

---

## 8. The Variance of a Dot Product

We now have every piece. Let us put them together.

The dot product of two $d_k$-dimensional vectors is a sum of $d_k$ independent component products:

$$
\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i
$$

The terms $q_1 k_1, q_2 k_2, \ldots, q_{d_k} k_{d_k}$ are independent of each other, because all $2d_k$ components are drawn independently. Applying the **Bienaymé formula**:

$$
\text{Var}(\mathbf{q} \cdot \mathbf{k}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i)
$$

From Section 6, each $\text{Var}(q_i k_i) = 1$. There are $d_k$ terms in the sum:

$$
\boxed{\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k \cdot 1 = d_k}
$$

The standard deviation is $\sqrt{d_k}$.

### Numerical check with $d_k = 2$

$\text{Var}(\mathbf{q} \cdot \mathbf{k}) = 2$. Our running example gave $\mathbf{q} \cdot \mathbf{k} = 0.36$, which is one draw from a distribution with mean 0 and standard deviation $\sqrt{2} \approx 1.414$. ✓

### Why this matters: the $\sqrt{d_k}$ scaling

At the head dimension used in the Transformer — $d_k = 64$ — the standard deviation of each dot product is $\sqrt{64} = 8$. The inputs to the softmax span a range of roughly $\pm 24$ (three standard deviations on either side of zero). When $e^{24} \approx 2.6 \times 10^{10}$ and $e^0 = 1$, the softmax concentrates all weight on the single largest score — it degenerates to a near-hard argmax. The gradient of the softmax is nearly zero in this regime, and training stalls. This is the **vanishing gradient problem** in the softmax.

Dividing by $\sqrt{d_k}$ before the softmax fixes this. The scaled score is $(\mathbf{q} \cdot \mathbf{k}) / \sqrt{d_k}$. When a constant $c$ divides a random variable, its variance is scaled by $1/c^2$ — this is the **scaling rule for variance**. Applying it:

$$
\text{Var}\!\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{\text{Var}(\mathbf{q} \cdot \mathbf{k})}{(\sqrt{d_k})^2} = \frac{d_k}{d_k} = 1
$$

The scaled scores have exactly unit variance and unit standard deviation. The softmax inputs are well-conditioned, gradients are healthy, and training proceeds.

### Numerical check of the scaling rule

With $d_k = 2$: scaled score $= 0.36 / \sqrt{2} \approx 0.254$. Variance of scaled scores $= 2 / 2 = 1$. Standard deviation $= 1$. The value $0.254$ is now $0.254$ standard deviations from zero — comfortably in the softmax's well-conditioned regime. ✓

---

## Summary

All seven tools were built from the same two vectors $\mathbf{q} = (1.2, -0.4)$ and $\mathbf{k} = (0.6, 0.9)$.

The **tanh function** $\tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})$ maps any real input to the interval $(-1, 1)$; we computed $\tanh(1.8) \approx 0.9468$ as a representative alignment score. The **dot product** $\mathbf{q}\cdot\mathbf{k} = \sum_i q_i k_i$ is a scalar measure of directional similarity; our two vectors gave $0.36$. The **standard normal distribution** $\mathcal{N}(0,1)$ has mean zero and variance one, with the key property $\mathbb{E}[X^2] = 1$. **Variance** $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ measures squared spread around the mean — a result derived in three lines from the binomial expansion and linearity of expectation. For **independent** random variables, the multiplication rule gives $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$, which combined with $\mathbb{E}[X^2] = 1$ yields $\text{Var}(q_i k_i) = 1$ for each component product. Finally, the **Bienaymé formula** (additivity of variance for independent variables) chains these single-component results together: $\text{Var}(\mathbf{q}\cdot\mathbf{k}) = d_k \cdot 1 = d_k$, so the standard deviation is $\sqrt{d_k}$, so we divide by $\sqrt{d_k}$ to restore unit variance at the softmax input.

With these tools in hand, we are ready for [What Attention is Really Doing](/blog/attention-what-is-it-really), where attention is built from scratch starting from the failure of fixed-length context vectors, and [From Soft Alignment to Queries, Keys, and Values](/blog/attention-q-k-v-from-scratch), where the $\sqrt{d_k}$ scaling is derived using exactly the results established in Sections 3–8.

---

*Previous: [Mixture of Experts from Scratch — Part 3](/blog/mixture-of-experts-part-3)*  
*Next: [What Attention is Really Doing](/blog/attention-what-is-it-really)*
