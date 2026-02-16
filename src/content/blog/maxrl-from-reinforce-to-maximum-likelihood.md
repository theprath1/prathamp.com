---
title: "MaxRL: From REINFORCE to Maximum Likelihood"
description: "Why dividing by the number of successes instead of the batch size changes what your gradient estimator optimizes — and how this connects REINFORCE, maximum likelihood, and pass@k through one clean mathematical identity."
date: 2026-02-16
tags: [reinforcement-learning, machine-learning, policy-gradient, mathematics]
---

This post explains the core ideas behind the [MaxRL paper](https://arxiv.org/abs/2502.07834) — a surprisingly clean result that connects Reinforcement Learning, maximum likelihood training, and pass@k evaluation through a single mathematical identity. We will derive everything from scratch using one running example, the same way we built RL from the ground up in the previous posts.

If you haven't read [Mathematical Prerequisites for Reinforcement Learning](/blog/math-prerequisites-for-rl) and [Reinforcement Learning from Scratch](/blog/reinforcement-learning-from-scratch), start there — we will use the same tools (expected value, the log trick, Monte Carlo estimation, and the policy gradient) throughout this post.

---

## The Running Example

We will use one prompt for the entire post: $x$ = "2 + 3 = ?". Our model $m_\theta$ can generate three possible outputs when given this prompt:

- Output $z = A$: the correct answer "5"
- Output $z = B$: the wrong answer "4"
- Output $z = C$: the wrong answer "6"

The model assigns probabilities to each output (these depend on the parameters $\theta$):

$$
m_\theta(A \mid x) = 0.1, \quad m_\theta(B \mid x) = 0.6, \quad m_\theta(C \mid x) = 0.3
$$

There is a verifier $f(\cdot)$ that checks whether an output is correct. We define a binary reward:

$$
r(x, z) = \begin{cases} 1 & \text{if } z \text{ is correct} \\ 0 & \text{if } z \text{ is wrong} \end{cases}
$$

In our example: $r(x, A) = 1$, $r(x, B) = 0$, $r(x, C) = 0$. The model currently assigns only 10% probability to the correct answer — it is a weak model that usually gets this question wrong.

---

## What Are We Optimizing?

Fix one prompt $x$. The model has a probability of producing a correct answer, which we write as:

$$
p_\theta(x) = \sum_{z : r(x,z) = 1} m_\theta(z \mid x)
$$

In our example, only output $A$ is correct, so $p_\theta(x) = m_\theta(A \mid x) = 0.1$. This is the probability of success on a single try.

Different training objectives amount to different functions of this probability.

**RL objective.** Standard RL maximizes the probability of success directly:

$$
J_{RL}(x) = p_\theta(x)
$$

If $p_\theta(x) = 0.1$, then $J_{RL}(x) = 0.1$.

**ML objective.** Maximum likelihood maximizes the log of the probability:

$$
J_{ML}(x) = \log p_\theta(x)
$$

If $p_\theta(x) = 0.1$, then $J_{ML}(x) = \log(0.1) \approx -2.3$.

**MaxRL objective.** The paper defines a new family of objectives parameterized by $N$:

$$
J^{(N)}_{\text{MaxRL}}(x) = - \sum_{k=1}^{N} \frac{(1 - p_\theta(x))^k}{k}
$$

This is a truncated version of $\log p_\theta(x)$, and $N$ controls how many terms you keep. When $N = 1$ you get something proportional to the RL objective. As $N \to \infty$ you recover the ML objective exactly. The reason this truncation matters — and why $N$ corresponds to the number of samples you draw — is what the rest of this post will explain.

---

## pass@k and the Bridge to Logarithms

Before touching any gradients, we need one key mathematical connection. It starts with a simple idea: what if instead of asking the model once, you ask it $k$ times and accept if at least one answer is correct?

### pass@k

If the model has probability $p$ of being correct on a single try, then the probability that all $k$ tries fail is $(1 - p)^k$. So the probability that at least one try succeeds is:

$$
\text{pass@}k = 1 - (1 - p)^k
$$

In our example with $p = 0.1$: pass@1 is just $0.1$, pass@3 is $1 - 0.9^3 = 1 - 0.729 = 0.271$, and pass@10 is $1 - 0.9^{10} \approx 0.651$. Even a weak model starts looking decent if you give it enough tries.

We can also define the complement — the probability that all $k$ tries fail:

$$
\text{fail@}k = (1 - p)^k
$$

### The power series identity

There is a standard identity in mathematics:

$$
\log(1 - q) = - \sum_{k=1}^{\infty} \frac{q^k}{k} \quad \text{for } |q| < 1
$$

Now let $q = 1 - p$. Then $1 - q = p$, and:

$$
\log(p) = \log(1 - (1-p)) = - \sum_{k=1}^{\infty} \frac{(1-p)^k}{k}
$$

But we already said that $(1-p)^k = \text{fail@}k$. So:

$$
\log(p) = - \sum_{k=1}^{\infty} \frac{\text{fail@}k}{k}
$$

This is a surprising identity. It says that the log probability of success — which is the ML objective — secretly encodes information about failure probabilities across all possible numbers of attempts. Each term $\frac{\text{fail@}k}{k}$ measures how likely it is that $k$ independent tries all fail, weighted down by $1/k$.

### Numerical verification

Take $p = 0.1$, so $1 - p = 0.9$. The first few terms of the series are:

$$
k=1: \frac{0.9}{1} = 0.9, \qquad k=2: \frac{0.81}{2} = 0.405, \qquad k=3: \frac{0.729}{3} = 0.243, \qquad k=4: \frac{0.6561}{4} \approx 0.164
$$

The partial sums with the minus sign: $-0.9$, $-1.305$, $-1.548$, $-1.712$, and so on. The true value is $\log(0.1) \approx -2.303$. The series converges slowly because $p$ is small (the model is weak), but it does converge. For stronger models with $p$ closer to 1, the series converges much faster because $(1-p)^k$ shrinks rapidly.

---

## Differentiating the Series: Why ML Mixes pass@k Gradients

Now we take the derivative of $\log p$ with respect to the model parameters $\theta$. This is where the connection between ML and pass@k becomes concrete.

Start from the series:

$$
\log p_\theta(x) = - \sum_{k=1}^{\infty} \frac{(1-p_\theta(x))^k}{k}
$$

Focus on one term and differentiate:

$$
\nabla_\theta \left( -\frac{(1 - p_\theta(x))^k}{k} \right)
$$

Using the chain rule, the derivative of $(1 - p_\theta(x))^k$ is $k(1 - p_\theta(x))^{k-1} \cdot (-\nabla_\theta p_\theta(x))$. Multiplying by $-1/k$:

$$
-\frac{1}{k} \cdot k(1 - p_\theta(x))^{k-1}(-\nabla_\theta p_\theta(x)) = (1 - p_\theta(x))^{k-1} \nabla_\theta p_\theta(x)
$$

The $k$ in the numerator cancels the $k$ in the denominator. Summing over all $k$:

$$
\nabla_\theta \log p_\theta(x) = \sum_{k=1}^{\infty} (1 - p_\theta(x))^{k-1} \nabla_\theta p_\theta(x)
$$

Now recall that $\text{pass@}k = 1 - (1 - p_\theta(x))^k$. Differentiating:

$$
\nabla_\theta \text{pass@}k = k(1 - p_\theta(x))^{k-1} \nabla_\theta p_\theta(x)
$$

So $(1 - p_\theta(x))^{k-1} \nabla_\theta p_\theta(x) = \frac{1}{k} \nabla_\theta \text{pass@}k$. Substituting into the gradient:

$$
\boxed{\nabla_\theta \log p_\theta(x) = \sum_{k=1}^{\infty} \frac{1}{k} \nabla_\theta \text{pass@}k}
$$

This is the core identity. The ML gradient is an infinite weighted sum of pass@k gradients, where the weight on the $k$-th term is $1/k$.

Standard RL only optimizes $\nabla_\theta p_\theta(x) = \nabla_\theta \text{pass@}1$. It uses only the first-order information — how to improve single-try success. Maximum likelihood, on the other hand, simultaneously pushes the model to improve pass@1, pass@2, pass@3, and so on, with decreasing weights. ML is a richer training signal because it cares about multi-try behavior, not just single-try performance.

If we truncate the series at $N$ terms instead of going to infinity, we get the MaxRL objective:

$$
\nabla_\theta J^{(N)}_{\text{MaxRL}}(x) = \sum_{k=1}^{N} \frac{1}{k} \nabla_\theta \text{pass@}k = \left( \sum_{k=1}^{N} (1 - p_\theta(x))^{k-1} \right) \nabla_\theta p_\theta(x)
$$

This is a finite approximation to the ML gradient that keeps the first $N$ pass@k terms. The paper's key result is that there is a simple estimator — one you can compute from samples — whose expectation equals exactly this truncated gradient.

---

## Theorem 1: The ML Gradient Is the Conditional Expectation Over Successes

Before proving the main result about the estimator, we need a foundational fact about what the ML gradient actually looks like. This is Theorem 1 in the paper, and it says something elegant: to compute the gradient of $\log p_\theta(x)$, you only need to look at successful outputs.

### Setup

Define the success set $\mathcal{S} := \{ z : f(z) = y^*(x) \}$ — all outputs that are correct. In our example, $\mathcal{S} = \{A\}$. The probability of success is:

$$
p_\theta(x) = \sum_{z \in \mathcal{S}} m_\theta(z \mid x)
$$

### Derivation

We want $\nabla_\theta \log p_\theta(x)$. Start by differentiating $p_\theta(x)$:

$$
\nabla_\theta p_\theta(x) = \nabla_\theta \sum_{z \in \mathcal{S}} m_\theta(z \mid x) = \sum_{z \in \mathcal{S}} \nabla_\theta m_\theta(z \mid x)
$$

The gradient passes inside the sum because differentiation is linear. Now apply the log-derivative identity — the same log trick from the previous posts. We derived in Part 1 that $\nabla_\theta m = m \cdot \nabla_\theta \log m$, so:

$$
\nabla_\theta p_\theta(x) = \sum_{z \in \mathcal{S}} m_\theta(z \mid x) \nabla_\theta \log m_\theta(z \mid x)
$$

To get the gradient of $\log p_\theta(x)$ rather than $p_\theta(x)$, apply the chain rule for logarithms:

$$
\nabla_\theta \log p_\theta(x) = \frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x) = \frac{1}{p_\theta(x)} \sum_{z \in \mathcal{S}} m_\theta(z \mid x) \nabla_\theta \log m_\theta(z \mid x)
$$

Now recognize that $\frac{m_\theta(z \mid x)}{p_\theta(x)}$ is the conditional probability of output $z$ given that it is correct:

$$
m_\theta(z \mid x, \mathcal{S}) = \frac{m_\theta(z \mid x)}{p_\theta(x)} \quad \text{for } z \in \mathcal{S}
$$

This is just Bayes' rule. If $z \in \mathcal{S}$, then the event "$z$ and success" is the same as the event "$z$" (because choosing this $z$ automatically means success). So $\Pr(z \text{ and } \mathcal{S}) = m_\theta(z \mid x)$, and dividing by $\Pr(\mathcal{S}) = p_\theta(x)$ gives the conditional probability.

Substituting:

$$
\nabla_\theta \log p_\theta(x) = \sum_{z \in \mathcal{S}} m_\theta(z \mid x, \mathcal{S}) \nabla_\theta \log m_\theta(z \mid x)
$$

The right side is the definition of a conditional expectation — it sums a function of $z$ weighted by the conditional probability of $z$ given success. So:

$$
\boxed{\nabla_\theta \log p_\theta(x) = \mathbb{E}_{z \sim m_\theta(\cdot \mid x)} \left[ \nabla_\theta \log m_\theta(z \mid x) \ \middle|\ \mathcal{S} \right]}
$$

The ML gradient is the expected score function, conditioned on the output being correct. In plain language: to increase $\log p_\theta(x)$, look only at correct outputs and push up their log-probabilities on average. This is the mathematical reason why, in the estimator, we will average only over successes rather than over the entire batch.

### Numerical check

In our running example, $\mathcal{S} = \{A\}$ and $p_\theta(x) = 0.1$. The conditional distribution given success puts all its mass on $A$: $m_\theta(A \mid x, \mathcal{S}) = 1$. So the conditional expectation is just $\nabla_\theta \log m_\theta(A \mid x)$. Meanwhile, $\nabla_\theta \log p_\theta(x) = \frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x) = \frac{1}{0.1} \nabla_\theta m_\theta(A \mid x)$. Using the log trick, $\nabla_\theta m_\theta(A \mid x) = m_\theta(A \mid x) \nabla_\theta \log m_\theta(A \mid x) = 0.1 \cdot \nabla_\theta \log m_\theta(A \mid x)$. So $\frac{1}{0.1} \cdot 0.1 \cdot \nabla_\theta \log m_\theta(A \mid x) = \nabla_\theta \log m_\theta(A \mid x)$. Both sides match.

---

## Theorem 2: The Divide-by-K Estimator

This is the main result of the paper. It says that a simple modification to REINFORCE — dividing by the number of successes $K$ instead of the batch size $N$ — produces an unbiased estimator for the truncated MaxRL gradient. The number of samples $N$ is not just a variance knob; it literally determines how many terms of the infinite series you are optimizing.

### The estimator

Fix a prompt $x$. Sample $N$ outputs independently from the model:

$$
z_1, z_2, \ldots, z_N \sim m_\theta(\cdot \mid x)
$$

For each sample, compute:

- The success indicator: $r_i = \mathbf{1}\{f(z_i) = y^*(x)\}$, which is 1 if the output is correct and 0 otherwise.
- The score function: $S_i = \nabla_\theta \log m_\theta(z_i \mid x)$, the gradient of the log-probability with respect to parameters.
- The total number of successes: $K = \sum_{i=1}^{N} r_i$.

The paper's estimator is:

$$
g_N^b(x) = \begin{cases} \frac{1}{K} \sum_{i=1}^{N} r_i S_i & \text{if } K \geq 1 \\ 0 & \text{if } K = 0 \end{cases}
$$

Compare this to what REINFORCE would do. Standard REINFORCE averages over the entire batch: $\frac{1}{N} \sum_{i=1}^{N} r_i S_i$. The MaxRL estimator $g_N^b(x)$ instead averages only over the successful samples — the numerator $\sum r_i S_i$ picks out the score functions of correct outputs (since $r_i = 0$ zeroes out the wrong ones), and the denominator $K$ counts how many correct outputs there were.

### Concrete example

Suppose we sample $N = 10$ outputs from our model on the prompt "2 + 3 = ?". With $p_\theta(x) = 0.1$, we typically get 0 or 1 correct answer.

**Case: exactly one success.** Suppose only sample 7 is correct ($z_7 = A$), so $r_7 = 1$ and all other $r_i = 0$. Then $K = 1$, the numerator is $r_7 S_7 = S_7$, and:

- REINFORCE gives $S_7 / 10$ — the correct gradient signal diluted by a factor of 10 because nine failures are in the average.
- MaxRL gives $S_7 / 1 = S_7$ — the full gradient signal from the one success.

MaxRL amplifies the rare success by a factor of 10 in this case. This factor is not arbitrary — it corresponds to $1/p_\theta(x) = 1/0.1 = 10$, which is exactly the amplification factor in the ML gradient from Theorem 1.

**Case: two successes.** Suppose samples 3 and 9 are correct. Then $K = 2$, and:

- REINFORCE gives $(S_3 + S_9) / 10$.
- MaxRL gives $(S_3 + S_9) / 2$ — the average over the two successes only.

Again, MaxRL does not dilute the signal by counting failures in the denominator.

### The theorem

The claim is:

$$
\mathbb{E}[g_N^b(x)] = \nabla_\theta J^{(N)}_{\text{MaxRL}}(x)
$$

We already know the right side equals $\left( \sum_{k=1}^{N} (1 - p_\theta(x))^{k-1} \right) \nabla_\theta p_\theta(x)$ from differentiating the truncated series. So our job is to show the left side equals the same expression.

### Proof

**Step 1 — Symmetry.** Start from the definition and take the expectation:

$$
\mathbb{E}[g_N^b(x)] = \mathbb{E}\left[ \frac{1}{K} \sum_{i=1}^{N} r_i S_i \right] = \sum_{i=1}^{N} \mathbb{E}\left[ \frac{r_i S_i}{K} \right]
$$

All $N$ samples are i.i.d., so each term in the sum has the same expectation. Therefore:

$$
\mathbb{E}[g_N^b(x)] = N \cdot \mathbb{E}\left[ \frac{r_1 S_1}{K} \right]
$$

**Step 2 — Condition on sample 1.** If $r_1 = 0$, the numerator $r_1 S_1$ is zero, so only the event $r_1 = 1$ contributes:

$$
\mathbb{E}\left[ \frac{r_1 S_1}{K} \right] = \Pr(r_1 = 1) \cdot \mathbb{E}\left[ \frac{S_1}{K} \ \middle|\ r_1 = 1 \right] = p_\theta(x) \cdot \mathbb{E}\left[ \frac{S_1}{K} \ \middle|\ r_1 = 1 \right]
$$

**Step 3 — Split $K$ into sample 1 and the rest.** When $r_1 = 1$, the total number of successes is $K = 1 + K_{-1}$, where $K_{-1} = \sum_{i=2}^{N} r_i$ counts successes among the other $N - 1$ samples. Since the samples are independent, $S_1$ (which depends only on $z_1$) is independent of $K_{-1}$ (which depends only on $z_2, \ldots, z_N$). So the expectation factors:

$$
\mathbb{E}\left[ \frac{S_1}{1 + K_{-1}} \ \middle|\ r_1 = 1 \right] = \mathbb{E}[S_1 \mid r_1 = 1] \cdot \mathbb{E}\left[ \frac{1}{1 + K_{-1}} \right]
$$

**Step 4 — Recognize $p_\theta(x) \cdot \mathbb{E}[S_1 \mid r_1 = 1] = \nabla_\theta p_\theta(x)$.** This follows from the single-sample policy gradient identity. For one sample $z \sim m_\theta(\cdot \mid x)$ with $r = \mathbf{1}\{z \in \mathcal{S}\}$ and $S = \nabla_\theta \log m_\theta(z \mid x)$:

$$
\mathbb{E}[rS] = \sum_{z \in \mathcal{S}} m_\theta(z \mid x) \nabla_\theta \log m_\theta(z \mid x) = \sum_{z \in \mathcal{S}} \nabla_\theta m_\theta(z \mid x) = \nabla_\theta p_\theta(x)
$$

The first equality restricts to successes (since $r = 0$ elsewhere), the second uses the log trick in reverse, and the third uses the definition of $p_\theta(x)$. But also $\mathbb{E}[rS] = \Pr(r=1) \cdot \mathbb{E}[S \mid r=1] = p_\theta(x) \cdot \mathbb{E}[S \mid r=1]$. Combining:

$$
p_\theta(x) \cdot \mathbb{E}[S_1 \mid r_1 = 1] = \nabla_\theta p_\theta(x)
$$

Substituting into our expression:

$$
\mathbb{E}[g_N^b(x)] = N \cdot \nabla_\theta p_\theta(x) \cdot \mathbb{E}\left[ \frac{1}{1 + K_{-1}} \right]
$$

Everything now reduces to computing one scalar quantity: $\mathbb{E}\left[ \frac{1}{1 + K_{-1}} \right]$.

**Step 5 — Compute $\mathbb{E}\left[\frac{1}{1 + K_{-1}}\right]$ exactly.** Since each of the $N - 1$ remaining samples succeeds independently with probability $p_\theta(x)$, we have $K_{-1} \sim \text{Binomial}(N-1, p_\theta(x))$. Writing out the expectation:

$$
\mathbb{E}\left[ \frac{1}{1 + K_{-1}} \right] = \sum_{j=0}^{N-1} \frac{1}{1+j} \binom{N-1}{j} p_\theta(x)^j (1 - p_\theta(x))^{N-1-j}
$$

This sum looks unwieldy, but there is a clean trick. The factor $\frac{1}{1+j}$ can be written as an integral:

$$
\frac{1}{1+j} = \int_0^1 t^j \, dt
$$

This is easy to verify: $\int_0^1 t^j \, dt = \left[\frac{t^{j+1}}{j+1}\right]_0^1 = \frac{1}{j+1}$. Substituting into the sum and swapping the (finite) sum and integral:

$$
\mathbb{E}\left[ \frac{1}{1 + K_{-1}} \right] = \int_0^1 \sum_{j=0}^{N-1} \binom{N-1}{j} (p_\theta(x) \cdot t)^j (1 - p_\theta(x))^{N-1-j} \, dt
$$

The sum inside the integral is a binomial expansion — it equals $((1 - p_\theta(x)) + p_\theta(x) \cdot t)^{N-1}$. So:

$$
\mathbb{E}\left[ \frac{1}{1 + K_{-1}} \right] = \int_0^1 \left( (1 - p_\theta(x)) + p_\theta(x) \cdot t \right)^{N-1} dt
$$

To evaluate this integral, substitute $u = (1 - p_\theta(x)) + p_\theta(x) \cdot t$, so $du = p_\theta(x) \, dt$. When $t = 0$, $u = 1 - p_\theta(x)$. When $t = 1$, $u = 1$. The integral becomes:

$$
\frac{1}{p_\theta(x)} \int_{1 - p_\theta(x)}^{1} u^{N-1} \, du = \frac{1}{p_\theta(x)} \left[ \frac{u^N}{N} \right]_{1 - p_\theta(x)}^{1} = \frac{1 - (1 - p_\theta(x))^N}{N \cdot p_\theta(x)}
$$

**Step 6 — Combine everything.** Plugging back:

$$
\mathbb{E}[g_N^b(x)] = N \cdot \nabla_\theta p_\theta(x) \cdot \frac{1 - (1 - p_\theta(x))^N}{N \cdot p_\theta(x)}
$$

The $N$ cancels:

$$
\mathbb{E}[g_N^b(x)] = \frac{1 - (1 - p_\theta(x))^N}{p_\theta(x)} \cdot \nabla_\theta p_\theta(x)
$$

Now use the geometric series identity. For any $q \neq 1$, $\frac{1 - q^N}{1 - q} = \sum_{k=1}^{N} q^{k-1}$. Here $q = 1 - p_\theta(x)$ and $1 - q = p_\theta(x)$, so:

$$
\frac{1 - (1 - p_\theta(x))^N}{p_\theta(x)} = \sum_{k=1}^{N} (1 - p_\theta(x))^{k-1}
$$

Therefore:

$$
\boxed{\mathbb{E}[g_N^b(x)] = \left( \sum_{k=1}^{N} (1 - p_\theta(x))^{k-1} \right) \nabla_\theta p_\theta(x) = \nabla_\theta J^{(N)}_{\text{MaxRL}}(x)}
$$

### What this means

The divide-by-$K$ estimator $g_N^b(x)$ is an unbiased estimator for the gradient of the truncated MaxRL objective $J^{(N)}_{\text{MaxRL}}(x)$. The number of samples $N$ is not just a computational budget — it determines how many terms of the infinite series $\log p_\theta(x) = -\sum_{k=1}^{\infty} \frac{(1-p)^k}{k}$ you are unbiasedly optimizing. With $N = 1$, you optimize only $\nabla_\theta \text{pass@}1$ (standard RL). With $N = 10$, you optimize a weighted combination of $\nabla_\theta \text{pass@}1$ through $\nabla_\theta \text{pass@}10$. As $N \to \infty$, you recover the full ML gradient.

The intuition for why $N$ shows up as the truncation level is that your batch of $N$ samples can directly witness events like "first success happens at try $k$" only up to $k = N$. The estimator naturally encodes pass@$k$ information for $k \leq N$ because those are the multi-try success events that $N$ samples can represent.

---

## Variance Reduction: The Control Variate

The estimator $g_N^b(x)$ is unbiased, but it can have high variance — especially when $p_\theta(x)$ is small and most batches contain zero successes. The paper introduces a modified estimator that reduces this variance without changing the expected value.

### The modified estimator

Define:

$$
g_N^e(x) = \frac{1}{K} \sum_{i=1}^{N} r_i S_i - \frac{1}{N} \sum_{i=1}^{N} S_i
$$

The first term is the original divide-by-$K$ estimator. The second term is the average of all score functions across the entire batch — both successes and failures. This second term is the **control variate**.

### Why the control variate has expectation zero

We need to show that $\mathbb{E}\left[\frac{1}{N} \sum_{i=1}^{N} S_i\right] = 0$. By linearity, it suffices to show $\mathbb{E}[S_i] = 0$ for a single sample.

Take one sample $z \sim m_\theta(\cdot \mid x)$ and expand the expectation of $S = \nabla_\theta \log m_\theta(z \mid x)$:

$$
\mathbb{E}[S] = \sum_z m_\theta(z \mid x) \nabla_\theta \log m_\theta(z \mid x)
$$

Using the log trick in reverse, $m_\theta(z \mid x) \nabla_\theta \log m_\theta(z \mid x) = \nabla_\theta m_\theta(z \mid x)$. So:

$$
\mathbb{E}[S] = \sum_z \nabla_\theta m_\theta(z \mid x) = \nabla_\theta \sum_z m_\theta(z \mid x) = \nabla_\theta(1) = 0
$$

The key step is that the probabilities $m_\theta(z \mid x)$ sum to 1 for any $\theta$ — this is the normalization constraint of a probability distribution. Differentiating a constant gives zero. This is the same score function identity we used in the REINFORCE derivation: the expected score under any distribution is always zero.

Therefore $\mathbb{E}\left[\frac{1}{N} \sum_{i=1}^{N} S_i\right] = 0$, and:

$$
\mathbb{E}[g_N^e(x)] = \mathbb{E}[g_N^b(x)] - 0 = \nabla_\theta J^{(N)}_{\text{MaxRL}}(x)
$$

The modified estimator is still unbiased for the same truncated objective.

### Why subtracting a zero-mean term helps

This is a general principle in statistics. If you have an estimator $G$ of some quantity and you subtract a zero-mean random variable $V$ that is correlated with $G$, the variance of $G - V$ can be smaller than the variance of $G$ alone. The formula is:

$$
\text{Var}(G - V) = \text{Var}(G) - 2 \text{Cov}(G, V) + \text{Var}(V)
$$

If $G$ and $V$ are positively correlated — meaning they tend to fluctuate in the same direction — then the covariance term is positive, and the reduction from $-2\text{Cov}(G, V)$ can outweigh the addition of $\text{Var}(V)$, making the overall variance smaller.

The intuition for why $\frac{1}{N} \sum S_i$ correlates with $\frac{1}{K} \sum r_i S_i$ is that both terms are driven by the same underlying samples. When the batch happens to contain outputs with large score function magnitudes, both the weighted (success-only) average and the unweighted (all-sample) average tend to be large. Subtracting the unweighted average removes some of this common fluctuation, leaving a cleaner signal.

### Compact form

The paper also writes the estimator in a combined form:

$$
g_N^e(x) = \sum_{i=1}^{N} \left( \frac{r_i}{K} - \frac{1}{N} \right) S_i
$$

This follows from simple algebra: factor $S_i$ out of both terms. The weight $\frac{r_i}{K} - \frac{1}{N}$ is positive for successful samples (they get upweighted) and negative for failed samples (they get slightly downweighted). This is reminiscent of a baseline in REINFORCE — except here the "baseline" is $1/N$ rather than an estimated value function.

---

## The Full Picture

Let's step back and see how all the pieces fit together.

Standard RL (REINFORCE) computes $\frac{1}{N} \sum r_i S_i$. This estimates $\nabla_\theta p_\theta(x)$ — the gradient of pass@1. It treats each sample equally and only cares about single-try success.

Maximum likelihood computes $\frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x)$, which is equivalent to $\mathbb{E}[S \mid \text{success}]$. This implicitly optimizes a weighted combination of all pass@$k$ gradients. But it requires knowing $p_\theta(x)$ exactly, which we don't have.

MaxRL bridges the gap. By dividing by $K$ (the observed number of successes) instead of $N$ (the batch size), we get an estimator that is computable from samples alone — no knowledge of $p_\theta(x)$ required — and whose expectation equals the gradient of a truncated approximation to $\log p_\theta(x)$. The truncation level equals $N$, the number of samples. More samples means a closer approximation to ML, which means richer multi-try training signal.

The control variate (subtracting the average score) keeps the estimator unbiased while reducing the variance that comes from the randomness of sampling. The result is a practical algorithm: sample $N$ outputs, check which ones are correct, average the score functions of the correct ones, subtract the average score function of all outputs, and use that as your gradient estimate.

The paper's central message is that REINFORCE and maximum likelihood are not separate training paradigms — they are endpoints of a single spectrum parameterized by $N$. At $N = 1$, you get REINFORCE. At $N = \infty$, you get ML. And for any finite $N$ in between, you get a well-defined objective $J^{(N)}_{\text{MaxRL}}$ with a simple, unbiased gradient estimator. The choice of $N$ lets you smoothly trade off between the computational cost of generating more samples and the richness of the training signal you extract from them.

---

## References

1. Blondel, M., Roulet, V., Sessa, P. G., & Thomson, M. (2025). *MaxRL: A Unified Framework for Maximum Likelihood and Reinforcement Learning.* arXiv:2502.07834.
