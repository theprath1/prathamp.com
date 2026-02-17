---
title: "MaxRL: From REINFORCE to Maximum Likelihood"
description: "Why dividing by the number of successes instead of the batch size changes what your gradient estimator optimizes — and how this connects REINFORCE, maximum likelihood, and pass@k through one clean mathematical identity."
date: 2026-02-16
tags: [reinforcement-learning, machine-learning, policy-gradient, mathematics]
---

This post explains the core ideas behind the [MaxRL paper](https://arxiv.org/abs/2602.02710) — a surprisingly clean result that connects Reinforcement Learning, maximum likelihood training, and pass@k evaluation through a single mathematical identity. We will derive everything from scratch using one running example, the same way we built RL from the ground up in the previous posts.

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

There is a standard identity in mathematics known as the **Maclaurin expansion** (Taylor series expanded around 0) of $\log(1 - q)$:

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

Focus on one term and differentiate. We want:

$$
\nabla_\theta \left( -\frac{(1 - p_\theta(x))^k}{k} \right)
$$

First, differentiate the inner expression $(1 - p_\theta(x))^k$ using the chain rule. Let $u = 1 - p_\theta(x)$, so we need $\nabla_\theta u^k$. The chain rule gives:

$$
\nabla_\theta (1 - p_\theta(x))^k = k(1 - p_\theta(x))^{k-1} \cdot \nabla_\theta(1 - p_\theta(x)) = k(1 - p_\theta(x))^{k-1} \cdot (-\nabla_\theta p_\theta(x))
$$

Now multiply by the outer factor $-1/k$:

$$
\nabla_\theta \left( -\frac{(1 - p_\theta(x))^k}{k} \right) = -\frac{1}{k} \cdot k(1 - p_\theta(x))^{k-1} \cdot (-\nabla_\theta p_\theta(x))
$$

The $k$ in the numerator from the chain rule cancels the $k$ in the denominator from the $1/k$ factor. The two minus signs also cancel. What remains is:

$$
= (1 - p_\theta(x))^{k-1} \nabla_\theta p_\theta(x)
$$

Now sum over all $k$ from 1 to $\infty$:

$$
\nabla_\theta \log p_\theta(x) = \sum_{k=1}^{\infty} (1 - p_\theta(x))^{k-1} \nabla_\theta p_\theta(x)
$$

We can connect each term in this sum to pass@$k$. Recall that $\text{pass@}k = 1 - (1 - p_\theta(x))^k$. Differentiating this with the same chain rule approach:

$$
\nabla_\theta \text{pass@}k = \nabla_\theta \left(1 - (1 - p_\theta(x))^k\right) = -k(1 - p_\theta(x))^{k-1} \cdot (-\nabla_\theta p_\theta(x))
$$

$$
= k(1 - p_\theta(x))^{k-1} \nabla_\theta p_\theta(x)
$$

From this we can solve for the factor that appears in our gradient sum:

$$
(1 - p_\theta(x))^{k-1} \nabla_\theta p_\theta(x) = \frac{1}{k} \nabla_\theta \text{pass@}k
$$

Substituting this into the gradient:

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

The gradient passes inside the sum because differentiation is linear. Now apply the log-derivative identity — the same log trick from the previous posts. For any function $m_\theta$:

$$
\nabla_\theta \log m_\theta(z \mid x) = \frac{1}{m_\theta(z \mid x)} \nabla_\theta m_\theta(z \mid x)
$$

Multiply both sides by $m_\theta(z \mid x)$:

$$
m_\theta(z \mid x) \cdot \nabla_\theta \log m_\theta(z \mid x) = \nabla_\theta m_\theta(z \mid x)
$$

This lets us replace each $\nabla_\theta m_\theta(z \mid x)$ in the sum:

$$
\nabla_\theta p_\theta(x) = \sum_{z \in \mathcal{S}} m_\theta(z \mid x) \nabla_\theta \log m_\theta(z \mid x)
$$

We want $\nabla_\theta \log p_\theta(x)$, not $\nabla_\theta p_\theta(x)$. The chain rule for logarithms says $\nabla_\theta \log u = \frac{1}{u} \nabla_\theta u$, so:

$$
\nabla_\theta \log p_\theta(x) = \frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x)
$$

Now substitute the expression for $\nabla_\theta p_\theta(x)$ that we just derived:

$$
= \frac{1}{p_\theta(x)} \sum_{z \in \mathcal{S}} m_\theta(z \mid x) \nabla_\theta \log m_\theta(z \mid x)
$$

We can push the $\frac{1}{p_\theta(x)}$ inside the sum and group it with $m_\theta(z \mid x)$:

$$
= \sum_{z \in \mathcal{S}} \frac{m_\theta(z \mid x)}{p_\theta(x)} \nabla_\theta \log m_\theta(z \mid x)
$$

Now recognize what $\frac{m_\theta(z \mid x)}{p_\theta(x)}$ means. The definition of conditional probability says:

$$
\Pr(z \mid \mathcal{S}) = \frac{\Pr(z \text{ and } \mathcal{S})}{\Pr(\mathcal{S})}
$$

If $z \in \mathcal{S}$, then choosing output $z$ automatically means success, so the event "$z$ and success" is just the event "$z$". Therefore $\Pr(z \text{ and } \mathcal{S}) = m_\theta(z \mid x)$. And $\Pr(\mathcal{S}) = p_\theta(x)$. So:

$$
m_\theta(z \mid x, \mathcal{S}) = \frac{m_\theta(z \mid x)}{p_\theta(x)} \quad \text{for } z \in \mathcal{S}
$$

Substituting this into our expression:

$$
\nabla_\theta \log p_\theta(x) = \sum_{z \in \mathcal{S}} m_\theta(z \mid x, \mathcal{S}) \nabla_\theta \log m_\theta(z \mid x)
$$

The right side is the definition of a conditional expectation — it sums a function of $z$ weighted by the conditional probability of $z$ given success. So:

$$
\boxed{\nabla_\theta \log p_\theta(x) = \mathbb{E}_{z \sim m_\theta(\cdot \mid x)} \left[ \nabla_\theta \log m_\theta(z \mid x) \ \middle|\ \mathcal{S} \right]}
$$

The ML gradient is the expected score function, conditioned on the output being correct. In plain language: to increase $\log p_\theta(x)$, look only at correct outputs and push up their log-probabilities on average. This is the mathematical reason why, in the estimator, we will average only over successes rather than over the entire batch.

### Numerical check

In our running example, $\mathcal{S} = \{A\}$ and $p_\theta(x) = 0.1$. Let's verify both sides of the boxed identity match.

**Right side (conditional expectation).** The conditional distribution given success puts all its mass on $A$, because $A$ is the only correct output:

$$
m_\theta(A \mid x, \mathcal{S}) = \frac{m_\theta(A \mid x)}{p_\theta(x)} = \frac{0.1}{0.1} = 1
$$

So the conditional expectation has only one term:

$$
\mathbb{E}[\nabla_\theta \log m_\theta(z \mid x) \mid \mathcal{S}] = 1 \cdot \nabla_\theta \log m_\theta(A \mid x) = \nabla_\theta \log m_\theta(A \mid x)
$$

**Left side ($\nabla_\theta \log p_\theta(x)$).** Start from the chain rule:

$$
\nabla_\theta \log p_\theta(x) = \frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x) = \frac{1}{0.1} \nabla_\theta m_\theta(A \mid x)
$$

Now apply the log trick to rewrite $\nabla_\theta m_\theta(A \mid x)$:

$$
\nabla_\theta m_\theta(A \mid x) = m_\theta(A \mid x) \cdot \nabla_\theta \log m_\theta(A \mid x) = 0.1 \cdot \nabla_\theta \log m_\theta(A \mid x)
$$

Substituting back:

$$
\nabla_\theta \log p_\theta(x) = \frac{1}{0.1} \cdot 0.1 \cdot \nabla_\theta \log m_\theta(A \mid x) = \nabla_\theta \log m_\theta(A \mid x)
$$

Both sides give $\nabla_\theta \log m_\theta(A \mid x)$. The identity checks out.

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

**Step 2 — Condition on sample 1.** We split the expectation into two cases using the law of total expectation:

$$
\mathbb{E}\left[ \frac{r_1 S_1}{K} \right] = \Pr(r_1 = 1) \cdot \mathbb{E}\left[ \frac{r_1 S_1}{K} \ \middle|\ r_1 = 1 \right] + \Pr(r_1 = 0) \cdot \mathbb{E}\left[ \frac{r_1 S_1}{K} \ \middle|\ r_1 = 0 \right]
$$

When $r_1 = 0$, the numerator $r_1 S_1 = 0 \cdot S_1 = 0$, so the entire second term vanishes. When $r_1 = 1$, we have $r_1 S_1 = S_1$. So:

$$
\mathbb{E}\left[ \frac{r_1 S_1}{K} \right] = \Pr(r_1 = 1) \cdot \mathbb{E}\left[ \frac{S_1}{K} \ \middle|\ r_1 = 1 \right]
$$

Since $\Pr(r_1 = 1) = p_\theta(x)$:

$$
= p_\theta(x) \cdot \mathbb{E}\left[ \frac{S_1}{K} \ \middle|\ r_1 = 1 \right]
$$

**Step 3 — Split $K$ into sample 1 and the rest.** When $r_1 = 1$, the total number of successes is $K = 1 + K_{-1}$, where $K_{-1} = \sum_{i=2}^{N} r_i$ counts successes among the other $N - 1$ samples. Since the samples are independent, $S_1$ (which depends only on $z_1$) is independent of $K_{-1}$ (which depends only on $z_2, \ldots, z_N$). So the expectation factors:

$$
\mathbb{E}\left[ \frac{S_1}{1 + K_{-1}} \ \middle|\ r_1 = 1 \right] = \mathbb{E}[S_1 \mid r_1 = 1] \cdot \mathbb{E}\left[ \frac{1}{1 + K_{-1}} \right]
$$

**Step 4 — Recognize $p_\theta(x) \cdot \mathbb{E}[S_1 \mid r_1 = 1] = \nabla_\theta p_\theta(x)$.** This follows from the single-sample policy gradient identity, which we derive now. Take one sample $z \sim m_\theta(\cdot \mid x)$ with $r = \mathbf{1}\{z \in \mathcal{S}\}$ and $S = \nabla_\theta \log m_\theta(z \mid x)$. Write out $\mathbb{E}[rS]$ as a sum over all outputs:

$$
\mathbb{E}[rS] = \sum_z m_\theta(z \mid x) \cdot r(x, z) \cdot \nabla_\theta \log m_\theta(z \mid x)
$$

Since $r(x, z) = 0$ for $z \notin \mathcal{S}$, the sum reduces to correct outputs only:

$$
= \sum_{z \in \mathcal{S}} m_\theta(z \mid x) \cdot \nabla_\theta \log m_\theta(z \mid x)
$$

Apply the log trick in reverse — replace $m \cdot \nabla \log m$ with $\nabla m$:

$$
= \sum_{z \in \mathcal{S}} \nabla_\theta m_\theta(z \mid x)
$$

Pull the gradient outside the sum:

$$
= \nabla_\theta \sum_{z \in \mathcal{S}} m_\theta(z \mid x) = \nabla_\theta p_\theta(x)
$$

So we have established that $\mathbb{E}[rS] = \nabla_\theta p_\theta(x)$. But we can also decompose $\mathbb{E}[rS]$ using the law of total expectation:

$$
\mathbb{E}[rS] = \Pr(r = 1) \cdot \mathbb{E}[S \mid r = 1] + \Pr(r = 0) \cdot \mathbb{E}[0 \cdot S \mid r = 0]
$$

The second term vanishes because $r = 0$ zeroes it out. So:

$$
\mathbb{E}[rS] = p_\theta(x) \cdot \mathbb{E}[S \mid r = 1]
$$

Combining both expressions for $\mathbb{E}[rS]$:

$$
p_\theta(x) \cdot \mathbb{E}[S_1 \mid r_1 = 1] = \nabla_\theta p_\theta(x)
$$

Now substitute this, along with the result from Step 3, into our running expression:

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

The sum inside the integral has the form $\sum_{j=0}^{n} \binom{n}{j} a^j b^{n-j}$ with $n = N-1$, $a = p_\theta(x) \cdot t$, and $b = 1 - p_\theta(x)$. By the binomial theorem, this equals $(a + b)^n = ((1 - p_\theta(x)) + p_\theta(x) \cdot t)^{N-1}$. So:

$$
\mathbb{E}\left[ \frac{1}{1 + K_{-1}} \right] = \int_0^1 \left( (1 - p_\theta(x)) + p_\theta(x) \cdot t \right)^{N-1} dt
$$

To evaluate this integral, substitute $u = (1 - p_\theta(x)) + p_\theta(x) \cdot t$. Then $du = p_\theta(x) \, dt$, which means $dt = \frac{du}{p_\theta(x)}$. The limits change: when $t = 0$, $u = 1 - p_\theta(x)$; when $t = 1$, $u = (1 - p_\theta(x)) + p_\theta(x) = 1$. The integral becomes:

$$
\int_0^1 \left( (1 - p_\theta(x)) + p_\theta(x) \cdot t \right)^{N-1} dt = \frac{1}{p_\theta(x)} \int_{1 - p_\theta(x)}^{1} u^{N-1} \, du
$$

Now integrate $u^{N-1}$:

$$
= \frac{1}{p_\theta(x)} \left[ \frac{u^N}{N} \right]_{1 - p_\theta(x)}^{1} = \frac{1}{p_\theta(x)} \cdot \frac{1^N - (1 - p_\theta(x))^N}{N}
$$

Since $1^N = 1$:

$$
= \frac{1 - (1 - p_\theta(x))^N}{N \cdot p_\theta(x)}
$$

**Step 6 — Combine everything.** Plugging back:

$$
\mathbb{E}[g_N^b(x)] = N \cdot \nabla_\theta p_\theta(x) \cdot \frac{1 - (1 - p_\theta(x))^N}{N \cdot p_\theta(x)}
$$

The $N$ cancels:

$$
\mathbb{E}[g_N^b(x)] = \frac{1 - (1 - p_\theta(x))^N}{p_\theta(x)} \cdot \nabla_\theta p_\theta(x)
$$

Now use the finite geometric series identity. For any $q \neq 1$:

$$
1 + q + q^2 + \cdots + q^{N-1} = \sum_{k=1}^{N} q^{k-1} = \frac{1 - q^N}{1 - q}
$$

Set $q = 1 - p_\theta(x)$, so $1 - q = p_\theta(x)$:

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
g_N^e(x) = g_N^b(x) - \frac{1}{N} \sum_{i=1}^{N} S_i
= \begin{cases}
\frac{1}{K} \sum_{i=1}^{N} r_i S_i - \frac{1}{N} \sum_{i=1}^{N} S_i & \text{if } K \geq 1 \\
- \frac{1}{N} \sum_{i=1}^{N} S_i & \text{if } K = 0
\end{cases}
$$

The first term is the original divide-by-$K$ estimator. The second term is the average of all score functions across the entire batch — both successes and failures. This second term is the **control variate**.

### Why the control variate has expectation zero

We need to show that $\mathbb{E}\left[\frac{1}{N} \sum_{i=1}^{N} S_i\right] = 0$. By linearity, it suffices to show $\mathbb{E}[S_i] = 0$ for a single sample.

Take one sample $z \sim m_\theta(\cdot \mid x)$ and expand the expectation of $S = \nabla_\theta \log m_\theta(z \mid x)$:

$$
\mathbb{E}[S] = \sum_z m_\theta(z \mid x) \nabla_\theta \log m_\theta(z \mid x)
$$

Now apply the log trick in reverse to each term. Recall that $m \cdot \nabla \log m = \nabla m$:

$$
\mathbb{E}[S] = \sum_z \nabla_\theta m_\theta(z \mid x)
$$

Pull the gradient outside the sum (differentiation is linear):

$$
= \nabla_\theta \left( \sum_z m_\theta(z \mid x) \right)
$$

The sum $\sum_z m_\theta(z \mid x)$ equals 1 for any value of $\theta$ — this is the normalization constraint of a probability distribution. Differentiating a constant gives zero:

$$
= \nabla_\theta(1) = 0
$$

So $\mathbb{E}[S] = 0$. This is the same score function identity we used in the REINFORCE derivation: the expected score under any distribution is always zero.

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

The paper also writes the estimator in a combined form (for $K \geq 1$):

$$
g_N^e(x) = \sum_{i=1}^{N} \left( \frac{r_i}{K} - \frac{1}{N} \right) S_i
$$

When $K = 0$, use the piecewise definition above (equivalently, adopt the convention $\frac{r_i}{K} := 0$ in that case).

This follows from distributing the sums:

$$
\frac{1}{K} \sum_{i=1}^{N} r_i S_i - \frac{1}{N} \sum_{i=1}^{N} S_i = \sum_{i=1}^{N} \frac{r_i}{K} S_i - \sum_{i=1}^{N} \frac{1}{N} S_i = \sum_{i=1}^{N} \left( \frac{r_i}{K} - \frac{1}{N} \right) S_i
$$

The weight $\frac{r_i}{K} - \frac{1}{N}$ is positive for successful samples (they get upweighted) and negative for failed samples (they get slightly downweighted). This is reminiscent of a baseline in REINFORCE — except here the "baseline" is $1/N$ rather than an estimated value function.

---

## The Unified Weight Function View

There is another way to see the relationship between REINFORCE, ML, and MaxRL — one that makes the conceptual difference between the three methods completely transparent. All three objectives produce gradients that can be written in the form:

$$
\nabla_\theta J = \mathbb{E}_x \left[ w(p_\theta(x)) \cdot \nabla_\theta p_\theta(x) \right]
$$

Here $p_\theta(x)$ is the single-sample pass rate, $\nabla_\theta p_\theta(x)$ is the basic REINFORCE direction that increases success probability, and $w(p)$ is a weight function that depends only on how hard the prompt currently is. The gradient direction is the same across all three methods — it is always $\nabla_\theta p_\theta(x)$, the raw signal for improving single-try success. What differs is how strongly each prompt contributes to the overall update, depending on its current difficulty.

For REINFORCE, the objective is $J_{RL} = \mathbb{E}_x[p_\theta(x)]$, which differentiates to:

$$
\nabla J_{RL} = \mathbb{E}_x[\nabla p_\theta(x)]
$$

The weight function is $w(p) = 1$ — all prompts contribute equally regardless of whether they are easy or hard.

For maximum likelihood, the objective is $J_{ML} = \mathbb{E}_x[\log p_\theta(x)]$, and differentiating using the chain rule $\nabla \log p = \frac{1}{p} \nabla p$ gives:

$$
\nabla J_{ML} = \mathbb{E}_x\left[\frac{1}{p_\theta(x)} \nabla p_\theta(x)\right]
$$

The weight function is $w(p) = 1/p$, which means a prompt with $p = 0.1$ gets weight 10, while a prompt with $p = 0.01$ gets weight 100. ML aggressively amplifies the gradient signal from hard prompts.

For MaxRL with $N$ samples, we showed in Theorem 2 that:

$$
\mathbb{E}[g_N^b(x)] = \frac{1 - (1-p)^N}{p} \nabla_\theta p_\theta(x)
$$

So the weight function is $w(p) = \frac{1 - (1-p)^N}{p}$. When $N = 1$, this reduces to:

$$
w(p) = \frac{1 - (1-p)}{p} = \frac{p}{p} = 1
$$

recovering REINFORCE. As $N \to \infty$, $(1-p)^N \to 0$, so $w(p) \to 1/p$, recovering ML. For any finite $N$ in between, the weight function interpolates smoothly. Taking our running example with $p = 0.1$ and $N = 10$:

$$
w(0.1) = \frac{1 - 0.9^{10}}{0.1} = \frac{1 - 0.3487}{0.1} \approx 6.5
$$

Compare this to REINFORCE's weight of 1 and ML's weight of 10.

This unified view reveals that the entire paper reduces to a single question: how aggressively should you emphasize hard examples? REINFORCE treats all prompts uniformly. ML pushes hardest on the prompts the model currently struggles with, which can destabilize training. MaxRL provides a controlled middle ground — the parameter $N$ directly governs how much extra attention hard prompts receive, letting you dial the difficulty emphasis up or down depending on your training needs.

---

## The Full Picture

Let's step back and see how all the pieces fit together.

Standard RL (REINFORCE) computes $\frac{1}{N} \sum r_i S_i$. This estimates $\nabla_\theta p_\theta(x)$ — the gradient of pass@1. It treats each sample equally and only cares about single-try success.

Maximum likelihood computes $\frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x)$, which is equivalent to $\mathbb{E}[S \mid \text{success}]$. This implicitly optimizes a weighted combination of all pass@$k$ gradients. But it requires knowing $p_\theta(x)$ exactly, which we don't have.

MaxRL bridges the gap. By dividing by $K$ (the observed number of successes) instead of $N$ (the batch size), we get an estimator that is computable from samples alone — no knowledge of $p_\theta(x)$ required — and whose expectation equals the gradient of a truncated approximation to $\log p_\theta(x)$. The truncation level equals $N$, the number of samples. More samples means a closer approximation to ML, which means richer multi-try training signal.

The control variate (subtracting the average score) keeps the estimator unbiased while reducing the variance that comes from the randomness of sampling. The result is a practical algorithm: sample $N$ outputs, check which ones are correct, average the score functions of the correct ones, subtract the average score function of all outputs, and use that as your gradient estimate.

The paper's central message is that REINFORCE and maximum likelihood are not separate training paradigms — they are endpoints of a single spectrum parameterized by $N$. At $N = 1$, you get REINFORCE. At $N = \infty$, you get ML. And for any finite $N$ in between, you get a well-defined objective $J^{(N)}_{\text{MaxRL}}$ with a simple, unbiased gradient estimator. The choice of $N$ lets you smoothly trade off between the computational cost of generating more samples and the richness of the training signal you extract from them. As the unified weight function view showed, this entire spectrum boils down to one question: how aggressively should you emphasize hard examples? The weight function $w(p)$ goes from uniform ($w = 1$) at $N = 1$ to maximal hard-example amplification ($w = 1/p$) at $N = \infty$, with $N$ controlling where you sit on that curve.
