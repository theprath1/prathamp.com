---
title: "Foundation Priors: Why LLM Outputs Are Priors, Not Data"
description: "Why treating LLM-generated synthetic data as real observations is wrong, and how a single trust parameter λ fixes everything — derived step by step from Bayes' theorem to the full framework using one coin flip example."
date: 2026-02-18
tags: [bayesian-inference, machine-learning, statistics, mathematics]
draft: true
---

This post explains the core ideas behind the [Foundation Priors paper](https://arxiv.org/abs/2512.01107) by Sanjog Misra — a clean framework for incorporating LLM-generated data into statistical inference. The paper's central claim is that outputs from foundation models are not objective data. They are subjective, shaped by the model's training and the user's prompts, and should be treated as components of a Bayesian prior distribution with a trust parameter that controls their influence. We will derive the entire framework from scratch using one running example.

---

## The Running Example

We will use one setup for the entire post. You find a coin on the street and want to estimate $\theta$, the probability it lands heads.

**Real data.** You flip the coin $n_r = 5$ times and observe: H, H, T, H, H. That is $k_r = 4$ heads and $n_r - k_r = 1$ tails. This is your real, objective data — you physically flipped the coin and recorded the outcomes.

**Synthetic data from an LLM.** Before (or after) flipping, you ask an LLM: "I found a coin on a sidewalk. What do you think $P(\text{heads})$ is?" The LLM, trained on vast amounts of text about coins, responds in a way that is equivalent to saying "most coins are roughly fair." We formalize this as synthetic data: the LLM produces $n_s = 10$ synthetic flips — 5 heads and 5 tails. So $k_s = 5$.

**Base prior.** Before consulting either the coin or the LLM, you start with total ignorance: $\theta$ is equally likely to be any value between 0 and 1. We write this as $\pi_0(\theta) = 1$ for $\theta \in [0, 1]$.

The central question is: how should you combine the 5 real flips with the 10 LLM-generated flips? Should you pool them together as if all 15 were real? Or should you treat the LLM data differently — and if so, how?

---

## Bayesian Updating: A Derivation from Scratch

Before we can formalize the foundation prior, we need the tool it builds on: **Bayes' theorem**. This is the rule for updating beliefs in light of data.

### Deriving Bayes' theorem

Start from the definition of **conditional probability** — the probability of event $A$ given that event $B$ has occurred:

$$
P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}
$$

This is a definition, not something we derive. But we can write the joint probability $P(A \text{ and } B)$ in two ways, depending on which event we condition on:

$$
P(A \text{ and } B) = P(B \mid A) \cdot P(A) = P(A \mid B) \cdot P(B)
$$

Set the two expressions equal and solve for $P(A \mid B)$:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

Now replace $A$ with a parameter $\theta$ and $B$ with observed data $D$:

$$
\boxed{\pi(\theta \mid D) = \frac{L(D \mid \theta) \cdot \pi(\theta)}{P(D)}}
$$

Each piece has a name:

- $\pi(\theta)$ is the **prior** — what we believed about $\theta$ before seeing data.
- $L(D \mid \theta)$ is the **likelihood** — the probability of observing our data if $\theta$ were the true value.
- $P(D) = \int L(D \mid \theta') \pi(\theta') \, d\theta'$ is the **marginal likelihood** — a normalizing constant that ensures the posterior integrates to 1.
- $\pi(\theta \mid D)$ is the **posterior** — our updated belief after seeing data.

Since $P(D)$ does not depend on $\theta$, we often write the proportional form:

$$
\pi(\theta \mid D) \propto L(D \mid \theta) \cdot \pi(\theta)
$$

The posterior is proportional to the likelihood times the prior. This is the core of Bayesian inference.

### The Beta-Binomial model

Now we apply Bayes' theorem to our coin. Each flip is an independent Bernoulli trial with parameter $\theta$. The **likelihood** of observing $k$ heads in $n$ flips is:

$$
L(\theta; k, n) = \binom{n}{k} \theta^k (1 - \theta)^{n-k}
$$

The binomial coefficient $\binom{n}{k}$ does not depend on $\theta$, so for the purpose of computing the posterior it is just a constant. We write:

$$
L(\theta; k, n) \propto \theta^k (1 - \theta)^{n-k}
$$

Our prior is $\pi_0(\theta) = 1$ (the uniform distribution on $[0, 1]$). The posterior is:

$$
\pi(\theta \mid D_r) \propto \theta^k (1 - \theta)^{n-k} \cdot 1 = \theta^{k}(1-\theta)^{n-k}
$$

For our real data with $k = 4$ heads and $n - k = 1$ tails:

$$
\pi(\theta \mid D_r) \propto \theta^4(1 - \theta)^1
$$

We need to recognize what distribution this is. A function of the form $\theta^{a-1}(1-\theta)^{b-1}$ on $[0, 1]$ is the kernel of the **Beta distribution**, written $\text{Beta}(a, b)$. Its full density is:

$$
\text{Beta}(\theta; a, b) = \frac{\theta^{a-1}(1-\theta)^{b-1}}{B(a, b)}
$$

where $B(a, b) = \int_0^1 \theta^{a-1}(1-\theta)^{b-1} \, d\theta$ is the **Beta function**, a normalizing constant. The mean of $\text{Beta}(a, b)$ is $a/(a+b)$.

Matching our posterior $\theta^4(1-\theta)^1 = \theta^{5-1}(1-\theta)^{2-1}$, we get $a = 5$ and $b = 2$. So:

$$
\pi(\theta \mid D_r) = \text{Beta}(5, 2)
$$

The posterior mean is $5 / (5 + 2) = 5/7 \approx 0.714$. After seeing 4 heads in 5 flips, we believe the coin is biased toward heads.

### Numerical check

Let us verify the normalizing constant. We need $B(5, 2) = \int_0^1 \theta^4(1-\theta) \, d\theta$. For integer arguments, there is a formula: $B(a, b) = \frac{(a-1)!(b-1)!}{(a+b-1)!}$. So:

$$
B(5, 2) = \frac{4! \cdot 1!}{6!} = \frac{24 \cdot 1}{720} = \frac{1}{30}
$$

Let us check by direct integration:

$$
\int_0^1 \theta^4(1-\theta) \, d\theta = \int_0^1 \theta^4 - \theta^5 \, d\theta = \frac{1}{5} - \frac{1}{6} = \frac{6 - 5}{30} = \frac{1}{30}
$$

Both give $1/30$. The posterior density at $\theta = 0.8$ is $30 \cdot 0.8^4 \cdot 0.2 = 30 \cdot 0.4096 \cdot 0.2 = 30 \cdot 0.08192 = 2.458$. This is a proper density — it peaks near $\theta = 0.8$, consistent with 4 heads in 5 flips.

### The general update rule

Notice the pattern. Starting from a $\text{Beta}(\alpha_0, \beta_0)$ prior and observing $k$ heads in $n$ flips:

$$
\text{Prior: } \text{Beta}(\alpha_0, \beta_0) \quad \xrightarrow{\text{observe } k \text{ heads, } n-k \text{ tails}} \quad \text{Posterior: } \text{Beta}(\alpha_0 + k, \beta_0 + n - k)
$$

Each head adds 1 to the first parameter. Each tails adds 1 to the second. The prior parameters $\alpha_0$ and $\beta_0$ act as "pseudo-observations" — the uniform prior $\text{Beta}(1, 1)$ corresponds to 0 pseudo-heads and 0 pseudo-tails (with one of each baked into the parameter values). This additivity is what makes the Beta-Binomial model so useful: updating is just counting.

---

## The Problem: Synthetic Data from an LLM

Now consider the LLM's synthetic data: $n_s = 10$ flips with $k_s = 5$ heads and 5 tails. If we treated this as real data and pooled it with our 5 actual flips, we would have $n = 15$ total flips with $k = 9$ heads and 6 tails. The posterior would be:

$$
\text{Beta}(1 + 9, 1 + 6) = \text{Beta}(10, 7), \quad \text{mean} = \frac{10}{17} \approx 0.588
$$

The LLM data has pulled our estimate from 0.714 down toward 0.5 — substantially. The 10 synthetic flips are literally outweighing the 5 real flips.

This should worry us. The LLM did not flip any coin. It generated text based on patterns in its training data. Its output reflects a compressed summary of what "coins" look like across millions of documents — not the specific coin in your hand. The LLM's data is:

- **Subjective**: it depends on which model you use (GPT, Claude, Gemini), how you phrase the prompt, and what was in the training data.
- **Not exchangeable with real data**: the LLM's "flips" did not come from the same physical process as your real flips.
- **Potentially useful**: the LLM's belief that coins are roughly fair is reasonable prior information.

The problem with pooling is that it gives synthetic data the same status as real data. Ten synthetic flips count as much as ten real flips. This is too much trust. What we need is a way to incorporate the LLM's information while controlling how much influence it has. The foundation prior provides exactly this.

---

## The Foundation Prior

### Definition

The **foundation prior** treats the LLM's synthetic data as a source of prior information, not as objective evidence. It is defined as:

$$
\boxed{\rho(\theta \mid D_s, \lambda) = \frac{\pi_0(\theta) \cdot L(D_s \mid \theta)^\lambda}{\int \pi_0(\theta') \cdot L(D_s \mid \theta')^\lambda \, d\theta'}}
$$

Here $\pi_0(\theta)$ is the base prior (our initial belief before any LLM involvement), $L(D_s \mid \theta)$ is the likelihood of the synthetic data, $\lambda \geq 0$ is the **trust parameter**, and the denominator is a normalizing constant.

The only difference from a standard Bayesian update is the exponent $\lambda$ on the likelihood. When you update with real data, the likelihood enters with exponent 1. The foundation prior raises it to the power $\lambda$, which controls how seriously we take the synthetic evidence.

### Concrete example

In our coin setup:

- Base prior: $\pi_0(\theta) = 1$ (uniform)
- Synthetic data likelihood: $L(D_s \mid \theta) \propto \theta^5(1-\theta)^5$ (from 5 heads, 5 tails in 10 synthetic flips)
- Foundation prior:

$$
\rho(\theta \mid D_s, \lambda) \propto 1 \cdot \left(\theta^5(1-\theta)^5\right)^\lambda = \theta^{5\lambda}(1-\theta)^{5\lambda}
$$

This is the kernel of $\text{Beta}(5\lambda + 1, 5\lambda + 1)$. Notice the symmetry: both parameters are equal, so the foundation prior is always centered at $\theta = 0.5$. This makes sense — the LLM's synthetic data had equal heads and tails, encoding a belief that the coin is fair.

What changes with $\lambda$ is the **concentration** — how tightly the prior clusters around 0.5:

| $\lambda$ | Foundation prior | Mean | Interpretation |
|---|---|---|---|
| 0 | $\text{Beta}(1, 1)$ = Uniform | 0.5 | No trust in LLM — prior is flat |
| 0.2 | $\text{Beta}(2, 2)$ | 0.5 | Mild preference for fairness |
| 0.5 | $\text{Beta}(3.5, 3.5)$ | 0.5 | Moderate trust |
| 1 | $\text{Beta}(6, 6)$ | 0.5 | Full trust — synthetic data treated as real |
| 2 | $\text{Beta}(11, 11)$ | 0.5 | Over-trust — LLM dominates |

At $\lambda = 0$, the foundation prior collapses to the base prior — the LLM is completely ignored. At $\lambda = 1$, raising the likelihood to the power 1 recovers a standard Bayesian update, meaning we treat the synthetic data exactly as if it were real. Values of $\lambda$ between 0 and 1 express partial trust: the synthetic data informs the prior but does not carry as much weight as actual observations.

---

## Deriving the Foundation Prior from KL Divergence

The definition above may look arbitrary — why raise the likelihood to the power $\lambda$? This section shows that the foundation prior is not an ad hoc construction. It emerges as the unique solution to a natural optimization problem: find the distribution closest to your base prior that also respects the LLM's synthetic data to a controlled degree.

### The optimization problem

We want a distribution $\rho(\theta)$ that:

1. Stays as close as possible to our base prior $\pi_0(\theta)$, because we don't want the LLM to completely override our existing beliefs.
2. Assigns reasonable likelihood to the synthetic data $D_s$, because we do want to incorporate the LLM's information.

We measure "closeness" using the **Kullback-Leibler (KL) divergence**, defined as:

$$
\text{KL}(\rho \| \pi_0) = \int \rho(\theta) \log \frac{\rho(\theta)}{\pi_0(\theta)} \, d\theta
$$

KL divergence is always non-negative and equals zero only when $\rho = \pi_0$. It measures how much $\rho$ deviates from $\pi_0$ in an information-theoretic sense.

The constraint on synthetic data is that the expected log-likelihood under $\rho$ must be at least some value $C$:

$$
\mathbb{E}_\rho[\log L(D_s \mid \theta)] = \int \rho(\theta) \log L(D_s \mid \theta) \, d\theta \geq C
$$

The full optimization problem is:

$$
\min_\rho \text{KL}(\rho \| \pi_0) \quad \text{subject to} \quad \mathbb{E}_\rho[\log L(D_s \mid \theta)] \geq C, \quad \int \rho(\theta) \, d\theta = 1
$$

### Solving with the Lagrangian

We form the **Lagrangian** by introducing a multiplier $\lambda \geq 0$ for the data constraint and $\mu$ for the normalization constraint:

$$
\mathcal{L}[\rho] = \int \rho(\theta) \log \frac{\rho(\theta)}{\pi_0(\theta)} \, d\theta - \lambda \left( \int \rho(\theta) \log L(D_s \mid \theta) \, d\theta - C \right) - \mu \left( \int \rho(\theta) \, d\theta - 1 \right)
$$

To find the minimum, we take the **functional derivative** with respect to $\rho(\theta)$ and set it to zero. The functional derivative of $\int \rho \log(\rho / \pi_0) \, d\theta$ with respect to $\rho(\theta)$ is $\log(\rho(\theta) / \pi_0(\theta)) + 1$. (This follows from differentiating $\rho \log \rho$ to get $\log \rho + 1$, and $-\rho \log \pi_0$ to get $-\log \pi_0$.) Setting the full derivative to zero:

$$
\log \frac{\rho(\theta)}{\pi_0(\theta)} + 1 - \lambda \log L(D_s \mid \theta) - \mu = 0
$$

Solve for $\log \rho(\theta)$:

$$
\log \rho(\theta) = \log \pi_0(\theta) + \lambda \log L(D_s \mid \theta) + (\mu - 1)
$$

Exponentiate both sides:

$$
\rho(\theta) = \pi_0(\theta) \cdot L(D_s \mid \theta)^\lambda \cdot e^{\mu - 1}
$$

The constant $e^{\mu - 1}$ is determined by the normalization constraint $\int \rho(\theta) \, d\theta = 1$:

$$
e^{\mu - 1} = \frac{1}{\int \pi_0(\theta') \cdot L(D_s \mid \theta')^\lambda \, d\theta'}
$$

Substituting back:

$$
\rho(\theta) = \frac{\pi_0(\theta) \cdot L(D_s \mid \theta)^\lambda}{\int \pi_0(\theta') \cdot L(D_s \mid \theta')^\lambda \, d\theta'}
$$

This is exactly the foundation prior. The trust parameter $\lambda$ is the Lagrange multiplier — it is not an arbitrary hyperparameter but the mathematical consequence of how tightly we want to enforce the synthetic data constraint. A larger $\lambda$ means a tighter constraint (we demand that $\rho$ assigns higher likelihood to $D_s$), which pulls $\rho$ further from $\pi_0$ and closer to the synthetic evidence.

### Numerical check

With our coin example, $\pi_0(\theta) = 1$ and $L(D_s \mid \theta) \propto \theta^5(1-\theta)^5$. Set $\lambda = 0.5$. Then:

$$
\rho(\theta) \propto \theta^{2.5}(1-\theta)^{2.5} = \text{Beta}(3.5, 3.5)
$$

The KL divergence from $\text{Beta}(3.5, 3.5)$ to Uniform is:

$$
\text{KL} = \log B(1,1) - \log B(3.5, 3.5) + (3.5 - 1)\psi(3.5) + (3.5 - 1)\psi(3.5) - (7 - 2)\psi(7)
$$

where $\psi$ is the digamma function. We don't need the exact number — the point is that this KL divergence is smaller than what we would get at $\lambda = 1$ (the $\text{Beta}(6, 6)$ prior), confirming that less trust means staying closer to the base prior.

At $\lambda = 0$, $\rho = \pi_0$ and $\text{KL} = 0$ — zero deviation, as expected.

---

## The Trust Parameter $\lambda$

The trust parameter $\lambda$ is the single most important quantity in the framework. It controls the transition between two extremes.

**At $\lambda = 0$:** The synthetic data is completely ignored. The foundation prior equals the base prior. You flip your coin, do standard Bayesian inference, and the LLM plays no role. The posterior would be $\text{Beta}(5, 2)$ with mean $0.714$.

**At $\lambda = 1$:** The synthetic data carries full weight, as if it were real data. This is equivalent to pooling — the posterior would be $\text{Beta}(10, 7)$ with mean $0.588$. Ten synthetic flips count as ten real flips.

**At $0 < \lambda < 1$:** Partial trust. The synthetic data informs the prior but is discounted. This is the regime the paper argues we should use for LLM-generated data — the LLM provides useful information, but it should not carry the same authority as physical measurements.

**At $\lambda > 1$:** Over-trust. The synthetic data counts for more than real data of the same size. The paper warns against this — it would mean trusting the LLM more than your own observations, which is rarely justified.

### Why $\lambda$ matters: the posterior under different trust levels

When we combine the foundation prior with real data (which we derive fully in the next section), the posterior is:

$$
\pi(\theta \mid D_r, D_s, \lambda) = \text{Beta}(5 + 5\lambda, \, 2 + 5\lambda)
$$

The posterior mean is $(5 + 5\lambda) / (7 + 10\lambda)$. Let us compute this for several values of $\lambda$:

| $\lambda$ | Posterior | Mean | What happens |
|---|---|---|---|
| 0 | $\text{Beta}(5, 2)$ | $5/7 = 0.714$ | LLM ignored — data says coin is biased |
| 0.2 | $\text{Beta}(6, 3)$ | $6/9 = 0.667$ | Slight pull toward fairness |
| 0.5 | $\text{Beta}(7.5, 4.5)$ | $7.5/12 = 0.625$ | Moderate pull |
| 1 | $\text{Beta}(10, 7)$ | $10/17 = 0.588$ | Full trust — same as pooling |
| 2 | $\text{Beta}(15, 12)$ | $15/27 = 0.556$ | Over-trust |
| $\infty$ | — | $0.5$ | LLM completely dominates |

As $\lambda$ increases, the posterior mean slides from $0.714$ (dictated by real data) toward $0.5$ (dictated by the LLM). Each value of $\lambda$ represents a different answer to the question: "How much do you trust the LLM relative to the data in your hand?"

The limit as $\lambda \to \infty$ deserves attention. The posterior mean is $(5 + 5\lambda)/(7 + 10\lambda)$. Dividing numerator and denominator by $\lambda$:

$$
\frac{5/\lambda + 5}{7/\lambda + 10} \xrightarrow{\lambda \to \infty} \frac{5}{10} = 0.5
$$

In the extreme, the LLM's belief (fair coin) overrides the real data entirely. Real observations become irrelevant — which is why $\lambda > 1$ is generally problematic.

---

## The Full Posterior: Combining Real Data with the Foundation Prior

### Derivation

Now we combine everything. The foundation prior $\rho(\theta \mid D_s, \lambda)$ serves as our prior, and we update it with the real data $D_r$ using Bayes' theorem:

$$
\pi(\theta \mid D_r, D_s, \lambda) \propto L(D_r \mid \theta) \cdot \rho(\theta \mid D_s, \lambda)
$$

Substituting the definition of the foundation prior:

$$
\propto L(D_r \mid \theta) \cdot \pi_0(\theta) \cdot L(D_s \mid \theta)^\lambda
$$

This is a **three-way product**: real data likelihood $\times$ tempered synthetic likelihood $\times$ base prior. In our coin example:

$$
\pi(\theta \mid D_r, D_s, \lambda) \propto \underbrace{\theta^4(1-\theta)^1}_{\text{real: 4H, 1T}} \cdot \underbrace{1}_{\text{base prior}} \cdot \underbrace{\left(\theta^5(1-\theta)^5\right)^\lambda}_{\text{synthetic: 5H, 5T, tempered by } \lambda}
$$

Combine the exponents. For $\theta$: $4 + 5\lambda$. For $(1-\theta)$: $1 + 5\lambda$.

$$
\pi(\theta \mid D_r, D_s, \lambda) \propto \theta^{4 + 5\lambda}(1-\theta)^{1 + 5\lambda}
$$

This is $\text{Beta}(5 + 5\lambda, \, 2 + 5\lambda)$.

### Numerical check

Set $\lambda = 0.5$. The posterior is $\text{Beta}(7.5, 4.5)$.

**Mean:** $7.5 / (7.5 + 4.5) = 7.5 / 12 = 0.625$.

Let us verify this makes sense. The real data alone gives a mean of $0.714$. The LLM says $0.5$. With $\lambda = 0.5$, the synthetic data carries half-weight, so we expect the posterior to be pulled partway toward $0.5$ but not as far as full pooling ($0.588$). Indeed, $0.625$ sits between $0.714$ and $0.588$ — closer to the data-only estimate, because the synthetic data is discounted.

**Mode:** For $\text{Beta}(a, b)$ with $a, b > 1$, the mode is $(a - 1)/(a + b - 2)$. Here: $(7.5 - 1)/(7.5 + 4.5 - 2) = 6.5/10 = 0.65$. The mode is slightly lower than the mean because the posterior is right-skewed (more mass above 0.5 than below, but a long tail toward 0).

### Interpretation

The posterior $\text{Beta}(5 + 5\lambda, 2 + 5\lambda)$ reveals the accounting clearly:

- The first parameter $5 + 5\lambda$ represents $k_r + \lambda k_s = 4 + 5\lambda$ heads (plus 1 from the base prior).
- The second parameter $2 + 5\lambda$ represents $(n_r - k_r) + \lambda(n_s - k_s) = 1 + 5\lambda$ tails (plus 1 from the base prior).

Each real head counts as 1. Each synthetic head counts as $\lambda$. This is the mechanical effect of the trust parameter: it converts synthetic observations into fractional real observations. With $\lambda = 0.5$, each of the LLM's 5 synthetic heads counts as half a real head, contributing $5 \times 0.5 = 2.5$ effective heads to the posterior.

---

## Effective Sample Size

### Derivation

The **effective sample size** of the synthetic data tells us how many real observations the LLM data is worth, given a trust level $\lambda$.

The posterior $\text{Beta}(5 + 5\lambda, 2 + 5\lambda)$ has total parameter count:

$$
(5 + 5\lambda) + (2 + 5\lambda) = 7 + 10\lambda
$$

The base prior $\text{Beta}(1, 1)$ contributes a parameter count of $1 + 1 = 2$. The total evidence beyond the base prior is:

$$
(7 + 10\lambda) - 2 = 5 + 10\lambda
$$

This decomposes as:

- Real data contribution: $n_r = 5$
- Synthetic data contribution: $\lambda \cdot n_s = \lambda \cdot 10 = 10\lambda$

So the effective sample size of the synthetic data is:

$$
\boxed{n_{\text{eff}} = \lambda \cdot n_s}
$$

### Numerical check

| $\lambda$ | Effective synthetic observations $\lambda \cdot 10$ | Total evidence | Synthetic as fraction of total |
|---|---|---|---|
| 0 | 0 | 5 | 0% |
| 0.1 | 1 | 6 | 17% |
| 0.5 | 5 | 10 | 50% |
| 1 | 10 | 15 | 67% |
| 2 | 20 | 25 | 80% |

At $\lambda = 0.1$, the LLM's 10 synthetic flips count as just 1 real flip. At $\lambda = 0.5$, they count as 5 — equal to our real sample size. At $\lambda = 1$ (pooling), they count as 10 — double our real data — giving the LLM a two-thirds vote in the final estimate.

This table makes the problem with naive pooling concrete. When you pool real and synthetic data without discounting, you are implicitly setting $\lambda = 1$. For our coin, this means the LLM's opinion counts for twice as much as your actual flips. Unless you have very strong reasons to trust the LLM that much, this is too aggressive.

### The unit information prior alternative

The paper mentions an alternative to tuning $\lambda$: fix $\lambda = 1$ but control the number of synthetic observations $n_s$. The **unit information prior** approach sets $n_s$ so that the effective contribution of the synthetic data equals one real observation. In our setting, this means generating just $n_s = 1$ synthetic flip instead of 10. The foundation prior would be $\text{Beta}(1 + k_s, 1 + 1 - k_s)$ where $k_s \in \{0, 1\}$.

Both approaches achieve the same goal — controlling the influence of synthetic data — through different levers. Tuning $\lambda$ lets you generate a large synthetic dataset and then discount it. Tuning $n_s$ lets you generate less data in the first place. The paper uses the $\lambda$ approach because it is more flexible: you can calibrate $\lambda$ after seeing both datasets, rather than committing to a synthetic sample size upfront.

---

## Why Not Just Pool the Data?

This section makes explicit what we have been building toward. There are three reasons why pooling ($\lambda = 1$) is the wrong default.

**First, subjectivity.** The synthetic data depends on which LLM you use, how you phrase the prompt, and what was in the training data. Ask GPT and Claude the same question, and you may get different synthetic datasets. This is not a property of real data — your 5 coin flips are the same regardless of who observes them. Treating subjective outputs as objective evidence inflates your certainty.

**Second, double-counting.** The LLM's training data likely included information about coins. If your real data came from the same distribution the LLM learned from, then the LLM's synthetic data partly reflects the same evidence as your real data. Pooling counts this information twice, leading to an overconfident posterior.

**Third, misalignment.** The LLM generates data according to its learned distribution $\pi_{\text{LLM}}(D_s \mid q)$, which depends on the prompt $q$. This distribution may not match the true data-generating process. In our coin example, the LLM assumes coins are roughly fair, but your specific coin might be genuinely biased. Pooling forces the posterior to compromise between the truth and the LLM's generic belief, with the compromise controlled by sample sizes rather than by a principled trust assessment.

The foundation prior framework addresses all three problems by making the trust explicit through $\lambda$ and letting the analyst (or a calibration procedure) determine how much the LLM's opinion should count.

---

## Calibrating $\lambda$ from Real Data

So far we have treated $\lambda$ as something the analyst chooses. The paper also provides a principled, data-driven approach: choose $\lambda$ to maximize the **marginal likelihood** of the real data under the foundation prior.

### The idea

The marginal likelihood of the real data given a trust level $\lambda$ is:

$$
m(D_r \mid \lambda) = \int L(D_r \mid \theta) \cdot \rho(\theta \mid D_s, \lambda) \, d\theta
$$

This measures how well the foundation prior (with trust $\lambda$) predicts the real data. If the LLM's synthetic data is useful — meaning it points toward parameter values that also explain the real data — then incorporating it (via $\lambda > 0$) should increase the marginal likelihood. If the synthetic data is misleading, incorporating it should decrease it.

The optimal trust level is:

$$
\lambda^* = \arg\max_\lambda \, m(D_r \mid \lambda)
$$

### Concrete example

In our coin setting, the foundation prior at trust $\lambda$ is $\text{Beta}(5\lambda + 1, 5\lambda + 1)$. The marginal likelihood of the real data (4 heads, 1 tails) under this prior is:

$$
m(D_r \mid \lambda) = \int_0^1 \theta^4(1-\theta)^1 \cdot \frac{\theta^{5\lambda}(1-\theta)^{5\lambda}}{B(5\lambda + 1, 5\lambda + 1)} \, d\theta
$$

$$
= \frac{1}{B(5\lambda + 1, 5\lambda + 1)} \int_0^1 \theta^{4 + 5\lambda}(1-\theta)^{1 + 5\lambda} \, d\theta
$$

$$
= \frac{B(5 + 5\lambda, \, 2 + 5\lambda)}{B(5\lambda + 1, 5\lambda + 1)}
$$

Both Beta functions can be evaluated for any $\lambda$. At $\lambda = 0$:

$$
m(D_r \mid 0) = \frac{B(5, 2)}{B(1, 1)} = \frac{1/30}{1} = \frac{1}{30} \approx 0.0333
$$

At $\lambda = 1$:

$$
m(D_r \mid 1) = \frac{B(10, 7)}{B(6, 6)} = \frac{\frac{9! \cdot 6!}{16!}}{\frac{5! \cdot 5!}{11!}} = \frac{\frac{362880 \cdot 720}{20922789888000}}{\frac{120 \cdot 120}{39916800}}
$$

Let us compute these step by step. For $B(10, 7) = \frac{9! \cdot 6!}{16!}$:

$$
9! = 362880, \quad 6! = 720, \quad 16! = 20922789888000
$$

$$
B(10, 7) = \frac{362880 \times 720}{20922789888000} = \frac{261273600}{20922789888000} \approx 1.249 \times 10^{-5}
$$

For $B(6, 6) = \frac{5! \cdot 5!}{11!}$:

$$
5! = 120, \quad 11! = 39916800
$$

$$
B(6, 6) = \frac{120 \times 120}{39916800} = \frac{14400}{39916800} \approx 3.608 \times 10^{-4}
$$

So:

$$
m(D_r \mid 1) = \frac{1.249 \times 10^{-5}}{3.608 \times 10^{-4}} \approx 0.0346
$$

Comparing: $m(D_r \mid 0) \approx 0.0333$ and $m(D_r \mid 1) \approx 0.0346$. The marginal likelihood is slightly higher at $\lambda = 1$ than at $\lambda = 0$, suggesting that the LLM's "fair coin" belief has mild positive value in predicting the real data. But the improvement is small, because our real data (4 heads in 5 flips) actually conflicts with the LLM's fairness belief. The optimal $\lambda^*$ in this case would be a small positive number — some trust in the LLM, but not full trust.

### Interpretation

The marginal likelihood calibration embodies a self-consistency check: trust the LLM to the extent that doing so helps explain the real data you actually observed. If the LLM's synthetic data is well-aligned with reality, $\lambda^*$ will be closer to 1. If the synthetic data conflicts with reality, $\lambda^*$ will be pushed toward 0. The real data acts as an automatic corrective for misplaced trust.

---

## The Unified View: A Spectrum of Trust

We can now see the full picture as a single spectrum. Every choice of $\lambda$ corresponds to a specific answer to the question: "How many real observations is each synthetic observation worth?"

At one end ($\lambda = 0$): ignore the LLM. Standard Bayesian inference using only real data. The posterior is determined entirely by what you observed.

At the midpoint ($\lambda = 1$): full trust. Pooling. The LLM's synthetic data is treated as if you had generated it by actually flipping the coin. Each synthetic observation counts as one real observation.

In between ($0 < \lambda < 1$): partial trust. Each synthetic observation counts as a fraction $\lambda$ of a real observation. The effective sample size of the synthetic data is $\lambda \cdot n_s$.

Beyond ($\lambda > 1$): over-trust. Each synthetic observation counts for more than a real one. This is almost never justified for LLM outputs.

The gradient of the posterior mean with respect to $\lambda$ tells us how sensitive the inference is to trust. In our example, the posterior mean is $f(\lambda) = (5 + 5\lambda)/(7 + 10\lambda)$. Differentiating:

$$
f'(\lambda) = \frac{5(7 + 10\lambda) - 10(5 + 5\lambda)}{(7 + 10\lambda)^2} = \frac{35 + 50\lambda - 50 - 50\lambda}{(7 + 10\lambda)^2} = \frac{-15}{(7 + 10\lambda)^2}
$$

The derivative is always negative (the posterior mean decreases as we trust the LLM's "fair coin" belief more) and its magnitude decreases as $\lambda$ grows. The inference is most sensitive to trust at small $\lambda$ values — the first bit of trust changes the estimate more than later increments. At $\lambda = 0$: $f'(0) = -15/49 \approx -0.306$. At $\lambda = 1$: $f'(1) = -15/289 \approx -0.052$. The marginal impact of additional trust diminishes.

This diminishing sensitivity is a general feature of the framework, not specific to our coin example. It follows from the denominator growing as $(a + b\lambda)^2$. Practically, it means that the distinction between $\lambda = 0$ and $\lambda = 0.2$ is far more consequential than between $\lambda = 0.8$ and $\lambda = 1.0$. Getting the right order of magnitude for $\lambda$ matters more than getting the exact value.

---

## The Full Picture

Let us step back and connect all the pieces.

The foundation prior framework rests on a simple observation: LLM outputs are not objective data. They are shaped by training corpora, prompt engineering, and the inherent biases of the generative process. Treating them as data — pooling them with real observations — gives them unearned authority and inflates statistical certainty.

The fix is to treat LLM outputs as components of a Bayesian prior with a trust parameter $\lambda$. The resulting **foundation prior** $\rho(\theta) \propto \pi_0(\theta) \cdot L(D_s \mid \theta)^\lambda$ is not an ad hoc recipe — it is the unique distribution that minimizes KL divergence from the base prior while respecting the synthetic evidence to degree $\lambda$. This is the same variational principle that underlies standard Bayesian updating, except that the Lagrange multiplier $\lambda$ is no longer fixed at 1. It is a free parameter because the synthetic data does not carry the same epistemic weight as real observations.

The trust parameter $\lambda$ controls one thing: how many real observations each synthetic observation is worth. The effective sample size of the synthetic data is $\lambda \cdot n_s$, giving practitioners a concrete dial. With $\lambda = 0$, the LLM is ignored. With $\lambda = 1$, it is fully trusted. The optimal $\lambda^*$ can be calibrated from real data by maximizing the marginal likelihood — a self-consistency check that trusts the LLM exactly to the extent that doing so helps explain the real evidence.

In our coin example, the entire framework reduced to a single family of posteriors: $\text{Beta}(5 + 5\lambda, 2 + 5\lambda)$, with $\lambda$ smoothly interpolating the posterior mean from $0.714$ (data alone says biased) to $0.5$ (LLM says fair). The right answer depends on how much you trust the LLM — and now you have the mathematical machinery to make that trust precise.
