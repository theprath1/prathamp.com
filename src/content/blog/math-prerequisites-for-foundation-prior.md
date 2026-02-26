---
title: "Mathematical Prerequisites for Foundation Prior"
description: "Building the math foundations for understanding how LLMs reshape Bayesian priors — parameters, likelihood, Beta distributions, KL divergence, entropy, exponential tilting, and marginal likelihood — all derived step by step with one coin example."
date: 2026-02-24
tags: [bayesian-inference, mathematics, machine-learning, probability]
---

Before diving into the Foundation Prior paper, you need a handful of mathematical ideas. This post builds every one of them inside a single, consistent coin-flip example so nothing feels abstract or disconnected. By the end, you will have all the tools required to understand how synthetic data from LLMs can be incorporated into Bayesian inference in Part 2.

---

## The Running Example

Imagine we flip a coin 10 times and observe:

$$
\text{7 Heads, 3 Tails}
$$

The coin might be biased. We do not know the true probability of Heads. Our goal throughout this post is to figure out what that probability might be, and to build every mathematical tool we need along the way.

---

## What Is a Parameter?

A **parameter** is just an unknown quantity that controls how the data is generated. That is the entire definition. It does not automatically mean probability. It does not automatically mean mean or variance. It means: "the hidden setting of the system."

In our coin example, we introduce a model where each flip has probability $p$ of being Heads and probability $1 - p$ of being Tails, and the probability stays the same across flips. Here $p$ is the parameter. In this model, the parameter happens to be a probability. That is allowed — a parameter is just an unknown number, and sometimes that number represents a probability.

Two outcomes does not mean $P(H) = 0.5$. All we know is:

$$
P(H) + P(T) = 1
$$

The coin might be biased. So we allow $0 \le p \le 1$ and try to learn $p$ from data.

In other models, the parameter could be a mean ($\mu$), a variance ($\sigma^2$), a slope ($\beta$), a rate ($\lambda$), or a vector of many numbers. The symbol $\theta$ just stands for "unknown." Its meaning depends on the model.

---

## What Is a Model?

A **model** is a mathematical description of how the data is generated. For the coin, the model says that the probability of Heads is $p$ and that flips are independent. That is the entire model. Without a model, you cannot compute probabilities. Without probabilities, you cannot construct likelihood.

---

## Constructing the Likelihood

We observed HHHHHHHTTT — 7 Heads and 3 Tails. We ask: if the coin's true probability of Heads were $p$, what is the probability of seeing exactly this outcome?

For one Head, $P(H \mid p) = p$. For one Tail, $P(T \mid p) = 1 - p$. Since flips are independent, we multiply probabilities:

$$
\boxed{L(p) = p^7(1 - p)^3}
$$

We did not invent a special formula. We simply wrote the joint probability of the observed data.

### Numerical check

If we plug in $p = 0.7$:

$$
L(0.7) = 0.7^7 \times 0.3^3 = 0.0823543 \times 0.027 \approx 0.0022
$$

That means: if the coin truly gives Heads 70% of the time, the probability of observing 7H3T is about 0.22%. That is an ordinary probability statement.

---

## Likelihood vs. Probability

Then why call it **likelihood**? Because we now treat that same probability formula as a function of $p$. We compare different guesses: $p = 0.1$ gives a very small value, $p = 0.7$ gives a larger value, and $p = 0.9$ gives a smaller value again. The value of $p$ that makes the observed data most plausible is the best explanation. That comparison process is what we call likelihood.

**Probability** asks: if $p$ is fixed, what data might happen? **Likelihood** asks: given what happened, which $p$ makes sense? Same formula. Different question.

### Why are likelihood values small?

All values are small. That is normal. Exact sequences are rare. Likelihood is not about absolute size — it is about relative comparison.

$$
\frac{L(0.7)}{L(0.1)} \approx 30{,}000
$$

The observed data is 30,000 times more likely under $p = 0.7$ than under $p = 0.1$. That comparison is powerful.

### The general rule for constructing likelihood

Whenever you want to construct a likelihood, you define the model (how data is generated), write the probability of one observation, multiply across observations (if independent), and treat the result as a function of the parameter. That is all likelihood ever is.

**One-sentence summary:** Likelihood is the joint probability of the observed data, viewed as a function of the unknown parameter.

---

## The Beta Distribution

### What Beta really is

The **Beta distribution** is a mathematical way to represent prior fake observations of heads and tails. Nothing mystical.

### Why it works perfectly for coins

When you flip a coin, the likelihood looks like $\theta^H(1 - \theta)^T$. A Beta prior looks like $\theta^{\alpha - 1}(1 - \theta)^{\beta - 1}$. Multiply them:

$$
\theta^{\alpha - 1 + H}(1 - \theta)^{\beta - 1 + T}
$$

Which is again a Beta distribution. That is why it is called a **conjugate prior** — the math stays in the same family.

### Shape intuition

Beta can look flat (uniform), peaked in the middle, skewed toward 0, skewed toward 1, U-shaped, or extremely concentrated — all controlled by $\alpha$ and $\beta$. So it is extremely flexible.

Think of it this way: if $\alpha$ is large, you have "seen many heads." If $\beta$ is large, you have "seen many tails." So Beta is just memory of imaginary past flips. Not real flips — belief flips.

---

## Deriving the Beta Distribution From Scratch

### Step 1 — What do we want?

We want a distribution over $\theta$ (probability of heads). After seeing $H$ heads and $T$ tails, we know the likelihood is:

$$
\theta^H(1 - \theta)^T
$$

That is not a definition — that is just how probabilities multiply. So any distribution that wants to behave like "fake data" must look like $\theta^{\text{something}}(1 - \theta)^{\text{something}}$.

### Step 2 — Build it from scratch

Suppose we say: "Before seeing data, I want my belief to behave exactly like I had already seen $A$ heads and $B$ tails." Then the function representing that belief should look like:

$$
\theta^A(1 - \theta)^B
$$

Because that is exactly what real data would produce.

### Step 3 — Rename the parameters

Instead of calling them $A$ and $B$, statisticians define:

$$
\alpha = A + 1, \quad \beta = B + 1
$$

So $A = \alpha - 1$ and $B = \beta - 1$. Substitute back:

$$
\theta^A(1 - \theta)^B = \theta^{\alpha - 1}(1 - \theta)^{\beta - 1}
$$

That is literally it.

### Step 4 — Why the $-1$?

It is just a definition choice. They could have defined Beta using $\theta^A(1-\theta)^B$. But historically, mathematicians defined the Beta function as:

$$
\int_0^1 \theta^{\alpha - 1}(1 - \theta)^{\beta - 1}\,d\theta
$$

So statistics inherited that form. The $-1$ exists because the Beta distribution parameters are defined one higher than the fake counts.

### Step 5 — Why define it that way?

Because of a convenience: when $\alpha = 1, \beta = 1$, the distribution becomes:

$$
\theta^0(1 - \theta)^0 = 1
$$

That gives a flat (uniform) distribution. If they had used $\theta^\alpha(1 - \theta)^\beta$, then uniform would occur at $\alpha = 0, \beta = 0$, and zero parameters are awkward. So they shifted everything by 1.

The $-1$ exists because Beta parameters are defined so that Beta(1,1) = uniform, and $\alpha, \beta$ are always positive. It is a parameterization choice, not a deep mystery.

### The mean of a Beta distribution

The mean of $\text{Beta}(\alpha, \beta)$ is:

$$
\boxed{\mu = \frac{\alpha}{\alpha + \beta}}
$$

It represents the expected value or balance point of the distribution. For example, if $\alpha = 2$ and $\beta = 2$, then $\mu = 2/4 = 0.5$.

### The variance of a Beta distribution

The variance of $\text{Beta}(\alpha, \beta)$ is:

$$
\boxed{\text{Var} = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}}
$$

This formula tells you how spread out the distribution is. As $\alpha + \beta$ grows (more pseudo-observations), the variance shrinks — you become more certain. For example, $\text{Beta}(2, 2)$ has variance $\frac{2 \times 2}{4^2 \times 5} = \frac{4}{80} = 0.05$, while $\text{Beta}(20, 20)$ has variance $\frac{400}{1600 \times 41} \approx 0.006$ — ten times more pseudo-observations gives roughly ten times less variance. We will use this formula in the law of total variance section below and again in Part 2 when computing mixture uncertainty.

---

## What Is Entropy?

**Entropy** measures how uncertain you are. More spread out belief means more entropy. More concentrated belief means less entropy.

### Simple example

Suppose $\theta$ can only be $\{0.2, 0.5, 0.8\}$. Consider Belief A, which assigns probability 0.05 to $\theta = 0.2$, probability 0.05 to $\theta = 0.5$, and probability 0.90 to $\theta = 0.8$. This belief is almost sure $\theta = 0.8$, so it has low uncertainty and low entropy. Now consider Belief B, which assigns probability 0.33 to each value. This belief is very spread out, so it has high uncertainty and high entropy.

### The formula

For discrete distributions, entropy is $H(p) = -\sum_i p_i \log p_i$. For continuous distributions:

$$
H(\rho) = -\int \rho(\theta)\log\rho(\theta)\,d\theta
$$

### Why log appears

Log measures information. If something is very unlikely, $\log(\text{probability})$ is very negative. Entropy averages "how surprising outcomes are" weighted by probability. So entropy measures average surprise. More spread distribution means more surprise means more entropy.

### The maximum entropy principle

Suppose you only know one thing: the average of $\theta$ is 0.6. But you know nothing else. There are infinitely many distributions with mean 0.6.

The **maximum entropy principle** says: choose the most spread out distribution that satisfies the constraint. Meaning: do not assume anything extra beyond what you are forced to assume. This will connect directly to the paper's core optimization in Part 2.

---

## KL Divergence

**KL divergence** measures how different two probability distributions are. It is defined as:

$$
\boxed{KL(\rho \| \pi_0) = \int \rho(\theta)\log\frac{\rho(\theta)}{\pi_0(\theta)}\,d\theta}
$$

If $\rho = \pi_0$, then $KL = 0$ (no change). Bigger KL means you moved farther from the reference distribution.

### Three critical properties

First, it respects probability structure. It compares distributions multiplicatively, not additively. Probabilities combine multiplicatively (likelihoods multiply), and KL respects that.

Second, it measures information gain. KL equals the expected log difference. It literally measures: how many extra bits are needed if I encode reality using $\pi_0$ but truth is $\rho$? So minimizing KL means: add as little new information as possible.

Third, it uniquely produces exponential tilting. There is a theorem: if you minimize KL subject to expectation constraints, the solution must be exponential in the constraint function. No other divergence gives that clean structure.

### Why not just subtract distributions?

Suppose we measured $\int(\rho(\theta) - \pi_0(\theta))^2\,d\theta$. That measures squared difference. The problem: it ignores that probabilities must remain positive, it does not respect the geometry of probability distributions, and it does not lead to Bayesian updating form.

### Connection to maximum entropy

When you already have a prior $\pi_0(\theta)$, the correct generalization of maximum entropy is: minimize $KL(\rho \| \pi_0)$ subject to constraints. This is called **relative entropy maximization** — the Bayesian version of maximum entropy. You are saying: given my old belief $\pi_0$, and given that some constraint must hold, what is the least informative update?

---

## Exponential Tilting

Start with a prior distribution $\pi_0(\theta)$. Suppose you want to favor larger values of some function $g(\theta)$. Instead of redefining everything from scratch, you **tilt** the prior by multiplying it by $\exp(\lambda g(\theta))$, then normalize:

$$
\boxed{\rho(\theta) = \frac{\pi_0(\theta)\exp(\lambda g(\theta))}{\int \pi_0(u)\exp(\lambda g(u))\,du}}
$$

You are "tilting" the original distribution toward higher values of $g(\theta)$.

### Discrete coin example

Let $\theta$ only take three values: $\{0.2, 0.5, 0.8\}$, with prior probabilities 0.3, 0.4, and 0.3 respectively. Now suppose synthetic data prefers larger $\theta$. Let $g(\theta) = \theta$ and choose $\lambda = 2$.

First, compute $\exp(2\theta)$ for each value:

$$
\exp(2 \times 0.2) = \exp(0.4) \approx 1.49, \quad \exp(2 \times 0.5) = \exp(1.0) \approx 2.72, \quad \exp(2 \times 0.8) = \exp(1.6) \approx 4.95
$$

Now multiply each prior probability by its corresponding tilt factor:

$$
0.3 \times 1.49 = 0.447, \quad 0.4 \times 2.72 = 1.088, \quad 0.3 \times 4.95 = 1.485
$$

The total is $0.447 + 1.088 + 1.485 = 3.02$. Dividing each product by this total gives the new (normalized) probabilities:

$$
\frac{0.447}{3.02} \approx 0.148, \quad \frac{1.088}{3.02} \approx 0.360, \quad \frac{1.485}{3.02} \approx 0.492
$$

The distribution shifted toward larger $\theta$. Before tilting, $\theta = 0.8$ had probability 0.30. After tilting, it has probability 0.49. That is exponential tilting — the prior was reweighted toward values that score higher on $g(\theta)$.

### The theorem (Csiszar's I-projection)

If you solve:

$$
\min_\rho KL(\rho \| \pi_0) \quad \text{subject to} \quad \mathbb{E}_\rho[g(\theta)] = C
$$

Then the unique solution must be:

$$
\rho(\theta) = \frac{\pi_0(\theta)\exp(\lambda g(\theta))}{\int \pi_0(u)\exp(\lambda g(u))\,du}
$$

This result is known as **Csiszar's I-projection theorem**. No other shape works — the exponential form is not assumed, it is forced by the structure of KL. You will see this theorem invoked by name in Part 2 when we derive the foundation prior.

### Why exponential form appears

KL divergence contains $\rho(\theta)\log\rho(\theta)$. When you take the derivative with respect to $\rho$, you get $\log(\rho)$. Solving for $\rho$ requires exponentiating. That is why the exponential form appears.

### Where exponential tilting appears

This is not just used in the Foundation Prior paper. It appears in maximum entropy distributions, exponential families, large deviation theory, variational inference, and statistical mechanics. It is a fundamental geometric property of probability distributions.

---

## Why We Use Log-Likelihood

If we have $n$ independent observations, the likelihood multiplies:

$$
L(\theta) = \prod_{i=1}^n L_i(\theta)
$$

Taking the log turns products into sums:

$$
\log L(\theta) = \sum_{i=1}^n \log L_i(\theta)
$$

This matters for three deep reasons. First, additivity: sums are far easier to optimize than products, which makes optimization tractable. Second, information geometry: KL divergence and log-likelihood live in the same exponential family geometry, and using log ensures duality between the constraint and the objective. Third, convexity: log-likelihood is often concave, making optimization easier, whereas if raw likelihood were used, the problem would be non-convex.

### Coin example

For our coin with 7 Heads and 3 Tails:

$$
\log L(\theta) = 7\log\theta + 3\log(1 - \theta)
$$

This is just a score: bigger (less negative) means $\theta$ explains the data better.

---

## Marginal Likelihood

**Marginal likelihood** answers this question: if my belief about $\theta$ is $p(\theta)$, how probable is the observed data overall? It is defined as:

$$
\boxed{p(D) = \int p(D \mid \theta)\,p(\theta)\,d\theta}
$$

You do not know the true $\theta$. So to compute how likely the data is under your model, you assume $\theta$ could be many values, compute the likelihood $p(D \mid \theta)$ for each possible $\theta$, weight it by how plausible $\theta$ is under your prior, and average across all $\theta$. That average is the marginal likelihood.

### Coin example

Suppose $\theta$ can only be $\{0.4, 0.6\}$ with equal prior probabilities. Real data: 6 heads out of 10. The likelihoods are:

$$
p(D \mid 0.4) = 0.4^6 \times 0.6^4 = 0.004096 \times 0.1296 \approx 0.000531
$$

$$
p(D \mid 0.6) = 0.6^6 \times 0.4^4 = 0.046656 \times 0.0256 \approx 0.001194
$$

The marginal likelihood averages these weighted by prior belief:

$$
p(D) = 0.5 \times 0.000531 + 0.5 \times 0.001194 \approx 0.000863
$$

Notice that $p(D \mid 0.6)$ is about 2.25 times larger than $p(D \mid 0.4)$ — the data (6 heads out of 10) favors $\theta = 0.6$ over $\theta = 0.4$. But the marginal likelihood combines both possibilities into a single number that measures how well the overall model (both values of $\theta$ together) predicts the data.

### Why it is called "marginal"

Because we integrated out $\theta$. We removed $\theta$ from the expression. So we now have probability of data alone — $\theta$ disappears.

### Why marginal likelihood matters

It answers: how good is my entire model at explaining the data? Not just "what $\theta$ is best," but: how well does the model as a whole predict reality? That is why it is used for model comparison, mixture weight updating, and trust parameter calibration — all things we will need in Part 2.

The important distinction to keep in mind: **likelihood** $p(D \mid \theta)$ depends on $\theta$, while **marginal likelihood** $p(D)$ has no $\theta$ left — it is averaged over uncertainty.

---

## The Law of Total Variance

This result will be needed when we average foundation priors across multiple prompts in Part 2.

### Setup

Suppose you have a **mixture distribution** — a weighted average of component distributions. To sample $\theta$ from a mixture, you first randomly pick a component (say A, B, or C), then sample $\theta$ from that component's distribution. This is equivalent to introducing a random variable $Z \in \{A, B, C\}$ with known probabilities, and then drawing $\theta \mid Z = i \sim p_i(\theta)$.

### The law of total expectation

The mean of a mixture is simply the weighted average of component means. If each component has mean $\mu_i$ and weight $w_i$, then:

$$
m = \sum_i w_i \mu_i
$$

This is the **law of total expectation**: $\mathbb{E}[\theta] = \mathbb{E}_Z[\mathbb{E}[\theta \mid Z]]$.

### The law of total variance

The variance identity is:

$$
\boxed{\text{Var}(\theta) = \mathbb{E}_Z[\text{Var}(\theta \mid Z)] + \text{Var}_Z(\mathbb{E}[\theta \mid Z])}
$$

This says: total variance = average within-component variance + variance of component means.

### Concrete example

Suppose three components with equal weight $1/3$, means $\mu_A = 0.6154$, $\mu_B = 0.5385$, $\mu_C = 0.6538$, and variances $\text{Var}_A = 0.03156$, $\text{Var}_B = 0.03314$, $\text{Var}_C = 0.03018$.

The mixture mean is:

$$
m = \frac{0.6154 + 0.5385 + 0.6538}{3} = \frac{1.8077}{3} \approx 0.6026
$$

The first term (average within-component variance) is:

$$
\frac{1}{3}(0.03156 + 0.03314 + 0.03018) = \frac{0.09488}{3} \approx 0.03163
$$

The second term (variance of component means) is:

$$
\frac{1}{3}\left[(0.6154 - 0.6026)^2 + (0.5385 - 0.6026)^2 + (0.6538 - 0.6026)^2\right]
$$

$$
= \frac{1}{3}[0.000164 + 0.004109 + 0.002630] = \frac{0.006903}{3} \approx 0.00230
$$

So the total mixture variance is approximately $0.03163 + 0.00230 = 0.03393$.

### Interpretation

Total uncertainty has two sources. **Within-component uncertainty** means that even if you knew which component was correct, you would still be uncertain about $\theta$. **Between-component uncertainty** means you are also uncertain about which component is right. The mixture variance captures both.

---

## Bayesian Updating (The Core Mechanic)

Before we move to the Foundation Prior paper, let us establish the standard Bayesian update — the baseline that the paper modifies.

You start with a **prior** $\pi(\theta)$, which represents your belief before seeing data. The **likelihood** $L(D \mid \theta)$ tells you how likely the data is under parameter $\theta$. The **posterior** $\pi(\theta \mid D)$ is your belief after seeing data. Bayes' rule says:

$$
\boxed{\pi(\theta \mid D) \propto L(D \mid \theta)\,\pi(\theta)}
$$

The prior is your belief, the likelihood is what the world tells you, and the posterior is their combination.

### Coin example

Take a prior of $\theta \sim \text{Beta}(2, 2)$, meaning $\pi_0(\theta) \propto \theta^1(1 - \theta)^1$. The data is 7 Heads and 3 Tails, giving likelihood $L(D \mid \theta) = \theta^7(1 - \theta)^3$. The posterior is:

$$
\pi(\theta \mid D) \propto \theta^1(1 - \theta)^1 \times \theta^7(1 - \theta)^3 = \theta^8(1 - \theta)^4
$$

That is $\text{Beta}(9, 5)$. This is **Beta-Bernoulli conjugacy** — the Beta prior and the Binomial likelihood multiply to give another Beta. The prior contributed $(\alpha_0 - 1) = 1$ pseudo-head and $(\beta_0 - 1) = 1$ pseudo-tail, and the data contributed 7 real heads and 3 real tails, giving $\text{Beta}(1 + 7 + 1,\; 1 + 3 + 1) = \text{Beta}(9, 5)$. The posterior mean is $9/14 \approx 0.643$.

Everything is clean because data comes from reality. The Foundation Prior paper (Part 2) asks: what happens when data comes from an LLM instead?

---

## Summary

We built seven mathematical tools, all from one coin example. A **parameter** is an unknown number controlling data generation. **Likelihood** is the joint probability of observed data viewed as a function of the parameter. The **Beta distribution** encodes prior fake observations of heads and tails and stays Beta after updating — that is conjugacy. **Entropy** measures uncertainty, and the **maximum entropy principle** says: do not assume more than your constraints force. **KL divergence** measures how far a new belief is from an old one, and minimizing it under expectation constraints uniquely produces **exponential tilting** — multiplying the prior by $\exp(\lambda g(\theta))$. **Marginal likelihood** integrates out the parameter to give a single number measuring how well the entire model predicts data. And the **law of total variance** decomposes mixture uncertainty into within-component and between-component parts. In Part 2, every one of these tools will be used to derive the Foundation Prior framework from scratch.
