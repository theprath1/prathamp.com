---
title: "Mathematical Prerequisites for Reinforcement Learning"
description: "Building the math foundations you need for RL — probability, expected value, derivatives, the log trick, and Monte Carlo estimation — all through one consistent example."
date: 2026-02-14
tags: [reinforcement-learning, mathematics, machine-learning, probability]
---

Before diving into Reinforcement Learning, you need a handful of mathematical ideas. This post builds every one of them inside a single, consistent example so nothing feels abstract or disconnected. By the end, you will have all the tools required to derive the core RL algorithm from scratch in Part 2.

---

## The Running Example

Imagine a coin that is **not** fair. It lands Heads with probability $p$ and Tails with probability $1 - p$. If it lands Heads you win ₹10, and if it lands Tails you win ₹0. We will derive every mathematical concept in this post from this single setup.

---

## 1 — Probability Distributions

A **probability distribution** assigns a probability to every possible outcome. For our coin, the distribution says that Heads occurs with probability $p$ and Tails occurs with probability $1 - p$.

Two rules must hold for any valid distribution. First, every probability must be between 0 and 1. Second, all probabilities must sum to 1, which ours do: $p + (1 - p) = 1$. That is the entire definition of a probability distribution — nothing more. It is simply a table (or function) that tells you how likely each outcome is.

---

## 2 — Expected Value

Expected value answers a natural question: **on average, how much do I win per toss?** The formula multiplies each possible value by its probability, then sums everything up:

$$
\mathbb{E}[X] = \sum (\text{value}) \times (\text{probability})
$$

For our coin, this becomes:

$$
\mathbb{E}[X] = 10 \times p + 0 \times (1 - p) = 10p
$$

If $p = 0.3$, then $\mathbb{E}[X] = 10 \times 0.3 = 3$. Notice that you never actually win ₹3 on any single toss — you either win ₹10 or ₹0. The expected value is a theoretical quantity: it tells you what your average winnings per toss would converge to if you kept tossing the coin thousands of times. That convergence is not a vague intuition — it is a mathematical guarantee we will see later in this post.

---

## 3 — The Sigmoid Function

We need a way to turn any real number into a probability, meaning a number between 0 and 1. The **sigmoid function** does exactly this:

$$
\sigma(\theta) = \frac{1}{1 + e^{-\theta}}
$$

When $\theta$ is a very large negative number, $e^{-\theta}$ becomes enormous and $\sigma(\theta)$ approaches 0. When $\theta = 0$, we get $\sigma(0) = \frac{1}{1+1} = 0.5$. And when $\theta$ is large and positive, $e^{-\theta}$ vanishes and $\sigma(\theta)$ approaches 1.

**$\theta \to -\infty$:** $\sigma(\theta) \to 0$

**$\theta = 0$:** $\sigma(\theta) = 0.5$

**$\theta \to +\infty$:** $\sigma(\theta) \to 1$

Now we can connect this to our coin. If we define $p = \sigma(\theta)$, then $\theta$ becomes a **knob** that controls the coin's bias. Setting $\theta = 0$ gives a fair coin with $p = 0.5$. Cranking $\theta$ to a large positive value makes the coin land Heads almost every time. Pushing $\theta$ far negative makes it almost always land Tails. The sigmoid simply converts an unconstrained real number into a valid probability.

---

## 4 — Parameterised Expected Value

Since $p = \sigma(\theta)$, our expected value from Section 2 now becomes a function of $\theta$:

$$
J(\theta) = 10 \cdot \sigma(\theta)
$$

We give it the name $J(\theta)$ because in RL this will become the **objective function** — the quantity we want to maximise. If we want to maximise our winnings, we need $J(\theta)$ as large as possible, which means pushing $\sigma(\theta)$ towards 1, which means making $\theta$ large and positive.

But here is the question: how do we know *which direction* to push $\theta$? In this toy example the answer is obvious — just increase it. But in real problems with millions of parameters, you cannot eyeball the right direction. You need a principled tool, and that tool is the derivative.

---

## 5 — Derivatives

The **derivative** of $J$ with respect to $\theta$ answers a precise question: if I nudge $\theta$ up by a tiny amount, does $J$ increase or decrease, and by how much? We write this as:

$$
\frac{dJ}{d\theta}
$$

The derivative of the sigmoid function is a known result from calculus:

$$
\frac{d}{d\theta}\sigma(\theta) = \sigma(\theta)(1 - \sigma(\theta))
$$

Since $J(\theta) = 10 \cdot \sigma(\theta)$, we get:

$$
\frac{dJ}{d\theta} = 10 \cdot \sigma(\theta)(1 - \sigma(\theta))
$$

Because $\sigma(\theta)$ always lies between 0 and 1, both $\sigma(\theta)$ and $(1 - \sigma(\theta))$ are positive, so the derivative is always positive. This confirms what we expected: increasing $\theta$ always increases $J$.

### Numerical check

At $\theta = 0$, the derivative is $10 \times 0.5 \times 0.5 = 2.5$. At $\theta = 2$, we have $\sigma(2) \approx 0.88$, so the derivative is approximately $10 \times 0.88 \times 0.12 = 1.06$. The derivative gets smaller as $\sigma$ approaches 1, which makes geometric sense — the sigmoid curve is steepest in the middle and flattens out at the extremes.

---

## 6 — The Log Trick

This is the key algebraic move that makes Reinforcement Learning work. It looks like a minor rearrangement, but it is what transforms an intractable sum into something we can estimate from samples. Let's derive it carefully using our coin example.

### Setup

We have the expected value defined as:

$$
J(\theta) = P(\text{Heads}) \times 10 + P(\text{Tails}) \times 0 = P(\text{Heads}) \times 10
$$

Taking the derivative directly:

$$
\frac{dJ}{d\theta} = \frac{dP(\text{Heads})}{d\theta} \times 10
$$

### The trick

Now we do something that seems pointless at first: take $\frac{dP}{d\theta}$ and multiply-and-divide by $P$ itself.

$$
\frac{dP}{d\theta} = P \cdot \frac{1}{P} \cdot \frac{dP}{d\theta}
$$

Why would we do this? Because from calculus, the expression $\frac{1}{P} \cdot \frac{dP}{d\theta}$ is exactly the derivative of $\log P$:

$$
\frac{d}{d\theta} \log P = \frac{1}{P} \cdot \frac{dP}{d\theta}
$$

This is sometimes called the "chain rule applied to the logarithm," and it is one of those identities that shows up everywhere once you know to look for it. Substituting this identity gives us:

$$
\frac{dP}{d\theta} = P \cdot \frac{d}{d\theta} \log P
$$

Plugging this back into our derivative of $J$:

$$
\frac{dJ}{d\theta} = P(\text{Heads}) \cdot \frac{d}{d\theta}\log P(\text{Heads}) \times 10
$$

### Why this matters

Look at the structure of that expression. We have a probability $P(\text{Heads})$ multiplied by some quantity. That is precisely the form of an expectation — we can rewrite it as:

$$
\frac{dJ}{d\theta} = \mathbb{E}\left[ \text{reward} \cdot \frac{d}{d\theta} \log P(\text{outcome}) \right]
$$

This is the crucial insight. The derivative of the objective is itself an expected value. And expected values can be estimated by sampling — you don't need to enumerate every possible outcome. You just run the experiment many times and average. In our coin example with two outcomes, the enumeration is trivial. But in Reinforcement Learning, where the "outcomes" are entire game trajectories with billions of possibilities, this conversion from "sum over all outcomes" to "average over samples" is what makes the problem tractable at all.

---

## 7 — Monte Carlo Estimation

### The problem

When $p = 0.3$, we can compute $\mathbb{E}[X] = 10 \times 0.3 = 3$ directly. But what if we don't know $p$? Or what if the number of possible outcomes is enormous — say, all possible games of chess, or all possible conversations a chatbot could have? Then you cannot compute the expected value directly because the sum $\mathbb{E}[X] = \sum (\text{value}) \times (\text{probability})$ has too many terms, or the probabilities themselves are unknown.

### The solution: sample and average

The idea is beautifully simple. Instead of computing the theoretical average, you run the experiment many times and take the empirical average. Concretely, you toss the coin $N$ times, record the winnings $x_1, x_2, \ldots, x_N$, and compute:

$$
\hat{\mathbb{E}}[X] = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

This is the **Monte Carlo estimator**. The name comes from the famous casino in Monaco — the idea is that you are "gambling" to estimate a quantity, relying on randomness to give you an answer.

### Concrete example

Suppose $p = 0.3$. You toss the coin 10 times and get:

**Toss 1:** Tails → ₹0, \
**Toss 2:** Heads → ₹10, \
**Toss 3:** Tails → ₹0, \
**Toss 4:** Tails → ₹0, \
**Toss 5:** Heads → ₹10, \
**Toss 6:** Tails → ₹0, \
**Toss 7:** Tails → ₹0, \
**Toss 8:** Tails → ₹0, \
**Toss 9:** Heads → ₹10, \
**Toss 10:** Tails → ₹0

That is 3 Heads and 7 Tails — a total of ₹30.

The average is $\frac{30}{10} = 3.0$. With just 10 tosses we happened to land exactly on the true value, but usually you won't be this lucky. The estimate might come out to 2 or 4 on any given run. The point is that as you increase the number of samples, the estimate becomes increasingly reliable.

### Why does this work?

The theoretical justification is the [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), one of the foundational results in probability theory. It guarantees that:

$$
\frac{1}{N}\sum_{i=1}^{N} x_i \xrightarrow{N \to \infty} \mathbb{E}[X]
$$

In plain language: as you take more and more samples, the sample average converges to the true expected value. This is not an RL-specific result — it applies to any random process, from coin tosses to stock prices to weather measurements.

### Unbiasedness

The Monte Carlo estimator has a stronger property called **unbiasedness**, which means that the expected value of the estimator equals the true expected value:

$$
\mathbb{E}\left[\hat{\mathbb{E}}[X]\right] = \mathbb{E}[X]
$$

The proof is short and uses only basic properties of expectation. Start by writing out the estimator and taking its expectation:

$$
\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N} x_i\right] = \frac{1}{N}\sum_{i=1}^{N} \mathbb{E}[x_i] = \frac{1}{N} \cdot N \cdot \mathbb{E}[X] = \mathbb{E}[X]
$$

The first equality uses **linearity of expectation** — the expectation of a sum equals the sum of expectations, regardless of whether the terms are independent. The second equality uses the fact that each $x_i$ is drawn from the **same distribution**, so $\mathbb{E}[x_i] = \mathbb{E}[X]$ for every $i$. That "same distribution" assumption is so important that it deserves its own section.

### A second numerical example

Suppose there are only two possible outcomes: outcome A gives value 5 with probability 0.2, and outcome B gives value 1 with probability 0.8. The true expectation is $0.2 \times 5 + 0.8 \times 1 = 1.8$. If you sample many times, roughly 20% of your samples will be 5 and 80% will be 1, and the average will converge to 1.8. That's Monte Carlo estimation — replacing a theoretical weighted sum with an empirical average.

---

## 8 — "Sampled from the Same Distribution"

This phrase appears constantly in RL proofs, and it is easy to gloss over. But the entire mathematical machinery we just built depends on it, so let's pin down exactly what it means.

When we proved unbiasedness, the key step was writing $\mathbb{E}[x_i] = \mathbb{E}[X]$ for all $i$. This is only true if every sample $x_i$ comes from the **same probabilistic rule**. In our coin example, that means every toss uses the same coin with the same probability $p$. If you secretly swapped the coin between tosses — sometimes using one with $p = 0.3$ and other times one with $p = 0.9$ — then your sample average would not converge to the expected value of either coin. It would converge to some muddled mixture that doesn't correspond to any well-defined quantity.

Here's a helpful analogy. Suppose you want to estimate the average height of students in Class A. You measure 5 students from Class A and 5 students from Class B, then average all 10 measurements. That average is not the mean height of Class A, because the samples came from different populations. For the estimate to be valid, all samples must come from one consistent source.

This will matter enormously in Reinforcement Learning. The "coin" in RL is the policy — the rule that decides which actions to take. When we sample trajectories to estimate the gradient, all trajectories must come from the same policy. If we update the policy between samples, we are effectively swapping the coin mid-experiment, and the mathematical guarantees break down. This is why the RL training loop has a very specific structure: fix the policy, sample a batch of trajectories, compute the gradient estimate, *then* update the policy. The update only happens after all samples for that batch have been collected.

---

## Summary

Everything in this post was built from one coin-toss example. A **probability distribution** assigns probabilities to outcomes. The **expected value** $\mathbb{E}[X]$ computes the weighted average $\sum (\text{value})(\text{probability})$. The **sigmoid** $\sigma(\theta)$ maps any real number to a probability, which lets us define a parameterised objective $J(\theta)$ — the expected value as a function of parameters. The **derivative** $\frac{dJ}{d\theta}$ tells us how $J$ changes when $\theta$ changes, giving us a direction to improve. The **log trick** — the identity $\frac{dP}{d\theta} = P \cdot \frac{d}{d\theta}\log P$ — converts that derivative into an expectation, and **Monte Carlo estimation** lets us approximate that expectation by sampling and averaging. The key guarantee is **unbiasedness**: on average, the sample estimate equals the truth.

With these tools in hand, we're ready for [Reinforcement Learning from Scratch](/blog/reinforcement-learning-from-scratch), where we put them all together to derive the REINFORCE algorithm.
