---
title: "Reinforcement Learning from Scratch"
description: "Building RL from the ground up — actions, rewards, policies, expected reward, the policy gradient theorem, and REINFORCE — all derived step by step with concrete examples."
date: 2026-02-15
tags: [reinforcement-learning, machine-learning, policy-gradient, mathematics]
---

This is Part 2. If you haven't read [Mathematical Prerequisites for Reinforcement Learning](/blog/math-prerequisites-for-rl), start there — we will use the same mathematical tools (expected value, derivatives, the log trick, and Monte Carlo estimation) throughout this post.

---

## The Single-Step World

### A tiny AI

Imagine you are building a very small AI. It sees a math question — "2 + 3 = ?" — and must output one of two answers. Action 0 means the model answers "4", and Action 1 means the model answers "5". That's the entire world: one question, two possible actions. We will build Reinforcement Learning completely inside this tiny setup before scaling up to anything more complex.

---

### What Is an Action?

An **action** is simply a choice the model makes, denoted $a$. In our example, $a \in \{0, 1\}$, where $a = 0$ means the model answers "4" and $a = 1$ means the model answers "5". There is nothing deep about this definition — an action is just a number that labels a choice.

---

### What Is Reward?

After the model outputs an answer, we check whether it was correct. If the answer is right, we assign a reward of 1. If it is wrong, the reward is 0. We denote reward as $r$, so if $a = 1$ (the model answers "5", which is correct), then $r = 1$, and if $a = 0$ (the model answers "4", which is wrong), then $r = 0$.

Reward is a number that tells us how good the action was. In more complex settings, rewards can be negative (penalties), fractional, or delayed, but the idea is always the same: a scalar signal indicating quality.

---

### The Policy

The model does not always pick the same action deterministically. Instead, it chooses **randomly** according to a probability distribution over actions. We call this distribution the **policy**, written as:

$$
\pi_\theta(a)
$$

Read this as "the probability of choosing action $a$ under parameter $\theta$." The policy is the central object in RL — it is the thing we are trying to improve.

---

### What Is $\theta$?

$\theta$ is a number (or, in general, a vector of numbers) that controls the model's behaviour. We define the policy using the sigmoid function from Part 1:

$$
\pi_\theta(1) = \sigma(\theta), \qquad \pi_\theta(0) = 1 - \sigma(\theta)
$$

where $\sigma(\theta) = \frac{1}{1 + e^{-\theta}}$.

So $\theta$ acts as a knob. When $\theta = 0$, the sigmoid outputs 0.5, meaning the model picks each answer with equal probability — a completely random policy. As $\theta$ grows large and positive, $\sigma(\theta)$ approaches 1, and the model almost always answers "5" (the correct answer). As $\theta$ grows large and negative, the model almost always answers "4" (the wrong answer). Training the model means finding a value of $\theta$ that makes the policy produce the correct answer reliably.

---

### Expected Reward

Because the model chooses its action randomly, the reward it receives is also random. Sometimes it will answer correctly and get reward 1, other times it will answer incorrectly and get reward 0. The natural question is: **on average, how much reward does this policy earn?**

Using the expected value formula from Part 1:

$$
\mathbb{E}[r] = \pi_\theta(1) \times 1 + \pi_\theta(0) \times 0 = \pi_\theta(1) = \sigma(\theta)
$$

We give this quantity a name — the **objective function**:

$$
J(\theta) = \sigma(\theta)
$$

$J(\theta)$ represents the average reward the model earns under the current policy. Our goal is to make $J$ as large as possible, which means finding the $\theta$ that pushes $\sigma(\theta)$ close to 1.

---

### Computing the Gradient

To maximise $J$, we compute its derivative with respect to $\theta$. From Part 1, the derivative of the sigmoid is $\sigma(\theta)(1 - \sigma(\theta))$, so:

$$
\frac{dJ}{d\theta} = \sigma(\theta)(1 - \sigma(\theta))
$$

This derivative is always positive (since $\sigma$ is always between 0 and 1), which confirms that increasing $\theta$ always increases $J$. We can use **gradient ascent** — the opposite of gradient descent — to iteratively improve the policy:

$$
\theta \leftarrow \theta + \alpha \frac{dJ}{d\theta}
$$

where $\alpha$ is a small positive learning rate. Each update nudges $\theta$ upward, which increases the probability of the correct action. Repeat this enough times and $\sigma(\theta)$ will approach 1, meaning the model almost always answers "5".

---

### Deriving the Policy Gradient (Log Trick)

The direct derivative above works perfectly for this two-action example. But it relies on us being able to differentiate $J$ in closed form, which requires knowing the reward for every possible action and summing over all of them. In real problems with enormous action spaces, that sum is intractable. So let's derive the same gradient a different way — the way that generalises.

Start from the definition of expected reward:

$$
J(\theta) = \sum_a \pi_\theta(a) \cdot r(a)
$$

Differentiate with respect to $\theta$:

$$
\nabla_\theta J = \sum_a \nabla_\theta \pi_\theta(a) \cdot r(a)
$$

Now apply the log trick from Part 1. We multiply and divide by $\pi_\theta(a)$ inside the sum:

$$
\nabla_\theta \pi_\theta(a) = \pi_\theta(a) \cdot \nabla_\theta \log \pi_\theta(a)
$$

Substituting this into the gradient expression:

$$
\nabla_\theta J = \sum_a \pi_\theta(a) \cdot r(a) \cdot \nabla_\theta \log \pi_\theta(a)
$$

This sum has the form $\sum_a P(a) \cdot f(a)$, which is exactly the definition of an expectation. So we can rewrite it as:

$$
\boxed{\nabla_\theta J = \mathbb{E}_{a \sim \pi_\theta}\left[ r(a) \cdot \nabla_\theta \log \pi_\theta(a) \right]}
$$

This is the **policy gradient formula**, and it emerged purely from algebra — no approximations, no tricks beyond the log identity. The reason this form is so powerful is that it is an expectation under the policy. That means we can estimate it by sampling actions from the policy and averaging, rather than summing over every possible action. For our two-action example this distinction is trivial, but for a language model choosing among 50,000 tokens, or a robot choosing among continuous motor torques, it is the difference between feasible and impossible.

---

## The Multi-Step World

Everything so far involved a single decision followed by a single reward. Real Reinforcement Learning is different — you take multiple actions over time, like playing a game where each move leads to a new position, and the reward might only arrive at the very end. Let's extend our framework to handle this.

---

### A Tiny Grid World

Consider a number line. You start at position 0 and want to reach position 2. At each time step, you can either move right (+1) or move left (−1). We will stop after exactly 2 steps, regardless of where you end up.

The action at time $t$ is $a_t \in \{-1, +1\}$, and the **state** $s_t$ simply represents your current position. So $s_0 = 0$. If you move right at step 0, then $s_1 = 1$. If you then move right again, $s_2 = 2$. The reward structure is sparse: you receive a reward of 1 only if your final position is 2, and 0 otherwise.

$$
r_t = \begin{cases} 1 & \text{if } s_t = 2 \\ 0 & \text{otherwise} \end{cases}
$$

This is a minimal example of a multi-step problem. You don't get any feedback until the end, and you have to make two sequential decisions, each of which affects what states are available in the future.

---

### What Is a Trajectory?

A **trajectory** is the complete record of what happened during one episode — every state visited and every action taken:

$$
\tau = (s_0, a_0, s_1, a_1, s_2)
$$

For example, the trajectory $\tau = (0, +1, 1, +1, 2)$ represents starting at 0, moving right to 1, then moving right again to reach 2. The trajectory $\tau = (0, +1, 1, -1, 0)$ represents moving right and then left, ending back at the start. Each complete sequence from start to finish is one trajectory, denoted $\tau$.

---

### Probability of a Trajectory

In our grid world, the environment is **deterministic** — if you are at position 1 and choose +1, you always end up at position 2, never somewhere random. This means all the randomness in a trajectory comes from the policy's action choices. Suppose the policy assigns the same probabilities at every state:

$$
\pi_\theta(+1 \mid s) = p, \qquad \pi_\theta(-1 \mid s) = 1 - p
$$

The probability of an entire trajectory is simply the product of the probabilities of each action taken along the way:

$$
P_\theta(\tau) = \prod_t \pi_\theta(a_t \mid s_t)
$$

This is the same multiplication rule you use for independent coin tosses — the probability of getting Heads then Heads is $p \times p$. Let's enumerate all four possible 2-step trajectories.

**Right, right** $(0, +1, 1, +1, 2)$: ends at position 2, earns reward 1, with probability $p^2$.

**Right, left** $(0, +1, 1, -1, 0)$: ends at position 0, earns reward 0, with probability $p(1-p)$.

**Left, right** $(0, -1, -1, +1, 0)$: ends at position 0, earns reward 0, with probability $(1-p) \cdot p$.

**Left, left** $(0, -1, -1, -1, -2)$: ends at position $-2$, earns reward 0, with probability $(1-p)^2$.

Only the first trajectory reaches position 2 and earns a reward. All others end at 0 or $-2$ and earn nothing.

---

### The Objective

The total reward for a trajectory is the sum of all rewards collected along the way:

$$
R(\tau) = \sum_t r_t
$$

In our example, only the trajectory $(0, +1, 1, +1, 2)$ has $R(\tau) = 1$. All other trajectories have $R(\tau) = 0$.

The objective function $J(\theta)$ is the **expected total reward** — the average reward you would get if you ran the policy many times:

$$
J(\theta) = \sum_\tau P_\theta(\tau) \cdot R(\tau)
$$

This is exactly the same expected value formula from Part 1 — $\mathbb{E}[X] = \sum P(x) \cdot x$ — except now the "outcomes" are entire trajectories instead of individual coin tosses, and the "values" are total rewards instead of winnings.

### Numerical verification

Let $p = 0.5$, so the policy is completely random. Each of the four trajectories has probability $0.25$, and only one of them has reward 1. So:

$$
J = 0.25 \times 1 + 0.25 \times 0 + 0.25 \times 0 + 0.25 \times 0 = 0.25
$$

This makes intuitive sense: with a random policy, you have a 25% chance of going right twice, which is the only way to reach position 2. If we increase $p$ to 0.8 (a policy that strongly prefers moving right), then $J = 0.8^2 = 0.64$, which is much better. And if $p = 1$, then $J = 1$ — the policy always reaches the goal.

---

### Deriving the Multi-Step Policy Gradient

Now we derive the gradient of $J$ with respect to $\theta$ for the multi-step case. The derivation follows the exact same structure as the single-step case, just with trajectories instead of individual actions.

**Step 1 — Differentiate.** We start from $J(\theta) = \sum_\tau P_\theta(\tau) \cdot R(\tau)$ and move the derivative inside the sum:

$$
\nabla_\theta J = \sum_\tau \nabla_\theta P_\theta(\tau) \cdot R(\tau)
$$

**Step 2 — Apply the log trick.** Using the identity $\nabla P = P \cdot \nabla \log P$ (derived in Part 1):

$$
\nabla_\theta J = \sum_\tau P_\theta(\tau) \cdot R(\tau) \cdot \nabla_\theta \log P_\theta(\tau)
$$

**Step 3 — Recognise as expectation.** The sum $\sum_\tau P_\theta(\tau) \cdot (\cdots)$ is the definition of expectation over trajectories:

$$
\nabla_\theta J = \mathbb{E}_\tau \left[ R(\tau) \cdot \nabla_\theta \log P_\theta(\tau) \right]
$$

**Step 4 — Expand the log probability.** Recall that the probability of a trajectory is $P_\theta(\tau) = \prod_t \pi_\theta(a_t \mid s_t) \times (\text{environment transitions})$. Taking the logarithm turns this product into a sum:

$$
\log P_\theta(\tau) = \sum_t \log \pi_\theta(a_t \mid s_t) + \text{terms that don't depend on } \theta
$$

The environment transition probabilities (like "moving right from position 1 always leads to position 2") do not depend on $\theta$ at all — they are fixed properties of the environment. So when we differentiate, those terms vanish:

$$
\nabla_\theta \log P_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

**Step 5 — Substituting back.** This gives us:

$$
\nabla_\theta J = \mathbb{E}_\tau \left[ R(\tau) \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

This is already a valid policy gradient formula — it is correct and you could use it as-is. But it has a problem: every action in the trajectory is weighted by the *total* reward $R(\tau)$, including rewards that happened *before* that action was taken. An action at time step $t$ cannot possibly have caused a reward at time step $k < t$, so those past rewards are adding noise to the gradient without providing useful signal. We can do better.

### From total reward to per-time-step credit

Recall that $R(\tau) = \sum_{k=0}^{T} r_k$. Substituting this into the gradient:

$$
\nabla_\theta J = \mathbb{E}_\tau \left[ \left(\sum_{k=0}^{T} r_k\right) \left(\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right) \right]
$$

Expanding the product of these two sums gives a double sum — every reward term $r_k$ multiplies every gradient term $\nabla \log \pi_\theta(a_t \mid s_t)$:

$$
= \mathbb{E}_\tau \left[ \sum_{t=0}^{T} \sum_{k=0}^{T} r_k \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

For each time step $t$, we can split the inner sum over $k$ into past rewards ($k < t$) and present-or-future rewards ($k \geq t$):

$$
\nabla_\theta J = \mathbb{E}_\tau \left[ \sum_{t=0}^{T} \left(\sum_{k < t} r_k + \sum_{k \geq t} r_k \right) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

Now comes the key observation: for a fixed time step $t$, the rewards $r_k$ with $k < t$ happened in the past. They were determined by earlier actions and do not depend on the action $a_t$ at all. So the past-reward terms can be factored out of the expectation over $a_t$, leaving only $\mathbb{E}_{a_t}[\nabla_\theta \log \pi_\theta(a_t \mid s_t)]$ behind. We need to show that this expectation is zero.

### The score function identity

For any probability distribution, the expected value of the gradient of its log-probability is zero:

$$
\mathbb{E}_{a \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a) \right] = 0
$$

The proof is short. Start from the fact that probabilities sum to 1:

$$
\sum_a \pi_\theta(a) = 1
$$

Differentiate both sides with respect to $\theta$:

$$
\sum_a \nabla_\theta \pi_\theta(a) = 0
$$

Now apply the log trick — replace $\nabla \pi$ with $\pi \nabla \log \pi$:

$$
\sum_a \pi_\theta(a) \nabla_\theta \log \pi_\theta(a) = 0
$$

But the left side is exactly $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a)]$. So the identity holds. This is pure calculus — it follows directly from the constraint that probabilities must sum to 1.

### Applying the identity

Consider the past-reward contribution to the gradient at time step $t$:

$$
\mathbb{E}_\tau \left[ \sum_{k < t} r_k \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

The rewards $r_k$ for $k < t$ are already determined by the time we reach step $t$ — they depend on earlier states and actions, not on $a_t$. So with respect to the expectation over $a_t$, the past rewards are constants that can be factored out:

$$
= \mathbb{E}_{\tau \setminus a_t} \left[ \sum_{k < t} r_k \cdot \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] \right]
$$

By the score function identity we just proved, the inner expectation is zero. So the entire expression vanishes:

$$
\mathbb{E}_\tau \left[ \sum_{k < t} r_k \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = 0
$$

Past rewards contribute nothing to the gradient. Only future rewards matter.

### The refined policy gradient theorem

Dropping the past-reward terms and defining $R_t := \sum_{k \geq t} r_k$ as the **reward-to-go** from time step $t$ onward, we obtain:

$$
\boxed{\nabla_\theta J = \mathbb{E}_\tau \left[ \sum_{t=0}^{T} R_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]}
$$

This is the refined form of the **policy gradient theorem**. Each action at time $t$ is weighted only by the rewards that occur at or after time $t$ — never by rewards from the past. Compared to the total-reward version, this form has the same expected value (it is still unbiased) but lower variance, because we have removed the noise contributed by past rewards that the action could not have influenced.

The key insight from the full derivation is that we never needed to know the environment's transition dynamics. The environment terms dropped out when we took the derivative. This is what makes policy gradient methods so general — they work even when you have no model of the environment, as long as you can run episodes and observe rewards.

---

## From Theory to Algorithm: REINFORCE

The policy gradient formula tells us the direction to update $\theta$, but it is written as an expectation over all possible trajectories. In practice, we cannot sum over every possible trajectory — even in our tiny grid world there are 4, and in real problems the number is astronomical. This is where Monte Carlo estimation from Part 1 comes in.

### The algorithm

The idea is to **sample** trajectories by actually running the policy, then use those samples to estimate the gradient. The procedure, known as **REINFORCE**, works as follows.

First, fix the current policy parameters $\theta$. Then generate $N$ trajectories by running the policy in the environment: $\tau_1, \tau_2, \ldots, \tau_N$. For each trajectory and each time step $t$, compute the reward-to-go $R_t^{(i)} = \sum_{k \geq t} r_k^{(i)}$ and the gradient term $\nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})$. The gradient estimate is:

$$
\widehat{\nabla J} = \frac{1}{N} \sum_{i=1}^{N} \sum_t R_t^{(i)} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})
$$

Finally, update the parameters using gradient ascent: $\theta \leftarrow \theta + \alpha \widehat{\nabla J}$, where $\alpha$ is the learning rate. Then repeat the whole process — fix the new $\theta$, sample fresh trajectories, estimate the gradient, update.

### Why $\theta$ must be fixed during sampling

This is a subtle but critical point. All $N$ trajectories in a single batch must be sampled from the **same policy** $\pi_\theta$. If you updated $\theta$ after generating the first trajectory and before generating the second, those two trajectories would come from different distributions. As we discussed in Part 1, the unbiasedness proof requires all samples to come from the same distribution — mixing samples from different policies is like measuring heights from different classes and averaging them together.

So the correct training loop has this structure: fix $\theta$, collect an entire batch of trajectories, compute the gradient estimate from that batch, update $\theta$, and only then start collecting a new batch. The update happens *between* batches, never *within* a batch.

### Unbiasedness

The Monte Carlo estimator $\widehat{\nabla J}$ is **unbiased**, meaning $\mathbb{E}[\widehat{\nabla J}] = \nabla J$. The proof is identical to the one in Part 1: each trajectory $\tau_i$ is drawn from the same distribution $P_\theta$, so $\mathbb{E}[X(\tau_i)] = \nabla J$ for every $i$, and averaging over $N$ such terms does not change the expected value.

This means that even though any single estimate might be noisy (because we only sampled $N$ trajectories instead of considering all possible ones), the estimate is correct *on average*. Over many batches and many updates, the noise averages out and the policy converges to one that earns high reward.

### Intuition

For each action in a sampled trajectory, the gradient estimate includes the term $R_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)$. If the reward-to-go from that point onward is large, this term pushes the parameters $\theta$ in a direction that makes that action more probable in that state. If the reward-to-go is zero, the action contributes nothing to the gradient. Over many samples, trajectories that led to good outcomes get reinforced (made more likely), while trajectories that led to poor outcomes are effectively ignored. This is the fundamental mechanism of policy gradient methods, and it is why the field is called *reinforcement* learning — good behaviour is reinforced through probability increases.

---

## Summary

An **action** $a$ is a choice the model makes, and the **reward** $r$ is the scalar signal telling us how good that choice was. The **policy** $\pi_\theta(a \mid s)$ defines the probability of taking action $a$ in state $s$, controlled by the **parameter** $\theta$ — a knob (or vector of knobs) that we tune during training. The **objective** $J(\theta)$ is the average reward earned under the current policy, and the **gradient** $\nabla_\theta J$ tells us the direction to change $\theta$ to increase that average. A **trajectory** $\tau$ is the complete sequence of states and actions in one episode. The **log trick** — the identity $\nabla P = P \nabla \log P$ — converts the gradient into an expectation we can estimate by sampling. And **REINFORCE** is the algorithm that ties it all together: sample trajectories, estimate the gradient, update the parameters, repeat.

Everything was derived from two small examples — a math quiz for the single-step case and a grid world for the multi-step case. The same structure scales to language models with vocabularies of 50,000 tokens, game-playing agents navigating millions of states, and robots learning to walk. The algebra is identical; only the size of the action space and the length of the trajectories change.
