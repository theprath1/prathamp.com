---
title: "MoE Load Balancing from Scratch"
description: "Building Mixture-of-Experts routing from the ground up — sigmoid scores, top-K selection, expert biases, aux-loss-free balancing, and SMEBU — all derived step by step with a 4-expert toy model."
date: 2026-03-03
tags: [machine-learning, mixture-of-experts, load-balancing, transformers, mathematics]
---

The Arcee Trinity Large technical report introduces a 400-billion-parameter sparse Mixture-of-Experts language model that activates only 13 billion parameters per token. The model has 256 routed experts per layer, but only 4 fire for any given token. That means 252 experts sit idle on every forward pass. How does the model decide which 4 to activate? And how does it prevent all tokens from piling onto the same few "popular" experts while the rest gather dust?

We will derive the entire MoE routing and load balancing mechanism from scratch, culminating in the paper's novel contribution: **SMEBU** (Soft-clamped Momentum Expert Bias Updates), a new method for keeping experts balanced during training. We will use a single tiny example — 4 experts, 8 tokens — and trace every computation end to end.

---

## The Setup: Our Running Example

We will work with the simplest possible Mixture-of-Experts layer:

- **4 routed experts**: $N_r = 4$ (labeled experts 1, 2, 3, 4)
- **1 shared expert**: $N_s = 1$ (always active for every token)
- **Top-2 routing**: $K_r = 2$ (each token activates exactly 2 of the 4 routed experts)
- **Model dimension**: $d = 3$ (so every vector has 3 components)

Each expert is a small feedforward network (FFN). The shared expert processes every token. The routed experts compete for tokens — each token picks its top-2 favorites.

We have one token vector:

$$
\mathbf{u} = \begin{bmatrix} 1 \\ 0.5 \\ -1 \end{bmatrix}
$$

and four router vectors, one per routed expert:

$$
\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad
\mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}, \quad
\mathbf{e}_4 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
$$

That is the entire setup. Every derivation and numerical check in this post uses these exact vectors.

---

## What is a Mixture-of-Experts Layer?

A **Mixture-of-Experts (MoE) layer** replaces the single feedforward network in a standard transformer block with a collection of smaller feedforward networks (the "experts"), plus a routing mechanism that decides which experts process each token. The idea is simple: the model can have enormous total capacity (many experts = many parameters) while keeping per-token computation cheap (only a few experts fire per token).

The output of the MoE layer for token $t$ is:

$$
\mathbf{h}'_t = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \, \text{FFN}_i^{(r)}(\mathbf{u}_t)
$$

where $\mathbf{u}_t$ is the input to the MoE layer, $\text{FFN}_i^{(s)}$ are the shared experts (always active), $\text{FFN}_i^{(r)}$ are the routed experts, and $g_{i,t}$ is the **gating score** for routed expert $i$ on token $t$. Most of the $g_{i,t}$ are zero — only the top-$K_r$ selected experts have nonzero gates.

### Concrete example

In our setup ($N_s = 1$, $N_r = 4$, $K_r = 2$), this becomes:

$$
\mathbf{h}' = \mathbf{u} + \text{FFN}^{(s)}(\mathbf{u}) + g_1 \, \text{FFN}_1^{(r)}(\mathbf{u}) + g_2 \, \text{FFN}_2^{(r)}(\mathbf{u}) + g_3 \, \text{FFN}_3^{(r)}(\mathbf{u}) + g_4 \, \text{FFN}_4^{(r)}(\mathbf{u})
$$

Exactly 2 of the 4 gating scores $g_1, g_2, g_3, g_4$ will be nonzero (the top-2 selected experts), and the other 2 will be zero. The shared expert always contributes.

The entire challenge is computing those gating scores $g_{i,t}$. That requires three steps: (1) compute routing scores, (2) select the top-$K$ experts, and (3) normalize the scores into gates. We derive each step now.

---

## Step 1: Sigmoid Routing Scores

The **routing score** measures how much a given token "prefers" a given expert. We compute it by taking the dot product of the token vector $\mathbf{u}_t$ with the expert's router vector $\mathbf{e}_i$, then passing the result through the sigmoid function.

The **sigmoid function** maps any real number to the interval $(0, 1)$:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The routing score for routed expert $i$ on token $t$ is:

$$
\boxed{s_{i,t} = \sigma\!\left(\mathbf{u}_t^\top \mathbf{e}_i\right)}
$$

### Why sigmoid instead of softmax?

Many earlier MoE models use softmax over all expert scores, which forces all scores to sum to 1. This means pushing one expert's score up necessarily pushes others down — the scores are coupled. Trinity uses sigmoid routing, where each expert gets an independent score in $(0, 1)$. This decoupling leads to more stable router logits during training, which matters especially when using the Muon optimizer.

### Numerical check

Let us compute the four routing scores for our token $\mathbf{u} = [1, 0.5, -1]^\top$.

**Dot products:**

$$
\mathbf{u}^\top \mathbf{e}_1 = 1 \times 1 + 0.5 \times 0 + (-1) \times 0 = 1
$$

$$
\mathbf{u}^\top \mathbf{e}_2 = 1 \times 0 + 0.5 \times 1 + (-1) \times 0 = 0.5
$$

$$
\mathbf{u}^\top \mathbf{e}_3 = 1 \times 0 + 0.5 \times 0 + (-1) \times 1 = -1
$$

$$
\mathbf{u}^\top \mathbf{e}_4 = 1 \times 1 + 0.5 \times 1 + (-1) \times 0 = 1.5
$$

**Sigmoid scores:**

$$
s_1 = \sigma(1) = \frac{1}{1 + e^{-1}} = \frac{1}{1 + 0.368} = \frac{1}{1.368} = 0.731
$$

$$
s_2 = \sigma(0.5) = \frac{1}{1 + e^{-0.5}} = \frac{1}{1 + 0.607} = \frac{1}{1.607} = 0.622
$$

$$
s_3 = \sigma(-1) = \frac{1}{1 + e^{1}} = \frac{1}{1 + 2.718} = \frac{1}{3.718} = 0.269
$$

$$
s_4 = \sigma(1.5) = \frac{1}{1 + e^{-1.5}} = \frac{1}{1 + 0.223} = \frac{1}{1.223} = 0.818
$$

Ranking from highest to lowest: expert 4 ($0.818$) > expert 1 ($0.731$) > expert 2 ($0.622$) > expert 3 ($0.269$).

Notice that the scores are independent — they do not sum to 1 ($0.731 + 0.622 + 0.269 + 0.818 = 2.440$). Each expert gets its own "affinity" for this token.

Quick sanity check using the identity $\sigma(-x) = 1 - \sigma(x)$: we have $s_1 = \sigma(1) = 0.731$ and $s_3 = \sigma(-1) = 0.269$, and indeed $0.731 + 0.269 = 1.000$. ✓

---

## Step 2: Top-K Selection with Expert Bias

Now we select which experts to activate. The selection uses the routing score $s_{i,t}$ plus an **expert bias** $b_i$:

$$
\text{selection score for expert } i = s_{i,t} + b_i
$$

The **expert bias** is a scalar associated with each expert that gets updated during training (but outside the gradient computation — it is "decoupled" from backpropagation). Its purpose is load balancing: by increasing $b_i$ for underutilized experts and decreasing it for overutilized ones, we can steer tokens toward less popular experts.

We select the top-$K_r$ experts by their selection scores:

$$
g'_{i,t} = \begin{cases} s_{i,t}, & \text{if } s_{i,t} + b_i \in \text{Top-}K_r\!\left(\{s_{j,t} + b_j\}_{j=1}^{N_r},\; K_r\right), \\[4pt] 0, & \text{otherwise.} \end{cases}
$$

### This is the part that confuses almost everyone

Read the equation above carefully. The top-$K$ selection uses $s_{i,t} + b_i$ to decide WHICH experts get selected. But the gating value $g'_{i,t}$ for the selected experts is $s_{i,t}$ — the routing score WITHOUT the bias. The bias influences the selection, but not the weight.

Why? Because the expert bias is updated by a heuristic rule (not gradient descent). If the bias affected the gating weights, those heuristic updates would corrupt the gradient signal. By keeping the bias out of the gating computation, we ensure the gradient flows cleanly through the routing scores while still allowing the bias to redirect token-to-expert assignments.

A natural follow-up question: if the bias does not affect the gating weights, how can it change the model's behavior at all? The answer: it changes the SET of active experts. Selecting expert 2 instead of expert 4 means running a completely different FFN — even if the gating weights stay the same. The bias is a selection mechanism, not a weighting mechanism.

### Numerical check with zero bias

Starting with $b_1 = b_2 = b_3 = b_4 = 0$ (no load balancing intervention yet):

| Expert | $s_i$ | $b_i$ | $s_i + b_i$ | Selected? |
|--------|--------|--------|-------------|-----------|
| 1      | 0.731  | 0      | 0.731       | ✓ (2nd)   |
| 2      | 0.622  | 0      | 0.622       | ✗         |
| 3      | 0.269  | 0      | 0.269       | ✗         |
| 4      | 0.818  | 0      | 0.818       | ✓ (1st)   |

Top-2 by selection score: experts 4 and 1.

So $g'_4 = s_4 = 0.818$, $g'_1 = s_1 = 0.731$, $g'_2 = 0$, $g'_3 = 0$.

### Numerical check with nonzero bias

Now suppose training has been running for a while and the load balancer has set $b_1 = 0$, $b_2 = 0.2$, $b_3 = 0$, $b_4 = -0.2$. Expert 4 was overloaded, so its bias was decreased. Expert 2 was underloaded, so its bias was increased.

| Expert | $s_i$ | $b_i$ | $s_i + b_i$ | Selected? |
|--------|--------|--------|-------------|-----------|
| 1      | 0.731  | 0      | 0.731       | ✓ (2nd)   |
| 2      | 0.622  | 0.2    | 0.822       | ✓ (1st)   |
| 3      | 0.269  | 0      | 0.269       | ✗         |
| 4      | 0.818  | -0.2   | 0.618       | ✗         |

Top-2 by selection score: experts 2 and 1.

The bias flipped the selection. Expert 4, which had the highest routing score ($0.818$), got demoted because its negative bias ($-0.2$) dragged its selection score below expert 2's boosted score ($0.622 + 0.2 = 0.822$).

The gating values: $g'_2 = s_2 = 0.622$, $g'_1 = s_1 = 0.731$, $g'_3 = 0$, $g'_4 = 0$.

Notice that $g'_2 = 0.622$, not $0.822$. The bias affected selection but not the gate value. This is the decoupled design in action.

---

## Step 3: The Gating Mechanism

The gating values $g'_{i,t}$ need to be normalized so they sum to 1. We divide each nonzero gate by the sum of all nonzero gates:

$$
\boxed{g_{i,t} = \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}}}
$$

### Numerical check (zero bias case)

The nonzero gates are $g'_4 = 0.818$ and $g'_1 = 0.731$. Their sum is:

$$
0.818 + 0.731 = 1.549
$$

Normalized:

$$
g_4 = \frac{0.818}{1.549} = 0.528, \qquad g_1 = \frac{0.731}{1.549} = 0.472
$$

Check: $0.528 + 0.472 = 1.000$. ✓

### Numerical check (nonzero bias case)

The nonzero gates are $g'_2 = 0.622$ and $g'_1 = 0.731$. Their sum is:

$$
0.622 + 0.731 = 1.353
$$

Normalized:

$$
g_1 = \frac{0.731}{1.353} = 0.540, \qquad g_2 = \frac{0.622}{1.353} = 0.460
$$

Check: $0.540 + 0.460 = 1.000$. ✓

---

## The MoE Output

We now have everything we need. The MoE layer output for our token is (zero-bias case):

$$
\mathbf{h}' = \mathbf{u} + \text{FFN}^{(s)}(\mathbf{u}) + 0.472 \cdot \text{FFN}_1^{(r)}(\mathbf{u}) + 0.528 \cdot \text{FFN}_4^{(r)}(\mathbf{u})
$$

The shared expert always contributes. Of the 4 routed experts, only experts 1 and 4 fire. Expert 4 gets slightly more weight ($0.528$) than expert 1 ($0.472$) because its routing score was higher ($0.818$ vs $0.731$).

Experts 2 and 3 do nothing for this token. Their parameters are not accessed, their computation is skipped entirely. This is the source of MoE's efficiency: 400B total parameters, but only 13B worth of computation per token.

### Interpretation

Let us now step back and look at the full pipeline:

```
token u
  │
  ├─── dot product with each router vector e_i ──→ raw logits
  │
  ├─── sigmoid ──→ routing scores s_i ∈ (0,1)
  │
  ├─── add expert bias b_i ──→ selection scores (s_i + b_i)
  │
  ├─── top-K selection ──→ which experts fire (using s_i + b_i)
  │
  ├─── gate values = s_i for selected experts ──→ how much weight (using s_i only)
  │
  ├─── normalize gates ──→ g_i summing to 1
  │
  └─── weighted sum of expert outputs ──→ MoE output h'
```

The expert bias $b_i$ enters at exactly one point: the top-$K$ selection. It affects who gets chosen, not how much weight they carry. Everything else flows through the learned routing scores $s_{i,t}$.

The question that remains: how do we set the expert biases? That is the load balancing problem.

---

## The Load Balancing Problem

Imagine training our 4-expert model on many tokens. If the router learns to always prefer experts 1 and 4 (because early in training they happen to give slightly better representations), then experts 2 and 3 never get selected. Experts that never get selected never receive gradient updates. Experts that never update never improve. Experts that never improve never get selected. This is a death spiral.

The result is called **expert collapse**: a few experts handle all the work while the rest are wasted. In Trinity Large, with 256 routed experts per layer, a collapse would mean the model effectively has far fewer experts than designed — a massive waste of parameters and compute.

We need a mechanism that gently steers tokens toward underutilized experts and away from overutilized ones. This is load balancing.

### Our batch example

For the rest of this post, we track what happens across a batch of $T = 8$ tokens, all processed in a single training step. Suppose the router (with current biases) makes the following top-2 selections:

| Token | Expert selected 1 | Expert selected 2 |
|-------|-------------------|-------------------|
| 1     | 4                 | 1                 |
| 2     | 4                 | 2                 |
| 3     | 4                 | 1                 |
| 4     | 4                 | 3                 |
| 5     | 4                 | 1                 |
| 6     | 4                 | 2                 |
| 7     | 4                 | 2                 |
| 8     | 4                 | 1                 |

Every single token chose expert 4 as its first pick. Expert 4 is the "popular" expert. The load counts (number of times each expert was selected) are:

$$
n_1 = 4, \quad n_2 = 3, \quad n_3 = 1, \quad n_4 = 8
$$

Total selections: $4 + 3 + 1 + 8 = 16 = K_r \times T = 2 \times 8$. ✓

The **mean load** is:

$$
\bar{n} = \frac{1}{N_r} \sum_{i=1}^{N_r} n_i = \frac{4 + 3 + 1 + 8}{4} = \frac{16}{4} = 4
$$

In a perfectly balanced world, every expert would handle exactly 4 tokens. Instead, expert 4 handles 8 (twice the mean) and expert 3 handles just 1 (a quarter of the mean). Expert 4 is severely overloaded. Expert 3 is starving.

---

## Aux-Loss-Free Load Balancing (The Sign-Based Method)

The standard aux-loss-free approach maintains a bias vector $\mathbf{b} = [b_1, \ldots, b_{N_r}]$ that is updated after each training step using a simple rule: increase the bias for underloaded experts, decrease it for overloaded ones.

**Step 1.** Compute the mean load:

$$
\bar{n} = \frac{1}{N_r} \sum_{i=1}^{N_r} n_i
$$

**Step 2.** Update each bias using the sign of the deviation from the mean:

$$
\Delta b_i = \gamma \cdot \text{sign}(\bar{n} - n_i)
$$

where $\gamma$ is a small step size (the "bias update speed"), and the **sign function** returns $+1$ if the argument is positive, $-1$ if negative, and $0$ if zero.

**Step 3.** Apply the update:

$$
b_i \leftarrow b_i + \Delta b_i
$$

**Step 4.** Center the biases (subtract the mean so they sum to zero):

$$
b_i \leftarrow b_i - \frac{1}{N_r} \sum_{j=1}^{N_r} b_j
$$

The centering step prevents the biases from drifting collectively upward or downward, which would shift the overall selection threshold without improving balance.

### Numerical check

Using our batch loads $n_1 = 4, n_2 = 3, n_3 = 1, n_4 = 8$ and $\bar{n} = 4$, starting from $b_i = 0$, with $\gamma = 0.1$:

**Deviations** $\bar{n} - n_i$:

$$
\bar{n} - n_1 = 4 - 4 = 0, \quad \bar{n} - n_2 = 4 - 3 = 1, \quad \bar{n} - n_3 = 4 - 1 = 3, \quad \bar{n} - n_4 = 4 - 8 = -4
$$

**Sign of deviations:**

$$
\text{sign}(0) = 0, \quad \text{sign}(1) = 1, \quad \text{sign}(3) = 1, \quad \text{sign}(-4) = -1
$$

**Updates:**

$$
\Delta b_1 = 0.1 \times 0 = 0, \quad \Delta b_2 = 0.1 \times 1 = 0.1, \quad \Delta b_3 = 0.1 \times 1 = 0.1, \quad \Delta b_4 = 0.1 \times (-1) = -0.1
$$

**After applying updates** (Step 3):

$$
b_1 = 0, \quad b_2 = 0.1, \quad b_3 = 0.1, \quad b_4 = -0.1
$$

**Centering** (Step 4):

$$
\text{mean}(\mathbf{b}) = \frac{0 + 0.1 + 0.1 + (-0.1)}{4} = \frac{0.1}{4} = 0.025
$$

$$
b_1 = 0 - 0.025 = -0.025
$$

$$
b_2 = 0.1 - 0.025 = 0.075
$$

$$
b_3 = 0.1 - 0.025 = 0.075
$$

$$
b_4 = -0.1 - 0.025 = -0.125
$$

Check that the centered biases sum to zero: $(-0.025) + 0.075 + 0.075 + (-0.125) = 0$. ✓

### Interpretation

Expert 3 was the most underloaded (1 token vs mean 4) and expert 4 was the most overloaded (8 tokens vs mean 4). After the update, expert 3 has the second-highest bias ($0.075$) and expert 4 has the lowest ($-0.125$). On the next training step, the biases will push tokens toward experts 2 and 3 and away from expert 4. This is exactly the rebalancing behavior we want.

But there is a problem hiding in the sign function.

---

## Why Sign-Based Updates Oscillate

Look again at the updates: expert 2 had a deviation of $\bar{n} - n_2 = 1$ (slightly underloaded) and expert 3 had a deviation of $\bar{n} - n_3 = 3$ (severely underloaded). Both received the exact same update $\Delta b = +0.1$, because $\text{sign}(1) = \text{sign}(3) = 1$.

The sign function is blind to magnitude. It treats a tiny imbalance and a massive imbalance identically.

This becomes a serious problem near convergence. Suppose after many training steps the loads become nearly balanced: $n_1 = 4, n_2 = 4, n_3 = 3, n_4 = 5$. The mean is still $\bar{n} = 4$.

**Sign-based updates for the nearly balanced case:**

$$
\Delta b_3 = 0.1 \times \text{sign}(4 - 3) = 0.1 \times 1 = 0.1
$$

$$
\Delta b_4 = 0.1 \times \text{sign}(4 - 5) = 0.1 \times (-1) = -0.1
$$

Expert 3 is only 1 token below average, yet it gets the full $+0.1$ boost — the same magnitude as when it was 3 tokens below average. Expert 4 is only 1 token above average, yet it gets the full $-0.1$ penalty.

These large updates overshoot. On the next step, expert 3 might become slightly overloaded, triggering a $-0.1$ swing in the other direction. Then it undershoots again. The biases oscillate around the equilibrium, never settling.

The paper puts it precisely: "Under the assumption that the ideal expert bias value is a fixed value, we note that the standard aux-loss-free load balancing cannot precisely converge on that value, as each local update under the $\text{sign}(\cdot)$ operator is always $\pm\gamma$."

As the total number of experts increases (Trinity Large has 256), the per-layer bias norm grows, making the oscillations larger and contributing to training instability.

We need an update rule that is aggressive when the imbalance is large and gentle when the imbalance is small. We need SMEBU.

---

## SMEBU: Soft-Clamped Momentum Expert Bias Updates

SMEBU replaces the sign-based update with three modifications: (1) a normalized, magnitude-aware update via $\tanh$, (2) centering, and (3) momentum smoothing. We derive each step.

### Step 1: Normalize the Violation

First, we compute how far each expert's load deviates from the mean, as a fraction of the mean:

$$
\boxed{v_i = \frac{\bar{n} - n_i}{\bar{n}}}
$$

We call $v_i$ the **normalized violation** for expert $i$. A positive $v_i$ means the expert is underloaded (fewer tokens than average). A negative $v_i$ means overloaded.

Dividing by $\bar{n}$ makes the violation scale-independent. Whether the batch has 8 tokens or 8 million, $v_i$ lives on the same scale. An expert handling twice the mean load always has $v_i = -1$, regardless of the absolute numbers.

#### Numerical check (heavily imbalanced)

Using our loads $n_1 = 4, n_2 = 3, n_3 = 1, n_4 = 8$ with $\bar{n} = 4$:

$$
v_1 = \frac{4 - 4}{4} = 0, \quad v_2 = \frac{4 - 3}{4} = 0.25, \quad v_3 = \frac{4 - 1}{4} = 0.75, \quad v_4 = \frac{4 - 8}{4} = -1
$$

Expert 3 has $v_3 = 0.75$: it handled only 25% of its fair share. Expert 4 has $v_4 = -1$: it handled twice its fair share.

#### Numerical check (nearly balanced)

Using loads $n_1 = 4, n_2 = 4, n_3 = 3, n_4 = 5$ with $\bar{n} = 4$:

$$
v_1 = 0, \quad v_2 = 0, \quad v_3 = \frac{4-3}{4} = 0.25, \quad v_4 = \frac{4-5}{4} = -0.25
$$

The violations are much smaller now. Under the sign-based method, these would all produce the same $\pm\gamma$ updates. Under SMEBU, they produce proportionally smaller updates, as we will see next.

### Step 2: Soft-Clamp with tanh

We apply the **hyperbolic tangent** function, scaled by a parameter $\kappa$:

$$
\boxed{\tilde{v}_i = \tanh(\kappa \, v_i)}
$$

The **hyperbolic tangent** function is:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

It maps any real number to the interval $(-1, 1)$. Three properties make it perfect for this job:

**Property 1: Near zero, tanh is approximately the identity.** For small $|x|$:

$$
\tanh(x) \approx x
$$

We can see why. When $|x|$ is small, $e^x \approx 1 + x$ and $e^{-x} \approx 1 - x$. Substituting:

$$
\tanh(x) \approx \frac{(1+x) - (1-x)}{(1+x) + (1-x)} = \frac{2x}{2} = x
$$

So near balance (small $v_i$), the update is proportional to the violation itself. A tiny imbalance produces a tiny update.

**Property 2: Far from zero, tanh saturates at $\pm 1$.** As $|x| \to \infty$, $\tanh(x) \to \pm 1$. So for large imbalances, the update is bounded — we never apply an update larger than $\lambda$ (the learning rate from Step 3).

**Property 3: tanh is a smooth approximation of sign.** In fact:

$$
\text{sign}(x) = \lim_{a \to \infty} \tanh(a \, x)
$$

The parameter $\kappa$ controls how quickly tanh transitions from the linear regime to the saturated regime. Large $\kappa$ makes it behave more like sign. Small $\kappa$ makes it more linear. Trinity Large uses $\kappa = 2$.

Here is a comparison table with $\kappa = 2$:

| $v_i$ (violation) | $\kappa v_i$ | $\tanh(\kappa v_i)$ | $\text{sign}(v_i)$ | Ratio |
|--------------------|-------------|----------------------|---------------------|-------|
| $-1.00$ | $-2.00$ | $-0.964$ | $-1$ | $96.4\%$ |
| $-0.75$ | $-1.50$ | $-0.905$ | $-1$ | $90.5\%$ |
| $-0.50$ | $-1.00$ | $-0.762$ | $-1$ | $76.2\%$ |
| $-0.25$ | $-0.50$ | $-0.462$ | $-1$ | $46.2\%$ |
| $-0.10$ | $-0.20$ | $-0.197$ | $-1$ | $19.7\%$ |
| $\phantom{-}0.00$ | $\phantom{-}0.00$ | $\phantom{-}0.000$ | $\phantom{-}0$ | — |
| $+0.10$ | $+0.20$ | $+0.197$ | $+1$ | $19.7\%$ |
| $+0.25$ | $+0.50$ | $+0.462$ | $+1$ | $46.2\%$ |
| $+0.50$ | $+1.00$ | $+0.762$ | $+1$ | $76.2\%$ |
| $+0.75$ | $+1.50$ | $+0.905$ | $+1$ | $90.5\%$ |
| $+1.00$ | $+2.00$ | $+0.964$ | $+1$ | $96.4\%$ |

The "Ratio" column shows how much of the full sign-step SMEBU applies. At large violations ($|v| = 1$), SMEBU applies 96.4% of the sign step — nearly identical. At moderate violations ($|v| = 0.25$), it applies 46.2%. At tiny violations ($|v| = 0.1$), it applies only 19.7%.

This is exactly the behavior we wanted: aggressive for large imbalances, gentle near equilibrium.

#### Numerical check (heavily imbalanced)

Violations: $v_1 = 0, v_2 = 0.25, v_3 = 0.75, v_4 = -1$. With $\kappa = 2$:

$$
\tilde{v}_1 = \tanh(2 \times 0) = \tanh(0) = 0
$$

$$
\tilde{v}_2 = \tanh(2 \times 0.25) = \tanh(0.5) = 0.462
$$

$$
\tilde{v}_3 = \tanh(2 \times 0.75) = \tanh(1.5) = 0.905
$$

$$
\tilde{v}_4 = \tanh(2 \times (-1)) = \tanh(-2) = -0.964
$$

Let us verify $\tanh(0.5)$ explicitly:

$$
\tanh(0.5) = \frac{e^{0.5} - e^{-0.5}}{e^{0.5} + e^{-0.5}} = \frac{1.649 - 0.607}{1.649 + 0.607} = \frac{1.042}{2.256} = 0.462 \quad \checkmark
$$

And $\tanh(-2)$:

$$
\tanh(-2) = -\tanh(2) = -\frac{e^{2} - e^{-2}}{e^{2} + e^{-2}} = -\frac{7.389 - 0.135}{7.389 + 0.135} = -\frac{7.254}{7.524} = -0.964 \quad \checkmark
$$

#### Numerical check (nearly balanced)

Violations: $v_1 = 0, v_2 = 0, v_3 = 0.25, v_4 = -0.25$. With $\kappa = 2$:

$$
\tilde{v}_1 = 0, \quad \tilde{v}_2 = 0, \quad \tilde{v}_3 = \tanh(0.5) = 0.462, \quad \tilde{v}_4 = \tanh(-0.5) = -0.462
$$

Compare with sign: $\text{sign}(v_3) = 1$, $\text{sign}(v_4) = -1$.

SMEBU gives $0.462$ where sign gives $1.0$. The update is less than half the sign-based step, because the imbalance is moderate. Near perfect balance, the ratio would shrink further — for $v = 0.05$, SMEBU gives only $\tanh(0.1) = 0.100$, which is 10% of the sign step.

### Step 3: Scale, Center, and Apply Momentum

The remaining three operations turn the soft-clamped violations into actual bias updates.

**Scale** by the load-balance learning rate $\lambda$:

$$
\Delta b_i = \lambda \, \tilde{v}_i
$$

**Center** the updates so they sum to zero:

$$
\Delta b_i \leftarrow \Delta b_i - \frac{1}{N_r} \sum_{j=1}^{N_r} \Delta b_j
$$

**Apply momentum**, maintaining a momentum buffer $m_i$ (initialized to 0):

$$
m_i \leftarrow \beta \, m_i + (1 - \beta) \, \Delta b_i
$$

**Update the bias:**

$$
b_i \leftarrow b_i + m_i
$$

The **momentum** here works exactly like momentum in SGD. Instead of applying the raw update $\Delta b_i$ directly, we maintain a running average $m_i$ that blends the current update with past updates. When the updates are noisy (pointing in different directions on different steps), the momentum averages out the noise. When the updates are consistent (pointing in the same direction), the momentum accumulates and accelerates convergence.

The parameter $\beta$ controls the memory: $\beta = 0$ means no momentum (use the raw update), $\beta = 1$ means infinite memory (ignore new updates). Trinity Large uses $\beta = 0.5$, giving equal weight to the current update and the accumulated history.

Why does momentum help here? Near convergence, expert loads fluctuate randomly around the mean — sometimes expert $i$ gets one extra token, sometimes one fewer. These fluctuations produce small, noisy, rapidly-alternating bias updates. Without momentum, the biases jitter. With momentum, consecutive opposing updates ($+\epsilon, -\epsilon, +\epsilon, \ldots$) cancel in the running average, and the bias stays steady.

---

## Full Numerical Walkthrough: SMEBU vs Sign-Based

Let us trace both methods through our heavily imbalanced batch ($n_1 = 4, n_2 = 3, n_3 = 1, n_4 = 8$, $\bar{n} = 4$), starting from $b_i = 0$ and $m_i = 0$.

**Hyperparameters:** $\gamma = 0.1$ for sign-based, $\lambda = 0.1$, $\kappa = 2$, $\beta = 0.5$ for SMEBU.

### Sign-based method

| Step | Formula | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|------|---------|----------|----------|----------|----------|
| Loads $n_i$ | — | 4 | 3 | 1 | 8 |
| $\bar{n} - n_i$ | — | 0 | 1 | 3 | $-4$ |
| $\text{sign}(\bar{n} - n_i)$ | — | 0 | 1 | 1 | $-1$ |
| $\Delta b_i$ | $\gamma \cdot \text{sign}$ | 0 | 0.1 | 0.1 | $-0.1$ |
| After add | $b_i + \Delta b_i$ | 0 | 0.1 | 0.1 | $-0.1$ |
| Mean of $\mathbf{b}$ | — | | $0.025$ | | |
| After center | $b_i - \text{mean}$ | $-0.025$ | $0.075$ | $0.075$ | $-0.125$ |

Experts 2 and 3 got the same update ($0.1$), despite expert 3 being far more underloaded. The sign function erased the magnitude information.

### SMEBU method

| Step | Formula | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|------|---------|----------|----------|----------|----------|
| Loads $n_i$ | — | 4 | 3 | 1 | 8 |
| $v_i$ | $\frac{\bar{n}-n_i}{\bar{n}}$ | 0 | 0.25 | 0.75 | $-1.00$ |
| $\kappa v_i$ | $2 v_i$ | 0 | 0.50 | 1.50 | $-2.00$ |
| $\tilde{v}_i$ | $\tanh(\kappa v_i)$ | 0 | 0.462 | 0.905 | $-0.964$ |
| $\Delta b_i$ | $\lambda \tilde{v}_i$ | 0 | 0.0462 | 0.0905 | $-0.0964$ |
| Mean of $\Delta\mathbf{b}$ | — | | $0.0101$ | | |
| Centered $\Delta b_i$ | $\Delta b_i - \text{mean}$ | $-0.0101$ | $0.0361$ | $0.0804$ | $-0.1065$ |
| $m_i$ | $0.5 \times 0 + 0.5 \times \Delta b_i$ | $-0.0051$ | $0.0181$ | $0.0402$ | $-0.0532$ |
| $b_i$ | $0 + m_i$ | $-0.0051$ | $0.0181$ | $0.0402$ | $-0.0532$ |

Let us verify the mean of $\Delta\mathbf{b}$ before centering:

$$
\frac{0 + 0.0462 + 0.0905 + (-0.0964)}{4} = \frac{0.0403}{4} = 0.0101 \quad \checkmark
$$

And verify the centered updates sum to zero:

$$
(-0.0101) + 0.0361 + 0.0804 + (-0.1065) = -0.0001 \approx 0 \quad \checkmark
$$

(The tiny residual is from rounding to 4 decimal places.)

### Comparing the results

| Expert | Sign-based $b_i$ | SMEBU $b_i$ |
|--------|-------------------|-------------|
| 1 (balanced) | $-0.025$ | $-0.005$ |
| 2 (slightly under) | $+0.075$ | $+0.018$ |
| 3 (severely under) | $+0.075$ | $+0.040$ |
| 4 (severely over) | $-0.125$ | $-0.053$ |

Two critical differences:

**1. SMEBU differentiates by severity.** Under sign-based, experts 2 and 3 both got $b_i = 0.075$ — the same bias despite very different loads (3 vs 1). Under SMEBU, expert 3 got $0.040$ and expert 2 got $0.018$. SMEBU gave more help to the expert that needed it more.

**2. SMEBU gives smaller updates overall.** The momentum halves the first-step update (since $\beta = 0.5$ and $m_i$ starts at 0). On subsequent steps, the momentum accumulates consistent signals and dampens noise. The sign-based method applies the full $\gamma$ every step regardless.

---

## Near-Balance Behavior: Where SMEBU Truly Shines

The comparison above shows SMEBU's advantages for a heavily imbalanced batch. But the difference becomes even more dramatic near convergence.

Suppose after many training steps the loads are nearly balanced: $n_1 = 4, n_2 = 4, n_3 = 3, n_4 = 5$, $\bar{n} = 4$. This is only a tiny deviation from perfect balance.

### Sign-based

$$
\Delta b_3 = 0.1 \times \text{sign}(4-3) = 0.1 \times 1 = 0.1
$$

$$
\Delta b_4 = 0.1 \times \text{sign}(4-5) = 0.1 \times (-1) = -0.1
$$

The full $\pm 0.1$ step. The same magnitude as when expert 4 was carrying double the load. The update does not know that balance is almost achieved.

### SMEBU

$$
v_3 = 0.25, \quad v_4 = -0.25
$$

$$
\tilde{v}_3 = \tanh(0.5) = 0.462, \quad \tilde{v}_4 = \tanh(-0.5) = -0.462
$$

$$
\Delta b_3 = 0.1 \times 0.462 = 0.0462, \quad \Delta b_4 = 0.1 \times (-0.462) = -0.0462
$$

After centering and momentum, the effective step is even smaller. SMEBU recognizes that the imbalance is mild and responds gently.

If the imbalance were even tinier — say $n_3 = 4, n_4 = 4$ with one token fluctuation — the violation would be $v \approx 0.06$, giving $\tanh(0.12) \approx 0.119$, which is only 11.9% of the sign step. The biases would barely budge, because there is barely anything to fix. This is convergence.

The fundamental problem with the sign function, stated precisely: it maps the continuous violation signal to the discrete set $\{-1, 0, +1\}$, destroying all magnitude information. The tanh function preserves magnitude while still bounding the updates to $(-1, +1)$, preventing any single step from being catastrophically large. It is a "continuous relaxation of the discrete update," exactly as the paper describes.

---

## Connecting It All: The Unified View

Let us step back and see the sign-based and SMEBU methods as special cases of a single framework. Both methods compute a bias update of the form:

$$
\Delta b_i = \lambda \cdot f\!\left(\frac{\bar{n} - n_i}{\bar{n}}\right)
$$

where $f(\cdot)$ is a function that maps the normalized violation to an update magnitude.

For the sign-based method:

$$
f(v) = \text{sign}(v)
$$

For SMEBU:

$$
f(v) = \tanh(\kappa \, v)
$$

Both functions are odd ($f(-v) = -f(v)$), both are bounded ($|f(v)| \leq 1$), and both have $f(0) = 0$. The difference is entirely in how they treat intermediate values:

- $\text{sign}$ is a step function: it jumps from 0 to $\pm 1$ at $v = 0$, with no values in between.
- $\tanh$ is a smooth S-curve: it transitions gradually, with $f(v) \approx \kappa v$ near zero and $f(v) \to \pm 1$ far from zero.

As $\kappa \to \infty$, the tanh curve becomes steeper and approaches the sign function. As $\kappa \to 0$, the tanh curve becomes shallower and approaches a pure linear update $f(v) = \kappa v$. The parameter $\kappa$ controls where on this spectrum we sit.

Trinity Large uses $\kappa = 2$, which is in the moderate range: the update is noticeably different from sign for violations smaller than about 0.5, but behaves almost identically to sign for violations larger than 1.

Adding momentum is the second key difference. The momentum buffer $m_i$ acts as a low-pass filter on the update sequence. High-frequency noise (random fluctuations in expert loads) gets attenuated, while low-frequency signals (persistent imbalances) pass through and accumulate. This is the same principle behind why momentum SGD converges faster than vanilla SGD in noisy settings.

---

## The Full SMEBU Algorithm

For reference, here is the complete SMEBU update, combining all the pieces we derived:

**Given:** Expert loads $n_1, \ldots, n_{N_r}$ from the current training step. Maintained state: bias vector $\mathbf{b}$, momentum buffer $\mathbf{m}$ (both initialized to zero). Hyperparameters: $\lambda$ (learning rate), $\kappa$ (tanh scale), $\beta$ (momentum).

$$
\bar{n} = \frac{1}{N_r}\sum_{i=1}^{N_r} n_i
$$

$$
v_i = \frac{\bar{n} - n_i}{\bar{n}} \qquad \text{(normalized violation)}
$$

$$
\tilde{v}_i = \tanh(\kappa \, v_i) \qquad \text{(soft clamp)}
$$

$$
\Delta b_i = \lambda \, \tilde{v}_i \qquad \text{(scale)}
$$

$$
\Delta b_i \leftarrow \Delta b_i - \frac{1}{N_r}\sum_{j=1}^{N_r} \Delta b_j \qquad \text{(center)}
$$

$$
m_i \leftarrow \beta \, m_i + (1 - \beta)\,\Delta b_i \qquad \text{(momentum)}
$$

$$
\boxed{b_i \leftarrow b_i + m_i} \qquad \text{(update)}
$$

Trinity Large uses $\lambda = 5 \times 10^{-4}$, $\kappa = 2$, $\beta = 0.5$.

---

## Summary

We built the Mixture-of-Experts routing mechanism from the ground up: a token vector hits each expert's router vector, the dot products pass through sigmoid to produce independent routing scores in $(0,1)$, the expert bias shifts the selection threshold without affecting the gating weights (the decoupled design), the top-$K$ experts fire, and their outputs are weighted by normalized routing scores. The expert bias is the lever for load balancing — increasing it for underused experts, decreasing it for overused ones — and SMEBU is the mechanism that adjusts that lever intelligently. By replacing the sign function with $\tanh$, SMEBU produces updates proportional to the severity of the imbalance: aggressive corrections for large deviations, gentle nudges near equilibrium, and convergence to zero updates at perfect balance. Momentum smooths out the noise from stochastic load fluctuations, preventing the oscillation that plagues sign-based methods. Together, these changes enabled Trinity Large to train stably with 256 experts per layer across 17 trillion tokens with zero loss spikes.
