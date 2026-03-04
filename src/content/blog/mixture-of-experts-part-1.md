---
title: "Mixture of Experts from Scratch — Part 1: The Foundations (1991–1993)"
description: "Building Mixture of Experts from the ground up — adaptive expert networks, gating functions, the mixture-of-Gaussians interpretation, hierarchical mixtures, and the EM algorithm — all derived step by step with a 2-expert regression example."
date: 2026-03-04
tags: [machine-learning, mixture-of-experts, neural-networks, em-algorithm, mathematics]
order: 3
---

We are going to build the Mixture of Experts (MoE) framework from scratch, starting from the two papers that created it: [Jacobs, Jordan, Nowlan & Hinton (1991)](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf) and [Jordan & Jacobs (1993)](https://www.cs.toronto.edu/~hinton/absps/hme.pdf). By the end of this post, we will have derived every equation in both papers using a single running example — a system of 2 experts trying to learn a piecewise-linear function.

---

## The Running Example

Suppose we have data from a function that behaves differently in two regions of the input space. For $x < 0$, the true function is $y = -x$ (a line with slope $-1$), and for $x \geq 0$, the true function is $y = 2x$ (a line with slope $+2$). A single linear model cannot capture both regions simultaneously. But two linear experts — one specializing in negative $x$, the other in positive $x$ — can.

We will use 2 experts, each a simple linear function:

$$
o_1 = w_1 x, \quad o_2 = w_2 x
$$

and a **gating network** that decides how much to trust each expert for a given input $x$. Concretely, let us work with three training cases:

| Case $c$ | Input $x^c$ | Target $d^c$ |
|-----------|-------------|--------------|
| 1         | $-1$        | $1$          |
| 2         | $0.5$       | $1$          |
| 3         | $1$         | $2$          |

Cases 2 and 3 come from the $y = 2x$ region; case 1 comes from the $y = -x$ region. We want our system to learn this decomposition automatically.

---

## 1. The Cooperative Error Function (and Why It Fails)

The most obvious approach is to linearly combine the expert outputs and compare the blend to the target. Hampshire & Waibel (1989) and Jacobs et al. (1990) used exactly this idea. Let $p_i^c$ be the proportion that the gating network assigns to expert $i$ for case $c$, and let $\mathbf{o}_i^c$ be the output of expert $i$ on case $c$. The **cooperative error** on case $c$ is:

$$
E^c = \left\| \mathbf{d}^c - \sum_i p_i^c \mathbf{o}_i^c \right\|^2
$$

This is the squared difference between the desired output $\mathbf{d}^c$ and the weighted combination of all expert outputs.

**The problem with cooperation.** To minimize this error, each expert must produce its output *and* cancel the residual error left by the combined effects of all other experts. The gradient of $E^c$ with respect to expert $i$'s output is:

$$
\frac{\partial E^c}{\partial \mathbf{o}_i^c} = -2p_i^c (\mathbf{d}^c - \mathbf{o}_i^c) \quad \text{... wait, this is wrong.}
$$

Let us derive it carefully. From the definition:

$$
E^c = \left\| \mathbf{d}^c - \sum_j p_j^c \mathbf{o}_j^c \right\|^2
$$

We apply the **chain rule of calculus**. The derivative of $\|u\|^2$ with respect to a parameter inside $u$ is $2u$ times the derivative of $u$. The quantity inside the norm that depends on $\mathbf{o}_i^c$ is $-p_i^c \mathbf{o}_i^c$. So:

$$
\frac{\partial E^c}{\partial \mathbf{o}_i^c} = -2p_i^c \left( \mathbf{d}^c - \sum_j p_j^c \mathbf{o}_j^c \right)
$$

This is equation (1.4) from Jacobs et al. (1991) — but notice the crucial coupling: the gradient for expert $i$ depends on the outputs of *all other experts* through the sum $\sum_j p_j^c \mathbf{o}_j^c$. When the weights $p_j^c$ in other experts change, the residual changes, and so the error derivatives for expert $i$ change too. This strong coupling causes the experts to cooperate — many experts contribute small pieces to each case — rather than specialize.

### Numerical check

Let us verify with our running example. Suppose at initialization both experts output 0 ($w_1 = w_2 = 0$) and the gating network assigns equal proportions $p_1^c = p_2^c = 0.5$ for all cases. For case 1 ($x = -1$, $d = 1$):

$$
E^1 = |1 - 0.5 \cdot 0 - 0.5 \cdot 0|^2 = |1|^2 = 1
$$

$$
\frac{\partial E^1}{\partial o_1^1} = -2 \times 0.5 \times (1 - 0) = -1
$$

Both experts get the exact same gradient — they are pushed in the same direction. There is no mechanism forcing them to specialize. This is the fundamental limitation of the cooperative error.

---

## 2. The Competitive Error Function

Jacobs et al. (1991) proposed a beautifully simple fix: instead of blending expert outputs, imagine that the gating network makes a **stochastic decision** about which single expert to use on each occasion. The error becomes the **expected squared difference** when the gating network stochastically selects one expert:

$$
E^c = \langle \| \mathbf{d}^c - \mathbf{o}_i^c \|^2 \rangle = \sum_i p_i^c \| \mathbf{d}^c - \mathbf{o}_i^c \|^2
$$

This is equation (1.2) in Jacobs et al. (1991). Notice the critical difference: each expert $i$ is now responsible for producing the *entire* output $\mathbf{d}^c$, not just a piece of it. The term $\|\mathbf{d}^c - \mathbf{o}_i^c\|^2$ is the error expert $i$ would make if it alone had to handle case $c$.

**Why this encourages specialization.** Let us take the derivative with respect to $\mathbf{o}_i^c$. Only the $i$-th term in the sum depends on $\mathbf{o}_i^c$:

$$
\frac{\partial E^c}{\partial \mathbf{o}_i^c} = -2p_i^c (\mathbf{d}^c - \mathbf{o}_i^c)
$$

This is equation (1.4) from the paper. Now the gradient for expert $i$ depends *only* on expert $i$'s own output and the target — no other experts appear. The experts are **decoupled**. Each expert is pushed toward the target independently, weighted by its responsibility $p_i^c$.

### Numerical check

With the same initialization ($w_1 = w_2 = 0$, $p_1 = p_2 = 0.5$), for case 1:

$$
E^1 = 0.5 \times |1 - 0|^2 + 0.5 \times |1 - 0|^2 = 0.5 + 0.5 = 1
$$

The total error is the same as before. But now:

$$
\frac{\partial E^1}{\partial o_1^1} = -2 \times 0.5 \times (1 - 0) = -1
$$

$$
\frac{\partial E^1}{\partial o_2^1} = -2 \times 0.5 \times (1 - 0) = -1
$$

At initialization, the gradients are identical — but as training proceeds and the gating network shifts responsibility (increasing $p_i^c$ for the expert that does better on case $c$), the expert that is losing responsibility gets a *smaller* gradient magnitude, while the winning expert gets a *larger* one. The competition is built into the weighting.

---

## 3. The Gating Network and Softmax

The gating network receives the same input $\mathbf{x}$ as the experts and produces the mixing proportions $p_1, p_2, \ldots, p_n$. These must be positive and sum to 1 — they represent probabilities. The standard way to achieve this is the **softmax function**.

The gating network computes a linear function of the input for each expert:

$$
s_j = \mathbf{v}_j^T \mathbf{x}
$$

where $\mathbf{v}_j$ is the weight vector for expert $j$ in the gating network, and $\mathbf{x}$ is the input (including a bias term of 1). Then:

$$
\boxed{p_j = \frac{e^{s_j}}{\sum_k e^{s_k}}}
$$

This is the **softmax function** (Bridle, 1989). It maps any real-valued vector $(s_1, \ldots, s_n)$ to a probability distribution: every $p_j > 0$ and $\sum_j p_j = 1$.

### Numerical check

In our 2-expert example, suppose the gating network weights for case 1 ($x = -1$) produce $s_1 = 2$ and $s_2 = -1$. Then:

$$
p_1 = \frac{e^{2}}{e^{2} + e^{-1}} = \frac{7.389}{7.389 + 0.368} = \frac{7.389}{7.757} = 0.953
$$

$$
p_2 = \frac{e^{-1}}{e^{2} + e^{-1}} = \frac{0.368}{7.757} = 0.047
$$

Expert 1 gets 95.3% of the responsibility for case 1. We can verify: $0.953 + 0.047 = 1.000$. The softmax has converted arbitrary real numbers into a valid probability distribution.

---

## 4. The Mixture of Gaussians Interpretation

Here is where Jacobs et al. (1991) make a deep connection. The competitive error function in equation (1.2) was motivated by a stochastic selection argument, but in practice the authors used a different error function that gives better performance:

$$
E^c = -\log \sum_i p_i^c \, e^{-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_i^c\|^2}
$$

This is equation (1.3) in the paper. Where does it come from?

The term $e^{-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_i^c\|^2}$ is (up to a normalizing constant) a **Gaussian probability density** centered at the expert's output $\mathbf{o}_i^c$ with unit variance, evaluated at the target $\mathbf{d}^c$. So the sum $\sum_i p_i^c \, e^{-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_i^c\|^2}$ is proportional to the probability of generating $\mathbf{d}^c$ under a **mixture of Gaussians** — a model where the output is generated by first picking expert $i$ with probability $p_i^c$, then drawing from a Gaussian centered at $\mathbf{o}_i^c$.

The negative log of a probability is the **negative log-likelihood**. Minimizing $E^c$ is equivalent to **maximizing the likelihood** that the mixture model generated the observed target. This is a fundamental principle: the **maximum likelihood estimation** (MLE) framework.

### Deriving the gradient under the log-likelihood error

Let us derive $\frac{\partial E^c}{\partial \mathbf{o}_i^c}$ for this new error function. We have:

$$
E^c = -\log \sum_j p_j^c \, e^{-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_j^c\|^2}
$$

Let $L^c = \sum_j p_j^c \, e^{-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_j^c\|^2}$, so $E^c = -\log L^c$.

By the **chain rule**:

$$
\frac{\partial E^c}{\partial \mathbf{o}_i^c} = -\frac{1}{L^c} \frac{\partial L^c}{\partial \mathbf{o}_i^c}
$$

Now $L^c = \sum_j p_j^c \, e^{-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_j^c\|^2}$. Only the $i$-th term depends on $\mathbf{o}_i^c$. Let $f_i = e^{-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_i^c\|^2}$. We need $\frac{\partial f_i}{\partial \mathbf{o}_i^c}$.

The exponent is $-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_i^c\|^2$. By the **chain rule** (derivative of $e^{g}$ is $e^{g} \cdot g'$, and derivative of $-\frac{1}{2}\|u\|^2$ with respect to $\mathbf{o}_i^c$ where $u = \mathbf{d}^c - \mathbf{o}_i^c$ is $+(\mathbf{d}^c - \mathbf{o}_i^c)$):

$$
\frac{\partial f_i}{\partial \mathbf{o}_i^c} = f_i \cdot (\mathbf{d}^c - \mathbf{o}_i^c)
$$

So:

$$
\frac{\partial L^c}{\partial \mathbf{o}_i^c} = p_i^c \, f_i \, (\mathbf{d}^c - \mathbf{o}_i^c)
$$

Putting it together:

$$
\frac{\partial E^c}{\partial \mathbf{o}_i^c} = -\frac{p_i^c \, f_i}{L^c} (\mathbf{d}^c - \mathbf{o}_i^c)
$$

Now define the **posterior probability** (or responsibility) of expert $i$ for case $c$:

$$
h_i^c = \frac{p_i^c \, e^{-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_i^c\|^2}}{\sum_j p_j^c \, e^{-\frac{1}{2}\|\mathbf{d}^c - \mathbf{o}_j^c\|^2}} = \frac{p_i^c \, f_i}{L^c}
$$

This is **Bayes' theorem** in action: $h_i^c$ is the posterior probability that expert $i$ generated the target $\mathbf{d}^c$, given the prior $p_i^c$ and the Gaussian likelihood $f_i$. So the gradient becomes:

$$
\boxed{\frac{\partial E^c}{\partial \mathbf{o}_i^c} = -h_i^c (\mathbf{d}^c - \mathbf{o}_i^c)}
$$

This is equation (1.5) from Jacobs et al. (1991). Compare this to the gradient from the simple competitive error (equation 1.4): $-2p_i^c(\mathbf{d}^c - \mathbf{o}_i^c)$. The crucial difference is that $p_i^c$ (the prior) has been replaced by $h_i^c$ (the posterior). The posterior takes into account *how well* expert $i$ actually fits the data, not just the gating network's prior assignment. An expert that fits the data well gets a large posterior even if its prior is moderate, and an expert that fits poorly gets a small posterior even if it has a large prior.

**This is the part that makes the system work.** Early in training, all experts have similar outputs, so the posteriors $h_i^c$ are close to the priors $p_i^c$. As training proceeds and one expert begins to fit a particular case better, its posterior for that case increases, giving it a larger gradient and thus faster learning on that case. The system spontaneously discovers which expert should handle which subset of the data.

### Numerical check

Suppose expert 1 has output $o_1^1 = 0.8$ on case 1 ($d^1 = 1$) and expert 2 has output $o_2^1 = 0.3$. The gating network gives $p_1^1 = 0.6$, $p_2^1 = 0.4$.

Errors: $\|\mathbf{d}^1 - \mathbf{o}_1^1\|^2 = |1 - 0.8|^2 = 0.04$, $\|\mathbf{d}^1 - \mathbf{o}_2^1\|^2 = |1 - 0.3|^2 = 0.49$.

Gaussian likelihoods: $f_1 = e^{-0.02} = 0.980$, $f_2 = e^{-0.245} = 0.783$.

Mixture: $L^1 = 0.6 \times 0.980 + 0.4 \times 0.783 = 0.588 + 0.313 = 0.901$.

Posteriors:

$$
h_1^1 = \frac{0.588}{0.901} = 0.653, \quad h_2^1 = \frac{0.313}{0.901} = 0.347
$$

Expert 1 started with prior $p_1 = 0.6$ and ended with posterior $h_1 = 0.653$. Its responsibility *increased* because it fits case 1 better ($o_1 = 0.8$ vs. $d = 1$, error $= 0.04$) compared to expert 2 ($o_2 = 0.3$ vs. $d = 1$, error $= 0.49$). The Bayesian update shifted responsibility toward the better-fitting expert.

Gradients:

$$
\frac{\partial E^1}{\partial o_1^1} = -0.653 \times (1 - 0.8) = -0.653 \times 0.2 = -0.131
$$

$$
\frac{\partial E^1}{\partial o_2^1} = -0.347 \times (1 - 0.3) = -0.347 \times 0.7 = -0.243
$$

Interestingly, expert 2 gets a *larger* gradient magnitude despite having lower responsibility, because it is farther from the target. But the gating network will increasingly route case 1 away from expert 2 as training continues, reducing its responsibility $h_2^1$ toward zero.

---

## 5. The Architecture

Let us now put the pieces together into a complete architecture, as shown in Figure 1 of Jacobs et al. (1991).

```
                           ô (output)
                            ↑
                     ┌──────┴──────┐
                     │  Stochastic │
                     │  Selector   │
                     ├──────┬──────┤
                   p₁       │     p₂
                    ↑       │      ↑
              ┌─────┴───┐   │  ┌───┴─────┐
              │Expert 1 │   │  │Expert 2 │
              │ o₁=w₁x  │   │  │ o₂=w₂x  │
              └────┬────┘   │  └────┬────┘
                   │        │       │
                   │   ┌────┴────┐  │
                   │   │ Gating  │  │
                   │   │ Network │  │
                   │   └────┬────┘  │
                   │        │       │
                   └────────┼───────┘
                            │
                          input x
```

All networks receive the same input $x$. The experts produce outputs $o_1, o_2$. The gating network produces mixing proportions $p_1, p_2$ via softmax. The selector stochastically chooses one expert according to $(p_1, p_2)$, or in practice we train using the log-likelihood error which uses the mixture of all experts weighted by their responsibilities.

---

## 6. From Flat to Hierarchical: The HME Architecture

Two years after the original paper, Jordan & Jacobs (1993) introduced the **Hierarchical Mixture of Experts** (HME). The key insight: instead of having a single flat layer of experts, arrange them in a tree.

In a two-level hierarchy with 2 branches at each level, we get 4 expert networks at the leaves and 3 gating networks (one at the top, two at the second level):

```
                         μ (output)
                         ↑
                   ┌─────┴─────┐
                   │Top Gating │
                   │  g₁  g₂   │
                   └──┬─────┬──┘
                      │     │
              ┌───────┴┐   ┌┴───────┐
              │Gating 1│   │Gating 2│
              │g₁|₁g₂|₁│   │g₁|₂g₂|₂│
              └┬─────┬─┘   └┬─────┬─┘
               │     │      │     │
            Expert Expert Expert Expert
            (1,1) (1,2)  (2,1) (2,2)
               ↑     ↑      ↑     ↑
               └─────┴──┬───┴─────┘
                        │
                      input x
```

The top-level gating network produces probabilities $g_1$ and $g_2$ (which branch to take). The lower-level gating networks produce conditional probabilities $g_{j|i}$ (which expert within branch $i$).

Each expert network $(i,j)$ produces output $\boldsymbol{\mu}_{ij}$ as a **generalized linear function** of the input:

$$
\boldsymbol{\mu}_{ij} = f(U_{ij} \mathbf{x})
$$

where $U_{ij}$ is a weight matrix and $f$ is a link function. For regression, $f$ is the identity (linear experts). For classification, $f$ could be the logistic function. This is the framework of **generalized linear models** (GLIMs) from statistics (McCullagh & Nelder, 1983).

The gating networks also use the generalized linear framework. At the top level:

$$
\xi_i = \mathbf{v}_i^T \mathbf{x}
$$

$$
g_i = \frac{e^{\xi_i}}{\sum_k e^{\xi_k}}
$$

This is the **softmax function** again — but now in the context of a **log-linear probability model**, a special case of GLIM commonly used for multiway classification.

The total output is:

$$
\boldsymbol{\mu} = \sum_i g_i \boldsymbol{\mu}_i = \sum_i g_i \sum_j g_{j|i} \boldsymbol{\mu}_{ij}
$$

### Numerical check with our running example

Let us extend our 2-expert flat example to a 2-level hierarchy with 4 experts. Suppose for input $x = -1$ the gating outputs are:

- Top level: $g_1 = 0.8$, $g_2 = 0.2$
- Branch 1: $g_{1|1} = 0.9$, $g_{2|1} = 0.1$
- Branch 2: $g_{1|2} = 0.5$, $g_{2|2} = 0.5$

Expert outputs: $\mu_{11} = 1.1$, $\mu_{12} = 0.3$, $\mu_{21} = -0.5$, $\mu_{22} = 0.7$.

Branch outputs:

$$
\mu_1 = 0.9 \times 1.1 + 0.1 \times 0.3 = 0.99 + 0.03 = 1.02
$$

$$
\mu_2 = 0.5 \times (-0.5) + 0.5 \times 0.7 = -0.25 + 0.35 = 0.10
$$

Total output:

$$
\mu = 0.8 \times 1.02 + 0.2 \times 0.10 = 0.816 + 0.020 = 0.836
$$

For target $d = 1$, this gives error $|1 - 0.836|^2 = 0.027$. Expert $(1,1)$ is doing the heavy lifting with output $1.1$, and the hierarchy is correctly routing most responsibility through branch 1 ($g_1 = 0.8$) and then to expert $(1,1)$ ($g_{1|1} = 0.9$).

---

## 7. The Probability Model

Jordan & Jacobs (1993) gave the hierarchy a precise probabilistic interpretation. The mechanism for generating data involves a **nested sequence of decisions**:

1. First, pick a top-level branch $i$ with probability $g_i(\mathbf{x}, \mathbf{v}_i)$ — this is a multinomial decision.
2. Then, within branch $i$, pick an expert $j$ with conditional probability $g_{j|i}(\mathbf{x}, \mathbf{v}_{ij})$ — another multinomial decision.
3. Finally, generate output $\mathbf{y}$ from the probability density $P(\mathbf{y} | \mathbf{x}, \theta_{ij})$ centered at expert $(i,j)$'s prediction.

The total probability of generating $\mathbf{y}$ from $\mathbf{x}$ is a **mixture of the component densities**, weighted by the multinomial probabilities:

$$
\boxed{P(\mathbf{y} | \mathbf{x}, \theta^0) = \sum_i g_i(\mathbf{x}, \mathbf{v}_i^0) \sum_j g_{j|i}(\mathbf{x}, \mathbf{v}_{ij}^0) P(\mathbf{y} | \mathbf{x}, \theta_{ij}^0)}
$$

This is equation (4) from Jordan & Jacobs (1993). For regression with Gaussian noise, the expert density is:

$$
P(\mathbf{y} | \mathbf{x}, \theta_{ij}) = \frac{1}{(2\pi)^{n/2} |\Sigma_{ij}|^{1/2}} \exp\left( -\frac{1}{2}(\mathbf{y} - \boldsymbol{\mu}_{ij})^T \Sigma_{ij}^{-1} (\mathbf{y} - \boldsymbol{\mu}_{ij}) \right)
$$

where $\boldsymbol{\mu}_{ij} = U_{ij}\mathbf{x}$ is the expert's predicted mean and $\Sigma_{ij}$ is the covariance (the **dispersion parameter** in GLIM terminology).

This model belongs to the **exponential family of densities**, which includes Gaussians, Bernoulli, Poisson, and many others. This is not a coincidence — Jordan & Jacobs deliberately designed the architecture to fit within the GLIM framework, which provides a unified treatment of regression, classification, and counting problems.

---

## 8. Posterior Probabilities via Bayes' Theorem

Given the probability model, we can compute how much each node should be "blamed" (or credited) for generating a particular data point. These are the **posterior probabilities**.

The gating outputs $g_i$ and $g_{j|i}$ are **prior probabilities** — they are computed from the input $\mathbf{x}$ alone, before seeing the target $\mathbf{y}$.

After observing both $\mathbf{x}$ and $\mathbf{y}$, we update our beliefs using **Bayes' theorem**:

**Top-level posterior:**

$$
h_i = \frac{g_i \sum_j g_{j|i} P_{ij}(\mathbf{y})}{\sum_k g_k \sum_j g_{j|k} P_{kj}(\mathbf{y})}
$$

This is equation (5) from Jordan & Jacobs (1993). The numerator is the joint probability that we chose branch $i$ AND generated $\mathbf{y}$. The denominator is the total probability of $\mathbf{y}$ (summing over all paths). The ratio gives the probability that branch $i$ was responsible, given the observed output.

**Lower-level conditional posterior:**

$$
h_{j|i} = \frac{g_{j|i} P_{ij}(\mathbf{y})}{\sum_l g_{l|i} P_{il}(\mathbf{y})}
$$

This is equation (6). It gives the probability that expert $j$ within branch $i$ generated the data.

**Joint posterior:**

$$
h_{ij} = h_i \cdot h_{j|i}
$$

This is the probability that expert $(i,j)$ specifically generated the data point, accounting for both levels of the hierarchy.

### Numerical check

Continuing our example with $x = -1$, $d = 1$, and assuming Gaussian densities with unit variance:

Expert likelihoods: $P_{11} \propto e^{-\frac{1}{2}(1-1.1)^2} = e^{-0.005} = 0.995$, $P_{12} \propto e^{-\frac{1}{2}(1-0.3)^2} = e^{-0.245} = 0.783$, $P_{21} \propto e^{-\frac{1}{2}(1+0.5)^2} = e^{-1.125} = 0.325$, $P_{22} \propto e^{-\frac{1}{2}(1-0.7)^2} = e^{-0.045} = 0.956$.

Branch likelihoods (weighted by lower gating):

$$
L_1 = g_{1|1} P_{11} + g_{2|1} P_{12} = 0.9 \times 0.995 + 0.1 \times 0.783 = 0.896 + 0.078 = 0.974
$$

$$
L_2 = g_{1|2} P_{21} + g_{2|2} P_{22} = 0.5 \times 0.325 + 0.5 \times 0.956 = 0.163 + 0.478 = 0.641
$$

Total: $L = g_1 L_1 + g_2 L_2 = 0.8 \times 0.974 + 0.2 \times 0.641 = 0.779 + 0.128 = 0.907$.

Top-level posteriors:

$$
h_1 = \frac{0.779}{0.907} = 0.859, \quad h_2 = \frac{0.128}{0.907} = 0.141
$$

Branch 1's responsibility *increased* from prior $g_1 = 0.8$ to posterior $h_1 = 0.859$ because expert $(1,1)$ fits the data well. Lower-level posteriors within branch 1:

$$
h_{1|1} = \frac{0.9 \times 0.995}{0.974} = \frac{0.896}{0.974} = 0.920, \quad h_{2|1} = \frac{0.1 \times 0.783}{0.974} = \frac{0.078}{0.974} = 0.080
$$

Joint posteriors: $h_{11} = 0.859 \times 0.920 = 0.790$, $h_{12} = 0.859 \times 0.080 = 0.069$.

Expert $(1,1)$ has a joint posterior of $0.790$ — it bears 79% of the responsibility for this data point. This makes sense: it predicted $1.1$ for a target of $1$, the closest of all experts.

---

## 9. The Log-Likelihood and Maximum Likelihood Estimation

The log-likelihood of the entire dataset $\mathcal{X} = \{(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})\}_{t=1}^N$ is obtained by taking the log of the product of $N$ densities of the form of equation (4):

$$
l(\theta; \mathcal{X}) = \sum_t \ln \sum_i g_i^{(t)} \sum_j g_{j|i}^{(t)} P_{ij}(\mathbf{y}^{(t)})
$$

This is equation (7) from Jordan & Jacobs (1993). We want to maximize this function with respect to all parameters $\theta$ (expert weights and gating weights).

The problem: the logarithm sits *outside* the summation over experts. This **log-of-a-sum** structure makes direct gradient computation messy — each parameter affects the log-likelihood through a complex ratio. This is where the EM algorithm comes in.

---

## 10. The EM Algorithm

The **Expectation-Maximization** (EM) algorithm (Dempster, Laird & Rubin, 1977) is an iterative approach to maximum likelihood estimation. It is designed for exactly the situation we face: the likelihood would be easy to maximize if we knew some hidden variables, but those variables are unknown.

### The key idea: missing data

Imagine we had **indicator variables** $z_{ij}$ that tell us which expert generated each data point: $z_{ij}^{(t)} = 1$ if expert $(i,j)$ generated data point $t$, and $z_{ij}^{(t)} = 0$ otherwise. Exactly one $z_{ij}$ is 1 for each data point.

If we knew the $z_{ij}$'s, the **complete-data log-likelihood** would be:

$$
l_c(\theta; \mathcal{Y}) = \sum_t \sum_i \sum_j z_{ij}^{(t)} \ln \{ g_i^{(t)} \, g_{j|i}^{(t)} \, P_{ij}(\mathbf{y}^{(t)}) \}
$$

This is equation (8) from Jordan & Jacobs (1993). Compare this to the incomplete-data log-likelihood in equation (7). The indicator variables $z_{ij}$ have allowed the logarithm to be **brought inside the summation signs**, by the **logarithm product rule** $\ln(ab) = \ln a + \ln b$. This substantially simplifies the maximization problem because the parameters of different experts and gating networks now appear in separate terms.

### The E step

We do not know the $z_{ij}$'s, so we take their expected value given the data and current parameters. The expected value of $z_{ij}^{(t)}$ is:

$$
E[z_{ij}^{(t)}] = P(z_{ij}^{(t)} = 1 | \mathbf{y}^{(t)}, \mathbf{x}^{(t)}, \theta^{(p)})
$$

This is just the posterior probability $h_{ij}^{(t)}$ that we computed in Section 8. So the E step simply computes the posterior probabilities at every node using the current parameter values.

The expected complete-data log-likelihood (the $Q$ function) is:

$$
Q(\theta, \theta^{(p)}) = \sum_t \sum_i \sum_j h_{ij}^{(t)} \ln \{ g_i^{(t)} \, g_{j|i}^{(t)} \, P_{ij}(\mathbf{y}^{(t)}) \}
$$

This is equation (9) from the paper.

### The M step

Now we maximize $Q$ with respect to $\theta$. The beauty of the complete-data formulation is that the parameters separate. Using $\ln(abc) = \ln a + \ln b + \ln c$ (the **logarithm product rule**):

$$
Q = \sum_t \sum_i \sum_j h_{ij}^{(t)} \ln g_i^{(t)} + \sum_t \sum_i \sum_j h_{ij}^{(t)} \ln g_{j|i}^{(t)} + \sum_t \sum_i \sum_j h_{ij}^{(t)} \ln P_{ij}(\mathbf{y}^{(t)})
$$

The first term depends only on the top-level gating parameters. The second depends only on the lower-level gating parameters. The third depends only on the expert parameters. We can maximize each separately.

**Expert parameters:** For each expert $(i,j)$:

$$
\theta_{ij}^{(p+1)} = \arg \max_{\theta_{ij}} \sum_t h_{ij}^{(t)} \ln P_{ij}(\mathbf{y}^{(t)})
$$

This is a **weighted maximum likelihood** problem for a generalized linear model, where the weights are the posterior probabilities $h_{ij}^{(t)}$. This can be solved by **iteratively reweighted least squares** (IRLS), a standard algorithm for GLIMs (McCullagh & Nelder, 1983).

**Top-level gating parameters:**

$$
\mathbf{v}_i^{(p+1)} = \arg \max_{\mathbf{v}_i} \sum_t \sum_k h_k^{(t)} \ln g_k^{(t)}
$$

This is a weighted maximum likelihood problem for a log-linear (softmax) model. The "observations" are the data points $\mathbf{x}^{(t)}$, the "targets" are the posterior probabilities $h_k^{(t)}$, and we are fitting a softmax model to predict them.

**Lower-level gating parameters** follow the same pattern.

### The complete HME algorithm

Putting it all together:

1. **E step:** For each data pair $(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})$, compute posteriors $h_i^{(t)}$ and $h_{j|i}^{(t)}$ using current parameters.
2. **M step (experts):** For each expert $(i,j)$, solve a weighted IRLS problem with observations $\{(\mathbf{x}^{(t)}, \mathbf{y}^{(t)})\}$ and weights $\{h_{ij}^{(t)}\}$.
3. **M step (top gating):** Solve a weighted IRLS problem with observations $\{(\mathbf{x}^{(t)}, h_k^{(t)})\}$ and weights $\{h_k^{(t)}\}$.
4. **M step (lower gating):** For each branch $i$, solve a weighted IRLS problem with observations $\{(\mathbf{x}^{(t)}, h_{l|k}^{(t)})\}$ and weights $\{h_k^{(t)}\}$.
5. **Iterate** with updated parameters.

### The convergence guarantee

Dempster, Laird & Rubin (1977) proved that every EM iteration increases the incomplete-data log-likelihood:

$$
l(\theta^{(p+1)}; \mathcal{X}) \geq l(\theta^{(p)}; \mathcal{X})
$$

with equality only at stationary points of $l$. This is a powerful result. It says the EM algorithm is guaranteed to climb the likelihood surface — it never goes downhill. In practice, this means convergence to a local maximum.

The proof relies on the relationship between the complete and incomplete likelihoods. An increase in $Q$ (the expected complete-data likelihood) implies an increase in $l$ (the incomplete-data likelihood). This is because $l = Q + H$ where $H$ is an entropy term that depends on the missing data distribution, and the EM update is designed to increase $Q$ while $H$ cannot decrease under the same update.

---

## 11. The On-line Algorithm

Jordan & Jacobs (1993) also developed an on-line version using **recursive estimation theory** (Ljung & Söderström, 1986). Instead of processing the entire dataset in each iteration, we update parameters after each individual data point.

The update rule for expert $(i,j)$'s weight matrix is:

$$
U_{ij}^{(t+1)} = U_{ij}^{(t)} + h_i^{(t)} h_{j|i}^{(t)} (\mathbf{y}^{(t)} - \boldsymbol{\mu}_{ij}^{(t)}) \mathbf{x}^{(t)T} R_{ij}^{(t)}
$$

This is equation (10) from the paper. Here $R_{ij}$ is the inverse covariance matrix for expert $(i,j)$, updated via:

$$
R_{ij}^{(t)} = \lambda^{-1} R_{ij}^{(t-1)} - \lambda^{-1} \frac{R_{ij}^{(t-1)} \mathbf{x}^{(t)} \mathbf{x}^{(t)T} R_{ij}^{(t-1)}}{\lambda [h_{ij}^{(t)}]^{-1} + \mathbf{x}^{(t)T} R_{ij}^{(t-1)} \mathbf{x}^{(t)}}
$$

where $\lambda$ is a decay parameter. This is the **Sherman-Morrison-Woodbury formula** applied to recursive least squares — it maintains a running estimate of the weighted covariance without storing all past data.

The structure of the update is intuitive: the change to the expert weights is proportional to:
- $h_i^{(t)} h_{j|i}^{(t)}$: the posterior probability (how responsible this expert is)
- $(\mathbf{y}^{(t)} - \boldsymbol{\mu}_{ij}^{(t)})$: the prediction error (how wrong the expert was)
- $\mathbf{x}^{(t)T} R_{ij}^{(t)}$: a curvature-adjusted input (second-order information)

---

## 12. Experimental Results: Speed and Task Decomposition

Jacobs et al. (1991) tested their system on a 4-class vowel discrimination task using formant data from 75 speakers. With 4 or 8 very simple experts (each restricted to a linear decision surface), the mixture achieved 90% test accuracy — matching a backpropagation network with 6 or 12 hidden units — but converging in roughly half the number of epochs.

| System | Train % | Test % | Avg. Epochs | SD |
|--------|---------|--------|-------------|-----|
| 4 Experts | 88 | 90 | 1124 | 23 |
| 8 Experts | 88 | 90 | 1083 | 12 |
| BP 6 Hid | 88 | 90 | 2209 | 83 |
| BP 12 Hid | 88 | 90 | 2435 | 124 |

The mixture system converged roughly twice as fast as backpropagation with less variance. The system automatically discovered the task decomposition: different experts specialized in different vowel pairs.

Jordan & Jacobs (1993) tested the HME on a robot dynamics identification problem (4-joint arm, 12 inputs, 4 outputs). The results were dramatic:

| Architecture | Relative Error | Epochs |
|-------------|---------------|--------|
| Linear | .31 | 1 |
| Backprop | .09 | 5,500 |
| HME (batch) | .10 | 35 |
| Backprop (on-line) | .08 | 63 |
| HME (on-line) | .12 | 2 |

The HME batch algorithm converged in **35 epochs** compared to backpropagation's **5,500 epochs** — a factor of 157x speedup — with comparable accuracy. The on-line HME converged in just **2 passes** through the data.

---

## Summary

We have built the Mixture of Experts framework from first principles. The cooperative error function couples experts and prevents specialization. The competitive error function decouples them by asking each expert to produce the entire output. The log-likelihood error function replaces priors with Bayesian posteriors, giving the system a principled way to discover which expert should handle which data. The Hierarchical Mixture of Experts extends this to a tree structure, and the EM algorithm provides a convergence-guaranteed learning procedure that decomposes into a collection of weighted generalized linear model fits. The on-line version brings recursive estimation theory to bear, enabling learning in a single pass through the data. These two papers — Jacobs et al. (1991) and Jordan & Jacobs (1993) — laid the mathematical foundations that every subsequent MoE paper builds upon.

In Part 2, we will see how Shazeer et al. (2017) scaled this framework to thousands of experts and billions of parameters, and how Fedus et al. (2021) simplified it further with the Switch Transformer.
