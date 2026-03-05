---
title: "Mathematical Prerequisites for Mixture of Experts"
description: "Building the math foundations you need for Mixture of Experts — expected value, Gaussian densities, likelihood, Bayes' theorem, softmax, mixture models, conditional probability, multinomial distributions, and the Sherman-Morrison-Woodbury formula — all derived step by step with one consistent example."
date: 2026-03-04
tags: [machine-learning, mixture-of-experts, mathematics, probability]
order: 2
---

Before diving into Mixture of Experts, you need a handful of mathematical ideas. This post builds every one of them inside a single, consistent example so nothing feels abstract or disconnected. By the end, you will have all the tools required to derive the MoE framework — cooperative and competitive errors, posterior responsibilities, the mixture-of-Gaussians interpretation, hierarchical mixtures, and the EM algorithm — from scratch in Part 1.

---

## The Running Example

Imagine two weather forecasters predicting tomorrow's temperature. Forecaster A predicts 20°C. Forecaster B predicts 26°C. The actual temperature turns out to be 22°C.

We will use this setup — two forecasters, one observed outcome — for every concept in this post. The forecasters will become "experts" in the MoE framework, the actual temperature will become the "target," and the question of how to combine or select between forecasters will motivate every mathematical tool we build.

Throughout, we will use the following concrete numbers:

| Forecaster | Prediction |
|------------|------------|
| A          | $\mu_A = 20$ |
| B          | $\mu_B = 26$ |

Observed temperature: $y = 22$.

---

## 1. Squared Error

We need a way to measure how wrong a prediction is. The most natural choice is the **squared error**: take the difference between the prediction and the actual value, then square it.

$$
\text{SE} = (y - \mu)^2
$$

Why square? Two reasons. First, squaring makes all errors positive — a prediction that is 2 degrees too high is just as "wrong" as one that is 2 degrees too low. Second, squaring penalises large errors more heavily than small ones: an error of 4 is penalised 16 times, not 4 times.

When dealing with vector-valued outputs (predictions with multiple components), we use the **squared norm**:

$$
\|\mathbf{y} - \boldsymbol{\mu}\|^2 = \sum_k (y_k - \mu_k)^2
$$

This sums the squared errors across all components. For our scalar example, $\|\mathbf{y} - \boldsymbol{\mu}\|^2$ reduces to $(y - \mu)^2$.

### Numerical check

For forecaster A: $(22 - 20)^2 = 4$.

For forecaster B: $(22 - 26)^2 = 16$.

Forecaster A has a squared error of 4, forecaster B has a squared error of 16. By this measure, A is four times better than B on this particular observation. We will use these two numbers — 4 and 16 — throughout the rest of the post.

---

## 2. Expected Value

Suppose we do not know which forecaster will be selected — a gating mechanism picks forecaster A with probability $p$ and forecaster B with probability $1 - p$. What is the average squared error we would see across many such selections?

The **expected value** answers this question. It multiplies each possible outcome by its probability, then sums:

$$
\boxed{\mathbb{E}[X] = \sum_i x_i \cdot P(X = x_i)}
$$

For the expected squared error when the gating mechanism assigns $p = 0.6$ to A and $1 - p = 0.4$ to B:

$$
\mathbb{E}[\text{SE}] = p \cdot (y - \mu_A)^2 + (1-p) \cdot (y - \mu_B)^2
$$

### Numerical check

$$
\mathbb{E}[\text{SE}] = 0.6 \times 4 + 0.4 \times 16 = 2.4 + 6.4 = 8.8
$$

On average, if we randomly pick a forecaster according to these probabilities and measure the squared error, we get 8.8. This formula is exactly the competitive error function in MoE — each expert is measured against the full target independently, and the errors are averaged using the gating probabilities as weights. We will see this in Part 1 as equation (1.2) from Jacobs et al. (1991).

---

## 3. The Softmax Function

In MoE, a gating network must produce probabilities $p_1, p_2, \ldots, p_n$ that are all positive and sum to 1. Given arbitrary real-valued scores $s_1, s_2, \ldots, s_n$ (which could be any real numbers — positive, negative, or zero), how do we convert them into valid probabilities?

The **softmax function** does exactly this:

$$
\boxed{p_j = \frac{e^{s_j}}{\sum_k e^{s_k}}}
$$

Each score is exponentiated (making it positive, since $e^x > 0$ for all $x$), then divided by the sum of all exponentiated scores (ensuring the result sums to 1).

Let us verify both requirements. Positivity: since $e^{s_j} > 0$ and $\sum_k e^{s_k} > 0$, we have $p_j > 0$ for all $j$. Summation: $\sum_j p_j = \sum_j \frac{e^{s_j}}{\sum_k e^{s_k}} = \frac{\sum_j e^{s_j}}{\sum_k e^{s_k}} = 1$.

**An important property.** Softmax preserves the ranking: if $s_1 > s_2$, then $p_1 > p_2$ (because the exponential is monotonically increasing: $e^{s_1} > e^{s_2}$, so $\frac{e^{s_1}}{\text{sum}} > \frac{e^{s_2}}{\text{sum}}$). The forecaster with the higher score gets the higher probability.

### Numerical check

Suppose the gating network produces scores $s_A = 1.5$ and $s_B = 0.5$ for our two forecasters. Then:

$$
e^{1.5} = 4.482, \quad e^{0.5} = 1.649
$$

$$
p_A = \frac{4.482}{4.482 + 1.649} = \frac{4.482}{6.131} = 0.731
$$

$$
p_B = \frac{1.649}{6.131} = 0.269
$$

Check: $0.731 + 0.269 = 1.000$. Both positive, sum to 1. Forecaster A, which had the higher score ($s_A = 1.5 > s_B = 0.5$), gets the higher probability ($p_A = 0.731 > p_B = 0.269$). In Part 1, the gating network computes these scores as linear functions of the input: $s_j = \mathbf{v}_j^T \mathbf{x}$.

---

## 4. The Gaussian Density

We now arrive at the concept that connects squared error to probability. The **Gaussian** (or **normal**) **probability density function** describes how likely a value $y$ is, given that it was drawn from a bell-shaped distribution centred at $\mu$ with spread controlled by the **variance** $\sigma^2$:

$$
\boxed{f(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(y - \mu)^2}{2\sigma^2} \right)}
$$

Let us unpack each piece:

- $(y - \mu)^2$ is the squared error between the observation and the centre — the same squared error from Section 1.
- Dividing by $2\sigma^2$ scales the error by the variance. Larger variance means the distribution is wider and more tolerant of errors.
- The exponential $\exp(-\cdot)$ converts the scaled squared error into a positive number (since $e^x > 0$ for all $x$). Smaller errors give larger density values.
- The factor $\frac{1}{\sqrt{2\pi\sigma^2}}$ is the **normalising constant** — it ensures the density integrates to 1 over all possible $y$ values.

**A common simplification.** When comparing two forecasters on the same observation, the normalising constant is the same for both (since $\sigma^2$ is the same). In MoE, Jacobs et al. (1991) use unit variance ($\sigma^2 = 1$) for simplicity. With $\sigma^2 = 1$, the Gaussian density simplifies to:

$$
f(y \mid \mu, 1) = \frac{1}{\sqrt{2\pi}} \exp\!\left( -\frac{(y - \mu)^2}{2} \right)
$$

The normalising constant $\frac{1}{\sqrt{2\pi}} \approx 0.399$ is now the same for all experts and does not affect which expert is "better." This is why the MoE papers often write $e^{-\frac{1}{2}\|d - o\|^2}$ and drop the constant.

### Numerical check

Using unit variance ($\sigma^2 = 1$):

Forecaster A ($\mu_A = 20$, $y = 22$):

$$
f(22 \mid 20, 1) = \frac{1}{\sqrt{2\pi}} \exp\!\left( -\frac{(22-20)^2}{2} \right) = 0.399 \times e^{-2} = 0.399 \times 0.135 = 0.054
$$

Forecaster B ($\mu_B = 26$, $y = 22$):

$$
f(22 \mid 26, 1) = \frac{1}{\sqrt{2\pi}} \exp\!\left( -\frac{(22-26)^2}{2} \right) = 0.399 \times e^{-8} = 0.399 \times 0.000335 = 0.000134
$$

Forecaster A's density is $0.054$, roughly 400 times larger than forecaster B's density of $0.000134$. The Gaussian density is telling us: if the true temperature-generating process is centred at 20 (forecaster A's prediction) with unit variance, the observed value of 22 is plausible. If the process is centred at 26, the value 22 is extremely unlikely. This ratio — 400 to 1 — is exactly the kind of comparison that will drive the posterior responsibilities in MoE.

### Why the densities are small

Both values — $0.054$ and $0.000134$ — are small numbers. That is normal. A continuous density is not a probability. It measures how densely the probability is concentrated around $y$. The actual probability of observing *exactly* 22.000... is zero for any continuous distribution. What matters is the *relative comparison* between densities, not their absolute size. Forecaster A's density is 400 times larger than forecaster B's — that is the relevant information.

---

## 5. Likelihood

We now flip the perspective. In Section 4, we fixed a forecaster ($\mu$) and asked how probable the observation $y$ is. Now we fix the observation ($y = 22$) and ask: which forecaster makes the observation more plausible?

The **likelihood function** uses the same formula as the probability density, but views it as a function of $\mu$ (the forecaster's prediction) rather than $y$ (the observation):

$$
\boxed{\mathcal{L}(\mu) = f(y \mid \mu) = \frac{1}{\sqrt{2\pi}} \exp\!\left( -\frac{(y - \mu)^2}{2} \right)}
$$

Same formula. Different question.

**Probability** asks: given forecaster A's prediction $\mu_A = 20$, how likely is the observation $y = 22$?

**Likelihood** asks: given the observation $y = 22$, how plausible is it that the generating process was centred at $\mu_A = 20$?

The answer is the same number — $0.054$ — but the interpretation is different. Probability varies $y$ while holding $\mu$ fixed. Likelihood varies $\mu$ while holding $y$ fixed.

### Numerical check

$$
\mathcal{L}(20) = 0.054, \quad \mathcal{L}(26) = 0.000134
$$

The likelihood of forecaster A is 400 times larger than the likelihood of forecaster B. If we had to pick one forecaster based solely on this observation, we would pick A — it assigns far more plausibility to the data we actually saw.

This is exactly what happens inside MoE. Each expert's output $o_i$ defines a Gaussian centre, and the likelihood $e^{-\frac{1}{2}\|d - o_i\|^2}$ measures how well that expert explains the observed target. Experts with higher likelihood get more responsibility. We will formalise this in the Bayes' theorem section.

---

## 6. Log-Likelihood and the Logarithm

Working directly with likelihoods leads to numerical problems — when you multiply many small numbers together (one per data point), the product quickly underflows to zero. The **logarithm** fixes this by converting products into sums.

The **natural logarithm** $\ln(x)$ is the inverse of the exponential: if $e^a = b$, then $\ln(b) = a$. We need three properties.

**Property 1: Monotonically increasing.** If $a > b > 0$, then $\ln a > \ln b$. Maximising $\ln f$ is equivalent to maximising $f$ — the logarithm preserves the location of maxima and minima.

**Property 2: Log of a product is a sum of logs.** This is the **logarithm product rule**:

$$
\boxed{\ln(a \cdot b) = \ln a + \ln b}
$$

More generally, $\ln(a \cdot b \cdot c) = \ln a + \ln b + \ln c$. This property is what makes the EM algorithm tractable — it will allow us to bring the logarithm inside the summation signs and separate the parameters of different experts.

**Property 3: Log of an exponential cancels.** $\ln(e^x) = x$. This converts the Gaussian density into a simple quadratic.

### The log-likelihood of a Gaussian

Applying property 3 to our Gaussian likelihood:

$$
\ln \mathcal{L}(\mu) = \ln \left[ \frac{1}{\sqrt{2\pi}} \exp\!\left( -\frac{(y - \mu)^2}{2} \right) \right]
$$

By property 2 (log of a product):

$$
= \ln \frac{1}{\sqrt{2\pi}} + \ln \exp\!\left( -\frac{(y - \mu)^2}{2} \right)
$$

By property 3 (log of exponential cancels):

$$
= -\frac{1}{2}\ln(2\pi) - \frac{(y - \mu)^2}{2}
$$

The first term is a constant (it does not depend on $\mu$). So maximising the log-likelihood is equivalent to minimising $(y - \mu)^2$ — the squared error from Section 1.

$$
\boxed{\ln \mathcal{L}(\mu) = \text{const} - \frac{(y - \mu)^2}{2}}
$$

This is a deep result: **maximum likelihood estimation with a Gaussian model is equivalent to minimising squared error.** When you see squared error in MoE, you are implicitly doing maximum likelihood under a Gaussian assumption.

### Numerical check

Forecaster A:

$$
\ln \mathcal{L}(20) = -\frac{1}{2}\ln(2\pi) - \frac{4}{2} = -0.919 - 2.0 = -2.919
$$

Verify: $\ln(0.054) = -2.919$. Correct.

Forecaster B:

$$
\ln \mathcal{L}(26) = -0.919 - \frac{16}{2} = -0.919 - 8.0 = -8.919
$$

Verify: $\ln(0.000134) = -8.919$. Correct.

The log-likelihood difference is $-2.919 - (-8.919) = 6.0$. We can verify: $\ln(0.054 / 0.000134) = \ln(403) = 5.999 \approx 6.0$. The logarithm has turned the ratio of 400:1 in likelihoods into an additive difference of 6.0 in log-likelihoods — much easier to work with numerically.

---

## 7. Negative Log-Likelihood as a Loss Function

In optimisation, we typically *minimise* a loss function. But likelihood and log-likelihood are things we want to *maximise* — higher likelihood means the model fits the data better. The standard trick is to negate the log-likelihood, turning maximisation into minimisation:

$$
\text{Loss} = -\ln \mathcal{L}(\mu)
$$

This is the **negative log-likelihood** (NLL). Minimising the NLL is mathematically identical to maximising the likelihood.

For a single Gaussian:

$$
-\ln \mathcal{L}(\mu) = \frac{1}{2}\ln(2\pi) + \frac{(y - \mu)^2}{2}
$$

The constant term does not affect the optimisation, so the NLL is proportional to the squared error.

In MoE, the error function from Jacobs et al. (1991) is:

$$
E = -\log \sum_i p_i \, e^{-\frac{1}{2}\|d - o_i\|^2}
$$

This is the negative log-likelihood of a *mixture* model — we will build up to it in Section 10. The negative sign converts "maximise likelihood" into "minimise error."

### Numerical check

Forecaster A: $-\ln \mathcal{L}(20) = 2.919$.

Forecaster B: $-\ln \mathcal{L}(26) = 8.919$.

Minimising the NLL means preferring forecaster A (loss 2.919) over forecaster B (loss 8.919). This is the same conclusion as before — just expressed as a minimisation problem.

---

## 8. Bayes' Theorem

We now have all the ingredients for the most important result in this post. Suppose we have two forecasters, and before seeing the actual temperature, a gating network assigns prior probabilities $p_A$ and $p_B = 1 - p_A$ to each. After seeing the actual temperature $y = 22$, we want to update these probabilities to reflect how well each forecaster did. The updated probabilities are called **posterior probabilities**.

**Bayes' theorem** tells us exactly how to update:

$$
\boxed{P(\text{forecaster } i \mid y) = \frac{p_i \cdot f(y \mid \mu_i)}{\sum_j p_j \cdot f(y \mid \mu_j)}}
$$

Let us read this formula piece by piece:

- **$p_i$** is the **prior probability** — the gating network's belief about forecaster $i$ *before* seeing the data. In MoE, these are the softmax outputs from Section 3.
- **$f(y \mid \mu_i)$** is the **likelihood** — how well forecaster $i$'s prediction explains the observed data. This is the Gaussian density from Section 4.
- **$p_i \cdot f(y \mid \mu_i)$** is the **joint probability** — the probability of choosing forecaster $i$ AND observing $y$ from it.
- **$\sum_j p_j \cdot f(y \mid \mu_j)$** is the **total probability** of observing $y$ under the full mixture. We will call this the **marginal likelihood**. It sums over all possible forecasters.
- The ratio gives the **posterior probability** $P(\text{forecaster } i \mid y)$ — the updated belief about forecaster $i$ *after* seeing the data.

### Derivation

Bayes' theorem follows directly from the definition of conditional probability. The **conditional probability** of event $A$ given event $B$ is:

$$
P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}
$$

Let $A$ = "forecaster $i$ was selected" and $B$ = "we observed $y$." Then:

- $P(A \text{ and } B) = p_i \cdot f(y \mid \mu_i)$. This is the probability of picking forecaster $i$ (probability $p_i$) and then observing $y$ from it (density $f(y \mid \mu_i)$).
- $P(B) = \sum_j p_j \cdot f(y \mid \mu_j)$. This is the total probability of observing $y$, obtained by summing over all possible forecasters. This step uses the **law of total probability**: the probability of $y$ is the sum of the probabilities of $y$ through each possible path.

Substituting:

$$
P(\text{forecaster } i \mid y) = \frac{p_i \cdot f(y \mid \mu_i)}{\sum_j p_j \cdot f(y \mid \mu_j)}
$$

That is Bayes' theorem. Nothing more.

### Numerical check

Let the gating network assign prior probabilities $p_A = 0.6$, $p_B = 0.4$ (same as Section 2). Using unit-variance Gaussian likelihoods (dropping the normalising constant, since it cancels in the ratio):

$$
f_A = e^{-\frac{1}{2}(22-20)^2} = e^{-2} = 0.135
$$

$$
f_B = e^{-\frac{1}{2}(22-26)^2} = e^{-8} = 0.000335
$$

Joint probabilities:

$$
p_A \cdot f_A = 0.6 \times 0.135 = 0.0812
$$

$$
p_B \cdot f_B = 0.4 \times 0.000335 = 0.000134
$$

Marginal likelihood (total probability):

$$
L = 0.0812 + 0.000134 = 0.08133
$$

Posterior probabilities:

$$
h_A = \frac{0.0812}{0.08133} = 0.9984
$$

$$
h_B = \frac{0.000134}{0.08133} = 0.0016
$$

Check: $0.9984 + 0.0016 = 1.0000$. The posteriors sum to 1, as they must.

**What happened.** Forecaster A started with a prior of $p_A = 0.6$ and ended with a posterior of $h_A = 0.9984$. Its responsibility *increased* dramatically because it fits the data much better (squared error 4 vs. 16). Forecaster B started with a prior of $p_B = 0.4$ and ended with a posterior of $h_B = 0.0016$ — nearly zero.

This is exactly the mechanism that drives specialisation in MoE. The posteriors $h_i$ replace the priors $p_i$ in the gradient computation, ensuring that experts who fit the data well get larger gradients and learn faster on the cases they handle best. In Part 1, these posteriors are called $h_i^c$ in equation (1.5).

### Prior vs. posterior: the key distinction

This distinction is easy to gloss over, but the entire MoE framework depends on it. Let us pin it down precisely.

**Prior $p_i$** depends on the input $\mathbf{x}$ only. It is computed *before* seeing the target $y$. It answers: "Based on the input alone, which expert should handle this case?"

**Posterior $h_i$** depends on both the input $\mathbf{x}$ and the target $y$. It is computed *after* seeing the target. It answers: "Given that we now see the target, which expert actually explains it best?"

The prior is the gating network's best guess. The posterior is reality's correction. In MoE, using posteriors instead of priors in the gradient is what makes the system discover the right task decomposition — it is not just a mathematical nicety, it is the mechanism that makes experts specialise.

---

## 9. Mixture Models

We now combine all the tools we have built. A **mixture model** says that the data was generated by one of $n$ component distributions, but we do not know which one. The overall probability of observing $y$ is a weighted sum of the component densities:

$$
\boxed{P(y) = \sum_{i=1}^{n} p_i \cdot f(y \mid \mu_i, \sigma_i^2)}
$$

Each term has two parts:

- $p_i$ is the **mixing weight** (or prior probability) for component $i$. These are non-negative and sum to 1 — exactly the output of a softmax (Section 3).
- $f(y \mid \mu_i, \sigma_i^2)$ is the **component density** — for Gaussians, this is the formula from Section 4.

This is the **law of total probability** applied to continuous densities. The observation $y$ could have come from any component, so we sum over all possible sources, weighting each by its probability.

### The mixture model as a generative story

A mixture model describes a two-step process for generating data:

1. **Select** a component $i$ with probability $p_i$. (Roll a weighted die.)
2. **Generate** the observation $y$ from component $i$'s density $f(y \mid \mu_i, \sigma_i^2)$. (Draw from the selected distribution.)

In MoE, step 1 is the gating network selecting an expert, and step 2 is the selected expert generating a prediction (with Gaussian noise). The entire MoE framework *is* a mixture model where the mixing weights and component means are input-dependent.

### Mixture of Gaussians

When every component density $f(y \mid \mu_i, \sigma_i^2)$ is a Gaussian (Section 4), the mixture model is called a **mixture of Gaussians** (also known as a **Gaussian mixture model** or GMM). The total density becomes:

$$
P(y) = \sum_{i=1}^{n} p_i \cdot \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\!\left( -\frac{(y - \mu_i)^2}{2\sigma_i^2} \right)
$$

Each component is a bell curve centred at $\mu_i$ with width $\sigma_i$, and the mixture is a weighted sum of these bell curves. The result is a density that can have multiple peaks — one per component — allowing the model to capture data that clusters in several regions.

This is exactly the interpretation Jacobs et al. (1991) give to MoE in Part 1: each expert defines a Gaussian centred at its output, the gating network provides the mixing weights, and the combined model is a mixture of Gaussians. The term $e^{-\frac{1}{2}\|d - o_i\|^2}$ in the MoE error function is the Gaussian component (with unit variance, normalising constant dropped), and $\sum_i p_i e^{-\frac{1}{2}\|d - o_i\|^2}$ is the mixture.

### Numerical check

Using our two forecasters with unit variance and mixing weights $p_A = 0.6$, $p_B = 0.4$:

$$
P(22) = p_A \cdot f(22 \mid 20, 1) + p_B \cdot f(22 \mid 26, 1)
$$

$$
= 0.6 \times 0.054 + 0.4 \times 0.000134
$$

$$
= 0.0324 + 0.0000536
$$

$$
= 0.0325
$$

This is the marginal likelihood — the total probability of observing $y = 22$ under the mixture. It is dominated by forecaster A's contribution ($0.0324$ out of $0.0325$), reflecting A's much better fit to the data.

Notice that we computed this same quantity (up to the normalising constant) in the Bayes' theorem section as the denominator $L = 0.08133$. The difference is the normalising constant $\frac{1}{\sqrt{2\pi}} = 0.399$: $0.08133 \times 0.399 = 0.0325$. Both are computing the total probability of the data under the mixture — one with the constant, one without.

---

## 10. The Log-Likelihood of a Mixture

The loss function in MoE is the negative log of the mixture probability:

$$
\boxed{E = -\ln P(y) = -\ln \sum_{i=1}^{n} p_i \cdot f(y \mid \mu_i)}
$$

This is the **negative log-likelihood of a mixture model**. It is equation (1.3) in Jacobs et al. (1991), and equation (7) in Jordan & Jacobs (1993).

For a dataset of $N$ observations, the total log-likelihood is the sum over all data points:

$$
l(\theta) = \sum_{t=1}^{N} \ln \sum_i p_i^{(t)} \cdot f(y^{(t)} \mid \mu_i^{(t)})
$$

Here the **logarithm product rule** (Section 6, property 2) has already done its work: the log of the product of $N$ independent likelihoods becomes a sum of $N$ individual log-likelihoods.

### The log-sum problem

Notice that the logarithm sits *outside* the sum over experts: $\ln \sum_i (\cdots)$. This is fundamentally different from $\sum_i \ln(\cdots)$, which would be much easier to work with. The **log of a sum** does not simplify nicely — we cannot separate the parameters of different experts.

Compare:

- **Log of a product** (easy): $\ln(a \cdot b) = \ln a + \ln b$ — the terms separate.
- **Log of a sum** (hard): $\ln(a + b) \neq \ln a + \ln b$ — the terms stay coupled.

This is the central computational difficulty of mixture models. The EM algorithm, which we will derive in Part 1, exists specifically to solve this problem. It introduces hidden indicator variables that tell us which expert generated each data point, converting the log-of-a-sum into a sum-of-logs.

### Numerical check

$$
E = -\ln(0.0325) = 3.426
$$

If we had used forecaster A alone: $E_A = -\ln(0.054) = 2.919$. If forecaster B alone: $E_B = -\ln(0.000134) = 8.919$. The mixture loss (3.426) is slightly worse than using A alone (2.919) because the mixture includes B, which hurts the likelihood. But the mixture model has a crucial advantage: it can learn which forecaster to trust for each input, achieving better overall performance across many data points from different regions.

---

## 11. Conditional Probability and Nested Decisions

In the flat MoE, the gating network makes a single decision: which expert to use. In the **Hierarchical Mixture of Experts** (HME), decisions are *nested*: first pick a branch, then pick an expert within that branch. To handle this, we need **conditional probability**.

The **conditional probability** of event $B$ given that event $A$ has occurred is:

$$
P(B \mid A) = \frac{P(A \text{ and } B)}{P(A)}
$$

Rearranging gives the **multiplication rule**:

$$
\boxed{P(A \text{ and } B) = P(A) \cdot P(B \mid A)}
$$

This says: the probability of both $A$ and $B$ happening is the probability that $A$ happens, times the probability that $B$ happens *given that $A$ has already happened*.

### Nested decisions in our example

Suppose we organise our two forecasters into a hierarchy. We have a "weather service" that first picks a forecasting *agency* (branch), then picks a specific *forecaster* within that agency.

- **Branch decision:** Pick agency 1 with probability $g_1 = 0.7$ or agency 2 with probability $g_2 = 0.3$.
- **Within-branch decision:** If agency 1 is picked, choose forecaster A with probability $g_{A|1} = 0.8$ or forecaster B with probability $g_{B|1} = 0.2$.

The probability of reaching forecaster A through agency 1 is a nested decision — two choices in sequence:

$$
P(\text{agency 1 and forecaster A}) = P(\text{agency 1}) \cdot P(\text{forecaster A} \mid \text{agency 1})
$$

$$
= g_1 \cdot g_{A|1} = 0.7 \times 0.8 = 0.56
$$

This is the **multiplication rule** in action. The notation $g_{A|1}$ means "the probability of choosing forecaster A *given* that we are in agency 1." The vertical bar $|$ always means "given that" — it separates what we are asking about from what we are conditioning on.

### The law of total probability with nested decisions

If the observation $y$ can be generated through multiple paths (agency 1 → forecaster A, agency 1 → forecaster B, agency 2 → forecaster A, etc.), the total probability of $y$ sums over all paths:

$$
P(y) = \sum_i g_i \sum_j g_{j|i} \cdot f(y \mid \mu_{ij})
$$

This is the **law of total probability** applied to a two-level hierarchy. Each path has probability $g_i \cdot g_{j|i}$ (by the multiplication rule), and generates $y$ with density $f(y \mid \mu_{ij})$. We sum over all possible paths.

In Part 1, this is exactly equation (4) from Jordan & Jacobs (1993) — the probability model for the Hierarchical Mixture of Experts.

### Numerical check

Suppose agency 2 has two different forecasters, C and D, with $g_{C|2} = 0.5$, $g_{D|2} = 0.5$, predictions $\mu_C = 19$, $\mu_D = 25$. Using unit-variance Gaussians:

Path probabilities and likelihoods for $y = 22$:

- Agency 1, forecaster A: $0.7 \times 0.8 \times e^{-\frac{1}{2}(22-20)^2} = 0.56 \times 0.135 = 0.0758$
- Agency 1, forecaster B: $0.7 \times 0.2 \times e^{-\frac{1}{2}(22-26)^2} = 0.14 \times 0.000335 = 0.0000469$
- Agency 2, forecaster C: $0.3 \times 0.5 \times e^{-\frac{1}{2}(22-19)^2} = 0.15 \times e^{-4.5} = 0.15 \times 0.0111 = 0.00167$
- Agency 2, forecaster D: $0.3 \times 0.5 \times e^{-\frac{1}{2}(22-25)^2} = 0.15 \times e^{-4.5} = 0.15 \times 0.0111 = 0.00167$

Total: $P(22) = 0.0758 + 0.0000469 + 0.00167 + 0.00167 = 0.0792$.

Forecaster A through agency 1 dominates, contributing $0.0758$ out of $0.0792$ — about 96% of the total probability. The hierarchy has identified the right path.

---

## 12. The Multinomial Distribution

When the gating network picks one of $n$ experts, it is making a **multinomial** (or **categorical**) decision. A **multinomial distribution** generalises a coin flip to $n$ outcomes. Instead of just Heads or Tails, we have outcomes $1, 2, \ldots, n$ with probabilities $p_1, p_2, \ldots, p_n$ where $\sum_i p_i = 1$.

For our two-forecaster example ($n = 2$), the multinomial reduces to a single coin flip: choose forecaster A with probability $p_A$ or forecaster B with probability $p_B = 1 - p_A$. But in MoE, the gating network may choose among many experts — 4, 8, or even thousands. The multinomial distribution handles any number of outcomes.

The probability of outcome $i$ in a single draw is simply:

$$
P(\text{outcome } = i) = p_i
$$

That is literally it. The term "multinomial" sounds imposing, but for a single draw it is just "pick option $i$ with probability $p_i$." The softmax function (Section 3) is the standard way to parameterise a multinomial distribution — it takes arbitrary scores and produces valid multinomial probabilities.

### Where this appears in MoE

In the HME, there are multiple multinomial decisions:

- The **top-level gating network** makes a multinomial decision over branches: pick branch $i$ with probability $g_i$.
- Each **lower-level gating network** makes a multinomial decision over experts within a branch: pick expert $j$ with probability $g_{j|i}$.

Each of these is a separate multinomial distribution, parameterised by its own softmax. The nested structure — first a multinomial over branches, then a conditional multinomial over experts — is what creates the tree-structured probability model from Section 11.

### Numerical check

With 4 forecasters and scores $s_1 = 2.0$, $s_2 = 1.0$, $s_3 = 0.5$, $s_4 = -0.5$:

$$
e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.5} = 1.649, \quad e^{-0.5} = 0.607
$$

Sum: $7.389 + 2.718 + 1.649 + 0.607 = 12.363$.

$$
p_1 = \frac{7.389}{12.363} = 0.598, \quad p_2 = \frac{2.718}{12.363} = 0.220, \quad p_3 = \frac{1.649}{12.363} = 0.133, \quad p_4 = \frac{0.607}{12.363} = 0.049
$$

Check: $0.598 + 0.220 + 0.133 + 0.049 = 1.000$. This is a valid multinomial distribution over 4 outcomes. The gating network rolls this 4-sided weighted die to pick an expert.

---

## 13. The Gradient of the Mixture Loss

To train MoE, we need the derivative of the loss $E = -\ln L$ with respect to each expert's prediction $\mu_i$, where $L = \sum_j p_j f_j$ is the mixture likelihood. Let us derive this step by step.

**Step 1.** Since $E = -\ln L$, by the chain rule:

$$
\frac{\partial E}{\partial \mu_i} = -\frac{1}{L} \cdot \frac{\partial L}{\partial \mu_i}
$$

**Step 2.** Since $L = \sum_j p_j f_j$ and only the $i$-th term depends on $\mu_i$:

$$
\frac{\partial L}{\partial \mu_i} = p_i \cdot \frac{\partial f_i}{\partial \mu_i}
$$

**Step 3.** For the Gaussian $f_i = \exp\!\left(-\frac{(y - \mu_i)^2}{2}\right)$ (dropping the constant), we need the derivative. Let $g = -\frac{(y - \mu_i)^2}{2}$, so $f_i = e^g$. By the chain rule, $\frac{\partial f_i}{\partial \mu_i} = e^g \cdot \frac{\partial g}{\partial \mu_i}$. Now:

$$
\frac{\partial g}{\partial \mu_i} = -\frac{1}{2} \cdot 2(y - \mu_i) \cdot (-1) = (y - \mu_i)
$$

The $-\frac{1}{2}$ and the $2$ from the power rule cancel, and the $(-1)$ comes from differentiating $(y - \mu_i)$ with respect to $\mu_i$. So:

$$
\frac{\partial f_i}{\partial \mu_i} = f_i \cdot (y - \mu_i)
$$

**Step 4.** Combining steps 1–3:

$$
\frac{\partial E}{\partial \mu_i} = -\frac{p_i \cdot f_i}{L} \cdot (y - \mu_i)
$$

**Step 5.** Recognise $\frac{p_i \cdot f_i}{L}$ as the posterior probability $h_i$ from Bayes' theorem (Section 8):

$$
\boxed{\frac{\partial E}{\partial \mu_i} = -h_i (y - \mu_i)}
$$

This is equation (1.5) from Jacobs et al. (1991). The gradient for expert $i$ is proportional to two things: the posterior probability $h_i$ (how responsible this expert is for the data point) and the prediction error $(y - \mu_i)$ (how far off the expert's prediction is). If $h_i$ is small — meaning the expert is not responsible — the gradient is small and the expert barely updates. If $h_i$ is large, the expert gets a strong push toward the target. This is the mathematical mechanism behind expert specialisation.

### Numerical check

Using $p_A = 0.6$, $f_A = 0.135$, $L = 0.08133$, $y = 22$, $\mu_A = 20$:

$$
h_A = \frac{0.6 \times 0.135}{0.08133} = \frac{0.0812}{0.08133} = 0.9984
$$

$$
\frac{\partial E}{\partial \mu_A} = -0.9984 \times (22 - 20) = -0.9984 \times 2 = -1.997
$$

The negative gradient means the loss decreases when $\mu_A$ increases — forecaster A should increase its prediction from 20 toward the target 22. This is the correct direction.

For forecaster B:

$$
h_B = 0.0016
$$

$$
\frac{\partial E}{\partial \mu_B} = -0.0016 \times (22 - 26) = -0.0016 \times (-4) = +0.0064
$$

The gradient for B is $+0.0064$. The gradient for A is $-1.997$. Which gradient is "larger"? This brings us to a subtle but important distinction.

---

## 14. Gradient Magnitude vs. Signed Value

Gradients are signed quantities — they can be positive or negative. The sign tells you the *direction*: a negative gradient means "increase the parameter to decrease the loss," and a positive gradient means "decrease the parameter to decrease the loss." But when we ask which expert is learning *faster*, we care about the **magnitude** (absolute value) of the gradient, not its sign.

The **magnitude** of a number $x$ is $|x|$ — its distance from zero on the number line, ignoring the sign:

$$
|{-5}| = 5, \quad |{+3}| = 3, \quad |{-5}| > |{+3}|
$$

Note that $-5 < +3$ as signed numbers, but $|{-5}| > |{+3}|$ — negative five is *smaller* than positive three, but it is *farther from zero*.

### Why this matters in MoE

From our gradient computation:

$$
\frac{\partial E}{\partial \mu_A} = -1.997, \quad \frac{\partial E}{\partial \mu_B} = +0.0064
$$

As signed numbers, $+0.0064 > -1.997$. But the magnitudes tell a different story:

$$
|{-1.997}| = 1.997, \quad |{+0.0064}| = 0.0064
$$

The magnitude of A's gradient ($1.997$) is 312 times larger than B's ($0.0064$). Forecaster A is learning 312 times faster from this data point.

When you read in Part 1 that "expert 2 gets a *larger* gradient magnitude," this means $|\text{gradient}_2| > |\text{gradient}_1|$ — the absolute value is bigger, meaning a stronger push. The sign tells the direction of the push; the magnitude tells how hard the push is. In MoE, the posterior $h_i$ controls the magnitude: an expert with high posterior gets a large $|h_i \cdot (y - \mu_i)|$ and learns aggressively; an expert with near-zero posterior barely moves.

---

## 15. Indicator Variables and the Complete-Data Log-Likelihood

The EM algorithm, which we will derive in Part 1, relies on a clever trick: imagine that for each data point, we know which expert generated it. We represent this knowledge with **indicator variables**.

Define $z_i = 1$ if forecaster $i$ generated the observation, and $z_i = 0$ otherwise. Exactly one indicator is 1 for each data point — the data came from exactly one forecaster.

If we knew the $z_i$'s, the **complete-data log-likelihood** would be:

$$
\ln P(y, z) = \sum_i z_i \ln(p_i \cdot f_i) = \sum_i z_i [\ln p_i + \ln f_i]
$$

The second step uses the **logarithm product rule** from Section 6: $\ln(p_i \cdot f_i) = \ln p_i + \ln f_i$.

Compare this to the incomplete-data log-likelihood from Section 10:

$$
\ln P(y) = \ln \sum_i p_i \cdot f_i
$$

The indicator variables $z_i$ have performed a crucial transformation: the logarithm has moved *inside* the summation. Instead of $\ln \sum_i (\cdots)$ (log of a sum — hard), we have $\sum_i z_i \ln(\cdots)$ (sum of logs — easy). The parameters of different experts now appear in separate terms, making maximisation straightforward.

### Numerical check

Suppose we knew forecaster A generated the data ($z_A = 1$, $z_B = 0$):

$$
\ln P(y, z) = 1 \cdot [\ln(0.6) + \ln(0.135)] + 0 \cdot [\ln(0.4) + \ln(0.000335)]
$$

$$
= \ln(0.6) + \ln(0.135) = -0.511 + (-2.0) = -2.511
$$

If instead forecaster B generated it ($z_A = 0$, $z_B = 1$):

$$
\ln P(y, z) = 0 + 1 \cdot [\ln(0.4) + \ln(0.000335)] = -0.916 + (-8.0) = -8.916
$$

The complete-data likelihood sharply favours the hypothesis that A generated the data ($-2.511$ vs. $-8.916$).

Of course, we do not know the $z_i$'s. The EM algorithm handles this by replacing $z_i$ with its expected value — the posterior probability $h_i$ from Bayes' theorem (Section 8). For our numbers: $E[z_A] = h_A = 0.9984$ and $E[z_B] = h_B = 0.0016$. This is the E step. The M step then maximises the expected complete-data log-likelihood with respect to the model parameters. The full derivation appears in Part 1.

---

## 16. The Matrix Inverse Update: Sherman-Morrison-Woodbury

The on-line version of MoE (Section 11 of Part 1) updates expert parameters after each data point, without re-processing the entire dataset. This requires maintaining a running estimate of an inverse matrix. The **Sherman-Morrison-Woodbury formula** tells us how to update a matrix inverse when a small change is made to the original matrix.

### The problem

Suppose we have a matrix $A$ and we know its inverse $A^{-1}$. Now $A$ changes by a small amount — specifically, a **rank-1 update**: $A_{\text{new}} = A + \mathbf{u}\mathbf{v}^T$, where $\mathbf{u}$ and $\mathbf{v}$ are column vectors and $\mathbf{u}\mathbf{v}^T$ is their **outer product** (a matrix where entry $(i,j)$ is $u_i v_j$).

Computing $A_{\text{new}}^{-1}$ from scratch is expensive — it takes $O(n^3)$ operations for an $n \times n$ matrix. But if we already know $A^{-1}$, the **Sherman-Morrison formula** gives us $A_{\text{new}}^{-1}$ in only $O(n^2)$ operations:

$$
\boxed{(A + \mathbf{u}\mathbf{v}^T)^{-1} = A^{-1} - \frac{A^{-1}\mathbf{u}\mathbf{v}^T A^{-1}}{1 + \mathbf{v}^T A^{-1} \mathbf{u}}}
$$

The formula says: start with the old inverse $A^{-1}$, then subtract a correction term. The correction term involves multiplying the old inverse by $\mathbf{u}$ and $\mathbf{v}$ — all $O(n^2)$ or cheaper operations.

### Verification

We can verify this by checking that $(A + \mathbf{u}\mathbf{v}^T) \cdot (A + \mathbf{u}\mathbf{v}^T)^{-1} = I$. Multiply the original matrix by the proposed inverse:

$$
(A + \mathbf{u}\mathbf{v}^T)\left(A^{-1} - \frac{A^{-1}\mathbf{u}\mathbf{v}^T A^{-1}}{1 + \mathbf{v}^T A^{-1}\mathbf{u}}\right)
$$

Distribute:

$$
= AA^{-1} - \frac{AA^{-1}\mathbf{u}\mathbf{v}^T A^{-1}}{1 + \mathbf{v}^T A^{-1}\mathbf{u}} + \mathbf{u}\mathbf{v}^T A^{-1} - \frac{\mathbf{u}\mathbf{v}^T A^{-1}\mathbf{u}\mathbf{v}^T A^{-1}}{1 + \mathbf{v}^T A^{-1}\mathbf{u}}
$$

Since $AA^{-1} = I$, the first term is $I$. Simplify $AA^{-1}\mathbf{u} = \mathbf{u}$ in the second term. In the fourth term, $\mathbf{v}^T A^{-1}\mathbf{u}$ is a scalar, call it $c$. So:

$$
= I - \frac{\mathbf{u}\mathbf{v}^T A^{-1}}{1 + c} + \mathbf{u}\mathbf{v}^T A^{-1} - \frac{c \cdot \mathbf{u}\mathbf{v}^T A^{-1}}{1 + c}
$$

Combine the last three terms (they all contain $\mathbf{u}\mathbf{v}^T A^{-1}$):

$$
= I + \mathbf{u}\mathbf{v}^T A^{-1}\left(- \frac{1}{1 + c} + 1 - \frac{c}{1 + c}\right)
$$

The expression in parentheses is $\frac{-(1) + (1+c) - c}{1+c} = \frac{0}{1+c} = 0$.

So the result is $I$. The formula is correct.

### Numerical check

Let $A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$, so $A^{-1} = \frac{1}{5}\begin{pmatrix} 3 & -1 \\ -1 & 2 \end{pmatrix} = \begin{pmatrix} 0.6 & -0.2 \\ -0.2 & 0.4 \end{pmatrix}$.

Let $\mathbf{u} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$. Then $\mathbf{u}\mathbf{v}^T = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$.

So $A_{\text{new}} = \begin{pmatrix} 2 & 2 \\ 1 & 3 \end{pmatrix}$.

Applying the formula:

$$
A^{-1}\mathbf{u} = \begin{pmatrix} 0.6 \\ -0.2 \end{pmatrix}, \quad \mathbf{v}^T A^{-1} = \begin{pmatrix} -0.2 & 0.4 \end{pmatrix}
$$

$$
\mathbf{v}^T A^{-1}\mathbf{u} = -0.2 \times 1 + 0.4 \times 0 = -0.2
$$

$$
1 + \mathbf{v}^T A^{-1}\mathbf{u} = 1 + (-0.2) = 0.8
$$

$$
A^{-1}\mathbf{u} \cdot \mathbf{v}^T A^{-1} = \begin{pmatrix} 0.6 \\ -0.2 \end{pmatrix} \begin{pmatrix} -0.2 & 0.4 \end{pmatrix} = \begin{pmatrix} -0.12 & 0.24 \\ 0.04 & -0.08 \end{pmatrix}
$$

$$
A_{\text{new}}^{-1} = \begin{pmatrix} 0.6 & -0.2 \\ -0.2 & 0.4 \end{pmatrix} - \frac{1}{0.8}\begin{pmatrix} -0.12 & 0.24 \\ 0.04 & -0.08 \end{pmatrix} = \begin{pmatrix} 0.6 & -0.2 \\ -0.2 & 0.4 \end{pmatrix} - \begin{pmatrix} -0.15 & 0.30 \\ 0.05 & -0.10 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0.75 & -0.50 \\ -0.25 & 0.50 \end{pmatrix}
$$

Verify: $A_{\text{new}} \cdot A_{\text{new}}^{-1} = \begin{pmatrix} 2 & 2 \\ 1 & 3 \end{pmatrix}\begin{pmatrix} 0.75 & -0.50 \\ -0.25 & 0.50 \end{pmatrix} = \begin{pmatrix} 1.5 - 0.5 & -1.0 + 1.0 \\ 0.75 - 0.75 & -0.5 + 1.5 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$. Correct.

### How this appears in MoE

In the on-line MoE algorithm, each expert maintains a matrix $R_{ij}$ (the inverse of a weighted sum of outer products $\mathbf{x}\mathbf{x}^T$). When a new data point $\mathbf{x}^{(t)}$ arrives, $R_{ij}$ must be updated. The Sherman-Morrison formula gives:

$$
R_{ij}^{(t)} = \lambda^{-1} R_{ij}^{(t-1)} - \lambda^{-1} \frac{R_{ij}^{(t-1)} \mathbf{x}^{(t)} \mathbf{x}^{(t)T} R_{ij}^{(t-1)}}{\lambda [h_{ij}^{(t)}]^{-1} + \mathbf{x}^{(t)T} R_{ij}^{(t-1)} \mathbf{x}^{(t)}}
$$

This has exactly the form of the Sherman-Morrison formula: the old inverse ($\lambda^{-1} R^{(t-1)}$) minus a correction term involving the outer product $\mathbf{x}^{(t)}\mathbf{x}^{(t)T}$. The parameter $\lambda$ is a decay factor that down-weights old data, and $h_{ij}^{(t)}$ is the posterior probability that controls how much this data point affects expert $(i,j)$'s update. Without this formula, the on-line algorithm would need to re-invert a matrix at every time step — making it impractical.

---

## Summary

We have built a complete mathematical toolkit for Mixture of Experts. Squared error measures prediction quality. Expected value averages over stochastic selections, giving us the competitive error function. The softmax function converts arbitrary scores into valid probabilities for the gating network. The Gaussian density connects squared error to probability, and the likelihood function flips the perspective from "how probable is the data" to "how plausible is the model." The logarithm converts products into sums, and the negative log-likelihood gives us a loss function to minimise. Bayes' theorem updates prior gating probabilities into posterior responsibilities — the mechanism that makes experts specialise. Mixture models describe the generative process, and the log-of-a-sum structure of the mixture log-likelihood creates the computational challenge that the EM algorithm solves. Conditional probability and the multiplication rule handle nested decisions in the hierarchical architecture, while the multinomial distribution formalises the gating network's multi-way choices. Differentiating the mixture loss yields the posterior-weighted gradient, where gradient magnitude — not signed value — determines how fast each expert learns. Indicator variables with the complete-data log-likelihood show how the EM algorithm brings the logarithm inside the sum. And the Sherman-Morrison-Woodbury formula enables efficient on-line updates without re-inverting matrices.

With these tools in hand, we are ready for [Part 1](/blog/mixture-of-experts-part-1), where we put them all together to derive the Mixture of Experts framework from scratch — cooperative and competitive errors, the mixture-of-Gaussians interpretation, hierarchical mixtures, and the EM algorithm.
