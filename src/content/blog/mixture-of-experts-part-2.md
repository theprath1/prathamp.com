---
title: "Mixture of Experts from Scratch — Part 2: Scaling to Billions (2017–2022)"
description: "From thousands of experts to trillion-parameter models — sparse gating, top-k routing, load balancing, the Switch Transformer, and the engineering behind scaling MoEs — all derived step by step with a concrete 4-expert example."
date: 2026-03-04
tags: [machine-learning, mixture-of-experts, transformers, sparse-models, scaling, mathematics]
order: 2
draft: true
---

In Part 1, we derived the Mixture of Experts framework from Jacobs et al. (1991) and Jordan & Jacobs (1993): experts, gating networks, the mixture-of-Gaussians interpretation, and the EM algorithm. Those foundational papers used 4–8 experts on small tasks. In this post, we follow two papers that scaled MoEs to thousands of experts and trillions of parameters: [Shazeer et al. (2017)](https://arxiv.org/abs/1701.06538) and [Fedus et al. (2021)](https://arxiv.org/abs/2101.03961).

We will continue using our running example, but now adapt it: a **4-expert MoE layer** embedded inside a neural network, processing tokens of dimension $d = 4$.

---

## The Running Example

We have a sequence of 8 tokens (like words in a sentence), each represented as a vector of dimension $d = 4$. The MoE layer has $n = 4$ experts, each a small feed-forward network. For simplicity, each expert is a single linear transformation: $E_i(x) = W_i x$ where $W_i$ is a $4 \times 4$ matrix.

The gating network is a linear layer followed by softmax: $G(x) = \text{Softmax}(W_g \cdot x)$ where $W_g$ is a $4 \times 4$ matrix (4 experts, input dimension 4).

Let us work with one specific token $x = [1.0, 0.5, -0.3, 0.8]^T$.

---

## 1. The Sparsely-Gated MoE Layer (Shazeer et al., 2017)

### 1.1 The basic MoE output

Recall from Part 1 that the output of a flat MoE with softmax gating is:

$$
y = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)
$$

This is equation (1) from Shazeer et al. (2017), identical in form to the flat mixture from Jacobs et al. (1991). Every expert computes its output, and the gating network determines the weighted combination. With $n = 4$ experts:

$$
y = G(x)_1 \cdot E_1(x) + G(x)_2 \cdot E_2(x) + G(x)_3 \cdot E_3(x) + G(x)_4 \cdot E_4(x)
$$

**The problem with dense gating.** If $n$ is large (say 4096 experts), we must compute all $n$ expert outputs $E_i(x)$, even if most gate values $G(x)_i$ are tiny. This is computationally wasteful. The key insight of Shazeer et al. is: **make the gating sparse**. If $G(x)_i = 0$ for most experts, we need not compute $E_i(x)$ for those experts at all.

### 1.2 Noisy Top-K Gating

Shazeer et al. introduce two modifications to the standard softmax gating: **sparsity** and **noise**.

**Step 1: Compute raw gate logits.**

$$
H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)
$$

This is equation (4). The first term $x \cdot W_g$ is the same linear transformation as in standard softmax gating from Part 1. The second term adds **tunable Gaussian noise**, where the noise magnitude is controlled by a second trainable weight matrix $W_{\text{noise}}$. The $\text{Softplus}$ function ensures the noise scale is positive. The **Softplus function** is $\text{Softplus}(z) = \ln(1 + e^z)$, a smooth approximation to the ReLU that is always positive.

**Step 2: Keep only the top $k$ values.**

$$
\text{KeepTopK}(v, k)_i = \begin{cases} v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v \\ -\infty & \text{otherwise} \end{cases}
$$

This is equation (5). Setting non-top-$k$ entries to $-\infty$ means that after softmax, they become $e^{-\infty} = 0$. These experts are completely zeroed out — we need not compute their outputs at all.

**Step 3: Apply softmax to the kept values.**

$$
G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))
$$

This is equation (3). The output is a vector with exactly $k$ nonzero entries that sum to 1.

### Numerical check

Suppose for our token $x = [1.0, 0.5, -0.3, 0.8]^T$, the raw logits (before noise) are:

$$
x \cdot W_g = [2.1, \; 0.5, \; 3.4, \; 1.2]
$$

and after adding noise, we get $H(x) = [2.3, \; 0.4, \; 3.1, \; 1.5]$.

With $k = 2$, the top-2 values are $H_3 = 3.1$ and $H_1 = 2.3$. After KeepTopK:

$$
\text{KeepTopK}(H, 2) = [2.3, \; -\infty, \; 3.1, \; -\infty]
$$

After softmax (only over the two finite values):

$$
G(x)_1 = \frac{e^{2.3}}{e^{2.3} + e^{3.1}} = \frac{9.974}{9.974 + 22.198} = \frac{9.974}{32.172} = 0.310
$$

$$
G(x)_3 = \frac{e^{3.1}}{e^{2.3} + e^{3.1}} = \frac{22.198}{32.172} = 0.690
$$

$$
G(x)_2 = G(x)_4 = 0
$$

The final output uses only experts 1 and 3:

$$
y = 0.310 \cdot E_1(x) + 0.690 \cdot E_3(x)
$$

We compute only 2 out of 4 expert outputs. With 4096 experts and $k = 4$, we would compute only 4 out of 4096 — a massive computational saving.

### 1.3 Why noise helps

The noise term serves a critical purpose: **exploration**. Without noise, the same experts would be selected for similar inputs every time, creating a self-reinforcing cycle. The noise breaks this determinism, giving other experts a chance to be selected and trained. This is analogous to $\epsilon$-greedy exploration in reinforcement learning.

The noise also helps with **gradient flow**. Since $G(x)$ is computed by softmax over the top-$k$ values, the gate values for the top-$k$ experts have nonzero derivatives with respect to the gating network weights $W_g$. These gradients flow back through the gating network via standard **backpropagation**. The noise does not block gradients — it is sampled independently and treated as a constant during backpropagation.

---

## 2. Balancing Expert Utilization

### 2.1 The collapse problem

Shazeer et al. observed a critical failure mode: the gating network tends to converge to a state where it always produces large weights for the same few experts. This is **self-reinforcing** — favored experts are trained more rapidly and thus are selected even more frequently. This phenomenon was also noted by Eigen et al. (2013) in the context of deep MoEs.

### 2.2 The importance loss

To combat this, Shazeer et al. define the **importance** of expert $i$ relative to a batch $X$ of training examples:

$$
\text{Importance}(X) = \sum_{x \in X} G(x)
$$

This is equation (6). The importance is a vector where the $i$-th component is the sum of gate values for expert $i$ across the entire batch. If all experts are equally important, this vector should have equal entries.

The **importance loss** is the square of the **coefficient of variation** (CV) of the importance values:

$$
L_{\text{importance}} = w_{\text{importance}} \cdot \text{CV}(\text{Importance}(X))^2
$$

This is equation (7). The **coefficient of variation** is the standard deviation divided by the mean: $\text{CV}(v) = \frac{\text{Std}(v)}{\text{Mean}(v)}$. It is zero when all entries are equal and increases as they diverge. Squaring the CV and multiplying by a hand-tuned weight $w_{\text{importance}}$ gives us a differentiable penalty that encourages uniform expert usage.

### Numerical check

Suppose in a batch of 8 tokens, the gate values (summed across the batch) are:

$$
\text{Importance} = [3.2, \; 2.8, \; 1.5, \; 0.5]
$$

Mean $= (3.2 + 2.8 + 1.5 + 0.5)/4 = 8.0/4 = 2.0$.

Variance $= \frac{1}{4}[(3.2-2.0)^2 + (2.8-2.0)^2 + (1.5-2.0)^2 + (0.5-2.0)^2]$
$= \frac{1}{4}[1.44 + 0.64 + 0.25 + 2.25] = \frac{4.58}{4} = 1.145$.

Std $= \sqrt{1.145} = 1.070$.

$\text{CV} = 1.070 / 2.0 = 0.535$.

$L_{\text{importance}} = w_{\text{importance}} \times 0.535^2 = w_{\text{importance}} \times 0.286$.

This is a significant penalty because expert 4 is barely used (importance $= 0.5$) while expert 1 is over-used (importance $= 3.2$). With a perfectly balanced allocation (importance $= [2.0, 2.0, 2.0, 2.0]$), the CV would be 0 and the loss would vanish.

### 2.3 The load loss

The importance loss ensures experts have equal total gate weight, but experts may still receive very different *numbers* of examples. One expert might receive a few examples with large weights, while another receives many examples with small weights. This creates memory and performance problems on distributed hardware. Shazeer et al. introduce a second loss $L_{\text{load}}$ that ensures balanced *counts* of examples routed to each expert. The details are in their Appendix A.

---

## 3. Embedding MoEs in Language Models

Shazeer et al. apply the MoE layer **between stacked LSTM layers** in a recurrent language model:

```
          ┌──────────┐
          │  LSTM 2  │
          └────┬─────┘
               │
          ┌────┴─────┐
          │ MoE Layer │ ← experts are FFNs
          └────┬─────┘
               │
          ┌────┴─────┐
          │  LSTM 1  │
          └────┬─────┘
               │
            input
```

The MoE layer is **applied convolutionally**: the same MoE (same experts, same gating network) processes each token position independently. At each position, the gating network selects a potentially different combination of experts. This is how the model achieves **conditional computation**: different experts are active for different tokens.

### The scale achieved

On the 1-Billion Word Language Modeling Benchmark, Shazeer et al. trained models with up to 137 billion parameters (using 131,072 experts in a hierarchical MoE), achieving a test perplexity of **28.0** — beating the previous best of **30.6** from Jozefowicz et al. (2016) — while using only 8 million ops per timestep (similar computational budget).

On machine translation (WMT'14 En→Fr), a model with 2048 experts achieved **40.56 BLEU**, surpassing GNMT (Wu et al., 2016) at **39.22 BLEU**, with 8.7B total parameters but only 85M ops per timestep versus GNMT's 214M.

The key insight: **model capacity can be dramatically increased (1000x) with only minor losses in computational efficiency**, provided the gating is sparse enough.

---

## 4. The Switch Transformer (Fedus et al., 2021)

### 4.1 The simplification: $k = 1$

Shazeer et al. conjectured that routing to $k > 1$ experts was necessary for non-trivial gradients to flow through the gating function. Fedus et al. challenged this assumption. They found that routing to a **single expert** ($k = 1$) not only works — it works *better*.

The output of a **Switch layer** for token $x$ is:

$$
y = p_i(x) \cdot E_i(x), \quad \text{where } i = \arg\max_j \; (W_r \cdot x)_j
$$

The token is routed to the single expert with the highest gate logit. The output is the expert's output multiplied by the softmax gate value (not just 1), which preserves differentiability.

The benefits of $k = 1$ routing are three-fold:
1. **Reduced router computation.** The router only needs to find the max, not the top-$k$.
2. **Halved expert batch size.** Each token goes to exactly one expert, so each expert's batch is at most half of what it would be with $k = 2$.
3. **Simplified implementation.** Communication costs are reduced because each token is sent to exactly one device.

### 4.2 Where the Switch layer goes in a Transformer

The Switch Transformer replaces the **dense feed-forward network** (FFN) in each Transformer block with a Switch FFN layer:

```
  Standard Transformer Block:       Switch Transformer Block:

  ┌───────────────────┐            ┌───────────────────────┐
  │  Add + Normalize  │            │   Add + Normalize     │
  └────────┬──────────┘            └──────────┬────────────┘
           │                                  │
  ┌────────┴──────────┐            ┌──────────┴────────────┐
  │   Dense FFN       │            │  Switch FFN Layer     │
  │  (same for all    │            │  ┌─────┬─────┬─────┐  │
  │   tokens)         │            │  │FFN1 │FFN2 │FFN3 │  │
  └────────┬──────────┘            │  └──┬──┘  │  └──┬──┘  │
           │                       │     │Router│     │     │
  ┌────────┴──────────┐            └─────┴─────┴─────┴─────┘
  │  Add + Normalize  │                       │
  └────────┬──────────┘            ┌──────────┴────────────┐
           │                       │   Add + Normalize     │
  ┌────────┴──────────┐            └──────────┬────────────┘
  │  Self-Attention   │                       │
  └────────┬──────────┘            ┌──────────┴────────────┐
           │                       │   Self-Attention      │
         input                     └──────────┬────────────┘
                                              │
                                            input
```

Each token is independently routed to one of the FFN experts by the router. The router is a simple linear layer: $h(x) = W_r \cdot x$, where $W_r \in \mathbb{R}^{N \times d_{\text{model}}}$ and $N$ is the number of experts.

### 4.3 Expert capacity and token dropping

In a distributed implementation, each expert has a fixed batch size called the **expert capacity**:

$$
\text{expert capacity} = \left(\frac{\text{tokens per batch}}{\text{number of experts}}\right) \times \text{capacity factor}
$$

This is equation (3) from Fedus et al. A **capacity factor** greater than 1.0 creates buffer space for when tokens are not perfectly balanced. If too many tokens are routed to an expert (exceeding its capacity), the overflow tokens are **dropped** — their representations pass directly to the next layer through the residual connection.

### Numerical check

With 8 tokens and 4 experts, if the capacity factor is 1.0:

$$
\text{expert capacity} = \frac{8}{4} \times 1.0 = 2
$$

Each expert can process at most 2 tokens. If the router sends 4 tokens to expert 1 and 0 tokens to expert 4, expert 1 will process only its first 2 tokens and drop the other 2.

With capacity factor 1.5:

$$
\text{expert capacity} = \frac{8}{4} \times 1.5 = 3
$$

Now each expert can handle up to 3 tokens, reducing the risk of dropping. But the total buffer space across experts is $4 \times 3 = 12 > 8$, so some compute is wasted on padding.

Fedus et al. found that capacity factors of 1.0 to 1.25 work best — smaller values reduce wasted compute while the load balancing loss keeps dropped tokens below 1%.

---

## 5. The Differentiable Load Balancing Loss

Fedus et al. simplified the load balancing approach from Shazeer et al. Instead of two separate losses (importance + load), they use a single **auxiliary loss** that is the scaled dot-product between two vectors $f$ and $P$:

$$
\boxed{\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i}
$$

This is equation (4) from Fedus et al. Here:

$f_i$ is the **fraction of tokens dispatched** to expert $i$:

$$
f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbb{1}\{\arg\max \; p(x) = i\}
$$

This is equation (5). The indicator $\mathbb{1}\{\cdot\}$ equals 1 when token $x$ is routed to expert $i$ and 0 otherwise.

$P_i$ is the **fraction of router probability** allocated to expert $i$:

$$
P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x)
$$

This is equation (6). Unlike $f_i$, this is **differentiable** with respect to the router weights, because $p_i(x)$ is the softmax output.

Under perfectly uniform routing, each expert receives a fraction $\frac{1}{N}$ of the tokens and a fraction $\frac{1}{N}$ of the probability mass. So:

$$
\sum_{i=1}^{N} f_i \cdot P_i = \sum_{i=1}^{N} \frac{1}{N} \cdot \frac{1}{N} = N \cdot \frac{1}{N^2} = \frac{1}{N}
$$

The loss becomes $\alpha \cdot N \cdot \frac{1}{N} = \alpha$. The factor of $N$ in the loss ensures it remains constant as the number of experts changes.

### Numerical check

With 4 experts and 8 tokens, suppose:
- Token routing: experts 1,1,2,3,1,3,2,4 (expert 1 gets 3 tokens, experts 2,3 get 2 each, expert 4 gets 1)
- So $f = [3/8, \; 2/8, \; 2/8, \; 1/8] = [0.375, \; 0.250, \; 0.250, \; 0.125]$

Suppose the average softmax probabilities are $P = [0.35, \; 0.25, \; 0.28, \; 0.12]$.

$$
\sum_i f_i \cdot P_i = 0.375 \times 0.35 + 0.250 \times 0.25 + 0.250 \times 0.28 + 0.125 \times 0.12
$$

$$
= 0.131 + 0.063 + 0.070 + 0.015 = 0.279
$$

$$
\text{loss} = \alpha \cdot 4 \cdot 0.279 = 1.116 \alpha
$$

For uniform routing, the loss would be $\alpha \cdot 4 \cdot 0.250 = 1.000 \alpha$. The actual loss is $1.116\alpha > 1.000\alpha$, penalizing the imbalance. The gradient of this loss (which flows only through the $P_i$ terms, since $f_i$ is non-differentiable) pushes the router to distribute probability more uniformly.

Fedus et al. use $\alpha = 10^{-2}$ throughout — large enough for load balancing, small enough to not overwhelm the cross-entropy training objective.

### Why this design is elegant

The key insight is that $f_i$ (the hard routing fraction) is **not differentiable**, but $P_i$ (the soft probability fraction) **is differentiable**. Their dot product $f_i \cdot P_i$ creates a loss where the gradient flows through the differentiable $P_i$ term. When expert $i$ receives too many tokens (high $f_i$), the gradient pushes $P_i$ down, which reduces the probability of routing future tokens to expert $i$.

---

## 6. Improved Training Techniques

### 6.1 Selective precision for stability

Sparse expert models are prone to **training instability** — the hard routing decisions create discontinuities that can be amplified by low-precision arithmetic. Fedus et al. found that training with bfloat16 precision causes Switch Transformers to diverge.

Their solution: **selective casting**. They cast the router input to float32 precision locally (within the routing function only), perform the softmax and dispatch in float32, then cast back to bfloat16 for the expert computation. This achieves nearly the same speed as pure bfloat16 with the stability of float32:

| Model | Quality (Neg. Log Perp.) | Speed (Examples/sec) |
|-------|--------------------------|---------------------|
| Switch-Base (float32) | -1.718 | 1160 |
| Switch-Base (bfloat16) | -3.780 [diverged] | 1390 |
| Switch-Base (Selective) | **-1.716** | **1390** |

The selective precision model matches float32 quality at bfloat16 speed.

### 6.2 Smaller initialization for stability

Fedus et al. initialize weights from a truncated normal with standard deviation $\sigma = \sqrt{s/n}$ where $s$ is a scale factor and $n$ is the fan-in. They found that reducing $s$ from the default 1.0 to **0.1** dramatically improves stability:

| Init Scale | Avg Quality | Std Dev of Quality |
|-----------|-------------|-------------------|
| 0.1x | **-2.72** | **0.01** |
| 1.0x | -3.60 | 0.68 |

The variance across runs drops from $0.68$ to $0.01$ — a 68x reduction. Smaller initialization keeps the router outputs closer to zero early in training, producing near-uniform routing that gradually sharpens as training proceeds.

### 6.3 Expert dropout for fine-tuning

Switch Transformers have many more parameters than dense models of similar FLOPs, making them prone to overfitting during fine-tuning on small datasets. Fedus et al. introduce **expert dropout**: during fine-tuning, they use a small dropout rate (0.1) at non-expert layers but a much larger rate (0.4) at expert layers.

| Dropout Config | GLUE | CNNDM | SQuAD | SuperGLUE |
|---------------|------|-------|-------|-----------|
| T5-Base (d=0.1) | 82.9 | **19.6** | 83.5 | 72.4 |
| Switch (d=0.1) | 84.7 | 19.1 | **83.7** | **73.0** |
| Switch (d=0.1, ed=0.4) | **85.2** | **19.6** | **83.7** | **73.0** |

The combination of low non-expert dropout and high expert dropout gives the best results across all benchmarks.

---

## 7. Distillation: Compressing Sparse Models

Deploying a trillion-parameter sparse model is impractical for many use cases. Fedus et al. study **distillation**: compressing large sparse models into small dense ones.

They find two techniques that stack:
1. **Initialize the dense model with non-expert weights** from the sparse teacher. Since non-expert layers (self-attention, layer norms, embeddings) are identical between Switch-Base and T5-Base, these weights can be copied directly.
2. **Use a mixture of teacher and ground-truth loss**: $0.25 \times \mathcal{L}_{\text{teacher}} + 0.75 \times \mathcal{L}_{\text{label}}$.

| Technique | Parameters | Quality ($\uparrow$) |
|-----------|-----------|---------------------|
| T5-Base | 223M | -1.636 |
| Switch-Base (teacher) | 3,800M | -1.444 |
| Distillation only | 223M | (3%) -1.631 |
| + Non-expert init | 223M | (20%) -1.598 |
| + Mixed loss | 223M | **(29%) -1.580** |

The quality gain percentage measures how much of the gap between T5-Base and Switch-Base the student recovers. By combining both techniques, a 223M parameter student preserves **~30% of the quality gain** from a 3.8B parameter teacher — a **17x compression** with meaningful quality retention.

At extreme compression, distilling a 14.7B-parameter Switch model into a 223M dense model (99% compression) still retains 28% of the teacher's improvement. This demonstrates that much of what experts learn can be transferred to a single dense model.

---

## 8. Multilingual Learning

Fedus et al. evaluate Switch Transformers on multilingual pre-training across **101 languages** using the mC4 dataset. Comparing mSwitch-Base (FLOP-matched) to mT5-Base:

- Switch Transformer improves negative log perplexity on **all 101 languages**
- Mean step speedup: **5x** over mT5-Base
- **91% of languages** achieve at least a **4x speedup**

This is particularly notable because multilingual data is naturally clustered by language — exactly the kind of structure where MoEs excel. Different experts can specialize on different languages or language families, while shared linguistic patterns are captured in the non-expert layers.

---

## 9. Parallelism Strategies for Scaling

Arbitrarily increasing the number of experts hits diminishing returns. Fedus et al. study **complementary** scaling strategies that combine expert parallelism with data and model parallelism.

With $N = n \times m$ total cores, where $n$ is the data-parallel dimension and $m$ is the model-parallel dimension:

**Data parallelism** ($n = N, m = 1$): Each core holds the full model and a shard of the batch. No communication until the end of the forward/backward pass (all-reduce for gradients).

**Model parallelism** ($n = 1, m = N$): Each core holds a slice of the weights. Communication at every layer (all-reduce for activations).

**Expert parallelism**: Each core holds a subset of experts. Tokens are dispatched via **all-to-all communication**: each core sends tokens to the core hosting the selected expert, then receives results back. This is unique to MoE — dense models don't need it.

**Expert + Data parallelism**: Used for Switch-C (1,571B params, 2048 experts). Each expert lives on one core. No model parallelism, so the per-core FFN size is limited, but expert count scales linearly with cores.

**Expert + Model + Data parallelism**: Used for Switch-XXL (395B params). Model parallelism increases per-token FLOPs (larger $d_{ff}$), expert parallelism increases parameters, and data parallelism increases throughput.

### Trillion-parameter results

| Model | Parameters | FLOPs/seq | Experts | Neg. Log Perp. @250k | @500k |
|-------|-----------|-----------|---------|---------------------|-------|
| T5-XXL | 11B | 6.3T | — | -1.147 | -1.095 |
| Switch-XXL | 395B | 6.3T | 64 | **-1.086** | **-1.008** |
| Switch-C | 1,571B | 890B | 2048 | -1.096 | -1.043 |

Switch-C uses expert-parallelism only and is **4x faster** to reach a given perplexity than T5-XXL. Interestingly, Switch-C exhibited **no training instability** despite having 1.6T parameters, while the FLOP-heavy Switch-XXL was sometimes unstable — suggesting that instability correlates more with FLOPs per token than total parameter count.

---

## 10. Scaling Properties

### 10.1 Scaling with number of experts

The key scaling axis for Switch Transformers is the number of experts. Increasing experts keeps the FLOPs per token approximately constant (since each token uses only one expert), but increases the total model parameters.

Fedus et al. trained Switch-Base models with 2, 4, 8, 16, 32, 64, 128, and 256 experts, all FLOP-matched to T5-Base. They observe:

1. **Consistent scaling**: every doubling of experts improves test loss.
2. **Sample efficiency**: the Switch-Base 64-expert model achieves the same quality as T5-Base at step 60k versus step 450k — a **7.5x speedup** in terms of training steps.
3. **Time efficiency**: on a wall-clock basis, the 64-expert Switch Transformer reaches T5-Base quality in **one-seventh** the time.

Even with as few as 2 experts, Switch Transformers improve over the dense T5-Base baseline — you do not need a supercomputer to benefit from sparse expert models.

### 10.2 Switch vs. dense scaling

A natural question: what if we used the compute budget for a bigger dense model instead?

Fedus et al. compared Switch-Base (64 experts, FLOP-matched to T5-Base) against T5-Large (which uses 3.5x more FLOPs per token):

$$
\text{Switch-Base: } 7x \text{ speedup over T5-Base, } 2.5x \text{ speedup over T5-Large}
$$

Despite T5-Large using 3.5x more computation per token, Switch-Base is still faster and more sample-efficient. This demonstrates that **parameter count** (via sparse activation) is a separate and important scaling axis beyond **compute per token**.

---

## 11. Switch Transformer vs. MoE Transformer: Head-to-Head

Fedus et al. provide a direct comparison with all models using 128 experts at every other feed-forward layer:

| Model | Capacity Factor | Quality (100k steps) | Time to Threshold | Speed |
|-------|----------------|---------------------|-------------------|-------|
| T5-Base | — | -1.731 | Not achieved | 1600 ex/s |
| MoE-Base | 2.0 | -1.547 | 68.7 hrs | 840 ex/s |
| Switch-Base | 2.0 | -1.554 | 72.8 hrs | 860 ex/s |
| MoE-Base | 1.0 | -1.572 | 80.1 hrs | 860 ex/s |
| Switch-Base | 1.0 | -1.561 | **62.8 hrs** | **1000 ex/s** |
| Switch-Base+ | 1.0 | **-1.534** | 67.6 hrs | 780 ex/s |

Three findings:
1. Switch Transformers outperform MoE on a speed-quality basis at capacity factor 1.0.
2. Switch Transformer has a smaller computational footprint than top-2 MoE.
3. Smaller capacity factors (1.0) work better for Switch, indicating efficient memory usage.

---

## 12. Fine-Tuning Results

Switch Transformers also excel on downstream tasks. Comparing FLOP-matched Switch models to T5 baselines across diverse benchmarks:

| Model | GLUE | SQuAD | SuperGLUE | Winogrande |
|-------|------|-------|-----------|------------|
| T5-Base | 84.3 | 85.5 | 75.1 | 66.6 |
| Switch-Base | **86.7** | **87.2** | **79.5** | **73.3** |
| T5-Large | 87.8 | 88.1 | 82.7 | 79.1 |
| Switch-Large | **88.5** | **88.6** | **84.7** | **83.0** |

The gains are consistent across both model sizes and across both reasoning tasks (SuperGLUE, Winogrande) and knowledge-heavy tasks (SQuAD, closed-book QA). On knowledge-based tasks, Switch-XXL achieves new state-of-the-art results: Natural Questions exact match increases to 34.4 (vs. prior best 32.8), WebQuestions to 41.0 (vs. 37.2), and TriviaQA to 47.5 (vs. 42.9).

---

## 13. The Unified View: From Dense to Sparse

Let us connect the dots from Part 1 to Part 2. The evolution follows a clear arc:

**Jacobs et al. (1991)**: All experts are active for all inputs (**dense MoE**). Outputs are blended by softmax gating. Training uses gradient descent on the mixture log-likelihood.

**Jordan & Jacobs (1993)**: Experts are arranged in a **tree hierarchy**. Training uses EM, which separates into weighted IRLS subproblems. Still dense — all experts contribute to every output.

**Shazeer et al. (2017)**: Only the **top-$k$** experts are active (**sparse MoE**). The rest are zeroed out by setting their logits to $-\infty$ before softmax. Noisy gating encourages exploration. Auxiliary losses prevent expert collapse. Scale: up to 131,072 experts, 137B parameters.

**Fedus et al. (2021)**: Only **one** expert is active (**Switch routing**, $k = 1$). The simplification reduces computation, halves expert batch sizes, and performs better than $k > 1$. Selective precision, smaller initialization, and expert dropout stabilize training. Scale: up to 1 trillion parameters.

The mathematical core is unchanged from Part 1:

$$
y = \sum_{i \in \mathcal{T}} G(x)_i \cdot E_i(x)
$$

where $\mathcal{T}$ is the set of selected experts. What changed is:
- $|\mathcal{T}|$ shrunk from $n$ (all experts) to $k$ (top-$k$) to $1$ (switch)
- $n$ grew from 4 to 131,072
- The training moved from EM to end-to-end backpropagation
- Auxiliary losses replaced the posterior-based responsibility assignment

---

## Summary

Shazeer et al. (2017) proved that conditional computation works at scale: 1000x more parameters with only minor computational overhead, achieving state-of-the-art results on language modeling and translation. The key innovations were noisy top-$k$ gating for sparsity, importance and load losses for balance, and mixing data and model parallelism for efficiency.

Fedus et al. (2021) simplified and improved the approach with the Switch Transformer: route each token to a single expert, use selective precision for stability, initialize smaller for consistent training, and apply expert dropout for fine-tuning. The result scales from 223M to 1 trillion parameters, achieving 7x speedups over dense T5 baselines.

In Part 3, we will examine why MoEs work from a theoretical perspective (Chen et al., 2022) and survey the modern MoE landscape in large language models (Cai et al., 2024).
