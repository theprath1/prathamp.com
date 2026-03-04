---
title: "Mixture of Experts from Scratch — Part 3: Why MoEs Work and the Modern Landscape (2022–2024)"
description: "Why experts specialize instead of collapsing, the role of nonlinearity, exploration and router learning, load balancing theory, and a complete taxonomy of modern MoE in LLMs — all derived from first principles with concrete examples."
date: 2026-03-04
tags: [machine-learning, mixture-of-experts, deep-learning, theory, large-language-models, mathematics]
order: 1
---

In Parts 1 and 2, we built the MoE framework from the ground up and scaled it to trillions of parameters. But a fundamental question remained unanswered: **why do experts diversify instead of collapsing into a single model?** All experts start with the same architecture, same initialization distribution, same training algorithm. The router starts uniform. Yet somehow, after training, different experts specialize on different data clusters and the router learns to dispatch data to the right expert.

In this final part, we examine [Chen et al. (2022)](https://arxiv.org/abs/2208.02813), who provide the first theoretical answer, and [Cai et al. (2024)](https://arxiv.org/abs/2407.06204), who survey the entire modern MoE landscape in LLMs.

We continue with our running example but adapt it to the theoretical setting: a **4-expert MoE** with 2-layer CNN experts trained on data from 4 clusters.

---

## The Running Example

We have data from $K = 4$ clusters. Each data point $(\mathbf{x}, y)$ has:
- A **feature signal** patch: $y \alpha \mathbf{v}_k$ (label $\times$ scaling $\times$ direction for cluster $k$)
- A **cluster-center signal** patch: $\beta \mathbf{c}_k$ (identifies which cluster)
- A **feature noise** patch: $\epsilon \gamma \mathbf{v}_{k'}$ (noise from a different cluster's direction)
- Random Gaussian noise patches

All signal vectors $\{\mathbf{v}_k\}$ and $\{\mathbf{c}_k\}$ are orthogonal to each other. The label $y \in \{+1, -1\}$ is binary.

The critical feature: the feature noise $\gamma$ can be as strong as the feature signal $\alpha$. This means a single expert, no matter how powerful, cannot reliably classify the data — it will confuse the noise patches with the signal patches. But an MoE can learn to first identify the cluster (via the center signal), then route to a specialized expert that has learned the feature signal for that specific cluster.

We use $M = 4$ experts and $K = 4$ clusters with $d = 50$ dimensional patches.

---

## 1. Why a Single Expert Fails

### Theorem 4.1 (Chen et al., 2022): Single expert performs poorly

If the feature noise has the same strength distribution as the feature signal ($\mathcal{D}_\alpha = \mathcal{D}_\gamma$), then **any** function of the form $F(\mathbf{x}) = \sum_{p=1}^{P} f(\mathbf{x}^{(p)})$ — which includes any single two-layer CNN with any activation function and any number of neurons — will have large test error:

$$
\mathbb{P}_{(\mathbf{x}, y) \sim \mathcal{D}}(y F(\mathbf{x}) \leq 0) \geq \frac{1}{8}
$$

In words: the best any single model can achieve is at most 87.5% accuracy on this data distribution. The single model is fundamentally limited because it applies the same function to every patch, and when the noise patch looks exactly like a signal patch from a different cluster, the model cannot distinguish them.

### Why this is the case

The single model computes $F(\mathbf{x}) = \sum_p f(\mathbf{x}^{(p)})$. It must use the same function $f$ on every patch. When the feature noise patch $\epsilon \gamma \mathbf{v}_{k'}$ has the same magnitude distribution as the feature signal patch $y \alpha \mathbf{v}_k$, the contribution of the noise patch to $F(\mathbf{x})$ can be as large as the signal patch's contribution. Since the noise label $\epsilon$ is independent of the true label $y$, the noise corrupts the prediction.

### Numerical check

Suppose in our example with 4 clusters, a single CNN achieves 80% accuracy. By Theorem 4.1, it cannot exceed 87.5%. This is confirmed experimentally by Chen et al.:

| Model | Test Accuracy |
|-------|--------------|
| Single linear CNN | 68.71% |
| Single nonlinear CNN | 79.48% |

Both are well below 87.5%, consistent with the theorem. The 87.5% bound is tight — there exist data distributions where a single model achieves exactly 87.5%.

---

## 2. Why a Nonlinear MoE Succeeds

### Theorem 4.2 (Chen et al., 2022): Nonlinear MoE performs well

With $M = \Theta(K \log K \log \log d)$ experts, filter size $J = \Theta(\log M \log \log d)$, appropriate initialization scale and learning rates, and training data of size $n = \Omega(d)$, gradient descent on the MoE model (with noise perturbation in the router) converges to a solution with:

1. **Zero training error:** $y_i F(\mathbf{x}_i; \boldsymbol{\Theta}^{(T)}, \mathbf{W}^{(T)}) > 0$ for all training points.
2. **Nearly zero test error:** $\mathbb{P}(yF(\mathbf{x}) \leq 0) = o(1)$, approaching 0 as the problem dimension grows.

More importantly, the experts can be divided into $K$ non-empty groups $[M] = \sqcup_{k \in [K]} \mathcal{M}_k$ such that:
- Each expert $m \in \mathcal{M}_k$ performs well on cluster $\Omega_k$
- The router dispatches examples from $\Omega_k$ to experts in $\mathcal{M}_k$ with high probability

In our example: the 4 experts will specialize into (at least) 4 groups, one per cluster, and the router will learn to send cluster-$k$ data to the experts in group $k$.

### Numerical verification

Chen et al. verify this experimentally:

| Model | Test Accuracy | Dispatch Entropy |
|-------|--------------|-----------------|
| Single linear | 68.71% | — |
| Single nonlinear | 79.48% | — |
| MoE linear | 92.99 ± 2.11% | 1.300 ± 0.044 |
| MoE nonlinear | **99.46 ± 0.55%** | **0.098 ± 0.087** |

The nonlinear MoE achieves 99.46% accuracy — far above the single-model ceiling of 87.5%. The **dispatch entropy** is nearly zero (0.098), meaning each expert receives data from essentially one cluster. The linear MoE also outperforms a single model but has high dispatch entropy (1.300), meaning the clusters are not well separated across experts.

---

## 3. The Three Training Stages

Chen et al. decompose the training process of the MoE into three stages, revealing the mechanism by which experts specialize.

### Stage 1: Exploration

At the beginning of training, the gating network $\boldsymbol{\Theta}$ is initialized to zero, so $\mathbf{h}(\mathbf{x}; \boldsymbol{\Theta}^{(0)}) = \mathbf{0}$ for all inputs. With the addition of uniform noise to the router, each expert is selected with probability $\approx 1/M$ for every input. The router is essentially performing **uniform random assignment**.

During this stage, each expert sees data from all clusters (approximately equally). The experts begin to specialize based on their random weight initialization. By the **law of large numbers**, different experts will have slightly different initial inner products $\langle \mathbf{w}_{m,j}^{(0)}, \mathbf{v}_k \rangle$ with the cluster signal vectors. An expert whose initial weights happen to align more with $\mathbf{v}_k$ will learn to classify cluster $k$ faster.

This exploration stage lasts until $T_1 = \lfloor \eta^{-1} \sigma_0^{0.5} \rfloor$ iterations — a relatively short period.

**Lemma 5.2:** At the end of the exploration stage, the experts have diverged into $K$ groups $\mathcal{M}_k$. Each expert $m \in \mathcal{M}_k$ achieves near-zero error on cluster $\Omega_k$ but error $\Omega(1/K)$ on other clusters. The grouping is determined by which cluster center each expert's initial weights happened to align with: $\mathcal{M}_k := \{m \mid \arg\max_{k' \in [K], j \in [J]} \langle \mathbf{v}_{k'}, \mathbf{w}_{m,j}^{(0)} \rangle = k\}$.

### Stage 2: Router Learning

Once the experts have specialized, the router needs to learn which expert to use for which input. The key signal the router uses is the **cluster-center vector** $\mathbf{c}_k$ in the input. The router learns to compute $\boldsymbol{\Theta}^T \mathbf{x}$, which includes terms like $\boldsymbol{\theta}_m^T \mathbf{c}_k$. When $\boldsymbol{\theta}_m$ aligns with $\mathbf{c}_k$ for the experts in $\mathcal{M}_k$, the router correctly routes cluster-$k$ data to group-$k$ experts.

### Stage 3: Generalization

The final stage combines the expert specialization (from Stage 1) with correct routing (from Stage 2) to achieve near-zero test error.

---

## 4. The Role of Nonlinearity

One of the most striking findings of Chen et al. is that **linear experts fail** even in the MoE setting:

| Setting | MoE (linear) | MoE (nonlinear) |
|---------|-------------|-----------------|
| Setting 1 | 92.99 ± 2.11 | **99.46 ± 0.55** |
| Setting 2 | 88.48 ± 1.96 | **98.09 ± 1.27** |
| Setting 3 | 95.93 ± 1.34 | **99.99 ± 0.02** |
| Setting 4 | 93.30 ± 1.48 | **98.92 ± 1.18** |

The dispatch entropy tells the same story:

| Setting | MoE (linear) entropy | MoE (nonlinear) entropy |
|---------|---------------------|------------------------|
| 1 | 1.300 ± 0.044 | **0.098 ± 0.087** |
| 2 | 1.294 ± 0.036 | **0.171 ± 0.103** |
| 3 | 1.160 ± 0.100 | **0.008 ± 0.011** |
| 4 | 1.160 ± 0.155 | **0.089 ± 0.120** |

The dispatch entropy of nonlinear MoEs is nearly zero in all settings, meaning each expert receives data from exactly one cluster. Linear MoEs have entropy above 1.0, meaning experts mix data from multiple clusters.

Why? A linear function $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$ responds to all patches equally — it cannot distinguish a signal patch from a noise patch in a different direction. A nonlinear function like $f(\mathbf{x}) = \sum_j \sum_p \sigma(\langle \mathbf{w}_j, \mathbf{x}^{(p)} \rangle)$ with cubic activation $\sigma(z) = z^3$ can amplify the response to aligned patches while suppressing others, because $z^3$ grows much faster than $z$ for large $|z|$.

---

## 5. The Three Key Techniques

Chen et al. identify three technical innovations that make the theoretical analysis possible.

### Technique 1: Stability by Smoothing (noise in the router)

The noise added to the router (from Shazeer et al., 2017) is not just a heuristic — it has a precise mathematical role. **Lemma 5.1** states: if $\mathbf{h}$ and $\hat{\mathbf{h}}$ are two gating network outputs with corresponding routing probabilities $\mathbf{p}$ and $\hat{\mathbf{p}}$, then:

$$
\|\mathbf{p} - \hat{\mathbf{p}}\|_\infty \leq M^2 \|\mathbf{h} - \hat{\mathbf{h}}\|_\infty
$$

This **Lipschitz bound** says that small changes in the gating network outputs cause at most $M^2$-scaled changes in the routing probabilities. Without noise, a tiny change in gating outputs could cause a discontinuous switch from one expert to another (hard routing). With noise, the transition is smooth because the routing probability is an average over many noise realizations.

At the beginning of training, $\boldsymbol{\Theta}^{(0)} = \mathbf{0}$, so $\mathbf{h}(\mathbf{x}; \boldsymbol{\Theta}^{(0)}) = \mathbf{0}$ for all inputs. The noise term dominates, and each expert gets selected with probability $\approx 1/M$. As training proceeds and $\boldsymbol{\Theta}$ grows, the gating signal dominates the noise and routing sharpens.

### Technique 2: Experts from Exploration

During the exploration stage, the router is nearly uniform but the experts are not — their random initializations give them different affinities for different clusters. This breaks the symmetry. The expert that is initially closest to cluster $k$'s signal learns fastest on cluster $k$, reinforcing its specialization.

### Technique 3: Normalized Gradient Descent

The experts use **normalized gradient descent**: the gradient is divided by its Frobenius norm before the update:

$$
\mathbf{W}_m^{(t+1)} = \mathbf{W}_m^{(t)} - \eta \cdot \frac{\nabla_{\mathbf{W}_m} \mathcal{L}}{\|\nabla_{\mathbf{W}_m} \mathcal{L}\|_F}
$$

This ensures all experts are trained at the **same speed** regardless of how many data points each receives. Without normalization, an expert that receives more data (due to router imbalance) would have a larger gradient and train faster, creating a self-reinforcing feedback loop. Normalized gradient descent breaks this cycle: even if expert $m$ receives twice as many data points, its update step has the same magnitude as every other expert's.

This is a theoretical justification for load balancing: load balancing losses ensure experts receive similar amounts of data, but normalized gradients ensure experts train at similar speeds *even when* loads are imbalanced.

---

## 6. Verification on Real Data

Chen et al. test their theory on CIFAR-10 and a variant called **CIFAR-10-Rotate** (images rotated by 30°, task is to predict rotation — a dataset with strong natural cluster structure).

| Architecture | Model | CIFAR-10 (%) | CIFAR-10-Rotate (%) |
|-------------|-------|-------------|-------------------|
| CNN | Single | 80.68 ± 0.45 | 76.78 ± 1.79 |
| CNN | MoE | 80.31 ± 0.62 | **79.60 ± 1.25** |
| MobileNetV2 | Single | 92.45 ± 0.25 | 85.76 ± 2.91 |
| MobileNetV2 | MoE | 92.23 ± 0.72 | **89.85 ± 2.54** |
| ResNet18 | Single | 95.51 ± 0.31 | 88.23 ± 0.96 |
| ResNet18 | MoE | 95.32 ± 0.68 | **92.60 ± 2.01** |

On standard CIFAR-10, MoEs provide no benefit — the task lacks strong cluster structure, so a single model suffices. On CIFAR-10-Rotate, which has intrinsic cluster structure, MoEs consistently outperform single models by 2.8–4.4 percentage points across all architectures.

This confirms the theory: **the advantage of MoE over single models depends on the cluster structure of the data**. When data naturally decomposes into clusters that require different processing, MoEs excel.

---

## 7. The Modern MoE Landscape (Cai et al., 2024)

Cai et al. (2024) provide a comprehensive survey of MoE in large language models, organizing the field into a taxonomy of algorithm design, system design, and applications. Let us walk through the key dimensions.

### 7.1 Gating Function Taxonomy

The gating function has evolved from the softmax gating of Jacobs et al. (1991) into three categories:

**Sparse gating** activates a subset of experts per token. This is the dominant approach:
- **Token-choice gating**: The router selects the top-$k$ experts for each token. Used by Shazeer et al. (2017), GShard, Switch Transformer, Mixtral-8x7B, DeepSeek-V2, and most production MoE models.
- **Expert-choice gating** (Zhou et al., 2022): Each expert selects which tokens to process. This guarantees perfect load balance but may leave some tokens unprocessed.
- **Non-trainable gating**: Hash layers, random routing, or domain-based routing. No gating parameters to train — load balance comes for free.

**Dense gating** activates all experts for every token (like the original Jacobs et al.):
- Used recently in DS-MoE, EvoMoE, LoRAMoE. The computational overhead is higher but avoids routing errors.

**Soft gating** (Puigcerver et al., 2023): Merges tokens or expert parameters rather than making discrete routing decisions:
- **Token merging**: Computes weighted averages of tokens before sending to experts.
- **Expert merging** (SMEAR, Lory): Merges expert parameters through weighted averages, circumventing discrete decisions entirely.

### 7.2 Expert Architecture

The experts themselves have evolved:

**Network type.** In transformer-based LLMs, MoE layers replace the **feed-forward network** (FFN) within each Transformer block. This is because FFN layers exhibit lower sparsity and less domain specificity than self-attention layers — Pan et al. found only 20% active expert engagement in FFN layers versus 80% in attention layers. Recent work extends MoE to attention (MoA), to the full Transformer block, and even to every layer.

**Expert size.** Initial work used full-sized FFN experts ($d_{\text{expert}} = d_{\text{ffn}}$). DeepSeekMoE introduced **fine-grained expert segmentation**: reducing $d_{\text{expert}}$ to $\frac{1}{8}$ of the dense FFN dimension while increasing the expert count from 16 to 128 and active experts from top-2 to top-16. This provides finer decomposition of knowledge.

**Shared experts.** DeepSpeed-MoE introduced the **Residual-MoE** architecture: each token is processed by both a fixed shared expert (a standard dense FFN) and a gating-selected expert. The shared expert provides a baseline that all tokens benefit from, while the routed expert provides specialization. This design has been adopted by DeepSeekMoE, OpenMoE, Qwen1.5-MoE, and others.

### 7.3 The Auxiliary Loss Landscape

The load balancing loss has diversified from the original importance + load losses of Shazeer et al.:

| Loss | Formula | Used by |
|------|---------|---------|
| $\mathcal{L}_{\text{importance}} + \mathcal{L}_{\text{load}}$ | CV² of importance + smooth load estimator | Shazeer et al. (2017) |
| $\mathcal{L}_{\text{aux}}$ | $N \sum_i f_i \cdot P_i$ (dot product of dispatch fraction and probability) | GShard, Switch-T, Mixtral, DBRX, DeepSeek-V2 |
| $\mathcal{L}_{\text{aux}} + \mathcal{L}_z$ | Above + penalty on large router logits | ST-MoE, JetMoE |
| $\mathcal{L}_{MI}$ | Maximize mutual information between experts and tasks | Mod-Squad, ModuleFormer, DS-MoE |

The $\mathcal{L}_z$ loss, introduced by ST-MoE, penalizes large logits entering the gating network:

$$
\mathcal{L}_z = \frac{1}{T} \sum_{x \in B} \left(\log \sum_{i=1}^{N} e^{x_i}\right)^2
$$

This encourages smaller gating values, reducing roundoff errors in exponential functions and improving training stability at large scale.

### 7.4 Hyperparameter Choices in Practice

A summary of hyperparameter trends across production MoE models:

**Expert count.** Early work (Shazeer, 2017) used thousands of experts. Modern models converge on 8–64 experts, with 8 being the most common choice (Mixtral-8x7B, DBRX, Jamba, Qwen1.5-MoE).

**Active experts.** Most models use top-1 or top-2 routing. DeepSeekMoE uses top-6 out of 64 (fine-grained). LLAMA-MoE uses top-4 out of 16.

**MoE layer frequency.** Most models alternate: MoE layer every other Transformer block (1/2 frequency). Some use every fourth block (1/4), others every block (1/1). The trend is shifting toward every block with fewer experts per block.

**Activation function.** Evolved from ReLU (GShard) → GEGLU (Switch-T) → GeLU (DeepSpeed-MoE) → SwiGLU (Mixtral onward).

### 7.5 Training and Inference Schemes

The relationship between dense and sparse models has spawned new training paradigms:

**Dense-to-Sparse (Dense2Sparse).** Start with a pre-trained dense model and convert it to MoE by splitting the FFN into experts. Used by MoEfication, Sparse Upcycling, LLaMA-MoE. This leverages existing dense model checkpoints and is cheaper than training from scratch.

**Sparse-to-Dense (Sparse2Dense).** Start with a sparse MoE and merge/prune it into a dense model for efficient inference. Used by OneS, MoE-Pruning. This trades quality for inference speed.

**Expert merging.** Branch-Train-Merge: independently train expert models on different data domains, then merge them into a single MoE with a learned router.

### 7.6 System Design for MoE

MoE models introduce unique system challenges:

**Computation.** Expert parallelism distributes experts across devices. Each device holds a subset of experts and receives tokens routed to them via all-to-all communication. FastMoE, DeepSpeed-MoE, and Tutel provide optimized MoE training frameworks.

**Communication.** The all-to-all communication pattern (tokens dispatched to expert devices, results returned) is the primary bottleneck. Techniques include overlapping communication with computation, reducing communication volume through expert placement optimization, and hierarchical routing to minimize cross-node communication.

**Storage.** MoE models have many more parameters than dense models of similar FLOPs. Pre-gated MoE precomputes routing decisions and stores only the active expert weights for each batch. EdgeMoE and MPipeMoE optimize storage for edge deployment.

### 7.7 The Production MoE Models

A snapshot of notable production MoE models:

| Model | Year | Active/Total Params | Experts | Top-$k$ | MMLU |
|-------|------|-------------------|---------|---------|------|
| Mixtral-8x7B | 2023.12 | 13B/47B | 8 | 2 | 70.6 |
| DeepSeekMoE-16B | 2024.1 | 3B/16B | 64 | 6 | 45.0 |
| Grok-1 | 2024.3 | 86B/314B | 8 | 2 | 73.0 |
| DBRX Instruct | 2024.3 | 36B/132B | 16 | 4 | 73.7 |
| Arctic Instruct | 2024.4 | 17B/480B | 128 | 2 | 67.3 |
| DeepSeek-V2 | 2024.5 | 21B/236B | 160 | 6 | 78.5 |
| Qwen1.5-MoE-A2.7B | 2024.3 | 3B/14B | 64 | 4 | 62.5 |
| DeepSeek-V3 | 2024.12 | 37B/671B | 257 | 9 | 87.1 |

The trend: models are getting larger in total parameters (up to 671B) while keeping active parameters modest (37B), using fine-grained experts with shared-expert architectures.

---

## 8. The Unified Arc: From 1991 to 2024

Let us trace the complete evolution across all six papers:

**1991 — Jacobs et al.**: "What if we had multiple experts and a gating network?" The competitive error function decouples experts. The mixture-of-Gaussians interpretation connects to maximum likelihood. Dense gating, 4–8 experts, vowel recognition.

**1993 — Jordan & Jacobs**: "What if experts are arranged in a tree?" Hierarchical MoE with the EM algorithm. Convergence-guaranteed learning. 16 experts, robot dynamics. The connection to GLIM and exponential families.

**2017 — Shazeer et al.**: "What if we scale to thousands of experts with sparse gating?" Top-$k$ routing with noise, importance and load balancing losses, mixed data and model parallelism. 131,072 experts, 137B parameters, language modeling and translation.

**2021 — Fedus et al.**: "What if $k = 1$?" The Switch Transformer simplifies routing, improves performance, and scales to 1 trillion parameters. Selective precision, small initialization, expert dropout. 7x speedup over T5.

**2022 — Chen et al.**: "Why does this work?" Nonlinear experts provably specialize on data clusters. Noise provides smooth exploration. Normalized gradient descent prevents self-reinforcing imbalance. Linear experts fail because they cannot distinguish signal from noise.

**2024 — Cai et al.**: "Where do we stand?" A comprehensive taxonomy: sparse/dense/soft gating, FFN/attention/full-block experts, shared experts, fine-grained experts, Dense2Sparse and Sparse2Dense training, expert/model/data parallelism. Production models from Mixtral to DeepSeek-V3.

The mathematical core has remained remarkably stable across three decades. The output is still a weighted sum of expert outputs:

$$
y = \sum_{i \in \mathcal{T}} G(x)_i \cdot E_i(x)
$$

What changed is the size of $\mathcal{T}$ (from all experts to top-$k$ to top-1), the scale of $n$ (from 4 to hundreds of thousands), the training algorithm (from EM to end-to-end backpropagation with auxiliary losses), and the integration point (from standalone systems to FFN replacements within Transformer blocks). But the fundamental principle — divide a complex problem into simpler subproblems, learn which expert handles which subproblem, and combine their outputs — is the same principle Jacobs, Jordan, Nowlan, and Hinton proposed in 1991.

---

## Summary

Chen et al. (2022) showed that MoEs succeed because of three properties: (1) the cluster structure of real data provides natural subproblems for individual experts, (2) nonlinear experts can distinguish signal from noise within their assigned cluster while linear experts cannot, and (3) random initialization provides the symmetry-breaking that allows experts to specialize during an exploration phase. The router then learns to identify clusters through cluster-center features and dispatches data accordingly.

Cai et al. (2024) surveyed the entire MoE landscape in the era of LLMs, revealing a field that has matured from a research curiosity into a production architecture powering models from Mixtral to DeepSeek-V3. The design space has expanded along every axis — gating functions (sparse, dense, soft), expert architectures (FFN, attention, shared, fine-grained, parameter-efficient), training schemes (original, Dense2Sparse, Sparse2Dense, expert merging), and system designs (expert parallelism, communication optimization, storage efficiency) — while the mathematical foundations laid in 1991 continue to underpin it all.
