---
title: "Mathematical Prerequisites for Mixture of Experts — Part 3"
description: "Building the math foundations you need for understanding why MoEs work — orthogonality, vector norms, asymptotic notation, Lipschitz continuity, and dispatch entropy — all derived step by step with one consistent example."
date: 2026-03-25
tags: [machine-learning, mixture-of-experts, mathematics, probability, theory]
order: 2
draft: false
---

In [Part 3](/blog/mixture-of-experts-part-3) of the Mixture of Experts series, we examine why experts specialize instead of collapsing, what role nonlinearity plays, and how training unfolds in three stages. The mathematics is different from Parts 1 and 2 — instead of Gaussian likelihoods and load balancing losses, we need tools for measuring vector alignment, bounding how functions change, reading theorem statements that describe asymptotic behaviour, and quantifying how sharply a router dispatches tokens. By the end of this post, you will have every mathematical tool required to follow Part 3 from the first theorem to the last.

We assume you have read the prerequisites for [Part 1](/blog/math-prerequisites-for-mixture-of-experts) (where we built softmax, Gaussian densities, Bayes' theorem, and the mixture log-likelihood) and [Part 2](/blog/math-prerequisites-for-mixture-of-experts-part-2) (where we built top-k masking, the coefficient of variation, indicator functions, and the dot-product loss). We also assume you have seen the definition of entropy from the [Foundation Prior prerequisites](/blog/math-prerequisites-for-foundation-prior). We will not re-derive any of those here. Instead, we build five new tools — each one earning its place by being directly used in Part 3.

---

## The Running Example

We have 2 clusters of data in $\mathbb{R}^2$. Each cluster has a **signal direction** — a vector that tells us "this is what cluster $k$ looks like." The two signal vectors are:

$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

We have two experts with weight vectors:

$$
\mathbf{w}_A = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad \mathbf{w}_B = \begin{bmatrix} 1 \\ 4 \end{bmatrix}
$$

The question we will answer throughout this post: does expert A "prefer" cluster 1 or cluster 2? Does expert B? How do we measure preference, distance, sensitivity, and routing sharpness? Each section builds one tool for answering these questions, and each tool will be used in Part 3.

---

## 1. Orthogonality

In the [Part 2 prerequisites](/blog/math-prerequisites-for-mixture-of-experts-part-2), we defined the dot product $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$ and showed that it measures alignment between two vectors. Here we need a stronger concept: what happens when two vectors have **zero** alignment?

### 1.1 The inner product as alignment

The **inner product** (another name for the dot product) of two vectors $\mathbf{u}$ and $\mathbf{v}$ in $\mathbb{R}^d$ is:

$$
\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^{d} u_i v_i
$$

The angle bracket notation $\langle \cdot, \cdot \rangle$ is used interchangeably with the dot notation $\mathbf{u} \cdot \mathbf{v}$ — they mean the same thing. Part 3 uses the angle bracket notation throughout, so we adopt it here.

When the inner product is large and positive, the vectors point in similar directions. When it is large and negative, they point in opposite directions. When it is zero, the vectors are completely unrelated — neither one tells you anything about the other. This last case has a name.

### 1.2 Definition of orthogonality

Two vectors $\mathbf{u}$ and $\mathbf{v}$ are **orthogonal** if their inner product is zero:

$$
\boxed{\mathbf{u} \perp \mathbf{v} \quad \iff \quad \langle \mathbf{u}, \mathbf{v} \rangle = 0}
$$

The symbol $\perp$ means "is orthogonal to." In two dimensions, orthogonal vectors are perpendicular — they meet at a right angle. In higher dimensions, the geometric intuition is the same: orthogonal vectors have no component along each other's direction.

### Numerical check

Let us verify that our signal vectors are orthogonal:

$$
\langle \mathbf{v}_1, \mathbf{v}_2 \rangle = 1 \times 0 + 0 \times 1 = 0
$$

The inner product is exactly zero, so $\mathbf{v}_1 \perp \mathbf{v}_2$. The two cluster signals are completely independent — a data point's component along $\mathbf{v}_1$ tells you nothing about its component along $\mathbf{v}_2$.

### 1.3 Why orthogonality matters for MoE

In Part 3, Chen et al. construct data from $K$ clusters where all signal vectors $\{\mathbf{v}_k\}$ and all cluster-center vectors $\{\mathbf{c}_k\}$ are orthogonal to each other. This is the key structural assumption that makes the theory work.

Why? Because orthogonal signals do not interfere. If an expert learns to respond to cluster 1's signal $\mathbf{v}_1$, that learning contributes nothing — positive or negative — to its response to cluster 2's signal $\mathbf{v}_2$. The expert's inner product with $\mathbf{v}_1$ can grow without affecting its inner product with $\mathbf{v}_2$. This clean separation is what allows different experts to specialise on different clusters without competing.

### 1.4 Expert alignment with clusters

Now let us compute which cluster each expert aligns with. Expert A's alignment with each signal:

$$
\langle \mathbf{w}_A, \mathbf{v}_1 \rangle = 3 \times 1 + 1 \times 0 = 3
$$

$$
\langle \mathbf{w}_A, \mathbf{v}_2 \rangle = 3 \times 0 + 1 \times 1 = 1
$$

Expert A has inner product 3 with cluster 1 and inner product 1 with cluster 2. It aligns more strongly with cluster 1.

Expert B's alignment:

$$
\langle \mathbf{w}_B, \mathbf{v}_1 \rangle = 1 \times 1 + 4 \times 0 = 1
$$

$$
\langle \mathbf{w}_B, \mathbf{v}_2 \rangle = 1 \times 0 + 4 \times 1 = 4
$$

Expert B has inner product 1 with cluster 1 and inner product 4 with cluster 2. It aligns more strongly with cluster 2.

This is precisely the mechanism from Part 3's Lemma 5.2: each expert is assigned to the cluster whose signal vector has the largest inner product with the expert's weights. Using the argmax notation from the [Part 2 prerequisites](/blog/math-prerequisites-for-mixture-of-experts-part-2):

$$
\text{Expert A} \in \mathcal{M}_k \text{ where } k = \arg\max_{k'} \langle \mathbf{v}_{k'}, \mathbf{w}_A \rangle = \arg\max(3, 1) = 1
$$

$$
\text{Expert B} \in \mathcal{M}_k \text{ where } k = \arg\max_{k'} \langle \mathbf{v}_{k'}, \mathbf{w}_B \rangle = \arg\max(1, 4) = 2
$$

Expert A specialises on cluster 1, expert B on cluster 2. The random initialisation of $\mathbf{w}_A$ and $\mathbf{w}_B$ determined this assignment — different initial weights would have produced a different grouping. This is the symmetry-breaking mechanism from Part 3: all experts start from the same initialisation distribution, but their specific random draws determine which cluster they align with.

---

## 2. Vector Norms

In Part 3, three different ways of measuring vector and matrix size appear: the $\ell_2$ norm (for normalising gradients), the $\ell_\infty$ norm (for bounding routing changes), and the Frobenius norm (for normalising gradient matrices). We derive all three.

### 2.1 The $\ell_2$ norm

The **$\ell_2$ norm** (also called the **Euclidean norm**) of a vector is its length — the distance from the origin to the point the vector represents:

$$
\boxed{\|\mathbf{w}\|_2 = \sqrt{\sum_{i=1}^{d} w_i^2}}
$$

This is the Pythagorean theorem generalised to $d$ dimensions. In two dimensions, $\|\mathbf{w}\|_2 = \sqrt{w_1^2 + w_2^2}$, which is the hypotenuse of a right triangle with legs $w_1$ and $w_2$.

### Numerical check

$$
\|\mathbf{w}_A\|_2 = \sqrt{3^2 + 1^2} = \sqrt{9 + 1} = \sqrt{10} \approx 3.162
$$

$$
\|\mathbf{w}_B\|_2 = \sqrt{1^2 + 4^2} = \sqrt{1 + 16} = \sqrt{17} \approx 4.123
$$

Expert B's weight vector is longer than expert A's. The $\ell_2$ norm tells us the overall magnitude of the weights, regardless of direction.

For the signal vectors:

$$
\|\mathbf{v}_1\|_2 = \sqrt{1^2 + 0^2} = 1, \quad \|\mathbf{v}_2\|_2 = \sqrt{0^2 + 1^2} = 1
$$

Both signal vectors have norm 1. A vector with $\ell_2$ norm equal to 1 is called a **unit vector**. Unit vectors encode pure direction with no magnitude. In Part 3, the signal vectors are unit vectors — they represent the direction of each cluster's signal, with a separate scalar ($\alpha$ or $\beta$) controlling the magnitude.

### 2.2 Unit vectors and normalisation

Given any nonzero vector $\mathbf{w}$, we can create a unit vector pointing in the same direction by dividing by the norm:

$$
\boxed{\hat{\mathbf{w}} = \frac{\mathbf{w}}{\|\mathbf{w}\|_2}}
$$

This operation is called **normalisation**. The hat notation $\hat{\mathbf{w}}$ denotes "the unit vector in the direction of $\mathbf{w}$."

### Numerical check

$$
\hat{\mathbf{w}}_A = \frac{1}{\sqrt{10}} \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.949 \\ 0.316 \end{bmatrix}
$$

Verify the norm: $\sqrt{0.949^2 + 0.316^2} = \sqrt{0.900 + 0.100} = \sqrt{1.000} = 1$. The normalised vector has length 1 but points in the same direction as $\mathbf{w}_A$.

In Part 3, normalised gradient descent divides the gradient by its norm before updating the weights. This is the same operation: it keeps the direction of the gradient but sets its magnitude to 1, ensuring all experts update at the same speed.

### 2.3 The $\ell_\infty$ norm

The **$\ell_\infty$ norm** (also called the **max norm** or **supremum norm**) of a vector is the largest absolute value among its entries:

$$
\boxed{\|\mathbf{w}\|_\infty = \max_{i} |w_i|}
$$

Where the $\ell_2$ norm aggregates all entries (via squaring and summing), the $\ell_\infty$ norm cares only about the single largest entry. It answers: what is the worst-case component?

### Numerical check

$$
\|\mathbf{w}_A\|_\infty = \max(|3|, |1|) = 3
$$

$$
\|\mathbf{w}_B\|_\infty = \max(|1|, |4|) = 4
$$

Compare with the $\ell_2$ norms: $\|\mathbf{w}_A\|_2 = 3.162$ vs. $\|\mathbf{w}_A\|_\infty = 3$. The $\ell_\infty$ norm is always less than or equal to the $\ell_2$ norm (the max of the absolute values cannot exceed the root-sum-of-squares). This relationship between norms is called a **norm equivalence** — different norms give different numbers, but they are always within a bounded ratio of each other.

The $\ell_\infty$ norm is used in Part 3's Lemma 5.1 (the Lipschitz bound on routing probabilities) because it measures the worst-case change across any single expert's routing probability: $\|\mathbf{p} - \hat{\mathbf{p}}\|_\infty$ is the largest change in routing probability for any single expert.

### 2.4 The Frobenius norm

The **Frobenius norm** extends the $\ell_2$ norm from vectors to matrices. For a matrix $\mathbf{M}$ with entries $M_{ij}$:

$$
\boxed{\|\mathbf{M}\|_F = \sqrt{\sum_{i} \sum_{j} M_{ij}^2}}
$$

The idea is identical to the $\ell_2$ norm: square every entry, sum them all, and take the square root. The only difference is that the entries are arranged in a grid (matrix) rather than a line (vector). If you "unroll" the matrix into a single long vector by stacking its columns, the Frobenius norm equals the $\ell_2$ norm of that vector.

### Numerical check

Suppose during training, the gradient of the loss with respect to expert A's weights is:

$$
\nabla_{\mathbf{W}_A} \mathcal{L} = \begin{bmatrix} 0.6 & -0.2 \\ 0.3 & 0.4 \end{bmatrix}
$$

The Frobenius norm is:

$$
\|\nabla_{\mathbf{W}_A} \mathcal{L}\|_F = \sqrt{0.6^2 + (-0.2)^2 + 0.3^2 + 0.4^2} = \sqrt{0.36 + 0.04 + 0.09 + 0.16} = \sqrt{0.65} \approx 0.806
$$

In Part 3's normalised gradient descent, the update rule divides the gradient by this norm:

$$
\mathbf{W}_A^{(t+1)} = \mathbf{W}_A^{(t)} - \eta \cdot \frac{\nabla_{\mathbf{W}_A} \mathcal{L}}{\|\nabla_{\mathbf{W}_A} \mathcal{L}\|_F}
$$

The normalised gradient is:

$$
\frac{1}{0.806} \begin{bmatrix} 0.6 & -0.2 \\ 0.3 & 0.4 \end{bmatrix} = \begin{bmatrix} 0.744 & -0.248 \\ 0.372 & 0.496 \end{bmatrix}
$$

Verify: $\sqrt{0.744^2 + 0.248^2 + 0.372^2 + 0.496^2} = \sqrt{0.554 + 0.062 + 0.138 + 0.246} = \sqrt{1.000} = 1$. The normalised gradient has Frobenius norm 1 — exactly the matrix analogue of a unit vector. Every expert takes a step of the same size regardless of how many data points contributed to its gradient. This is **normalised gradient descent**, the third key technique in Part 3.

---

## 3. Asymptotic Notation

Part 3 states its theorems using **asymptotic notation** — symbols that describe how quantities grow as the problem gets large, without committing to exact constants. Theorem 4.2 alone contains $\Theta(\cdot)$, $\Omega(\cdot)$, and $o(\cdot)$. We define each one.

### 3.1 Why we need this

Consider the statement from Theorem 4.2: "With $M = \Theta(K \log K \log \log d)$ experts... the test error is $o(1)$." Without understanding the notation, this is unreadable. With it, the statement becomes precise: the number of experts must grow proportionally to $K \log K \log \log d$, and the test error vanishes as the dimension grows. Every symbol has a specific meaning.

We will use the following concrete function to illustrate all four symbols:

$$
f(n) = 3n^2 + 5n + 2
$$

This is a function of $n$ (think of $n$ as the problem dimension or dataset size). As $n$ grows, we want to characterise how $f$ grows without worrying about the exact coefficients.

### 3.2 Big-$O$: upper bound on growth

We write $f(n) = O(g(n))$ and say "$f$ is **big-O** of $g$" if $f$ grows **at most as fast** as $g$, up to a constant factor. Formally:

$$
\boxed{f(n) = O(g(n)) \quad \iff \quad \text{there exist constants } C > 0 \text{ and } n_0 \text{ such that } f(n) \leq C \cdot g(n) \text{ for all } n \geq n_0}
$$

The constant $C$ absorbs the leading coefficient and all lower-order terms. The threshold $n_0$ means we only care about large $n$ — the bound does not need to hold for tiny values.

### Numerical check

We claim $f(n) = 3n^2 + 5n + 2 = O(n^2)$. To verify, we need to find $C$ and $n_0$ such that $3n^2 + 5n + 2 \leq C \cdot n^2$ for all $n \geq n_0$.

For $n \geq 1$: $5n \leq 5n^2$ and $2 \leq 2n^2$, so $3n^2 + 5n + 2 \leq 3n^2 + 5n^2 + 2n^2 = 10n^2$. This gives $C = 10$ and $n_0 = 1$.

Let us verify at a specific value. At $n = 3$: $f(3) = 3(9) + 5(3) + 2 = 27 + 15 + 2 = 44$. And $10 \cdot 3^2 = 90$. Indeed $44 \leq 90$.

At $n = 100$: $f(100) = 30{,}000 + 500 + 2 = 30{,}502$. And $10 \cdot 100^2 = 100{,}000$. Indeed $30{,}502 \leq 100{,}000$.

So $3n^2 + 5n + 2 = O(n^2)$. The big-$O$ says: "this function grows like $n^2$ or slower, ignoring constant factors."

### 3.3 Big-$\Omega$: lower bound on growth

We write $f(n) = \Omega(g(n))$ and say "$f$ is **big-Omega** of $g$" if $f$ grows **at least as fast** as $g$:

$$
\boxed{f(n) = \Omega(g(n)) \quad \iff \quad \text{there exist constants } c > 0 \text{ and } n_0 \text{ such that } f(n) \geq c \cdot g(n) \text{ for all } n \geq n_0}
$$

Big-$\Omega$ is the mirror image of big-$O$: it provides a floor rather than a ceiling.

### Numerical check

We claim $f(n) = 3n^2 + 5n + 2 = \Omega(n^2)$. Since $5n \geq 0$ and $2 \geq 0$ for all $n \geq 0$, we have $3n^2 + 5n + 2 \geq 3n^2$. This gives $c = 3$ and $n_0 = 0$.

At $n = 100$: $f(100) = 30{,}502 \geq 3 \times 10{,}000 = 30{,}000$. The lower bound holds.

In Part 3, the statement that a single expert has error $\Omega(1/K)$ on other clusters means: no matter how the expert is trained, its error on clusters it has not specialised on is at least proportional to $1/K$. The error cannot be made arbitrarily small — it has a floor.

### 3.4 Big-$\Theta$: tight bound

We write $f(n) = \Theta(g(n))$ and say "$f$ is **big-Theta** of $g$" if $f$ grows **at exactly the same rate** as $g$:

$$
\boxed{f(n) = \Theta(g(n)) \quad \iff \quad f(n) = O(g(n)) \text{ and } f(n) = \Omega(g(n))}
$$

Big-$\Theta$ combines both bounds: $f$ is sandwiched between $c \cdot g(n)$ and $C \cdot g(n)$ for large $n$. It is the tightest characterisation.

### Numerical check

We showed that $f(n) = O(n^2)$ with $C = 10$ and $f(n) = \Omega(n^2)$ with $c = 3$. Therefore $f(n) = \Theta(n^2)$. For all large $n$:

$$
3n^2 \leq 3n^2 + 5n + 2 \leq 10n^2
$$

The function grows exactly like $n^2$: the $5n$ and $2$ terms become negligible relative to $3n^2$.

In Part 3, "$M = \Theta(K \log K \log \log d)$" means the number of experts must scale proportionally to $K \log K \log \log d$ — not much more, not much less. Fewer experts and the proof fails; more are unnecessary.

### 3.5 Little-$o$: strictly slower growth

We write $f(n) = o(g(n))$ and say "$f$ is **little-o** of $g$" if $f$ grows **strictly slower** than $g$:

$$
\boxed{f(n) = o(g(n)) \quad \iff \quad \lim_{n \to \infty} \frac{f(n)}{g(n)} = 0}
$$

Little-$o$ is stronger than big-$O$. Big-$O$ allows $f$ to grow at the same rate as $g$ (the ratio can approach a nonzero constant). Little-$o$ requires the ratio to approach zero — $f$ becomes negligible compared to $g$.

### Numerical check

Consider $g(n) = n^2$ and $h(n) = n$. We claim $h(n) = o(g(n))$, meaning $n = o(n^2)$:

$$
\lim_{n \to \infty} \frac{n}{n^2} = \lim_{n \to \infty} \frac{1}{n} = 0
$$

The limit is zero, confirming $n = o(n^2)$. The linear function becomes negligible compared to the quadratic.

But $3n^2 + 5n + 2$ is **not** $o(n^2)$:

$$
\lim_{n \to \infty} \frac{3n^2 + 5n + 2}{n^2} = \lim_{n \to \infty} \left(3 + \frac{5}{n} + \frac{2}{n^2}\right) = 3
$$

The limit is 3, not 0. So $f(n) = O(n^2)$ but $f(n) \neq o(n^2)$.

The most important use in Part 3 is the statement that the test error is $o(1)$. Since $o(1)$ means "strictly slower than the constant function 1":

$$
\lim_{d \to \infty} \frac{\text{test error}}{1} = \lim_{d \to \infty} \text{test error} = 0
$$

That is literally it: $o(1)$ means "approaches zero." The test error vanishes as the problem dimension $d$ grows. This is how Part 3 encodes "nearly zero test error" in mathematical notation.

---

## 4. Lipschitz Continuity

Part 3's Technique 1 (stability by smoothing) rests on a property of the noisy router: small changes in gating outputs cause only small changes in routing probabilities. This property has a name.

### 4.1 Motivation

Imagine the gating network produces scores $\mathbf{h} = [2.0, 1.0]$ for two experts, and these are converted to routing probabilities $\mathbf{p}$ via softmax (with noise). Now suppose we perturb the gating scores slightly to $\hat{\mathbf{h}} = [2.1, 1.0]$. We changed the input by a small amount. The question is: **how much can the output change?**

If the output can change by an arbitrarily large amount in response to a tiny input change, the system is unstable — training would be chaotic. If the output change is bounded by a multiple of the input change, the system is stable. This is the idea behind **Lipschitz continuity**.

### 4.2 Definition

A function $f: \mathbb{R}^d \to \mathbb{R}^m$ is **Lipschitz continuous** with constant $L$ if:

$$
\boxed{\|f(\mathbf{x}) - f(\mathbf{y})\| \leq L \cdot \|\mathbf{x} - \mathbf{y}\| \quad \text{for all } \mathbf{x}, \mathbf{y}}
$$

The constant $L$ is called the **Lipschitz constant**. It bounds the ratio of output change to input change. A function with a small Lipschitz constant changes slowly; a function with a large Lipschitz constant can change quickly — but never faster than $L$ times the input change.

The norms can be any norms — $\ell_2$, $\ell_\infty$, or others. The choice of norm affects the value of $L$ but not the concept. In Part 3, the $\ell_\infty$ norm is used on both sides.

### 4.3 A simple example

Consider the scalar function $f(x) = 2x$. For any two inputs $x$ and $y$:

$$
|f(x) - f(y)| = |2x - 2y| = 2|x - y|
$$

So $f$ is Lipschitz with constant $L = 2$. The output always changes by exactly twice the input change.

### Numerical check

Take $x = 3$ and $y = 3.1$:

$$
|f(3) - f(3.1)| = |6 - 6.2| = 0.2
$$

$$
L \cdot |x - y| = 2 \times |3 - 3.1| = 2 \times 0.1 = 0.2
$$

The bound is tight: $0.2 \leq 0.2$.

### 4.4 A non-Lipschitz example

Consider $g(x) = x^2$. For inputs $x$ and $y$:

$$
|g(x) - g(y)| = |x^2 - y^2| = |x + y| \cdot |x - y|
$$

The factor $|x + y|$ grows without bound as $x$ and $y$ increase. There is no fixed constant $L$ that works for all $x$ and $y$ — we would need $L \geq |x + y|$ for every pair, which is impossible with a single constant. So $g(x) = x^2$ is **not** globally Lipschitz.

### Numerical check

Take $x = 100$ and $y = 100.1$:

$$
|g(100) - g(100.1)| = |10{,}000 - 10{,}020.01| = 20.01
$$

$$
|x - y| = 0.1
$$

The ratio is $20.01 / 0.1 = 200.1$. Now take $x = 1000$ and $y = 1000.1$:

$$
|g(1000) - g(1000.1)| = |1{,}000{,}000 - 1{,}000{,}200.01| = 200.01
$$

The ratio is $200.01 / 0.1 = 2000.1$. The ratio keeps growing — no fixed $L$ can bound it. The function $x^2$ amplifies small perturbations more and more as $x$ increases.

### 4.5 The Lipschitz bound in Part 3

Part 3's Lemma 5.1 states that the noisy routing function satisfies:

$$
\|\mathbf{p} - \hat{\mathbf{p}}\|_\infty \leq M^2 \|\mathbf{h} - \hat{\mathbf{h}}\|_\infty
$$

This is a Lipschitz bound with constant $L = M^2$, using the $\ell_\infty$ norm on both sides. Let us unpack what it says using our running example.

Suppose we have $M = 2$ experts. The Lipschitz constant is $M^2 = 4$. Now suppose the gating outputs change by:

$$
\|\mathbf{h} - \hat{\mathbf{h}}\|_\infty = 0.05
$$

meaning no single gating score changes by more than $0.05$. Then the routing probabilities change by at most:

$$
\|\mathbf{p} - \hat{\mathbf{p}}\|_\infty \leq 4 \times 0.05 = 0.20
$$

No single expert's routing probability changes by more than $0.20$. This is the stability guarantee: small perturbations in the gating network produce bounded changes in routing. Without noise, routing would be determined by argmax, which can switch discontinuously from one expert to another — a tiny change in gating scores could cause a complete routing change (from probability 1 to probability 0). The noise smooths this out, making the routing function Lipschitz.

### Numerical check

Suppose the original gating outputs are $\mathbf{h} = [2.0, 1.0]$ and the perturbed outputs are $\hat{\mathbf{h}} = [2.05, 1.0]$. The input change is:

$$
\|\mathbf{h} - \hat{\mathbf{h}}\|_\infty = \max(|2.0 - 2.05|, |1.0 - 1.0|) = \max(0.05, 0) = 0.05
$$

If the actual routing probabilities change from $\mathbf{p} = [0.731, 0.269]$ to $\hat{\mathbf{p}} = [0.738, 0.262]$, the output change is:

$$
\|\mathbf{p} - \hat{\mathbf{p}}\|_\infty = \max(|0.731 - 0.738|, |0.269 - 0.262|) = \max(0.007, 0.007) = 0.007
$$

Check the bound: $0.007 \leq 4 \times 0.05 = 0.20$. The bound holds with room to spare — the actual change ($0.007$) is much smaller than the worst case ($0.20$). The bound is conservative, but what matters is that it exists: it guarantees that routing can never change dramatically in response to a small gating perturbation.

---

## 5. Dispatch Entropy

Part 3 uses **dispatch entropy** as the primary metric for measuring how sharply the router dispatches tokens to experts. Low dispatch entropy means each token goes to essentially one expert (sharp routing); high dispatch entropy means tokens are spread across many experts (diffuse routing). We build this from the entropy definition in the [Foundation Prior prerequisites](/blog/math-prerequisites-for-foundation-prior).

### 5.1 Entropy recap

Entropy measures how uncertain a probability distribution is. For a discrete distribution $\mathbf{p} = [p_1, p_2, \ldots, p_M]$ over $M$ outcomes:

$$
H(\mathbf{p}) = -\sum_{i=1}^{M} p_i \log p_i
$$

where we use the convention $0 \log 0 = 0$ (the limit of $p \log p$ as $p \to 0$ is $0$). Entropy is always non-negative: $H(\mathbf{p}) \geq 0$.

Two extreme cases define the range:
- **Minimum entropy:** $H = 0$ when one probability is 1 and the rest are 0. The outcome is certain.
- **Maximum entropy:** $H = \log M$ when all probabilities are equal ($p_i = 1/M$ for all $i$). The outcome is maximally uncertain.

### 5.2 Dispatch entropy for a single token

When the router produces routing probabilities $\mathbf{p}(x)$ for a token $x$, the **dispatch entropy** for that token is:

$$
\boxed{H(\mathbf{p}(x)) = -\sum_{i=1}^{M} p_i(x) \log p_i(x)}
$$

This measures how concentrated the routing decision is. If the router is confident — sending the token almost entirely to one expert — the dispatch entropy is near zero. If the router is uncertain — spreading the token across all experts — the dispatch entropy is near $\log M$.

### Numerical check

Suppose the router produces probabilities $\mathbf{p} = [0.9, 0.1]$ for a token routed between $M = 2$ experts:

$$
H = -(0.9 \ln 0.9 + 0.1 \ln 0.1)
$$

Computing each term:

$$
0.9 \ln 0.9 = 0.9 \times (-0.105) = -0.095
$$

$$
0.1 \ln 0.1 = 0.1 \times (-2.303) = -0.230
$$

$$
H = -(-0.095 + (-0.230)) = -(-0.325) = 0.325
$$

Now suppose the router is completely uncertain, $\mathbf{p} = [0.5, 0.5]$:

$$
H = -(0.5 \ln 0.5 + 0.5 \ln 0.5) = -(2 \times 0.5 \times (-0.693)) = -(-0.693) = 0.693
$$

And $\log 2 = 0.693$. The uniform distribution achieves maximum entropy, as expected.

For a perfectly sharp routing, $\mathbf{p} = [1.0, 0.0]$:

$$
H = -(1.0 \ln 1.0 + 0 \ln 0) = -(0 + 0) = 0
$$

The three cases in order: $H = 0$ (sharp) $< 0.325$ (mostly one expert) $< 0.693$ (uniform). Entropy increases as routing becomes more diffuse.

### 5.3 Average dispatch entropy

Part 3 reports a single dispatch entropy number for the entire model, not for individual tokens. This is the average dispatch entropy across all tokens in the test set:

$$
\boxed{\bar{H} = \frac{1}{T} \sum_{x \in \mathcal{B}} H(\mathbf{p}(x))}
$$

This averages the per-token dispatch entropy over all $T$ tokens in the batch $\mathcal{B}$.

### Numerical check

Suppose we have $T = 4$ tokens with routing probabilities:

| Token | $\mathbf{p}$ | $H$ |
|-------|-------------|-----|
| 1 | $[0.95, 0.05]$ | $-(0.95 \ln 0.95 + 0.05 \ln 0.05) = 0.199$ |
| 2 | $[0.10, 0.90]$ | $-(0.10 \ln 0.10 + 0.90 \ln 0.90) = 0.325$ |
| 3 | $[0.99, 0.01]$ | $-(0.99 \ln 0.99 + 0.01 \ln 0.01) = 0.056$ |
| 4 | $[0.05, 0.95]$ | $-(0.05 \ln 0.05 + 0.95 \ln 0.95) = 0.199$ |

Let us verify token 1 explicitly: $0.95 \ln 0.95 = 0.95 \times (-0.051) = -0.049$ and $0.05 \ln 0.05 = 0.05 \times (-2.996) = -0.150$. So $H = -(-0.049 + (-0.150)) = 0.199$.

The average dispatch entropy:

$$
\bar{H} = \frac{0.199 + 0.325 + 0.056 + 0.199}{4} = \frac{0.779}{4} = 0.195
$$

### 5.4 Interpreting dispatch entropy in Part 3

In Part 3, the experimental results show:

| Model | Dispatch Entropy |
|-------|-----------------|
| MoE (linear) | 1.300 |
| MoE (nonlinear) | 0.098 |

With $M = 4$ experts, the maximum possible entropy is $\log 4 = 1.386$ (using natural log). The linear MoE's dispatch entropy (1.300) is close to the maximum — the router is nearly uniform, spreading tokens across all experts with little discrimination. The nonlinear MoE's dispatch entropy (0.098) is close to zero — the router sends each token to essentially one expert.

This is the quantitative signature of expert specialisation. An entropy of 0.098 means the routing distribution is extremely sharp — on average, the router is nearly certain about which expert should process each token. An entropy of 1.300 means the router has barely learned to distinguish between experts. The difference between 0.098 and 1.300 is the difference between a specialised MoE and a glorified ensemble.

---

## Summary

We have built five tools for Part 3. Orthogonality ($\langle \mathbf{u}, \mathbf{v} \rangle = 0$) ensures that cluster signals do not interfere, allowing experts to specialise on one cluster without degrading performance on another — the structural assumption underlying Chen et al.'s data model. Three vector norms measure size in different ways: the $\ell_2$ norm gives overall length and enables normalisation to unit vectors, the $\ell_\infty$ norm gives worst-case component magnitude, and the Frobenius norm extends $\ell_2$ to matrices — all three appear in the normalised gradient descent technique and the Lipschitz stability bound. Asymptotic notation ($O$, $\Omega$, $\Theta$, $o$) lets us read theorem statements that describe how quantities scale: $\Theta$ for tight bounds, $\Omega$ for lower bounds, and $o(1)$ for "vanishes as the problem grows." Lipschitz continuity bounds how much a function's output can change relative to its input, and the $M^2$ Lipschitz constant of the noisy router is what makes training stable — small gating perturbations cannot cause catastrophic routing changes. Dispatch entropy measures routing sharpness on a scale from 0 (deterministic) to $\log M$ (uniform), and the near-zero dispatch entropy of nonlinear MoEs is the quantitative proof that experts have truly specialised.

With these tools in hand, we are ready for [Part 3](/blog/mixture-of-experts-part-3), where we examine why experts specialise, why nonlinearity is essential, and how the three training stages — exploration, router learning, and generalisation — produce a working MoE from random initialisation.

---

*Previous: [Attention Residuals: Replacing Fixed Skip Connections with Learned Depth-Wise Attention](/blog/attention-residuals)*  
*Next: [Mixture of Experts from Scratch — Part 3](/blog/mixture-of-experts-part-3)*
