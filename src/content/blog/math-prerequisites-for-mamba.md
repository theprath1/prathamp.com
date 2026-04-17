---
title: "Mathematical Prerequisites for Mamba"
description: "Building the math foundations you need for Mamba — first-order linear ODEs, the integrating factor method, the matrix exponential, the integral identity used in zero-order hold, the ZOH discretization itself, the unrolled time-varying recurrence, and the 1-semiseparable matrix view — all derived step by step with one consistent leaky-integrator example."
date: 2026-04-08
tags: [machine-learning, mamba, state-space-models, mathematics, deep-learning]
order: 2
---

The next post in this series derives **Mamba** and **Mamba-2** from first principles. Both architectures start from a continuous-time differential equation, discretize it, and run the discretized version as a recurrence. To follow that derivation, we need seven mathematical tools that have not appeared in any earlier prerequisite post.

The most important is the **zero-order hold (ZOH)** discretization formula

$$
\bar{A} = e^{A\Delta}, \qquad \bar{B} = A^{-1}(e^{A\Delta} - I)\, B
$$

which converts the continuous SSM $h'(t) = Ah(t) + Bx(t)$ into the discrete recurrence $h_{n+1} = \bar{A} h_n + \bar{B} x_n$. The Mamba paper writes these formulas down in two lines, but every symbol in them earns its place through a chain of derivations: the integrating factor solves the linear ODE, the matrix exponential generalizes $e^{at}$ to vector-valued states, an integral identity collapses the convolution integral when the input is held constant, and the substitution rule glues the pieces together. We will build that chain end to end. We will also derive the **1-semiseparable matrix** view that turns the unrolled recurrence into a single matrix–vector product — the bridge that the second Mamba paper uses to expose the duality between SSMs and attention.

By the end of this post, you will have everything you need to read the boxed results in [Mamba and Mamba-2](/blog/attention-mamba) as derivations rather than as definitions.

---

## The Running Example

A single bucket with a hole in the bottom and a tap above it. Let $h(t)$ be the water level at time $t$. The hole drains the bucket at a rate proportional to how full it is, and the tap pours water in at a rate $x(t)$. With both rate constants set to 1, this gives:

$$
h'(t) = -h(t) + x(t), \qquad h(0) = 0
$$

We turn the tap on at $t = 0$ and leave it running at constant rate $x(t) \equiv 1$. By the end of Section 2 we will have derived the closed-form solution

$$
h(t) = 1 - e^{-t}
$$

and every numerical check in this post will land back on this curve. At $t = 1, 2, 3, 4$ this gives $0.632, 0.865, 0.950, 0.982$ — those four numbers are the ground truth we will reproduce from many different angles.

In the matrix-valued sections, $A$ becomes the $1 \times 1$ matrix $[-1]$, $B$ becomes $[1]$, and the scalar formulas drop out as the one-dimensional special case.

---

## 1. First-Order Linear Ordinary Differential Equation

A **differential equation** is an equation that relates a function to its own derivatives. The simplest interesting example is one that says "the rate of change of $h$ depends linearly on $h$ itself, plus an external driving term":

$$
h'(t) = a\, h(t) + b\, x(t)
$$

This is a **first-order linear ordinary differential equation** (ODE). "First-order" because only the first derivative appears. "Linear" because $h$ enters only to the first power, with no $h^2$ or $\sin h$. "Ordinary" because there is only one independent variable, $t$. The coefficient $a$ is the **decay rate**, the coefficient $b$ is the **input gain**, and $x(t)$ is the **forcing term**.

For our running example, $a = -1$, $b = 1$, and $x(t) = 1$. The equation becomes

$$
h'(t) = -h(t) + 1
$$

which says: if the bucket is empty, water rises at rate 1 (the tap dominates); if the bucket is full at $h = 1$, the inflow exactly cancels the outflow and the level stops changing. The fixed point is $h = 1$, and we expect $h(t)$ to climb from 0 toward 1.

We do not yet know what $h(t)$ looks like as a function. Section 2 derives the closed form.

---

## 2. The Integrating Factor Method

We want to solve

$$
h'(t) - a\, h(t) = b\, x(t)
$$

The left-hand side does not look like the derivative of anything obvious. The **integrating factor method** is a trick that fixes this: multiply both sides by a carefully chosen factor that turns the LHS into a single derivative.

The chosen factor is $e^{-at}$. Multiply through:

$$
e^{-at}\, h'(t) - a\, e^{-at}\, h(t) = e^{-at}\, b\, x(t)
$$

Now look at the LHS. By the **product rule**,

$$
\frac{d}{dt}\!\left[e^{-at}\, h(t)\right] = (-a)\, e^{-at}\, h(t) + e^{-at}\, h'(t) = e^{-at}\, h'(t) - a\, e^{-at}\, h(t)
$$

which is exactly the LHS. The equation has collapsed to

$$
\frac{d}{dt}\!\left[e^{-at}\, h(t)\right] = e^{-at}\, b\, x(t)
$$

Integrate both sides from $0$ to $t$. By the **fundamental theorem of calculus**, the LHS integrates to $e^{-at} h(t) - e^{0} h(0) = e^{-at} h(t) - h(0)$:

$$
e^{-at}\, h(t) - h(0) = \int_0^t e^{-as}\, b\, x(s)\, ds
$$

Multiply both sides by $e^{at}$:

$$
\boxed{\,h(t) = e^{at}\, h(0) + \int_0^t e^{a(t-s)}\, b\, x(s)\, ds\,}
$$

This is the **variation-of-constants formula** for a first-order linear ODE. The first term is the unforced response — what the bucket would do with the tap off, decaying from its starting level. The second term is the forced response — the cumulative effect of every drop that fell into the bucket, each one weighted by how much it has decayed by time $t$.

### Numerical check

Plug in $a = -1$, $b = 1$, $x(s) = 1$, $h(0) = 0$:

$$
h(t) = e^{-t}\cdot 0 + \int_0^t e^{-(t-s)}\cdot 1 \cdot 1\, ds = e^{-t}\int_0^t e^{s}\, ds = e^{-t}\,(e^t - 1) = 1 - e^{-t}
$$

At $t = 1$: $h(1) = 1 - e^{-1} \approx 1 - 0.368 = 0.632$. At $t = 4$: $h(4) = 1 - e^{-4} \approx 1 - 0.018 = 0.982$. These are the four ground-truth numbers we promised. ✓

### Why this matters

The boxed formula is the closed-form solution to *every* linear SSM with constant coefficients. In Mamba, $h$ is a vector and $a$ is a matrix $A$, but the structure of the formula is unchanged. We will need a way to make sense of "$e^{At}$" when $A$ is a matrix — that is the next section.

---

## 3. The Matrix Exponential

The scalar exponential $e^{at}$ is defined by the power series

$$
e^{at} = \sum_{k=0}^{\infty} \frac{(at)^k}{k!} = 1 + at + \frac{(at)^2}{2!} + \frac{(at)^3}{3!} + \cdots
$$

This power series uses only addition, multiplication, and scalar division — operations that are perfectly well-defined when $a$ is replaced by a matrix $A$. The **matrix exponential** is defined the same way:

$$
\boxed{\,e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots\,}
$$

Three properties that we will use repeatedly.

**Scalar special case.** When $A$ is the $1 \times 1$ matrix $[a]$, every power $A^k = [a^k]$, and the series reduces term-for-term to $e^{at}$. The matrix and scalar exponentials agree on $1 \times 1$ matrices.

**Diagonal case.** When $A = \mathrm{diag}(a_1, \ldots, a_n)$ is diagonal, every power $A^k = \mathrm{diag}(a_1^k, \ldots, a_n^k)$ is diagonal too, and the series gives

$$
e^{At} = \mathrm{diag}\!\left(e^{a_1 t},\, e^{a_2 t},\, \ldots,\, e^{a_n t}\right)
$$

This is why Mamba parameterizes $A$ as diagonal: every matrix exponential reduces to $n$ independent scalar exponentials, one per state channel.

**Derivative rule.** Differentiating the power series term by term,

$$
\frac{d}{dt} e^{At} = \frac{d}{dt}\!\left[I + At + \frac{A^2 t^2}{2!} + \frac{A^3 t^3}{3!} + \cdots\right] = A + A^2 t + \frac{A^3 t^2}{2!} + \cdots = A \cdot e^{At}
$$

So $\frac{d}{dt} e^{At} = A\, e^{At} = e^{At}\, A$. The matrix exponential commutes with $A$, just as $e^{at}$ commutes with $a$ in the scalar case (trivially).

### Numerical check

Take $A = [-1]$. The diagonal-case formula gives $e^{At} = [e^{-t}]$. At $t = 1$, $e^{-1} \approx 0.368$. The derivative rule gives $\frac{d}{dt} e^{-t} = -e^{-t}$, which equals $A \cdot e^{At} = (-1)(0.368) = -0.368$ at $t = 1$. ✓

### Why this matters

The integrating factor derivation in Section 2 used three facts about $e^{-at}$: the product rule, the fact that $\frac{d}{dt}e^{-at} = -a\, e^{-at}$, and the fact that we can multiply by it on both sides. All three carry over verbatim to the matrix case because of the derivative rule above. So the boxed solution from Section 2 generalizes:

$$
h(t) = e^{At}\, h(0) + \int_0^t e^{A(t-s)}\, B\, x(s)\, ds
$$

with the same derivation, $a$ swapped for $A$, $b$ swapped for $B$, and $h$ now vector-valued. We will use this in Section 5.

---

## 4. The Integral Identity for Matrix Exponentials

When the input $x(s)$ is held constant over the integration window, we can pull it out of the integral and the remaining piece is

$$
\int_0^{\Delta} e^{A u}\, du
$$

We claim this equals $A^{-1}(e^{A\Delta} - I)$ whenever $A$ is invertible.

**Derivation by term-by-term integration.** Substitute the power series and integrate each term:

$$
\int_0^{\Delta} e^{A u}\, du = \int_0^{\Delta} \sum_{k=0}^{\infty} \frac{A^k u^k}{k!}\, du = \sum_{k=0}^{\infty} \frac{A^k}{k!} \int_0^{\Delta} u^k\, du = \sum_{k=0}^{\infty} \frac{A^k \Delta^{k+1}}{(k+1)!}
$$

Pull a factor of $A^{-1}$ out of every term (note that $A^k = A^{-1} A^{k+1}$, so the first term — where $k=0$ — becomes $A^{-1} A \Delta = A^{-1}(A\Delta)^1 / 1!$, exactly what we want):

$$
= A^{-1} \sum_{k=0}^{\infty} \frac{(A\Delta)^{k+1}}{(k+1)!} = A^{-1} \sum_{j=1}^{\infty} \frac{(A\Delta)^j}{j!}
$$

where we relabeled $j = k + 1$. The sum starting from $j = 1$ is missing only the $j = 0$ term, which is $I$. So it equals $e^{A\Delta} - I$. Putting it together:

$$
\boxed{\,\int_0^{\Delta} e^{A u}\, du = A^{-1}\!\left(e^{A\Delta} - I\right)\,}
$$

### Numerical check

Take $A = [-1]$, $\Delta = 1$.

LHS: $\int_0^{1} e^{-u}\, du = -e^{-u}\Big|_0^1 = -(e^{-1} - 1) = 1 - e^{-1} \approx 0.632$.

RHS: $A^{-1}(e^{A\Delta} - I) = (-1)^{-1}(e^{-1} - 1) = -(0.368 - 1) = 0.632$. ✓

### Why this matters

This is the single algebraic identity that turns the convolution integral in Section 3's solution into a closed-form coefficient. The next section uses it.

---

## 5. Zero-Order Hold Discretization

A continuous SSM evolves at every instant, but a neural network only sees the input at discrete sample times $t_n = n\Delta$. We need a rule that converts the continuous equation into a discrete recurrence

$$
h_{n+1} = \bar{A}\, h_n + \bar{B}\, x_n
$$

that produces the *exact* trajectory at the sample points, given some assumption about what the input does between samples.

**Zero-order hold** is the simplest possible assumption: between consecutive samples, the input is held constant at the value it had at the start of the interval. So $x(t_n + s) = x_n$ for all $s \in [0, \Delta)$. This is the staircase reconstruction: each new sample sets a new constant level until the next sample arrives.

Apply the matrix solution from Section 3 over the interval $[t_n, t_n + \Delta]$, with starting state $h_n = h(t_n)$:

$$
h_{n+1} = e^{A\Delta}\, h_n + \int_0^{\Delta} e^{A(\Delta - s)}\, B\, x(t_n + s)\, ds
$$

Under ZOH, $x(t_n + s) = x_n$ is constant in $s$, so $B\, x_n$ comes out of the integral:

$$
h_{n+1} = e^{A\Delta}\, h_n + \left(\int_0^{\Delta} e^{A(\Delta - s)}\, ds\right) B\, x_n
$$

Substitute $u = \Delta - s$ in the integral. When $s = 0$, $u = \Delta$; when $s = \Delta$, $u = 0$; and $du = -ds$:

$$
\int_0^{\Delta} e^{A(\Delta - s)}\, ds = \int_{\Delta}^{0} e^{A u}\, (-du) = \int_0^{\Delta} e^{A u}\, du
$$

By Section 4, this equals $A^{-1}(e^{A\Delta} - I)$. Substituting back:

$$
h_{n+1} = e^{A\Delta}\, h_n + A^{-1}\!\left(e^{A\Delta} - I\right) B\, x_n
$$

Reading off the coefficients of $h_n$ and $x_n$ gives the discrete parameters:

$$
\boxed{\,\bar{A} = e^{A\Delta}, \qquad \bar{B} = A^{-1}\!\left(e^{A\Delta} - I\right) B, \qquad h_{n+1} = \bar{A}\, h_n + \bar{B}\, x_n\,}
$$

### Numerical check

With $A = [-1]$, $B = [1]$, $\Delta = 1$:

$$
\bar{A} = e^{-1} \approx 0.368, \qquad \bar{B} = (-1)^{-1}(e^{-1} - 1)\cdot 1 = -(0.368 - 1) = 0.632
$$

The discrete recurrence is $h_{n+1} = 0.368\, h_n + 0.632\, x_n$. With $h_0 = 0$ and $x_n = 1$ for all $n$:

$$
h_1 = 0.632, \quad h_2 = 0.368(0.632) + 0.632 = 0.864, \quad h_3 = 0.368(0.864) + 0.632 = 0.950, \quad h_4 = 0.368(0.950) + 0.632 = 0.982
$$

The continuous solution at $t = 1, 2, 3, 4$ was $0.632, 0.865, 0.950, 0.982$. The discrete values match the continuous values at every sample point, off only in the third decimal due to rounding $e^{-1}$ to $0.368$. ✓

### A useful identity in the scalar case

When $A = -1$ and $B = 1$, the formula simplifies further. Compute $\bar{A} + \bar{B} = e^{-\Delta} + (1 - e^{-\Delta}) = 1$. So $\bar{A}$ and $\bar{B}$ sum to 1, and the update

$$
h_{n+1} = \bar{A}\, h_n + (1 - \bar{A})\, x_n
$$

is a **convex combination** of the previous state and the current input. This is exactly the form of an exponential moving average — the same form a classical RNN gate uses. The Mamba post (Section 1.5) leans on this identity to argue that selective SSMs are equivalent to gated RNNs in the scalar case.

### Why this matters

These two boxed formulas are the connection between continuous control theory and the discrete recurrence that runs on a GPU. Every selective SSM in Mamba is exactly this recurrence, with $A$, $B$, and $\Delta$ made to depend on the input.

---

## 6. Unrolling a Time-Varying Recurrence

In Section 5, $\bar{A}$ and $\bar{B}$ are constants. In Mamba, the **selection mechanism** makes $\Delta_n$, $B_n$, and $C_n$ depend on the input $x_n$, which in turn makes $\bar{A}_n = e^{\bar A_n \Delta_n}$ and $\bar{B}_n$ vary across time steps. The recurrence becomes

$$
h_n = \bar{A}_n\, h_{n-1} + \bar{B}_n\, x_n
$$

To express $h_n$ as a closed-form function of the inputs $x_1, \ldots, x_n$, we unroll the recurrence step by step. Starting from $h_0 = 0$:

$$
h_1 = \bar{A}_1\, h_0 + \bar{B}_1\, x_1 = \bar{B}_1\, x_1
$$

$$
h_2 = \bar{A}_2\, h_1 + \bar{B}_2\, x_2 = \bar{A}_2 \bar{B}_1\, x_1 + \bar{B}_2\, x_2
$$

$$
h_3 = \bar{A}_3\, h_2 + \bar{B}_3\, x_3 = \bar{A}_3 \bar{A}_2 \bar{B}_1\, x_1 + \bar{A}_3 \bar{B}_2\, x_2 + \bar{B}_3\, x_3
$$

The pattern is now visible: each input $x_j$ contributes to $h_n$ multiplied by $\bar{B}_j$ (the gain at the time the input entered) and by the **cumulative product** of every $\bar{A}_k$ that came after, capturing how much that contribution has decayed by time $n$:

$$
\boxed{\,h_n = \sum_{j=1}^{n} \left(\prod_{k=j+1}^{n} \bar{A}_k\right) \bar{B}_j\, x_j\,}
$$

with the empty-product convention $\prod_{k=n+1}^{n} \bar{A}_k = 1$ (so the $j = n$ term is just $\bar{B}_n\, x_n$, with no decay applied).

### Numerical check

Apply the formula with the running example, where $\bar{A}_k = 0.368$ and $\bar{B}_j = 0.632$ for all $k, j$ (the constant-coefficient case), $x_j = 1$, $n = 4$:

$$
h_4 = (0.368)^3 (0.632) + (0.368)^2 (0.632) + (0.368)(0.632) + (0.632)
$$

$$
= 0.0315 + 0.0856 + 0.2326 + 0.6320 = 0.9817 \approx 0.982 \checkmark
$$

The unrolled sum reproduces the step-by-step recurrence value from Section 5.

### Why this matters

This formula is the matrix-attention bridge. The next section reads it as a single matrix–vector product and exposes its structure.

---

## 7. The 1-Semiseparable Matrix View

Stack the unrolled values $h_1, h_2, \ldots, h_T$ into a column vector $\mathbf{h}$, and the inputs $x_1, \ldots, x_T$ into a column vector $\mathbf{x}$. The formula from Section 6 says $\mathbf{h} = M \mathbf{x}$ with

$$
M_{nj} = \begin{cases} \left(\prod_{k=j+1}^{n} \bar{A}_k\right) \bar{B}_j & \text{if } j \le n \\ 0 & \text{if } j > n \end{cases}
$$

The matrix $M$ is lower-triangular by construction, because $h_n$ depends only on inputs at times $j \le n$ (causality). Factoring out the $\bar{B}_j$ values:

$$
M = L \cdot \mathrm{diag}(\bar{B}_1, \bar{B}_2, \ldots, \bar{B}_T), \qquad L_{nj} = \prod_{k=j+1}^{n} \bar{A}_k \text{ for } j \le n
$$

The matrix $L$ is the heart of the structure. A lower-triangular matrix is called **1-semiseparable** (1-SS) if every entry in its lower triangle factors as a product

$$
L_{nj} = c_n\, b_j \qquad \text{for } j \le n
$$

for some sequences $(c_n)$ and $(b_j)$. Equivalently, every submatrix contained entirely in the lower triangle has rank at most 1. The cumulative-product form above is exactly this structure: define $c_n = \prod_{k=1}^{n} \bar{A}_k$ and $b_j = 1/\prod_{k=1}^{j} \bar{A}_k$, and then

$$
c_n\, b_j = \frac{\prod_{k=1}^{n} \bar{A}_k}{\prod_{k=1}^{j} \bar{A}_k} = \prod_{k=j+1}^{n} \bar{A}_k = L_{nj}
$$

So the unrolled SSM recurrence *is* multiplication by a 1-semiseparable matrix. We have shown:

$$
\boxed{\,\mathbf{h} = L \cdot \mathrm{diag}(\bar{B})\, \mathbf{x}, \qquad L_{nj} = \prod_{k=j+1}^{n} \bar{A}_k\,}
$$

### Numerical check

Take the running example with constant $\bar{A}_k = 0.368$ and $\bar{B}_j = 0.632$, and $T = 4$. The matrix $L$ has $L_{nn} = 1$ on the diagonal (empty product), and below the diagonal the entries are powers of $0.368$:

$$
L = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0.368 & 1 & 0 & 0 \\
0.135 & 0.368 & 1 & 0 \\
0.050 & 0.135 & 0.368 & 1
\end{pmatrix}
$$

(Using $0.368^2 = 0.135$, $0.368^3 = 0.050$.) Multiplying by $\mathrm{diag}(0.632, 0.632, 0.632, 0.632)$ scales every column by $0.632$, giving $M = 0.632\, L$. Then $\mathbf{h} = M \mathbf{x}$ with $\mathbf{x} = (1, 1, 1, 1)^\top$:

$$
h_1 = 0.632 \cdot 1 = 0.632
$$

$$
h_2 = 0.632 \cdot (0.368 + 1) = 0.632 \cdot 1.368 = 0.864
$$

$$
h_3 = 0.632 \cdot (0.135 + 0.368 + 1) = 0.632 \cdot 1.503 = 0.950
$$

$$
h_4 = 0.632 \cdot (0.050 + 0.135 + 0.368 + 1) = 0.632 \cdot 1.553 = 0.981 \approx 0.982
$$

Same four numbers as Section 5 and Section 6. ✓ Three different computations — recurrent, unrolled-sum, and matrix-multiply — all produce the same trajectory, because they are three algorithms for evaluating the same function.

### Why this matters

This is the key insight of the second Mamba paper. The recurrent algorithm and the matrix-multiply algorithm compute the same output, so we can choose between them based on hardware. Recurrent is $O(T)$ time and $O(1)$ memory but inherently sequential. Matrix-multiply is $O(T^2)$ time but uses the GPU's matrix units. The **structured state space duality** (SSD) splits the sequence into chunks and uses matrix-multiply *within* each chunk, then connects chunks with the recurrence — getting the best of both. Every step of that algorithm starts from the boxed identity above.

---

## Summary

All seven tools were built from a single bucket with a hole, governed by $h'(t) = -h(t) + x(t)$ with the tap held at $x(t) \equiv 1$.

The **first-order linear ODE** $h'(t) = a h(t) + b x(t)$ is the continuous-time skeleton of every state space model. The **integrating factor method** — multiply by $e^{-at}$, recognize the LHS as $\frac{d}{dt}[e^{-at} h(t)]$, integrate, solve — gave the closed-form solution $h(t) = e^{at} h(0) + \int_0^t e^{a(t-s)} b x(s) ds$, verified at the four sample points $0.632, 0.865, 0.950, 0.982$. The **matrix exponential** $e^{At} = \sum_k (At)^k / k!$ generalizes $e^{at}$ verbatim, reduces to a per-channel scalar exponential when $A$ is diagonal (which is why Mamba parameterizes $A$ as diagonal), and satisfies the same derivative rule, which lets the integrating factor derivation extend to the matrix case unchanged. The **integral identity** $\int_0^\Delta e^{Au} du = A^{-1}(e^{A\Delta} - I)$ — derived by term-by-term integration of the power series — collapses the convolution integral whenever the input is held constant. **Zero-order hold discretization** combines the matrix solution with the integral identity: holding $x$ constant on $[t_n, t_n+\Delta]$ pulls $B x_n$ out of the integral and applies the identity to what remains, yielding the boxed pair $\bar{A} = e^{A\Delta}$, $\bar{B} = A^{-1}(e^{A\Delta} - I) B$. **Unrolling the time-varying recurrence** $h_n = \bar{A}_n h_{n-1} + \bar{B}_n x_n$ produced the closed form $h_n = \sum_{j=1}^n (\prod_{k=j+1}^n \bar{A}_k) \bar{B}_j x_j$, where each input is weighted by a cumulative product of subsequent decay factors. Finally, the **1-semiseparable matrix** view recognized this sum as $\mathbf{h} = L \cdot \mathrm{diag}(\bar{B}) \mathbf{x}$ with $L_{nj} = \prod_{k=j+1}^n \bar{A}_k$ — a structured lower-triangular matrix whose every lower-triangular submatrix has rank at most 1. The recurrence and the matrix-multiply are two algorithms for the same function, which is the foundation of structured state space duality.

With these tools in hand, we are ready for [Mamba and Mamba-2: Selective State Spaces and Structured State Space Duality](/blog/attention-mamba), where the boxed ZOH formulas are taken as the starting point, the selection mechanism makes $\Delta$, $B$, and $C$ input-dependent, and the 1-semiseparable mask becomes the structured replacement for the causal attention mask.

---

*Previous: [The Kernel Zoo: Performers, Fast Weight Programmers, and the Capacity-Approximation Tradeoff in Linear Attention](/blog/attention-linear-attention)*
*Next: [Mamba and Mamba-2: Selective State Spaces and Structured State Space Duality](/blog/attention-mamba)*
