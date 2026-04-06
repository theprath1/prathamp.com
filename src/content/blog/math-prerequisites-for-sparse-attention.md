---
title: "Mathematical Prerequisites for Sparse and Sliding Window Attention"
description: "Building the discrete math foundations for sparse attention patterns — the triangular number formula for counting pairs, modular arithmetic for strided patterns, the pigeonhole principle for existence proofs, and set union with inclusion-exclusion for combining connectivity sets — all derived step by step with one consistent 8-house example."
date: 2026-04-06
tags: [machine-learning, mathematics, attention, discrete-math, combinatorics]
order: 3
---

The next two posts in this series derive sparse factorized attention (the Sparse Transformer) and sliding window attention (Longformer). Both papers replace the dense $n \times n$ attention pattern with structured sparse patterns and then prove that these patterns preserve the model's ability to route information between any two positions.

To follow those derivations, we need four tools from discrete mathematics. The **triangular number formula** counts the total entries in a causal attention matrix — it appears every time we need the sum $1 + 2 + 3 + \cdots + n$. **Modular arithmetic** defines which positions are "aligned" at a given stride — it is the language of strided attention patterns. The **pigeonhole principle** proves that two-hop paths always exist in factorized attention — it turns a counting argument into an existence guarantee. And **set union with its cardinality rule** combines local windows with global tokens — it is how Longformer merges two attention patterns without double-counting.

We will build all four tools from a single running example and verify every formula numerically.

---

## The Running Example

We use 8 houses on a street, numbered 0 through 7, arranged in a row. The houses are grouped into 2 blocks of 4:

$$
\underbrace{0 \quad 1 \quad 2 \quad 3}_{\text{Block 0}} \qquad \underbrace{4 \quad 5 \quad 6 \quad 7}_{\text{Block 1}}
$$

We will use this setup to build every tool: counting connections between houses, determining which block a house belongs to, proving that a mail carrier can always find a relay, and combining neighborhoods.

---

## 1. The Triangular Number Formula

### 1.1 Motivation

Suppose every house wants to send a greeting card to every house that comes before it on the street (lower-numbered houses only). House 0 sends nothing. House 1 sends to house 0. House 2 sends to houses 0 and 1. And so on. How many greeting cards are sent in total?

This is exactly the question that arises in causal attention: position $i$ attends to all positions $j \leq i$, and we want the total number of attention entries.

### 1.2 Counting directly

Let $C(i)$ be the number of cards house $i$ sends. Since house $i$ sends to all houses $0, 1, \ldots, i - 1$ plus itself (the self-connection in attention), we have $C(i) = i + 1$:

| House $i$ | Recipients | $C(i) = i + 1$ |
|---|---|---|
| 0 | $\{0\}$ | 1 |
| 1 | $\{0, 1\}$ | 2 |
| 2 | $\{0, 1, 2\}$ | 3 |
| 3 | $\{0, 1, 2, 3\}$ | 4 |
| 4 | $\{0, 1, 2, 3, 4\}$ | 5 |
| 5 | $\{0, 1, 2, 3, 4, 5\}$ | 6 |
| 6 | $\{0, 1, 2, 3, 4, 5, 6\}$ | 7 |
| 7 | $\{0, 1, 2, 3, 4, 5, 6, 7\}$ | 8 |

The total is:

$$
T = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
$$

But we want a formula that works for any $n$, not just $n = 8$. We want a closed form for:

$$
T_n = \sum_{i=0}^{n-1} (i + 1) = \sum_{k=1}^{n} k = 1 + 2 + 3 + \cdots + n
$$

### 1.3 Derivation: Gauss's pairing trick

Write the sum forwards and backwards and add them:

$$
\begin{aligned}
T_n &= 1 + 2 + 3 + \cdots + (n-1) + n \\
T_n &= n + (n-1) + (n-2) + \cdots + 2 + 1
\end{aligned}
$$

Add the two rows term by term. Each column sums to $n + 1$:

$$
2 T_n = \underbrace{(1 + n)}_{= n+1} + \underbrace{(2 + (n-1))}_{= n+1} + \underbrace{(3 + (n-2))}_{= n+1} + \cdots + \underbrace{(n + 1)}_{= n+1}
$$

There are $n$ such pairs, so:

$$
2 T_n = n(n + 1)
$$

Divide both sides by 2:

$$
\boxed{T_n = \sum_{k=1}^{n} k = \frac{n(n+1)}{2}}
$$

This is the **triangular number formula**, also called **Gauss's summation formula**. The name "triangular" comes from the fact that $T_n$ counts the number of dots in a triangle with $n$ rows.

### 1.4 Numerical check

For $n = 8$:

$$
T_8 = \frac{8 \times 9}{2} = \frac{72}{2} = 36
$$

We computed $1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36$ directly. Both sides match. $\checkmark$

### 1.5 A second useful form

In the attention blogs, we often encounter the sum starting from 0:

$$
\sum_{i=0}^{n-1}(i + 1) = \frac{n(n+1)}{2}
$$

This is identical to $T_n$ — we are just re-indexing. Substituting $k = i + 1$, the sum runs from $k = 1$ to $k = n$, which is exactly $T_n$. The re-indexing changes nothing.

### 1.6 Interpretation

The triangular number formula converts a sum that takes $n$ additions into a single multiplication and division. In the sparse attention blogs, this formula appears every time we count total entries in a causal attention matrix. For a sequence of length $n$, full causal attention has $T_n = \frac{n(n+1)}{2} \approx \frac{n^2}{2}$ entries — the formula makes the quadratic scaling explicit.

---

## 2. Modular Arithmetic

### 2.1 Motivation

In the sparse attention blog, strided attention connects position $i$ to every position $j$ such that $i$ and $j$ are separated by a multiple of the stride $l$. The formal definition says: "$j$ is in position $i$'s strided set if $(i - j) \bmod l = 0$." To understand this, we need the $\bmod$ operator.

### 2.2 Definition

The **modulo operator** $a \bmod m$ returns the remainder when $a$ is divided by $m$. Formally, for integers $a$ and $m > 0$:

$$
a \bmod m = a - m \left\lfloor \frac{a}{m} \right\rfloor
$$

where $\lfloor \cdot \rfloor$ is the **floor function** (the largest integer $\leq$ the argument), which we derived in [Mathematical Prerequisites for Mixture of Experts — Part 2](/blog/math-prerequisites-for-mixture-of-experts-part-2).

The result is always in the set $\{0, 1, 2, \ldots, m - 1\}$.

### 2.3 Concrete examples with our houses

With $m = 4$ (block size):

| House $i$ | $\lfloor i / 4 \rfloor$ (block number) | $i \bmod 4$ (position within block) |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 0 | 3 |
| 4 | 1 | 0 |
| 5 | 1 | 1 |
| 6 | 1 | 2 |
| 7 | 1 | 3 |

The floor gives the block number. The mod gives the position within the block. Together they partition the house number into two pieces:

$$
i = \underbrace{\left\lfloor \frac{i}{m} \right\rfloor}_{\text{block}} \times m + \underbrace{i \bmod m}_{\text{offset}}
$$

This is the **Euclidean division identity**: every integer $i$ can be uniquely written as $qm + r$ where $q = \lfloor i/m \rfloor$ is the quotient and $r = i \bmod m$ is the remainder, with $0 \leq r < m$.

### 2.4 Numerical check

For house 7 with $m = 4$:

$$
7 = \underbrace{1}_{\lfloor 7/4 \rfloor} \times 4 + \underbrace{3}_{7 \bmod 4}
$$

Check: $1 \times 4 + 3 = 7$. $\checkmark$

### 2.5 Residue classes

A **residue class** modulo $m$ is the set of all integers with the same remainder when divided by $m$. With $m = 4$, there are exactly 4 residue classes:

$$
\begin{aligned}
\text{Class 0}: &\quad \{0, 4\} \quad (i \bmod 4 = 0) \\
\text{Class 1}: &\quad \{1, 5\} \quad (i \bmod 4 = 1) \\
\text{Class 2}: &\quad \{2, 6\} \quad (i \bmod 4 = 2) \\
\text{Class 3}: &\quad \{3, 7\} \quad (i \bmod 4 = 3)
\end{aligned}
$$

In the 8-house street, each residue class contains exactly $8 / 4 = 2$ houses. These are exactly the houses that are connected by strided attention with stride $l = 4$: house 0 and house 4 are in the same residue class, so strided attention connects them. House 3 and house 7 are in the same residue class, so strided attention connects them.

### 2.6 Congruence

Two integers $a$ and $b$ are **congruent modulo** $m$, written $a \equiv b \pmod{m}$, if they have the same remainder:

$$
a \equiv b \pmod{m} \quad \iff \quad a \bmod m = b \bmod m \quad \iff \quad m \mid (a - b)
$$

The notation $m \mid (a - b)$ means "$m$ divides $(a - b)$" — that is, $(a - b)$ is a multiple of $m$, which is equivalent to saying $(a - b) \bmod m = 0$.

### 2.7 Numerical check

Are houses 3 and 7 congruent modulo 4?

$$
7 - 3 = 4, \qquad 4 \bmod 4 = 0
$$

Yes: $7 \equiv 3 \pmod{4}$. And indeed, $7 \bmod 4 = 3$ and $3 \bmod 4 = 3$ — same remainder. $\checkmark$

Are houses 2 and 5 congruent modulo 4?

$$
5 - 2 = 3, \qquad 3 \bmod 4 = 3 \neq 0
$$

No: $5 \not\equiv 2 \pmod{4}$. And indeed, $5 \bmod 4 = 1$ while $2 \bmod 4 = 2$ — different remainders. $\checkmark$

### 2.8 The key property for strided attention

The strided connectivity set in the sparse attention blog is defined as:

$$
A_i^{(2)} = \{j : j \leq i \text{ and } (i - j) \bmod l = 0\}
$$

Using what we just derived, $(i - j) \bmod l = 0$ means $l \mid (i - j)$, which means $i \equiv j \pmod{l}$. So the strided set connects position $i$ to all earlier positions in the same residue class modulo $l$. The residue classes are exactly the columns of the image grid when $l$ equals the row width — which is why strided attention naturally attends along columns.

### 2.9 Spacing of residue classes

Within any residue class modulo $m$, consecutive elements are exactly $m$ apart. The class with remainder $r$ contains the elements $r, r + m, r + 2m, r + 3m, \ldots$ This is an **arithmetic sequence** with common difference $m$.

In our example with $m = 4$: class 2 contains $\{2, 6, 10, 14, \ldots\}$, with consecutive elements spaced exactly 4 apart.

This spacing property will be critical in the next section, where we prove that a relay position always exists within any interval of length $m$.

---

## 3. The Pigeonhole Principle

### 3.1 Motivation

The sparse attention blog proves that any two positions can communicate through a two-hop path. The proof relies on showing that an intermediate relay position must exist. The tool that guarantees existence is the pigeonhole principle.

### 3.2 Definition

The **pigeonhole principle** (also called the **Dirichlet box principle**) states:

$$
\boxed{\text{If } n \text{ items are placed into } m \text{ containers, and } n > m, \text{ then at least one container holds } \geq 2 \text{ items.}}
$$

The name comes from the physical setup: if 9 pigeons fly into 8 pigeonholes, at least one hole contains at least 2 pigeons. There is no way around it — the arithmetic forces a collision.

### 3.3 Proof

Suppose, for contradiction, that every container holds at most 1 item. Then the total number of items is at most $1 \times m = m$. But we have $n > m$ items. Contradiction. Therefore at least one container must hold $\geq 2$ items.

This is a proof by contradiction — we assumed the opposite of what we wanted to show and derived a false statement ($n \leq m$ when we know $n > m$), which means our assumption was wrong. $\square$

### 3.4 Concrete example

Suppose 5 packages must be delivered to the 4 houses in Block 0 (houses 0, 1, 2, 3). By the pigeonhole principle, at least one house receives $\geq 2$ packages.

Verify: the only way to distribute 5 packages among 4 houses is to give each house at least 1 and have 1 left over. That leftover must go to some house, giving it 2. $\checkmark$

### 3.5 The generalized pigeonhole principle

A stronger version gives a tighter bound:

$$
\boxed{\text{If } n \text{ items are placed into } m \text{ containers, then at least one container holds } \geq \left\lceil \frac{n}{m} \right\rceil \text{ items.}}
$$

Here $\lceil \cdot \rceil$ is the **ceiling function**: $\lceil x \rceil$ is the smallest integer $\geq x$. For example, $\lceil 5/4 \rceil = \lceil 1.25 \rceil = 2$.

### 3.6 Proof of the generalized version

Suppose every container holds at most $\lceil n/m \rceil - 1$ items. Then the total is at most $m \cdot (\lceil n/m \rceil - 1)$. We need to show this is less than $n$.

Since $\lceil n/m \rceil \leq n/m + 1$ (the ceiling exceeds the value by less than 1), we have:

$$
m \cdot (\lceil n/m \rceil - 1) \leq m \cdot (n/m + 1 - 1) = m \cdot (n/m) = n
$$

But this bound is not strict enough. We need the sharper fact: $\lceil n/m \rceil - 1 < n/m$, which gives:

$$
m \cdot (\lceil n/m \rceil - 1) < m \cdot \frac{n}{m} = n
$$

So the total is strictly less than $n$. But we have $n$ items. Contradiction. $\square$

### 3.7 Numerical check

With $n = 5$ packages and $m = 4$ houses:

$$
\left\lceil \frac{5}{4} \right\rceil = \lceil 1.25 \rceil = 2
$$

So at least one house gets $\geq 2$ packages. This matches our direct reasoning above. $\checkmark$

### 3.8 Application to strided attention

The sparse attention blog uses the pigeonhole principle in the following form: "In any interval of length $l$, there is at least one representative from each residue class modulo $l$."

Let us derive this. Consider an interval of $l$ consecutive integers: $[a, a + l - 1] = \{a, a+1, a+2, \ldots, a+l-1\}$. This interval contains exactly $l$ integers. When we compute each of these modulo $l$, we get $l$ remainders, each in $\{0, 1, \ldots, l-1\}$.

We claim these $l$ remainders are all distinct. The proof: take any two integers $a + s$ and $a + t$ from the interval with $0 \leq s < t \leq l-1$. Their difference is $t - s$, which satisfies $1 \leq t - s \leq l - 1$. Since $t - s$ is strictly between 0 and $l$, it is not divisible by $l$. By the definition of congruence, $a + s \not\equiv a + t \pmod{l}$, so they have different remainders.

Since $l$ consecutive integers produce $l$ distinct remainders out of $l$ possible values $\{0, 1, \ldots, l-1\}$, every residue class is represented exactly once.

### 3.9 Numerical check

Take the interval $[2, 5] = \{2, 3, 4, 5\}$ (length $l = 4$). Compute each modulo 4:

$$
2 \bmod 4 = 2, \quad 3 \bmod 4 = 3, \quad 4 \bmod 4 = 0, \quad 5 \bmod 4 = 1
$$

Remainders: $\{0, 1, 2, 3\}$ — all four residue classes are represented. $\checkmark$

Take the interval $[5, 8] = \{5, 6, 7, 8\}$:

$$
5 \bmod 4 = 1, \quad 6 \bmod 4 = 2, \quad 7 \bmod 4 = 3, \quad 8 \bmod 4 = 0
$$

Again: $\{0, 1, 2, 3\}$ — all four residue classes represented. $\checkmark$

### 3.10 Interpretation

This is the key result that makes the path-length argument work in the sparse attention blog. The strided attention head connects positions in the same residue class modulo $l$. The local attention head covers $l$ consecutive positions. Since any $l$ consecutive positions contain one representative from every residue class, there is always a relay point where the two heads overlap. The pigeonhole principle turns a counting fact into an existence guarantee: we do not need to search for the relay — the principle tells us it must be there.

---

## 4. Set Operations and Cardinality

### 4.1 Motivation

Both the sparse attention and sliding window blogs define attention patterns as sets and then combine them. The sliding window blog combines a local window set with a global token set. The sparse attention blog takes the union of two factorized patterns. To count the total entries correctly — especially when sets overlap — we need set operations and the inclusion-exclusion principle.

### 4.2 Basic definitions

A **set** is an unordered collection of distinct elements. We write sets with curly braces. For our running example:

- The neighborhood of house 3 with radius 2: $N_3 = \{1, 2, 3, 4, 5\}$
- The set of houses in Block 0: $B_0 = \{0, 1, 2, 3\}$
- The set of all houses: $U = \{0, 1, 2, 3, 4, 5, 6, 7\}$

The **cardinality** of a set $A$, written $|A|$, is the number of elements it contains:

$$
|N_3| = 5, \qquad |B_0| = 4, \qquad |U| = 8
$$

### 4.3 Union

The **union** of two sets $A$ and $B$, written $A \cup B$, is the set of elements that belong to $A$ or $B$ (or both):

$$
A \cup B = \{x : x \in A \text{ or } x \in B\}
$$

### 4.4 Intersection

The **intersection** of two sets $A$ and $B$, written $A \cap B$, is the set of elements that belong to both $A$ and $B$:

$$
A \cap B = \{x : x \in A \text{ and } x \in B\}
$$

### 4.5 Concrete example

Let $N_3 = \{1, 2, 3, 4, 5\}$ (neighborhood of house 3) and $G = \{0, 7\}$ (two "global" houses). Then:

$$
N_3 \cup G = \{0, 1, 2, 3, 4, 5, 7\}
$$

$$
N_3 \cap G = \emptyset
$$

The union has 7 elements. The intersection is empty because houses 0 and 7 are not in the neighborhood of house 3. Therefore the union's cardinality is simply the sum of the individual cardinalities: $|N_3 \cup G| = |N_3| + |G| = 5 + 2 = 7$.

Now consider $N_5 = \{3, 4, 5, 6, 7\}$ (neighborhood of house 5) and $G = \{0, 7\}$:

$$
N_5 \cup G = \{0, 3, 4, 5, 6, 7\}
$$

$$
N_5 \cap G = \{7\}
$$

The union has 6 elements, not $5 + 2 = 7$. The discrepancy is because house 7 appears in both sets. We counted it twice when we added the cardinalities and must subtract it once.

### 4.6 The inclusion-exclusion principle

The **inclusion-exclusion principle** gives the correct cardinality of a union:

$$
\boxed{|A \cup B| = |A| + |B| - |A \cap B|}
$$

In words: add the sizes of both sets, then subtract the overlap to avoid double-counting.

### 4.7 Derivation

Every element of $A \cup B$ falls into exactly one of three categories:

1. In $A$ only (not in $B$): there are $|A| - |A \cap B|$ such elements
2. In $B$ only (not in $A$): there are $|B| - |A \cap B|$ such elements
3. In both $A$ and $B$: there are $|A \cap B|$ such elements

The total is:

$$
|A \cup B| = (|A| - |A \cap B|) + (|B| - |A \cap B|) + |A \cap B|
$$

Simplify by collecting the $|A \cap B|$ terms. The first two groups contribute $-|A \cap B|$ each, the third contributes $+|A \cap B|$:

$$
= |A| + |B| - |A \cap B| - |A \cap B| + |A \cap B| = |A| + |B| - |A \cap B|
$$

This completes the derivation.

### 4.8 Numerical check

For $N_5 = \{3, 4, 5, 6, 7\}$ and $G = \{0, 7\}$:

$$
|N_5 \cup G| = |N_5| + |G| - |N_5 \cap G| = 5 + 2 - 1 = 6
$$

We computed $N_5 \cup G = \{0, 3, 4, 5, 6, 7\}$ directly, which has 6 elements. $\checkmark$

For $N_3 = \{1, 2, 3, 4, 5\}$ and $G = \{0, 7\}$:

$$
|N_3 \cup G| = |N_3| + |G| - |N_3 \cap G| = 5 + 2 - 0 = 7
$$

We computed $N_3 \cup G = \{0, 1, 2, 3, 4, 5, 7\}$ directly, which has 7 elements. $\checkmark$

### 4.9 Application to Longformer

In the sliding window attention blog, each local token's connectivity set is the union of its local window and the set of global tokens:

$$
S_i = W_i \cup G
$$

where $W_i$ is the local window and $G$ is the set of global tokens. By inclusion-exclusion:

$$
|S_i| = |W_i| + |G| - |W_i \cap G|
$$

When a global token happens to fall inside the local window, $|W_i \cap G|$ is nonzero, and the union is smaller than the naive sum $|W_i| + |G|$. This is exactly the overlap that the sliding window blog accounts for when tracing the connectivity table: "Note that when a global token is already in the local window, the union does not add a new entry."

### 4.10 Interpretation

The inclusion-exclusion principle is a bookkeeping tool: it ensures we count each position exactly once even when two connectivity patterns overlap. Without it, we would overcount the total number of attention entries by counting shared positions twice. In practice, the overlap between local windows and global tokens is small (global tokens are few relative to the window size), so the correction is minor — but the principle is what makes the counting rigorous.

---

## Summary

All four tools were built from the same 8 houses on a street.

The **triangular number formula** $T_n = \frac{n(n+1)}{2}$ — derived by Gauss's pairing trick — counts the sum $1 + 2 + \cdots + n$ in closed form. In the attention blogs, this formula counts the total entries in a causal attention matrix: $T_n \approx \frac{n^2}{2}$, making the quadratic scaling of full attention explicit. **Modular arithmetic** — the $\bmod$ operator, residue classes, and congruence — partitions positions into groups that are separated by a fixed stride $l$. In strided attention, positions in the same residue class modulo $l$ are connected: $(i - j) \bmod l = 0$ means $i \equiv j \pmod{l}$, and the residue classes correspond exactly to columns of an image grid when $l$ equals the row width. The **pigeonhole principle** guarantees that any $l$ consecutive positions contain one representative from every residue class modulo $l$, because $l$ consecutive integers produce $l$ distinct remainders. This turns the existence of a relay position in factorized attention from a search problem into a mathematical certainty. And **inclusion-exclusion** — $|A \cup B| = |A| + |B| - |A \cap B|$ — correctly counts the entries when local windows and global tokens overlap, preventing double-counting.

With these tools in hand, we are ready for [Why Full Attention Is Wasteful](/blog/attention-sparse-factorization), where we derive sparse factorized attention patterns and prove that two-hop paths always exist, and [Sliding Window Attention](/blog/attention-sliding-window), where we combine local windows with global attention and count every entry precisely.

---

*Previous: [DeepSeek-V2 from Scratch: Multi-head Latent Attention and DeepSeekMoE](/blog/deepseek-v2-mla-and-deepseekmoe)*
*Next: [Why Full Attention Is Wasteful: Sparse Factorization from Scratch](/blog/attention-sparse-factorization)*
