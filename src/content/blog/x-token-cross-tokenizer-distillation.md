---
title: "X-Token: Cross-Tokenizer Knowledge Distillation from Scratch"
description: "Building cross-tokenizer distillation from the ground up — why per-position KL breaks across tokenizers, DP span alignment, the chain-rule chunk merge, the projection matrix W, and two complementary losses (P-KL and H-KL) — with a full proof of GOLD's suppressive gradient and one running 2+3 example derived step by step."
date: 2026-05-31
tags: ["knowledge-distillation", "tokenizers", "llm", "deep-learning", "nlp"]
---

We want to teach a small Llama-3.2-1B student using a strong Qwen3-4B teacher. There is one problem, and it sounds trivial until you try it: the two models do not agree on what a token *is*. Qwen splits the number `201` into three tokens `2`, `0`, `1`. Llama packs it into a single token `201`. The standard distillation loss compares the student's probability of token $i$ against the teacher's probability of token $i$ — but token $i$ does not mean the same thing on both sides. The comparison is undefined.

This post builds the solution from NVIDIA's **X-Token** paper ([Turuvekere Sreenivas et al., 2026](https://arxiv.org/abs/2605.21699)) from scratch. We will derive every piece: why naive distillation fails, how to align two token streams with dynamic programming, how to merge per-token probabilities into per-chunk probabilities, how to build a projection matrix that maps one vocabulary into another, and the two complementary losses — **P-KL** and **H-KL** — that fix two distinct failure modes of the prior state of the art. We will also prove, in full, the suppressive-gradient pathology that motivates the whole design.

We use **one running example** throughout. The teacher and student both process the string `"2+3=6"`. The student tokenizes the answer `6` and the prompt naturally; the teacher uses a different tokenizer that, on the number `201` appearing elsewhere, splits digits. We will lean hardest on the single chunk where the text is the multi-digit number `201`, because that one chunk exposes every failure and every fix.

---

## 1. Mathematical setup

This post is self-contained, but it leans on a few tools that earlier posts in this series already derived from scratch. Rather than re-derive them, we link out and recall only what we need:

- **KL divergence and entropy** — the definition $\text{KL}(q\|p) = \sum_v q[v]\log\frac{q[v]}{p[v]}$, that it is zero iff $q = p$ (**Gibbs' inequality**), and the entropy $H(p) = -\sum_v p[v]\log p[v]$ — are built in [Mathematical Prerequisites for Foundation Prior](/blog/math-prerequisites-for-foundation-prior).
- **The softmax function** $p[s] = \exp(z_s)/\sum_{s'}\exp(z_{s'})$ and the **logarithm rules** are built in [Mathematical Prerequisites for Mixture of Experts](/blog/math-prerequisites-for-mixture-of-experts).
- **The chain rule of calculus** and **derivatives** are built in [Mathematical Prerequisites for Reinforcement Learning](/blog/math-prerequisites-for-rl); **argmax and the indicator function** in [Mathematical Prerequisites for Mixture of Experts (Part 2)](/blog/math-prerequisites-for-mixture-of-experts-part-2).

There is exactly **one** identity we will need that those posts do not state outright, and the suppressive-gradient proof in Section 9 depends on it entirely, so we derive it here once: the **softmax log-derivative identity**.

We want $\partial \log p_S[s] / \partial z_j$, the sensitivity of one log-probability to one logit, where $p_S[s] = \exp(z_s)/Z$ with the normalizer $Z = \sum_{s'}\exp(z_{s'})$. Take the log first:

$$
\log p_S[s] = z_s - \log Z.
$$

Differentiate with respect to $z_j$ term by term. The first term: $\partial z_s/\partial z_j = \mathbb{1}[s = j]$, the **indicator function** (1 if $s = j$, else 0), since distinct logits are independent variables. The second term, by the **chain rule of calculus**, is

$$
\frac{\partial \log Z}{\partial z_j} = \frac{1}{Z}\cdot\frac{\partial Z}{\partial z_j} = \frac{1}{Z}\cdot\exp(z_j) = \frac{\exp(z_j)}{Z} = p_S[j],
$$

where $\partial Z/\partial z_j = \exp(z_j)$ because every other term of the sum $Z = \sum_{s'}\exp(z_{s'})$ is constant in $z_j$. Subtracting:

$$
\boxed{\;\frac{\partial \log p_S[s]}{\partial z_j} = \mathbb{1}[s = j] - p_S[j].\;}
$$

**Numerical check.** Suppose three logits give $p_S = (0.5, 0.3, 0.2)$. Then $\partial \log p_S[1]/\partial z_1 = 1 - 0.5 = 0.5$ (raising logit 1 raises its own log-prob, but less than one-for-one because the normalizer also grows), while $\partial \log p_S[1]/\partial z_2 = 0 - 0.3 = -0.3$ (raising a *different* logit lowers $p_S[1]$, purely through the normalizer). That second case — the cross term being $-p_S[j]$ with no indicator — is the engine of the suppressive gradient we prove in Section 9. We restate this boxed identity there at the point of use.

---

## 2. The setup: standard knowledge distillation

**Knowledge distillation** (KD) is training a small *student* model to imitate the full output distribution of a large *teacher* model, not just its top answer ([Hinton, Vinyals, Dean, 2015](https://arxiv.org/abs/1503.02531)). The teacher's full distribution carries **dark knowledge** — the relative probabilities it assigns to *wrong* answers, which encode similarity structure the hard label throws away.

Concretely, at one position the teacher produces a probability distribution $p_T$ over its vocabulary $\mathcal{V}_T$, and the student produces $p_S$ over its vocabulary $\mathcal{V}_S$. The standard loss is the **Kullback–Leibler divergence** of the student from the teacher:

$$
\mathcal{L}_{\text{KD}} = \text{KL}(p_T \,\|\, p_S) = \sum_{v} p_T[v] \, \log \frac{p_T[v]}{p_S[v]}.
$$

Let us name what we just used. The KL divergence measures how many extra nats you pay to encode samples from $p_T$ using a code optimized for $p_S$. It is zero exactly when $p_S = p_T$ and positive otherwise — this is **Gibbs' inequality**, $\text{KL}(q\|p) \ge 0$ with equality iff $q = p$. Minimizing it drives the student toward the teacher.

This loss has a hidden assumption baked into the sum $\sum_v$: index $v$ must mean the same token in both distributions. When $\mathcal{V}_S = \mathcal{V}_T$ — same tokenizer, same vocabulary — that holds, and KD is a clean drop-in. The moment the tokenizers differ, the sum is comparing apples to oranges. That is the entire problem.

---

## 3. Why per-position KL is ill-defined across tokenizers

Take our running text `201`. Two tokenizers, two token streams:

- **Student (Llama-style):** one token, `[201]`. One position.
- **Teacher (Qwen-style):** three tokens, `[2, 0, 1]`. Three positions.

The student emits one distribution; the teacher emits three. There is no position $i$ shared between them. Even the *lengths* of the two sequences disagree. You cannot write $\sum_v p_T[v] \log p_S[v]$ because there is no single $p_T$ and $p_S$ aligned at a common position.

This is the part that trips people up: it is not merely that the vocabularies are different *sets of symbols*. It is that the **segmentation of the same underlying text** is different, so the sequences have different lengths and no positional correspondence. We need two things, in order:

1. A way to group tokens on each side into **chunks that cover the same underlying text**, so we have aligned units to compare. (Section 4 and 5.)
2. A way to compare a student chunk-distribution to a teacher chunk-distribution when the two are still over different vocabularies. (Section 7 onward.)

Let us solve them one at a time.

---

## 4. Span alignment with dynamic programming

We need to partition both token streams into **aligned chunks** $\{(A_k^S, A_k^T)\}_{k=1}^K$, where chunk $k$ on the student side ($A_k^S$, a run of student tokens) and chunk $k$ on the teacher side ($A_k^T$, a run of teacher tokens) **decode to the same substring of text**. For `201`, the aligned chunk is

$$
A_k^S = [\,201\,], \qquad A_k^T = [\,2, 0, 1\,],
$$

because `201` (one student token) and `2`,`0`,`1` (three teacher tokens) both decode to the string `"201"`.

How do we find these chunks automatically for an arbitrary sentence? X-Token uses **dynamic programming** (DP). Let $\mathbf{s}_{1:i}$ be the first $i$ student tokens and $\mathbf{t}_{1:j}$ the first $j$ teacher tokens. Define $D(i,j)$ as the maximum alignment score achievable over those two prefixes. The recurrence is

$$
D(i,j) = \max
\begin{cases}
D(i-1,\,j-1) + \text{match}(s_i, t_j) & \text{(diagonal, 1-to-1)} \\[4pt]
\max_{2 \le k \le L}\; D(i-1,\,j-k) + \alpha_{\text{comb}}\,k \cdot \mathbb{1}[\,s_i \equiv \mathbf{t}_{j-k+1:j}\,] & \text{(1-to-}k\text{)} \\[4pt]
\max_{2 \le k \le L}\; D(i-k,\,j-1) + \alpha_{\text{comb}}\,k \cdot \mathbb{1}[\,\mathbf{s}_{i-k+1:i} \equiv t_j\,] & \text{(}k\text{-to-1)} \\[4pt]
D(i-1,\,j) + \alpha_{\text{gap}} & \text{(gap in teacher)} \\[4pt]
D(i,\,j-1) + \alpha_{\text{gap}} & \text{(gap in student)}
\end{cases}
$$

with boundary conditions $D(i,0) = i \cdot \alpha_{\text{gap}}$ and $D(0,j) = j \cdot \alpha_{\text{gap}}$. Here $\equiv$ denotes **canonicalized string equality** between a single token and the concatenation of a span — two strings are equal after normalizing surface differences like space-prefix markers (we make this precise in Section 6). The scoring constants used throughout the paper are

$$
\alpha_{\text{exact}} = 3, \qquad \alpha_{\text{comb}} = 1.5, \qquad \alpha_{\text{gap}} = -1.5,
$$

where $\text{match}(s_i, t_j) = +\alpha_{\text{exact}}$ if the two canonicalized tokens are string-equal and $-\alpha_{\text{exact}}$ otherwise. After filling the table, a **backtrace** from $D(n,m)$ recovers the chosen chunk boundaries; transitions selected as gaps mark token positions as unaligned, and those positions are excluded from the loss.

### Why soft scoring, not hard alignment

A hard alignment — align-or-fail — has two failure modes on real text. First, a single local oddity (a byte-fallback token, an unusual whitespace glyph) can make the *entire* sequence misalign or propagate the error to neighbors. Second, two locally-plausible alignments can tie, and an arbitrary tie-break produces inconsistent alignments across training runs. The soft scoring resolves both. Gaps cost $|\alpha_{\text{gap}}| = 1.5$, so the DP prefers to insert *one* gap rather than distort a long stretch. And a $k$-token combination scores $+\alpha_{\text{comb}}\,k = 1.5k$, which competes favorably with $k$ individual exact matches ($+\alpha_{\text{exact}}\,k = 3k$) only when no exact match exists — so exact 1-to-1 matches are preferred when available, and span combinations are the fallback. The inequality $\alpha_{\text{exact}} > |\alpha_{\text{gap}}|$ (i.e. $3 > 1.5$) is what rewards walking *through* an alignment over walking *around* it.

### Numerical check on the running example

Suppose at the relevant region the student has the single token `201` and the teacher has `2`,`0`,`1`. The DP can either (a) take three gaps to skip them apart, scoring $3 \times (-1.5) = -4.5$, or (b) take one 1-to-$k$ combination with $k=3$, since `201` $\equiv$ `2`+`0`+`1` after concatenation, scoring $\alpha_{\text{comb}} \cdot 3 = 1.5 \times 3 = 4.5$. Since $4.5 > -4.5$, the DP chooses the combination and emits the aligned chunk $([201],\,[2,0,1])$. The numbers come out exactly as the design intends.

---

## 5. The chain-rule chunk merge

Span alignment gives us *which* tokens form a chunk. But the teacher chunk `[2, 0, 1]` is still three separate per-token distributions, while the student chunk `[201]` is one. To compare them we need **one distribution per chunk on each side**. We build it with the **chain rule of probability**.

A **chunk-level distribution** $\hat{p}^{(k)}$ is the probability the model assigns to producing the entire chunk's text, decomposed autoregressively over its tokens. For the teacher chunk $[2,0,1]$, the probability of that specific three-token string is

$$
\hat{p}_T^{(k)}(\text{``201''}) = p_T(2) \cdot p_T(0 \mid 2) \cdot p_T(1 \mid 2,0).
$$

This is the **chain rule of probability**: $P(ABC) = P(A)\,P(B\mid A)\,P(C\mid A,B)$. Each factor is exactly the per-position softmax the teacher already computed during its forward pass, so the merge is free — no extra model calls.

### Numerical check

Suppose the teacher is confident: $p_T(2) = 0.9$, $p_T(0\mid 2) = 0.8$, $p_T(1 \mid 2,0) = 0.95$. Then

$$
\hat{p}_T^{(k)}(\text{``201''}) = 0.9 \times 0.8 \times 0.95 = 0.684.
$$

The student, packing `201` as one token, directly reads off $\hat{p}_S^{(k)}(\text{``201''}) = p_S(201)$ from its single softmax — say $0.5$. Now both sides express the *same event* ("the model produces the text 201") as a single number, and we finally have aligned, comparable units: $\hat p_S^{(k)}$ and $\hat p_T^{(k)}$. This chunk-level view is what every loss below operates on.

---

## 6. Canonicalization: making "the same text" actually the same

Before we can declare two tokens string-equal in the DP or in the projection matrix below, we must normalize away cosmetic differences between tokenizer families. The **canonicalization function** $\text{canon}(\cdot)$ maps a token's decoded string to a normal form so functionally identical tokens compare equal. The rules, applied in order:

- **Space-prefix unification:** the GPT-2/Llama space marker `Ġ`, the SentencePiece marker `▁`, and a literal Unicode space all map to a single literal space at the start of a token.
- **Newline unification:** `Ċ`, the escaped `\n`, and a literal newline all map to `\n`.
- **Byte-fallback tokens:** SentencePiece byte tokens of the form `<0xHH>` are replaced by the literal character with that byte value.
- **Leading whitespace + punctuation pairs:** combinations like `Ġ,` are normalized to the punctuation alone when the whitespace interpretation is ambiguous.
- **Special tokens:** BOS, EOS, PAD, and chat-template tokens are handled by an explicit role-to-role mapping across families.

Canonicalization is **idempotent** (applying it twice changes nothing) and involves no learned parameters. It is applied at *both* projection-matrix construction time and inside the DP's string-equality check, so the two stages agree on what "the same text" means.

This matters more than it looks. The paper documents a concrete failure of an alternative surface-substring aligner (used in TRL's GOLD trainer) caused by exactly this: the Llama tokenizer auto-prepends a `<bos>` token (its config default `add_bos_token=True`) while Qwen and Phi-4-mini default to `False`. On the input `"Hello world."` the decoded streams differ on byte 0. The surface aligner extends per-side decoded buffers piece by piece and only flushes when the buffers compare equal as raw strings, but after the first piece, the student buffer is `"<|begin_of_text|>"` (16 chars) versus the teacher's `"Hello"` (5 chars). They never re-sync, and the end-of-sequence force-flush dumps everything into one mis-grouped **super-group** bundling all tokens together. The DP, by contrast, marks the spurious `<bos>` as a one-sided **gap** of unit cost and aligns the three content tokens diagonally as 1-to-1 matches. The disagreement is localized to one gap regardless of sentence length.

---

## 7. The projection matrix $W$

We have aligned chunks and chunk-level distributions, but $\hat p_S^{(k)}$ lives over $\mathcal{V}_S$ and $\hat p_T^{(k)}$ lives over $\mathcal{V}_T$. They are still distributions over **different vocabularies**. The final bridge is a **projection matrix** $W \in \mathbb{R}^{|\mathcal{V}_S| \times |\mathcal{V}_T|}$ that maps a student-vocabulary distribution into teacher-vocabulary space. Entry $W[s,t]$ is the weight with which student token $s$'s probability mass should be routed to teacher token $t$.

$W$ is built deterministically in **two passes**.

**Pass 1 — canonicalized exact match.** For every pair $(s,t) \in \mathcal{V}_S \times \mathcal{V}_T$ whose canonicalized decoded strings are equal, set $W[s,t] = 1$. This handles tokens that exist verbatim in both vocabularies (e.g. `_the`, `_cat`).

**Pass 2 — multi-token decoding rule.** For each remaining student token $s$ with no exact match, decode its text and re-tokenize it under the teacher tokenizer, yielding a sequence $(\tau_0, \tau_1, \dots, \tau_{\ell-1})$ of teacher sub-tokens. Assign exponentially decaying weights along that sequence:

$$
W[s, \tau_i] = \beta \, \gamma^{\,i}, \qquad i = 0, 1, \dots, \ell-1,
$$

with $(\beta, \gamma) = (0.9, 0.1)$. Then each row is truncated to its **top-$K$** entries ($K=4$) and **row-normalized**.

The decay concentrates mass on the *leading* sub-token, which typically carries the most informative probability mass for cross-tokenizer distillation (e.g. `_20` in `["_20", "24"]`, or the prefix in `["_inter", "national"]`), while trailing sub-tokens matter less given the prefix.

### Numerical check on the decay

For our `201` example, the student token `201` re-tokenizes under the teacher as $(2, 0, 1)$, length $\ell = 3$. Before normalization:

$$
\bar w_0 = 0.9 \cdot 0.1^0 = 0.9, \quad \bar w_1 = 0.9 \cdot 0.1^1 = 0.09, \quad \bar w_2 = 0.9 \cdot 0.1^2 = 0.009.
$$

The sum is $0.9 + 0.09 + 0.009 = 0.999$. Row-normalizing (dividing each by $0.999$):

$$
W[201, 2] = \frac{0.9}{0.999} = 0.9009, \quad W[201, 0] = \frac{0.09}{0.999} = 0.0901, \quad W[201, 1] = \frac{0.009}{0.999} = 0.0090.
$$

These are exactly the length-3 weights $(0.9009, 0.0901, 0.0090)$ the paper reports. Almost all of `201`'s mass routes to the teacher's leading sub-token `2`.

### $W$ is a probability-preserving operator

Here is the property that makes $W$ safe to use. Each row of $W$ is non-negative and sums to 1 (after normalization), so left-multiplication by $W^\top$ is a **convex combination of rows** — and a convex combination of probability vectors is a probability vector. Let us prove the projected student distribution is still a valid distribution. Writing $\mathbf{p}_S$ for the student chunk distribution,

$$
\sum_{t \in \mathcal{V}_T} (W^\top \mathbf{p}_S)[t]
= \sum_t \sum_s W[s,t]\, p_S[s]
= \sum_s p_S[s] \underbrace{\sum_t W[s,t]}_{=\,1}
= \sum_s p_S[s] = 1.
$$

The middle step swaps the order of summation (**Fubini's theorem** for finite sums — interchanging two finite sums is always valid), then uses row-normalization $\sum_t W[s,t] = 1$, then total probability $\sum_s p_S[s] = 1$. So $W^\top \mathbf p_S$ is a genuine distribution over $\mathcal V_T$ with no extra normalization tricks. We will use this in P-KL.

$W$ is constructed *once* before training. It can optionally be fine-tuned during KD for additional gains — we will see the ablation.

---

## 8. The baseline and its two failures: GOLD's hybrid loss

To appreciate X-Token's two losses we must first see precisely how the prior state of the art, **GOLD** ([Patiño et al., 2025](https://arxiv.org/abs/2504.13161)), fails. GOLD partitions the two vocabularies into a 1-to-1 string-matched **common set** $\mathcal{C}$ and **uncommon remainders** $\mathcal{U}_S, \mathcal{U}_T$ with $\mathcal{U} = \mathcal{U}_S \cup \mathcal{U}_T$. It applies direct KL on the common set and a rank-sorted $L_1$ match (a **Universal Logit Distillation**, ULD, term [Boizard et al., 2024](https://arxiv.org/abs/2402.12030)) on the uncommon remainder:

$$
\mathcal{L}_{\text{common}}^{(k)} = \sum_{(s,t) \in \mathcal{C}} \hat p_T^{(k)}[t]\,\big(\log \hat p_T^{(k)}[t] - \log \hat p_S^{(k)}[s]\big),
$$

$$
\mathcal{L}_{\text{ULD}}^{(k)} = \big\| \,\text{sort}_\downarrow(\hat p_S^{(k)}|_{\mathcal{U}_S}) - \text{sort}_\downarrow(\hat p_T^{(k)}|_{\mathcal{U}_T})\, \big\|_1,
$$

$$
\mathcal{L}_{\text{GOLD}}^{(k)} = \lambda_{\text{KL}}\,\mathcal{L}_{\text{common}}^{(k)} + \lambda_{\text{ULD}}\,\mathcal{L}_{\text{ULD}}^{(k)}.
$$

Now the two failures.

### Failure 1 — the uncommon-token failure

A **critical token** is a token whose correct prediction directly determines task accuracy — the multi-digit numerals in a math benchmark like GSM8k are the canonical example. Under the Qwen3-4B teacher, *all 1,100* of Llama's two- and three-digit numerals fall into the uncommon set $\mathcal{U}$, because Qwen digit-splits and Llama does not, so there is no 1-to-1 match (Table 8 in the paper: 0/100 two-digit and 0/1000 three-digit Llama numerals survive into $\mathcal{C}$).

These critical tokens are then handled only by the rank-sorted ULD term, which pairs the student's numeral with whatever teacher token happens to sit at the same *rank* — an unrelated special character, perhaps. This is **identity-agnostic noise**: it misaligns critical tokens with semantically unrelated teacher tokens. The supervision signal on exactly the tokens that matter most is garbage.

### Failure 2 — the suppressive gradient (proven below)

Worse, even though the uncommon tokens do not appear in $\mathcal{L}_{\text{common}}$, the common-KL term *still* pushes their probabilities **down**, because it is computed through the full-vocabulary softmax. We will prove this in Section 9.

The empirical cost is dramatic: on the Qwen pair, GSM8k drops to **2.56** under GOLD, versus **12.89** for same-tokenizer KD from a weaker Llama-3B teacher. Cross-tokenizer KD from a *stronger* teacher does *worse* than same-tokenizer KD from a *weaker* one. Something is actively harmful.

### Failure 3 — over-conservative matching

A third, subtler issue: GOLD's common set requires *exact* string equality. A pair like (`Hundreds`, `Hund`) — where the student token corresponds to the teacher's leading sub-token — is near-equivalent but not string-equal, so it is exiled to $\mathcal{U}$ and its clean alignment signal is wasted. Strict equality is too conservative even when the partition is otherwise sound.

X-Token attacks Failures 1–2 with **P-KL** and Failure 3 with **H-KL**.

<svg viewBox="0 0 720 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="GOLD versus P-KL on the critical token 201" style="max-width:100%;height:auto;background:#ffffff;border:1px solid #d0d0d0;border-radius:8px">
  <text x="180" y="28" font-family="sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#b00020">GOLD: 201 falls into the uncommon set</text>
  <text x="540" y="28" font-family="sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#1a7f37">P-KL: 201 routed through W</text>
  <line x1="360" y1="45" x2="360" y2="235" stroke="#cccccc" stroke-width="1"/>
  <!-- GOLD side -->
  <rect x="60" y="70" width="70" height="40" rx="6" fill="#fde2e2" stroke="#b00020"/>
  <text x="95" y="95" font-family="monospace" font-size="16" text-anchor="middle" fill="#b00020">201</text>
  <text x="95" y="128" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#666">student (critical)</text>
  <rect x="230" y="70" width="50" height="40" rx="6" fill="#f0f0f0" stroke="#999"/>
  <text x="255" y="95" font-family="monospace" font-size="14" text-anchor="middle" fill="#444">下午</text>
  <text x="255" y="128" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#666">teacher (rank match)</text>
  <line x1="130" y1="90" x2="230" y2="90" stroke="#b00020" stroke-width="2" stroke-dasharray="5,4"/>
  <text x="180" y="82" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#b00020">erroneous</text>
  <text x="180" y="175" font-family="sans-serif" font-size="13" text-anchor="middle" fill="#b00020">+ common-KL suppresses 201</text>
  <!-- P-KL side -->
  <rect x="420" y="70" width="70" height="40" rx="6" fill="#dcefe1" stroke="#1a7f37"/>
  <text x="455" y="95" font-family="monospace" font-size="16" text-anchor="middle" fill="#1a7f37">201</text>
  <rect x="600" y="60" width="40" height="26" rx="4" fill="#f0f0f0" stroke="#999"/>
  <text x="620" y="78" font-family="monospace" font-size="13" text-anchor="middle" fill="#444">2</text>
  <rect x="600" y="92" width="40" height="26" rx="4" fill="#f0f0f0" stroke="#999"/>
  <text x="620" y="110" font-family="monospace" font-size="13" text-anchor="middle" fill="#444">0</text>
  <rect x="600" y="124" width="40" height="26" rx="4" fill="#f0f0f0" stroke="#999"/>
  <text x="620" y="142" font-family="monospace" font-size="13" text-anchor="middle" fill="#444">1</text>
  <line x1="490" y1="86" x2="600" y2="73" stroke="#1a7f37" stroke-width="2.5"/>
  <line x1="490" y1="90" x2="600" y2="105" stroke="#1a7f37" stroke-width="1.5"/>
  <line x1="490" y1="94" x2="600" y2="137" stroke="#1a7f37" stroke-width="1"/>
  <text x="545" y="60" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a7f37">W: 0.90 / 0.09 / 0.01</text>
  <text x="540" y="190" font-family="sans-serif" font-size="13" text-anchor="middle" fill="#1a7f37">no partition, direct teacher signal</text>
</svg>

---

## 9. Proof: the common-KL term suppresses every uncommon token

This is the formal heart of the motivation. We prove **Proposition 1**: GOLD's common-KL term induces a non-negative gradient on *every* uncommon student logit, pushing all uncommon-token probabilities down, even though those tokens never appear in the loss.

**Setup.** Fix one chunk. Let $z \in \mathbb{R}^{|\mathcal{V}_S|}$ be the student logits, $p_S = \text{softmax}(z)$, and let $p_T$ be the fixed teacher distribution. Let $\mathcal{C}_S \subseteq \mathcal{V}_S$ and $\mathcal{C}_T \subseteq \mathcal{V}_T$ be the two sides of the common set $\mathcal{C}$ (a bijection: each $t \in \mathcal{C}_T$ appears in exactly one pair $(s,t) \in \mathcal{C}$), and $\mathcal{U} = \mathcal{V}_S \setminus \mathcal{C}_S$ the uncommon set. The full-vocabulary softmax is

$$
p_S[s] = \frac{\exp(z_s)}{Z_{\text{full}}}, \qquad Z_{\text{full}} = \sum_{s' \in \mathcal{V}_S} \exp(z_{s'}).
$$

The common-KL term, dropping the constant teacher-entropy part, is

$$
\mathcal{L}_{\text{common}}(z) = \sum_{(s,t) \in \mathcal{C}} p_T[t]\,\big(\log p_T[t] - \log p_S[s]\big).
$$

**Two preliminary identities.** Treating each logit as an independent variable, for any $s, j \in \mathcal{V}_S$:

$$
\frac{\partial z_s}{\partial z_j} = \mathbb{1}[s = j], \qquad \frac{\partial \log Z_{\text{full}}}{\partial z_j} = \frac{\exp(z_j)}{Z_{\text{full}}} = p_S[j].
$$

The first is immediate. The second follows from $\log Z_{\text{full}} = \log \sum_{s'} \exp(z_{s'})$ by the **chain rule of calculus**: differentiating $\log(\cdot)$ gives $\tfrac{1}{Z_{\text{full}}}$, and differentiating the sum picks out only the $j$-th term $\exp(z_j)$, leaving $\exp(z_j)/Z_{\text{full}} = p_S[j]$.

Combining with $\log p_S[s] = \log \exp(z_s) - \log Z_{\text{full}} = z_s - \log Z_{\text{full}}$:

$$
\frac{\partial \log p_S[s]}{\partial z_j} = \frac{\partial z_s}{\partial z_j} - \frac{\partial \log Z_{\text{full}}}{\partial z_j} = \mathbb{1}[s = j] - p_S[j].
$$

This is the standard **softmax log-derivative identity**.

**The proof.** Fix an uncommon logit $j \in \mathcal{U}$. Since $\mathcal{C}_S$ and $\mathcal{U}$ are disjoint, every $s \in \mathcal{C}_S$ satisfies $s \ne j$, so $\mathbb{1}[s = j] = 0$ and the identity above collapses to $\partial \log p_S[s] / \partial z_j = -p_S[j]$ for every $s \in \mathcal{C}_S$. The teacher factor $p_T[t]$ does not depend on $z$. Differentiating $\mathcal{L}_{\text{common}}$ with respect to $z_j$ (using **linearity of differentiation** to move the derivative inside the finite sum):

$$
\frac{\partial \mathcal{L}_{\text{common}}}{\partial z_j}
= -\sum_{(s,t) \in \mathcal{C}} p_T[t] \cdot \frac{\partial \log p_S[s]}{\partial z_j}
= -\sum_{(s,t) \in \mathcal{C}} p_T[t] \cdot \big(-p_S[j]\big)
= p_S[j] \sum_{t \in \mathcal{C}_T} p_T[t].
$$

Writing $M_{\mathcal{C}}(T) := \sum_{t \in \mathcal{C}_T} p_T[t] \in [0,1]$ (the teacher's total mass on the common set — a sub-sum of a probability distribution, hence between 0 and 1):

$$
\boxed{\;\frac{\partial \mathcal{L}_{\text{common}}}{\partial z_j} = p_S[j] \cdot M_{\mathcal{C}}(T) \ge 0 \quad \text{for every } j \in \mathcal{U}.\;}
$$

The gradient is non-negative because both factors are non-negative ($p_S[j] \ge 0$ and $M_{\mathcal C}(T) \ge 0$), and it vanishes only when one of them is zero.

**Interpretation.** Gradient descent with step $\eta > 0$ updates $\Delta z_j = -\eta\, p_S[j]\, M_{\mathcal{C}}(T) \le 0$. So every uncommon logit is driven *down* at every step. Because the softmax is monotonically increasing in each logit, shrinking $z_j$ shrinks $p_S[j]$ relative to all other probabilities. The probability mass of **every uncommon token is suppressed** — even though no uncommon token appears in $\mathcal{L}_{\text{common}}$, and the gradient depends only on $p_T$, making it **independent of the ground-truth token** at the position. When your critical numerals live in $\mathcal{U}$ (the Qwen case), GOLD is actively training the student to stop predicting them. This is why GSM8k collapses to 2.56.

### Numerical check

Say at a chunk $p_S[201] = 0.3$ and the teacher places $M_{\mathcal{C}}(T) = 0.7$ of its mass on common tokens. Then $\partial \mathcal{L}_{\text{common}} / \partial z_{201} = 0.3 \times 0.7 = 0.21 > 0$, and with $\eta = 0.1$ the logit moves by $\Delta z_{201} = -0.1 \times 0.21 = -0.021$ — downward, every step, regardless of whether `201` is the correct answer.

---

## 10. P-KL: remove the partition entirely

The fix for Failures 1 and 2 follows directly from the proof: the partition is the problem, so **delete the partition**. **P-KL** ("projection KL") projects the student's full chunk distribution into teacher-vocabulary space using $W$ and applies a single KL against the teacher's full distribution — no common set, no ULD term, nothing for an uncommon token to fall out of.

Define the projected student distribution $\tilde p_S^{(k)}$ over $\mathcal{V}_T$:

$$
\tilde p_S^{(k)}[t] = \sum_{s \in \mathcal{V}_S} W[s,t] \cdot \hat p_S^{(k)}[s], \qquad
\mathcal{L}_P^{(k)} = \text{KL}\big(\hat p_T^{(k)} \,\|\, \tilde p_S^{(k)}\big).
$$

The first equation is exactly the operator $\tilde p_S^{(k)} = W^\top \hat p_S^{(k)}$, which Section 7 proved is a valid distribution over $\mathcal{V}_T$. The second is plain KL between two distributions *over the same teacher vocabulary* — now well-defined. Because there is no partition, the critical token `201` is no longer exiled to $\mathcal{U}$; its mass is *routed* through $W$ onto the teacher's decomposition $\{2,0,1\}$ and compared directly. Both sources of error from Section 8–9 are replaced by teacher-aware supervision over all tokens.

### Numerical check

From Section 7, $W$ routes student `201` as $(0.9009, 0.0901, 0.0090)$ onto teacher $(2,0,1)$. If the student chunk distribution puts $\hat p_S^{(k)}[201] = 0.5$ (and we ignore other student tokens for illustration), the projected mass on teacher token `2` is $\tilde p_S^{(k)}[2] = 0.9009 \times 0.5 = 0.4505$, on `0` it is $0.0901 \times 0.5 = 0.0451$, and on `1` it is $0.0090 \times 0.5 = 0.0045$. These now sit in the *same* teacher vocabulary as $\hat p_T^{(k)}$, and KL compares them directly — no rank-matching, no suppression.

### When P-KL wins

P-KL is the right loss **when critical tokens fall outside the common set** — the Qwen3-4B regime where all multi-digit numerals are uncommon. Empirically P-KL improves over GOLD by **+3.82 average points**, and on GSM8k specifically from 2.56 to **15.54** — a $6\times$ jump that even surpasses same-tokenizer KD from Llama-3B (12.89). Notably, plain ULD (no partition, just rank-sort) already beats GOLD (36.77 vs 35.03 avg), confirming the partition is the primary source of failure; P-KL's identity-aware projection then adds another +2.08 over ULD.

---

## 11. H-KL: keep the partition, relax the matching

P-KL throws away the partition entirely. But sometimes the partition is *good* — when critical tokens already live in the common set, direct identity-aligned KL gives *sharper* supervision than projecting student mass through $W$'s multi-token rows. This is the Phi-4-mini regime: Phi-4-mini keeps all of Llama's multi-digit numerals in $\mathcal{C}$ (Table 8: 100/100 two-digit, 1000/1000 three-digit). Here the partition is structurally sound, and discarding it (using P-KL) would *sacrifice* identity-aligned signal — the paper measures this as a regression.

So the second loss, **H-KL** ("hybrid KL"), keeps GOLD's hybrid structure but fixes Failure 3 — the over-conservative exact-match requirement. Instead of requiring string equality to enter $\mathcal{C}$, H-KL admits each student token's **top-ranked teacher token under $W$**. For each student token $s$, select

$$
t^* = \arg\max_{t' \in \mathcal{V}_T} W[s, t'], \qquad W[s, t^*] > 0,
$$

and extend the common set with the pair $(s, t^*)$. Exact matches are preserved (they receive the highest weight, 1, in $W$), and additional near-equivalent pairs like (`Hundreds`, `Hund`) are now admitted — they get the *same* direct-KL signal as a native exact match. H-KL then applies the hybrid loss (the GOLD formula of Section 8) over this **expanded** common set.

### When H-KL wins

H-KL is the right loss **when token alignment is reliable** — the partition is sound and we want the sharper identity-aligned KL. On the Phi-4-mini teacher, H-KL improves over GOLD by **+0.5 average** and beats P-KL by **+1.68** on that teacher. The reversal is exactly symmetric to P-KL's: each loss exhibits a sharp drop when applied to the wrong teacher (Table 2 flips the per-teacher winner). Neither mode dominates; the loss must match the regime.

---

## 12. The unified view: P-KL and H-KL are two points on one axis

It is tempting to see P-KL and H-KL as two unrelated tricks. They are not. They are the two settings of a single design decision: **what to do with the partition**.

<svg viewBox="0 0 720 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="P-KL and H-KL as two settings of the partition decision" style="max-width:100%;height:auto;background:#ffffff;border:1px solid #d0d0d0;border-radius:8px">
  <text x="360" y="30" font-family="sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#222">One decision: keep the partition, or remove it?</text>
  <!-- P-KL box -->
  <rect x="40" y="60" width="290" height="120" rx="10" fill="#dcefe1" stroke="#1a7f37" stroke-width="1.5"/>
  <text x="185" y="86" font-family="sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#1a7f37">P-KL — remove partition</text>
  <text x="185" y="112" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#333">project full student dist through W</text>
  <text x="185" y="132" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#333">single KL over teacher vocab</text>
  <text x="185" y="160" font-family="sans-serif" font-size="12" font-style="italic" text-anchor="middle" fill="#1a7f37">use when critical tokens ∈ uncommon</text>
  <!-- H-KL box -->
  <rect x="390" y="60" width="290" height="120" rx="10" fill="#e2ecfb" stroke="#1a56b0" stroke-width="1.5"/>
  <text x="535" y="86" font-family="sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#1a56b0">H-KL — keep partition</text>
  <text x="535" y="112" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#333">expand common set via top-1 of W</text>
  <text x="535" y="132" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#333">hybrid common-KL + ULD on tail</text>
  <text x="535" y="160" font-family="sans-serif" font-size="12" font-style="italic" text-anchor="middle" fill="#1a56b0">use when partition is sound</text>
  <text x="360" y="120" font-family="sans-serif" font-size="20" text-anchor="middle" fill="#888">↔</text>
</svg>

Both share the same machinery: the same DP alignment, the same chunk merge, and the same projection matrix $W$. P-KL uses $W$ as a *full projection* ($\tilde p_S = W^\top \hat p_S$ then KL). H-KL uses $W$ only for its *top-1 entry per row* ($\arg\max_t W[s,t]$ to expand $\mathcal C$). The conceptual difference between the two methods is therefore completely transparent: **how much of $W$ you use, and whether you keep the partition.** This is what makes the selection rule simple.

### The selection rule: a coverage audit

We choose between them with a **coverage analysis**. Group tokens into character classes (digits by length, alphabetic, punctuation, multi-byte / non-ASCII) and measure each class's retention in the common set $\mathcal{C}$. The rule:

- If **critical tokens fall outside** $\mathcal{C}$ → use **P-KL** (Qwen3-4B: all multi-digit numerals are uncommon).
- If **critical tokens remain inside** $\mathcal{C}$ → use **H-KL** (Phi-4-mini: numerals stay common, punctuation fully covered).

This is a one-time, deterministic audit per teacher — no tuning loop.

---

## 13. Plugging the chunk loss into training

The per-chunk loss feeds the standard KD objective, averaged over the $K$ aligned chunks of a sequence (top-$K$ teacher logits with $K=8192$ for the KL itself):

$$
\mathcal{L}_{\text{KD}} = \frac{1}{K}\sum_{k=1}^{K} \mathcal{L}_*^{(k)}, \qquad \mathcal{L}_*^{(k)} \in \{\mathcal{L}_P^{(k)},\, \mathcal{L}_H^{(k)}\}.
$$

Two practical details complete the recipe.

**Dynamic KD/CE scaling.** Distillation is combined with ordinary next-token cross-entropy $\mathcal{L}_{\text{CE}}$ on the student. These two terms can differ wildly in magnitude and drift during training, so a fixed weight makes optimization unstable. X-Token rescales the KD term at every step to match the scale of $\mathcal{L}_{\text{CE}}$:

$$
\mathcal{L} = \text{sg}\!\left(\frac{\mathcal{L}_{\text{CE}}}{\mathcal{L}_{\text{KD}}}\right) \cdot \mathcal{L}_{\text{KD}} + \mathcal{L}_{\text{CE}},
$$

where $\text{sg}(\cdot)$ is the **stop-gradient** operator: the ratio is treated as a constant for differentiation, so it only rescales the magnitude and does not contribute its own gradient. The effect is that the KD contribution always carries roughly the same weight as CE, regardless of their raw scales. The ablation (Table 4) shows dynamic scaling reaching 36.39 avg vs 35.92–36.27 for the best fixed weights.

**Multi-teacher distillation.** With $M$ teachers, each with its own projection matrix $W_m$ and its own selected loss, aggregate per-teacher losses with static weights:

$$
\mathcal{L}_{\text{KD,multi}} = \sum_{m=1}^{M} \alpha_m \, \frac{1}{|\mathcal{K}_m|} \sum_{k \in \mathcal{K}_m} \mathcal{L}_{*,m}^{(k)}.
$$

The surprising finding: **static weighting beats adaptive weighting.** The paper tried confidence-adaptive $\alpha_m$ from cross-entropy, entropy, and max-probability scores, and a simple static ratio won every time (Table 5: static (0.2, 0.8) reaches 40.48 avg vs 40.16–40.21 for adaptive variants). Adaptive schemes add tuning complexity without consistent gains.

The deeper lesson is about *which* teachers to combine, not how to weight them. **Teacher complementarity drives the gains:** pairing Phi-4-mini (math/reasoning) with Llama-3B (commonsense) reaches 40.48 avg, beating the best single cross-tokenizer teacher by +1.3 — while pairing two reasoning teachers (Phi-4-mini + Qwen3-4B) gives only 38.49, where overlapping strengths interfere rather than add.

---

## 14. The full algorithm, end to end

Putting every piece in order, one X-Token training step is:

1. **Preprocess (cached across epochs):** tokenize input $x$ on both sides → $\mathbf{s} = \mathcal{T}_S(x)$, $\mathbf{t} = \mathcal{T}_T(x)$. Run the **DP alignment** (Section 4) to get aligned chunks $\{(A_k^S, A_k^T)\}$. Alignment is per-sequence and adds *no* per-step training overhead.
2. **Forward:** run the student $f_S(\mathbf{s})$ *with* gradient, the frozen teacher $f_T(\mathbf{t})$ *without*.
3. **Per chunk $k$:** merge per-token probabilities into chunk distributions $\hat p_S^{(k)}, \hat p_T^{(k)}$ via the **chain-rule merge** (Section 5).
4. **Apply the selected loss:** if P-KL, project $\tilde p_S^{(k)} = W^\top \hat p_S^{(k)}$ and take $\text{KL}(\hat p_T^{(k)} \| \tilde p_S^{(k)})$; if H-KL, apply the hybrid common-KL + ULD over the expanded $\mathcal{C}$.
5. **Aggregate** $\mathcal{L}_{\text{KD}} = \tau^2 \cdot \tfrac{1}{K}\sum_k \mathcal{L}^{(k)}$ (temperature $\tau = 1.0$), compute $\mathcal{L}_{\text{CE}}$, apply the **stop-gradient rescaling** $\gamma = \text{sg}(\mathcal{L}_{\text{CE}} / \mathcal{L}_{\text{KD}})$, and update $f_S$ via $\nabla_{f_S}(\gamma \mathcal{L}_{\text{KD}} + \mathcal{L}_{\text{CE}})$.

$W$ is initialized rule-based (Section 7), then **jointly learned with the student under P-KL** (learning rate $10^{-2}$, no gradient clipping) and **kept fixed under H-KL** (which only reads $\arg\max_t W$, a discrete operation that receives no gradient). The ablation (Table 3) confirms learning $W$ helps modestly: 38.85 vs 38.37 avg on the Qwen pair, winning 5/6 benchmark columns — so the rule-based construction is already a strong initialization that fine-tuning refines.

---

## 15. What the numbers say

Training a Llama-3.2-1B student on the Nemotron-ClimbMix dataset for 30,000 steps, evaluated 3-shot across MMLU, GSM8k, MATH, Winogrande, and HellaSwag:

- **Frozen baseline:** 33.96 avg. **Continued pre-training (no teacher):** 36.63 — modest, confirming the gains come from distillation, not extra compute.
- **Same-tokenizer KD** (Llama-3B → 1B): 38.40 avg — the same-family ceiling.
- **Qwen3-4B teacher:** GOLD 35.03 (below even no-teacher pre-training!) → **P-KL 38.85** (+3.82), with GSM8k 2.56 → 15.54.
- **Phi-4-mini teacher:** GOLD 38.66 → **H-KL 39.18** (+0.5).
- **Two teachers** (Phi-4-mini + Llama-3B): **40.48 avg**, beating the best single cross-tokenizer run by +1.3 and the same-family reference by +2.1.

The headline: cross-tokenizer KD, done right, **exceeds same-tokenizer KD** — you are no longer locked to teachers that share your tokenizer, and combining complementary teachers from different families adds gains a single teacher cannot.

---

## 16. Summary

Standard distillation breaks across tokenizers because per-position KL assumes a shared segmentation that does not exist; X-Token restores it with **DP span alignment** (grouping tokens into chunks that decode to the same text), a **chain-rule merge** (collapsing each chunk's per-token softmaxes into one chunk-level distribution), and a **projection matrix $W$** (a probability-preserving operator mapping student-vocabulary mass into teacher space, built from canonicalized exact matches plus exponentially-decayed re-tokenization rules). On top of this shared machinery sit two complementary losses chosen by a one-time coverage audit: **P-KL** deletes GOLD's partition and projects the full student distribution through $W$ — the cure for the *suppressive gradient* we proved drives every uncommon (and often critical) token's probability to zero — while **H-KL** keeps the partition but expands the common set via $W$'s top-1 mapping, recovering sharper identity-aligned KL whenever the partition is already sound. Together with dynamic KD/CE rescaling and complementary-teacher multi-distillation, these let a 1B student learn from any-family teachers and beat same-tokenizer distillation outright.
