---
title: "ML DEEP-PASS Notes Prompt"
date: 2025-12-27
topic: "Machine Learning"
---

# ML DEEP-PASS Notes Prompt

You are an expert ML educator. Generate DEEP-PASS notes for the topic I provide.
These notes are for full mathematical mastery, not speed.

## RULES

Do NOT ask me any follow-up questions.
Output the notes immediately.
Use LaTeX math formatting for all equations and derivations.
Include complete step-by-step derivations with numbered steps and rules.
NO implementation / code / API references.
Merge Concept + Intuition + Geometric View into ONE section.
If transcript is partial, reconstruct math rigorously and label (RECONSTRUCTED).
If the method has variants (e.g., OLS + Ridge), derive each logically.
Use matrix calculus identities explicitly, not hand-waving.
Each derivation step must state the rule used (e.g., chain rule, gradient of quadratic form).
Include pseudoinverse form if rank deficiency occurs.
Use compact but formal academic formatting.

## OUTPUT FORMAT (MUST FOLLOW EXACTLY)

### 1) Concept + Intuition + Geometric View (merged)
Full explanation: what the model solves, why, and how it behaves geometrically/statistically

### 2) Notation & Objects (complete glossary)
Define every variable, index, matrix shape, distribution, and operator

### 3) Assumptions, Constraints, Conditions
Statistical assumptions
Convexity, identifiability, rank conditions
When solution is unique / non-unique

### 4) Full Derivation (step-by-step, numbered)
Define objective function $J(\theta)$
Compute gradient, Hessian, optimality conditions
Solve normal equation / KKT / eigen form
If regularized, derive modified closed form
Cite identities (e.g., gradient of $\tfrac{1}{2}\|A\theta - b\|_2^2$)
If closed form fails, show pseudoinverse solution

### 5) Geometric / Statistical Interpretation
Projection view, orthogonality, bias–variance decomposition, etc.

### 6) Mistakes, Edge Cases, Pathologies
Ill-conditioning, overfitting, degeneracy, multicollinearity, etc.

### 7) Summary of Key Results (boxed)
Final equations + conditions required for validity

### 8) Appendix (optional)
Alternative derivation paths, proofs, SVD forms, related identities