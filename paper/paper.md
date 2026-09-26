---
title: "Expressible but Not Learnable? MDL Local-Rule Induction and the ARC-AGI-1 → ARC-AGI-2 Gap"
subtitle: "ARC Prize 2026 — Paper Track submission"
---

## Abstract

We present a CPU-only ARC solver built around one principle: **choose the shortest consistent
explanation, and say when there is none.** The solver searches over grid-level transforms and then
induces *local rules*. A local rule is a lookup table from a small tuple of per-cell features
(own colour, neighbours, rays, symmetry partners, object statistics, grid-level context) to an
output label. Candidate rules are scored by description length (DL). Predictions from all
consistent rules are pooled by Bayesian model averaging with weights ∝ 2^−DL. Three mechanisms
make the rule class generalise beyond memorisation:

1. **target re-encodings** (*absolute*, *keep-unless-changed* and *copy-from-feature*);
2. **monotone interval interpolation**, so rules keyed on ordinal features stay defined for
   values never seen in training;
3. a **compression gate** that rejects rules which do not compress the observed changes.

We also **meta-learn the DL prior over features** from the public training set, validated with
cross-validation.

Our main contribution is diagnostic. We separate *expressibility* (a compact rule exists, found by
an oracle that also sees the test outputs) from *learnability* (the rule is recovered from the
training pairs alone). On the 1,000 ARC-AGI-2 training tasks, compact local rules express
**RESULT_EXPR_TRAIN** tasks, and the solver learns **RESULT_LEARN_RATE** of those. On the 120
public evaluation tasks, the same rule class expresses **only RESULT_EXPR_EVAL**. The ARC-AGI-2
evaluation set is therefore not merely harder to *learn*: it is almost entirely outside the
*expressible* region of per-cell rules, even with a rich feature bank and grid-level context. We
use this to argue, quantitatively, where program-synthesis research effort should go.

## 1. Introduction

ARC-AGI asks a solver to infer a grid transformation from two to five demonstration pairs. Two
families dominate the literature. Neural methods use test-time-trained LLMs or recursive networks.
Symbolic methods search a DSL. Both families improve on ARC-AGI-1 and struggle on ARC-AGI-2. The
reason is usually described qualitatively: ARC-AGI-2 tasks are "more compositional".

We make this quantitative for one well-defined, interpretable hypothesis class: **local rules**.
A local rule states that each output cell is a function of a few features of the corresponding
input cell (after an optional global transform). The class is small enough to enumerate
exhaustively. It is rich enough to cover recolouring by object size, symmetry completion, hole
filling, line extension between markers, parity patterns, and conditioning on a "key" object
elsewhere in the grid. Because the class can be enumerated, we can answer two separate questions
for every task:

* **Expressibility.** Does *any* compact rule reproduce the train and test outputs? We answer this
  with an oracle that is given the test outputs.
* **Learnability.** Does MDL selection, from the train pairs alone, pick a rule that is right
  on the test?

The difference between the two is the *induction gap*. Tasks outside the expressible region
measure the *representation gap*. Keeping the two apart tells us whether to improve priors (the
induction gap) or the hypothesis class (the representation gap).

## 2. Method

### 2.1 Transform stage
A library of about 110 parameter-free grid transforms covers the following:

* rotations and reflections;
* crops to the non-background bounding box or to a selected object (largest, smallest, unique
  colour, unique shape, densest, most colours);
* integer up- and down-scaling;
* tilings and mirror tilings;
* a self-Kronecker product;
* separator-aware splitting into panels, combined by count, bit-code, AND, OR, XOR, first/last
  overlay, or selection of the most/least/unique panel;
* per-cell summaries of separator grids;
* de-duplication of rows and columns.

A transform is *exact* if it maps every train input to its output. Otherwise, if it produces
output-shaped grids, it is kept as the substrate for rule induction. Depth-2 compositions are
checked for exactness only. The six substrates closest to the targets (by pixel accuracy) are
passed to the rule stage.

### 2.2 Feature bank
For every cell we compute about 70 integer features:

* **local:** own colour, the 8 neighbours, first non-background colour along each of 8 rays,
  horizontal/vertical "between two same-coloured cells", 4- and 8-neighbour counts, same-colour
  neighbours;
* **geometric:** row and column indices from each edge, index mod 2 and 3, border and diagonal
  membership, mirror partners (horizontal, vertical, both, transpose, anti-transpose), dominant
  colour of the row and of the column;
* **object:** for single-colour 4-connected objects and multicolour 8-connected objects: size,
  size rank from the top and from the bottom, border contact, number of colours, minority
  ("marker") colour, enclosure;
* **grid context:** number of objects, rarest and commonest colour, colour and shape hash of the
  smallest and largest object;
* **frequency:** frequency rank of the own colour.

### 2.3 Rule induction under MDL
For a feature subset *S* and a target encoding *E*, the rule is the table key_S(cell) → label_E(cell).
It is *consistent* if no key maps to two labels across all training cells. The target encodings are:

* **abs**: the output colour;
* **keep**: `KEEP` if the cell is unchanged, else its new colour;
* **copy_f**: `KEEP` if unchanged, else `COPY`, meaning "take the colour of feature *f*". This is
  offered only when *f* explains every changed cell.

We enumerate all singletons and pairs exhaustively. We extend the 24 least-impure pairs to
triples (beam search). The description length of a rule is

  DL = |table| + Σ_{f∈S} cost(f) + cost(E) + 4·#unseen + interp_cost.

Here #unseen counts test keys that the table does not cover (they default to `KEEP`).
interp_cost is charged per key resolved by interval interpolation (§2.4). Rules with more than
40 table entries fail the **compression gate**. They are kept only as a last resort.

### 2.4 Monotone interval interpolation
Object sizes, counts and ranks in the test grid are often values never seen in training. For an
unseen test key, we look for seen keys that match it on all positions except one ordinal
feature. If the nearest seen values below and above agree on the label, the key inherits that
label (cost 1). One-sided extrapolation costs 2. Conflicts leave the key unresolved.

### 2.5 Bayesian model averaging
Every consistent rule, on every substrate, votes for its predicted test grids with log-weight
−DL/2 − cost(transform). Identical predictions pool their evidence with log-sum-exp. The two
highest-scoring distinct predictions are submitted as `attempt_1` and `attempt_2`.

### 2.6 Meta-learned feature prior
The uniform cost(f) = 1.5 is replaced by cost(f) = −log₂ p(f) / log₂ 10, measured in
table-entry units. p(f) is the Laplace-smoothed frequency of *f* in the minimal-DL oracle rules
of held-in tasks. We report 2-fold cross-validation on the training set, so the prior is never
estimated from the tasks it is evaluated on.

## 3. Experiments

RESULTS_SECTION

## 4. Discussion

DISCUSSION_SECTION

## 5. Limitations

* The rule class is per-cell. It cannot express counting-to-output-size, object re-arrangement,
  or multi-step programs unless one of the substrate transforms already does the non-local work.
* The shape hashes in the context features are local to a grid. Matching shapes *across*
  demonstrations works by hash equality, which is brittle to rotation.
* Accuracy on ARC-AGI-2 is low in absolute terms. We make no claim that this approach is
  competitive with test-time-trained LLMs. Our claim is about *where* the difficulty lies.

## 6. Reproducibility

All code is open source (Apache-2.0) in this repository. The whole pipeline runs on four CPU
cores in under ten minutes for the 1,000 training tasks.

* `evaluate.py` scores any task directory.
* `analysis_oracle.py` produces the expressibility/learnability numbers.
* `learn_prior.py` produces the feature prior.
* `kaggle/arc_mdl_rules.ipynb` is the offline Kaggle submission (no internet, no GPU, no
  pretrained weights).
