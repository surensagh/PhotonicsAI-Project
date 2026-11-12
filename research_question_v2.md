# Research question and pre-registered hypotheses — v2

v1 committed 2026-11-12 before any results existed. v2 committed 2026-11-13: identical to v1 except the amendment at the end.
Rule: v1 and v2 are never edited. Further changes go into a dated v3 with a reason.

## Research question

Can a physically realizable optical nonlinear feature transformation (phase encoding → coherent
propagation → intensity detection) improve the low-data downstream performance of small frozen
pretrained language models, compared with the raw representation and with digital transformations
matched in output dimension, trainable parameters and tuning budget?

## Primary endpoint (the single claim the project stands on)

- **Task:** SciQ, multiple-choice accuracy on the locked 1,000-question test set.
- **Model:** Pythia-160M (frozen), final-layer mean-pooled representation.
- **Regime:** N = 100 labeled training questions, 10 paired subsamples × 5 seeds.
- **Comparison:** optical map F (D = 4d) + linear head versus the best of {RFF, quadratic random map}
  at the same D + linear head.
- **Test:** paired bootstrap over test questions and subsamples, two-sided, α = 0.05.
- **Success:** mean difference > 0 with the 95% CI excluding 0.

The primary model is Pythia-160M rather than GPT-2 Small because Pythia offers a controlled size
ladder for E6. GPT-2 Small stays as the debugging sandbox.

## Hypotheses

| ID | Hypothesis (stated so it can fail) | Tested in | Falsified if |
|---|---|---|---|
| H1 Performance | F + linear head beats raw h + linear head | E2, E5 | CI of (F − identity) includes 0 at N = 100 |
| H2 Optical specificity | F beats matched digital random features (RFF, quadratic, tanh) | E2, E5 | F within CI of the best digital map |
| H3 Data efficiency | The gain (F − best baseline) shrinks as N grows | E5 | Gain at N = 50 not larger than at full data |
| H4 Small-model benefit | The gain shrinks as model size grows | E6 | Slope of gain vs log(params) ≥ 0 |
| H5 Mechanism | Gains track increases in kernel–label alignment and class separability | E10 | Spearman ρ between alignment change and accuracy change < 0.3 |
| H6 Physical robustness | The gain survives realistic noise, quantization and drift | E8 | Gain vanishes at noise levels measured on the real setup |
| H7 Generality | The effect appears on ≥ 2 of 3 tasks | E9 | Effect on SciQ only |
| H8 Hardware transfer | Real SLM–camera features reproduce simulated behaviour | E13 | Hardware accuracy below raw-h baseline |

H2 is the hypothesis that makes this research rather than engineering. If H1 holds but H2 fails, the
honest conclusion is "an optically implementable random-feature map performs like its digital
equivalent". That is still a publishable, defensible result, because it means the optical hardware
could perform this step physically.

## Amendment v2 (2026-11-13): phase encoding of the optical map F

**Change.** The phase encoding of F (protocol 3.4) changes from

    v1:  phi = beta * (tanh(h) + 1) / 2,        knob beta in {pi/4, pi/2, pi, 2 pi}

to

    v2:  phi = beta * (tanh(h / tau) + 1) / 2,  beta = 2 pi fixed (full SLM phase range),
                                                knob tau in {1, 4, 16, 64}

for both F-scatter and F-fourier. The optics are unchanged (same transmission matrix U / far-field
pixels, same map seeds). The tuning budget is unchanged: 4 knob values x 7 lambda = 28 configurations.
tau = 1 reproduces v1 at beta = 2 pi. The "untuned" control row becomes tau = 4 (fixed in advance).
The primary endpoint, hypotheses, data, splits, heads, statistics and gates are otherwise unchanged;
"optical map F" in the primary endpoint now means F with the v2 encoding.

**Reason (evidence, all recorded before this amendment).** Under v1, E2 did not pass gate G2.
An exploratory decomposition (`scripts/diag_linear_limit.py`, deviations.md 2026-11-12) showed that
in its small-beta limit F matches its theoretical linear limit exactly (46.1% vs 46.3% dev accuracy,
full data), so the optics simulation is correct, and that the entire ~6-point loss relative to the raw
representation comes from tanh squashing of the standardized features (h: 51.7%, tanh(h): 46.1%);
the global-phase invariance of intensity detection costs nothing (51.7% -> 51.7%). Dividing by tau
before the tanh keeps the encoding in tanh's near-linear range, which removes that loss while keeping
the phase-encoding physics; the phase depth per standard deviation of h is about pi / tau.

**What stays reported.** The v1 E2 result (Table 2, gate G2 not passed) is reported as-is, alongside
the v2 re-run. E2 under v2 is run with the pre-registered E2 design (same lambda grid, seeds, paired
subsamples, selection rule and statistics); records go to `runs/E2v2/`.

**Predictions written before the v2 run.**
1. At full data, F's accuracy rises with tau and at tau = 64 approaches the raw representation A
   (51.7%), because the encoding becomes nearly linear and no longer squashes.
2. Whether F beats A or the best digital map at N = 100 is not predicted; that is what E2 v2 tests.
