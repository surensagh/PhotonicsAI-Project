# Research question and pre-registered hypotheses — v1

Committed: 2026-11-12, before any results exist.
Rule: this file is never edited. Changes go into a dated `research_question_v2.md` with a reason.

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
