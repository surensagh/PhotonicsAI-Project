# Protocol deviations log

Every departure from the pre-registered protocol (`research_question_v1.md` + the research protocol) is
recorded here with its date, reason and status, before its results are looked at. Exploratory runs
never replace a pre-registered analysis; both are reported.

## 2026-11-12 · E2diag: widened tuning grids (exploratory)

- **What:** E2 re-run as a separate experiment (`configs/e2_diag.yaml`, records in `runs/E2diag/`) with
  λ ∈ {1e-4 … 1e4} (9 values) for every linear map and 6 knob values per knob map, extended toward
  the near-linear end: F β ∈ {π/32, π/16, π/8, π/4, π/2, π}; D s ∈ {1/16 … 2}; H σ/√d ∈ {0.5 … 16}.
  G (trainable MLP) is not re-run.
- **Why:** In the pre-registered E2 (gate G2 not passed) several selected values sat on the edge of
  the protocol grids: F always chose the smallest β (π/4); C and E chose the largest λ (100) at N = 100.
  The E2 interpretation-matrix row "F < A" prescribes checking the β range. Theory gives a testable
  prediction: as β → 0, F reduces to a random linear map of tanh(h), so its accuracy should approach
  the linear maps' accuracy. If it does not, the map loses information (a bug or a real property).
- **Status:** exploratory. The pre-registered E2 result (Table 2, G2 not passed) stands as reported.
  Any later decision that uses these grids (e.g. for E3–E5) needs its own dated entry here.

## 2026-11-12 · E2diag follow-up: linear-limit decomposition (exploratory)

- **What:** `scripts/diag_linear_limit.py` trains the same linear head (E2diag λ grid, dev selection) on
  h, h minus its common-mode direction, tanh(h), tanh(h) minus common mode, F's analytic first-order
  term, and F-scatter at β = π/256.
- **Why:** In E2diag F did not approach the linear maps as β → 0 (full data, D = d: 44.8% vs ~51.5%),
  failing the theory check written before the run. To first order in β, F is a linear map of
  tanh(h) whose rows sum to zero (global-phase invariance), so the loss can come from the tanh
  squashing, from the lost common-mode direction, or from a bug. Each is isolated here.
- **Status:** exploratory; informs debugging and the E3/E4 design, not a confirmatory claim.

## 2026-11-13 · Protocol amendment v2: F's phase encoding (confirmatory going forward)

- **What:** `research_question_v2.md`. F's encoding becomes phi = 2 pi (tanh(h / tau) + 1) / 2 with knob
  tau in {1, 4, 16, 64} (replacing beta); same optics, seeds and 28-configuration budget. E2 is re-run
  under v2 as `E2v2` with the pre-registered E2 design; non-F maps reuse their E2 records (identical
  seeds, grids and data, so re-running them would reproduce the same numbers).
- **Why:** the 2026-11-12 diagnostic located F's loss in the tanh squashing of the encoding, not in the
  optics (details in the v2 amendment).
- **Status:** amendment made before the Dec 26 architecture freeze and before any locked-test use.
  The v1 E2 result stays reported. Later experiments (E3 onward) use the v2 encoding.
