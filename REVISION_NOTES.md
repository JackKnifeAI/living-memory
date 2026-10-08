# Living Memory revision notes

## Version 0.2 — October 8, 2026

This revision preserves the central JackKnifeAI research question: can characterized physical memory effects contribute useful, input-conditioned neural computation? It turns the initial proposal into an explicit model and experimental program while correcting claims that did not follow from the original assumptions.

The [original manuscript](living-memory.md) remains byte-identical to the file at commit [`1aee254913f2375255346ec12b9ce7d4e1ca9ea8`](https://github.com/JackKnifeAI/living-memory/tree/1aee254913f2375255346ec12b9ce7d4e1ca9ea8). The [revised manuscript](living-memory-v0.2.md) is a separate file; no historical commit is rewritten.

| Original statement or modeling choice | Revision and reason |
|---|---|
| Refresh restores pretrained weights and guarantees a bounded orbit. | Ordinary refresh replenishes the sensed value; it does not recover an already-corrupted bit from a model checkpoint. Add an explicit, verified restoration policy and account for its cost. |
| The perturbation is defined as an expectation, then treated as a random update. | Separate realized updates, conditional means, covariances, and persistent stored state. Model joint transitions and history. |
| Nonzero-mean loss expansion uses covariance alone. | Use the second moment, covariance plus the mean outer product. State smoothness and remainder assumptions. |
| Substrate noise automatically supplies beneficial Tikhonov regularization. | Distinguish a conditional loss expansion from a positive regularizer and from demonstrated training or inference benefit. Correct the scope of the Bishop connection. |
| Stability holds iff every coupling-Jacobian eigenvalue has real part between -2 and 0. | For a fixed-input smooth map, the strict linearization condition is magnitude of 1 plus the eigenvalue less than 1. Complex eigenvalues require a disk condition. Boundary cases need nonlinear analysis. |
| Stable Jacobians for every input imply sequence stability. | Include a counterexample with unstable products of stable matrices. Supply stronger common-contraction assumptions and a bound under persistent noise. |
| At most two neighboring rows imply rank at most two. | Physical locality does not imply low matrix rank. Give a rank-three counterexample to that implication. Propose a factorized adapter that enforces a chosen rank explicitly. |
| Conditional per-cell flip entropy is a channel-capacity bound. | Define input, output, side information, and input-distribution-dependent marginal probabilities. Separate single-use information from channels with history. |
| Unsigned bit decoding is used for signed quantized weights. | State two's-complement decoding and exact pre/post-bit-change differences, including quantization metadata. |
| Higher temperature universally increases disturbance and supplies annealing. | Treat thermal response as an experimentally characterized, cell-dependent relationship. Do not infer a useful annealing schedule. |
| Huge pages guarantee one-bank placement and simple row adjacency. | Require evidence for controller mapping and device-internal topology. Include packing and decoding costs. |
| Perturbations have zero additional energy cost. | Require complete workload measurements and profiling amortization. Distinguish ordinary inference activity from added characterization or control activity. |
| Prior art is absent. | Describe constructive RowHammer, DRAM randomness, processing-in-memory, approximate-memory neural networks, and adaptation precedents. Narrow the research question without claiming an exhaustive priority search. |
| Mathematical claims and proposed experiments are presented as established outcomes. | Label this version as a proposal with analytical statements under explicit assumptions. Report no unperformed experiments or measured benefits. |

## What is added, beyond corrections

1. A conditional transition model with physical layout, control schedule, environment, and memory history.
2. A Living LoRA design with a stable base and factor, plus a substrate-influenced adapter factor. Low rank follows from the representation; usefulness remains to be tested.
3. Restored and persistent experimental modes, matched software controls, and tests that distinguish input dependence from access activity.
4. Quality-matched energy and latency accounting, including profile collection, data conversion, monitoring, and explicit restoration.
5. A proposed JackKnife integration contract for profiles, effect authority, numerical intent, state management, and replay above the deterministic self-hosted baseline.

## Reference and publication corrections

- The current citation identifies a **GitHub research preprint**, not an arXiv paper.
- EIM-TRNG uses its published title and author attribution; MCEL is dated 2026, matching its submission record.
- QUAC-TRNG is distinguished from ordinary adjacent-row RowHammer disturbance.
- Added primary sources for DRAM address mapping and temperature sensitivity.
- Original author attribution is retained. Correspondence uses `JackKnifeAI@proton.me`, consistent with the repository's latest pre-revision contact correction.

Version 0.2 was prepared with Codex assistance at Alexander Casavant's request. Publication records an evolution of the idea, not an empirical validation of the proposed hardware architecture.
