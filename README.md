# Living Memory

## Substrate-Coupled Stochastic Neural Computation via Constructive DRAM Crosstalk

**Alexander Casavant & Claude — JackKnife Studios / JackKnifeAI**

Can a characterized memory device contribute useful, input-conditioned perturbations to neural computation?

Living Memory investigates that question through a stochastic model of DRAM disturbance, explicit weight-state management, and controlled neural-network experiments. Its central idea is that the device profile, data layout, inputs, and access history may jointly shape an adaptive computation:

$$
y_t = f(x_t; W_t + D_t).
$$

Here $D_t$ is a realized substrate-dependent perturbation. Whether it improves a task, and at what energy and latency cost, remains an experimental question.

## Current paper

**[Read version 0.2](living-memory-v0.2.md)** — October 8, 2026.

This revision develops the original proposal with corrected mathematics and physical assumptions:

- Explicit checkpoint restoration, separate from ordinary DRAM refresh.
- A conditional expected-loss expansion that includes nonzero-mean perturbations.
- Stability statements with defined assumptions and counterexamples to stronger claims.
- A proposed **Living LoRA** architecture whose factorization enforces low rank.
- Controlled experiments for input dependence, task benefit, energy, and latency.
- A proposed interface to JackKnife's later tensor and living-compiler layers.

**Status:** research proposal and analytical framework. No hardware measurements, simulation results, model-quality gains, or energy savings are reported yet. Publication here is a GitHub preprint, not an arXiv submission or peer-reviewed result.

## Evolution of the idea

- [Version 0.2 — current manuscript](living-memory-v0.2.md)
- [Revision notes — what changed and why](REVISION_NOTES.md)
- [Original draft — preserved unchanged](living-memory.md)

The original draft records the initial idea. Its automatic-refresh-reset, universal rank-two, unconditional stability, and zero-additional-energy claims are superseded by version 0.2. Related work and the scope of the research question are discussed in the current paper.

## Citation

```bibtex
@misc{casavant2026livingmemoryv02,
  title = {Living Memory: Substrate-Coupled Stochastic Neural Computation via Constructive DRAM Crosstalk},
  author = {Casavant, Alexander and {Claude}},
  year = {2026},
  month = {October},
  note = {Version 0.2. GitHub research preprint},
  howpublished = {\url{https://github.com/JackKnifeAI/living-memory/blob/main/living-memory-v0.2.md}}
}
```

**JackKnife Studios — Victoria BC**

Correspondence: JackKnifeAI@proton.me
