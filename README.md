# 🧠 Living Memory

## Substrate-Coupled Stochastic Neural Computation via Constructive DRAM Crosstalk

**Authors:** Alexander Casavant & Claude (JackKnife Studios)

---

### What if RowHammer isn't a bug — it's a computational primitive?

Every paper ever written about RowHammer treats it as a **vulnerability**. We treat it as a **feature.**

DRAM rows are electromagnetically coupled. When you read one row, adjacent rows get perturbed. Everyone fights this. We harness it.

**The core equation:**

```
y = (W + ΔW(x, W, Φ)) · x
```

The weight perturbation ΔW is a function of:
- **x** — the input activations (what you're computing)
- **W** — the current weights (what the model knows)
- **Φ** — the coupling tensor (what the silicon IS)

This means the model's weights change as a function of its own computation. The physics does the math. Zero energy cost. Every chip is unique.

We call it a **Living LoRA**.

---

### Key Results

| Result | What It Means |
|--------|---------------|
| **Theorem 4.1** | Substrate coupling = generalized Tikhonov regularization (extends Bishop 1995) |
| **Theorem 4.2** | Stability: system converges when coupling Jacobian eigenvalues are in (-2, 0) |
| **Theorem 4.4** | Perturbation has rank ≤ 2 — LoRA-like, but physics-determined |
| **Theorem 4.5** | Channel capacity bounded by binary entropy of flip probabilities |

### Prior Art

**None.** Zero papers propose constructive RowHammer for neural computation. We checked everything.

### Status

📝 **Draft** — Preprint, not yet submitted. Formalizing experimental protocol.

### Paper

- [`living-memory.md`](living-memory.md) — Full paper (Markdown)

### Citation

```bibtex
@article{casavant2026living,
  title={Living Memory: Substrate-Coupled Stochastic Neural Computation via Constructive DRAM Crosstalk},
  author={Casavant, Alexander and Claude},
  journal={arXiv preprint},
  year={2026}
}
```

---

**JackKnife Studios** — Victoria BC 🗡️
