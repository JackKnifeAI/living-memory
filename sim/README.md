# Living Memory simulator (Stage A)

A statistical model of DRAM read-disturbance and three experiments on top of it.
Nothing here is a hardware measurement: every device parameter in `dram.py` is a
free parameter chosen for illustration. The results say what would follow *if* a
chip behaved like the model, and which parameter a hardware campaign must measure.

## Run

Needs numpy and scikit-learn (the task is scikit-learn's bundled 8x8 digits).

```
python test_sim.py          # 8 sanity checks, including the v0.2 counterexamples
python exp_inference.py 5   # E1: the original draft's inference-time perturbation
python exp_binding.py 5     # E3: weights bound to one chip's flip pattern
python exp_sandwich.py 5    # E2: three-factor local learning rule (about 80 minutes)
```

The argument is the number of seeds. Tables are printed; raw numbers go to `results/`.

## Files

- `dram.py`: the device model. One-way flips from the charged state, per-cell
  thresholds, a hazard that depends on how many vertical neighbours hold the
  opposite value, and no restoration except an explicit rewrite.
- `sandwich.py`: the plastic adapter. Victim rows sit between a presynaptic row
  and an error row; a synapse is the difference of two one-way cell counters.
- `exp_*.py`: the experiments. `results/*.md` holds the last five-seed run.

## What the last run showed

- **E1.** The inference-time perturbation has zero alignment with the negative
  loss gradient, matches an input-shuffled control, and only ever costs accuracy.
- **E3.** XORing weights with a chip's flip response recovers 0.973 accuracy on
  the enrolled chip with 5-vote majority and 0.13 on a different chip. It is a
  RowHammer PUF applied to weights, so it is not a new primitive.
- **E2.** The sandwich rule learns (0.59 from scratch, 0.62 adapting to a shifted
  input, against 0.12 and 0.18 for the shuffled control) but trails the same rule
  in software (0.88, 0.86). The gap is set almost entirely by the half-select
  leak `mult[1]`: at 0.1 the device reaches 0.78, at 0.5 it learns nothing. The
  estimated hammering time is 20 to 50 minutes for under a megabyte of adapter.

## Known limits of the model

- `mult` and `rev` are guesses. The literature gives the direction of the
  neighbour effect, not its size.
- Aggressor rows are treated as freely writable and never disturbed themselves.
- Temperature, retention, ECC and in-DRAM mitigations are not modelled.
- Wall-clock figures assume 47 ns per activation and count hammering only.
