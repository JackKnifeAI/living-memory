"""E3: chip binding. The stored weights are XORed with a response that only the
enrolled chip produces under a fixed hammer recipe. This is a RowHammer PUF
(Schaller et al. 2017) applied to model weights, close to EIM-TRNG (2025); it is
included to measure how well the idea survives stochastic cells and drift, not
as a new primitive.

Usage: python exp_binding.py [seeds]
"""
import json
import sys

import numpy as np

from data import accuracy, digits, fit_linear
from dram import Chip, from_bits, profile, to_bits, victim_rows

HC50, ROWS, COLS, RECIPE = 50_000.0, 3 * 400, 2048, 100_000


def response(chip, victims, pol, pos, n, votes=1):
    """Write the enrolment pattern, hammer, and read which key cells flipped."""
    total = np.zeros(len(pos[0]))
    for _ in range(votes):
        chip.write(victims, pol)
        chip.write(victims - 1, 1 - pol)
        chip.write(victims + 1, 1 - pol)
        total += chip.hammer(victims, n)[pos]
    return (total * 2 > votes).astype(np.uint8)


def run(seed):
    rng = np.random.default_rng(seed)
    imgs, y = digits(seed)
    x = imgs.reshape(len(imgs), -1) / 16.0
    w, b = fit_linear(x[:1400], y[:1400])
    s = np.abs(w).max() / 127
    wbits = to_bits(np.round(w / s).astype(np.int64)).reshape(-1)
    acc = lambda bits: accuracy(s * x[1400:] @ from_bits(bits.reshape(10, 64, 8)).T + b, y[1400:])

    home, other = Chip(ROWS, COLS, seed), Chip(ROWS, COLS, seed + 100)
    victims = victim_rows(home)
    p = profile(home, victims, RECIPE, 16)
    pol = np.repeat(p.sum(2).argmax(0)[:, None], COLS, axis=1).astype(np.uint8)  # row polarity
    p = np.take_along_axis(p, pol[None], axis=0)[0]
    half = len(wbits) // 2
    ones, zeros = np.argwhere(p >= 1.0), np.argwhere(p == 0.0)
    pick = np.concatenate([ones[rng.choice(len(ones), half, replace=False)],
                           zeros[rng.choice(len(zeros), half, replace=False)]])
    pick = pick[rng.permutation(len(pick))]
    pos = (pick[:, 0], pick[:, 1])
    key = (p[pos] >= 1.0).astype(np.uint8)
    stored = wbits ^ key

    aged = Chip(ROWS, COLS, seed)  # the same chip after its thresholds drift by ~15%
    aged.hc *= np.exp(0.15 * rng.standard_normal(aged.hc.shape))
    cases = {
        "home chip, 1 shot": response(home, victims, pol, pos, RECIPE),
        "home chip, 5 votes": response(home, victims, pol, pos, RECIPE, 5),
        "home chip, 20% fewer hammers": response(home, victims, pol, pos, int(0.8 * RECIPE)),
        "home chip, thresholds drifted 15%": response(aged, victims, pol, pos, RECIPE),
        "different chip": response(other, victims, pol, pos, RECIPE),
        "no chip (all-zero response)": np.zeros_like(key),
    }
    out = {"plaintext": dict(key_ber=0.0, acc=acc(wbits)),
           "ciphertext read directly": dict(key_ber=float(key.mean()), acc=acc(stored))}
    for name, r in cases.items():
        out[name] = dict(key_ber=float((r != key).mean()), acc=acc(stored ^ r))
    return out


def main(seeds=5):
    runs = [run(s) for s in range(seeds)]
    res = {k: {m: float(np.mean([r[k][m] for r in runs])) for m in ("key_ber", "acc")} for k in runs[0]}
    json.dump(res, open("results/binding.json", "w"), indent=1)
    print("| condition | key bit error | test acc |\n|---|---|---|")
    for k, v in res.items():
        print(f"| {k} | {v['key_ber']:.4f} | {v['acc']:.3f} |")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
