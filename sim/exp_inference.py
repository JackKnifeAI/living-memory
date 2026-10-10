"""E1: the original draft's scheme. INT8 weights sit in victim rows, the input's
bits sit in the rows above and below, and inference-time hammering perturbs the
weights. Does the input-conditioned perturbation help, and does it point along
the negative gradient as the placement algorithm assumed?

Usage: python exp_inference.py [seeds]
"""
import json
import sys

import numpy as np

from data import accuracy, digits, fit_linear
from dram import Chip, from_bits, profile, to_bits, victim_rows

N_IN, N_OUT, HC50 = 64, 10, 50_000.0


def softmax_loss(logits, y):
    z = logits - logits.max(-1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(-1))).mean())


class Deployed:
    def __init__(self, seed, density, placement):
        imgs, y = digits(seed)
        self.x = imgs.reshape(len(imgs), -1) / 16.0
        self.xq = np.minimum(imgs.reshape(len(imgs), -1) * 16, 255).astype(np.int64)
        self.y, self.test = y, np.arange(1400, len(y))
        w, self.b = fit_linear(self.x[:1400], y[:1400])
        self.s = np.abs(w).max() / 127
        self.wq = np.round(w / self.s).astype(np.int64)
        self.chip = Chip(3 * N_OUT, N_IN * 8, seed, vuln_density=density, hc50=HC50)
        self.victims = victim_rows(self.chip)
        self.col = np.tile(np.arange(N_IN * 8), (N_OUT, 1))  # column of weight bit (i, k), row-major
        if placement == "profile":  # most significant bit planes on the least susceptible cells
            p = profile(self.chip, self.victims, int(2 * HC50), 8).max(0)
            by_plane = np.argsort(np.tile(np.arange(8)[::-1], N_IN), kind="stable")  # bit 7 first
            for k in range(N_OUT):
                self.col[k, by_plane] = np.argsort(p[k], kind="stable")
        self.restore()

    def restore(self):
        bits = to_bits(self.wq).reshape(N_OUT, -1)
        row = np.zeros_like(bits)
        np.put_along_axis(row, self.col, bits, axis=1)
        self.chip.write(self.victims, row)

    def weights(self):
        bits = np.take_along_axis(self.chip.read(self.victims), self.col, axis=1)
        return from_bits(bits.reshape(N_OUT, N_IN, 8))

    def logits(self, wq, idx):
        return self.s * self.x[idx] @ wq.T + self.b

    def expose(self, t, n):
        """Store sample t's bits in every aggressor row, then hammer every victim."""
        a = to_bits(self.xq[t]).reshape(-1)
        self.chip.write(self.victims - 1, np.tile(a, (N_OUT, 1)))
        self.chip.write(self.victims + 1, np.tile(a, (N_OUT, 1)))
        return self.chip.hammer(self.victims, n)


def run(seed, density, n, placement):
    rng = np.random.default_rng(seed)
    d = Deployed(seed, density, placement)
    clean = accuracy(d.logits(d.wq, d.test), d.y[d.test])
    out = dict(clean=clean)
    for mode in ("matched", "shuffled"):
        d.restore()
        hits = 0
        for t in d.test:
            src = t if mode == "matched" else rng.choice(d.test)
            d.expose(src, n)
            hits += int(d.logits(d.weights(), [t]).argmax() == d.y[t])
        out[f"{mode}_stream"] = hits / len(d.test)
        out[f"{mode}_final"] = accuracy(d.logits(d.weights(), d.test), d.y[d.test])
        planes = (to_bits(d.weights()) != to_bits(d.wq)).sum((0, 1))  # net changed bits per plane
        out[f"{mode}_flips"] = float(planes.sum())
        if mode == "matched":
            device_planes = planes
    # software control: the same number of flips per bit plane, at random positions
    soft = []
    for _ in range(20):
        bits = to_bits(d.wq).reshape(-1, 8)
        for k in range(8):
            pos = rng.choice(N_OUT * N_IN, min(device_planes[k], N_OUT * N_IN), replace=False)
            bits[pos, k] ^= 1
        soft.append(accuracy(d.logits(from_bits(bits).reshape(N_OUT, N_IN), d.test), d.y[d.test]))
    out["software_final"] = float(np.mean(soft))
    # restored trials: is one exposure's update aligned with the negative loss gradient?
    cos, dloss = [], []
    for t in d.test[:150]:
        d.restore()
        d.expose(t, n)
        dw = (d.weights() - d.wq).astype(float)
        if not dw.any():
            continue
        z = d.logits(d.wq, [t])[0]
        p = np.exp(z - z.max())
        p /= p.sum()
        p[d.y[t]] -= 1
        grad = np.outer(p, d.x[t])
        cos.append(-(dw * grad).sum() / (np.linalg.norm(dw) * np.linalg.norm(grad) + 1e-12))
        dloss.append(softmax_loss(d.logits(d.weights(), [t]), d.y[[t]]) - softmax_loss(z[None], d.y[[t]]))
    out["cos_neg_grad"] = float(np.mean(cos)) if cos else float("nan")
    out["dloss"] = float(np.mean(dloss)) if dloss else 0.0
    out["exposures_with_flips"] = len(cos) / 150
    return out


def main(seeds=5):
    results = {}
    print("| density | hammer/HC50 | placement | clean | device stream | shuffled stream | device final |"
          " software final | flips | cos(-grad) | dloss/exposure |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for density in (0.001, 0.01, 0.1):
        for h in (1.0, 2.0):
            for placement in ("naive", "profile"):
                runs = [run(s, density, int(h * HC50), placement) for s in range(seeds)]
                r = {k: float(np.nanmean([x[k] for x in runs])) for k in runs[0]}
                results[f"{density}/{h}/{placement}"] = r
                print(f"| {density} | {h} | {placement} | {r['clean']:.3f} | {r['matched_stream']:.3f} | "
                      f"{r['shuffled_stream']:.3f} | {r['matched_final']:.3f} | {r['software_final']:.3f} | "
                      f"{r['matched_flips']:.0f} | {r['cos_neg_grad']:+.3f} | {r['dloss']:+.3f} |", flush=True)
    json.dump(results, open("results/inference.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
