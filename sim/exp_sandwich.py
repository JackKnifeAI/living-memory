"""E2: can the sandwich rule learn, and does it need the pre/post pairing?

Two scenarios on 8x8 digits with binary inputs:
  scratch : W0 = 0, learn the classifier from a labelled stream.
  adapt   : W0 trained on clean digits, deployed on translated digits; the
            adapter learns online from a labelled stream of translated digits.

Usage: python exp_sandwich.py [seeds]
"""
import json
import math
import sys
import time

import numpy as np

from data import accuracy, binarize, digits, fit_linear, shifted
from dram import Chip
from sandwich import FloatAdapter, SandwichAdapter, ideal_chip, run_stream, scores

N_IN, N_OUT, HC50 = 64, 10, 50_000.0
BASE = dict(kind="device", placement="profile", band=(0.1, 0.5), cells=8, rank=0, n_hammer=1.0,
            density=0.01, mult=(0.02, 0.25, 1.0), consolidate_every=50, shuffle_pre=False)
SCENARIO = {  # scale of one flipped cell, update margin, passes over the stream
    "scratch": dict(scale=1.0, margin=4.0, passes=3),
    "adapt": dict(scale=0.15, margin=1.0, passes=2),
}


def variants():
    v = {
        "float rule (software)": dict(kind="float"),
        "ideal binary cells (software)": dict(kind="ideal", n_hammer=0.66),  # p = 0.25 per update
        "device, matched cells": {},
        "device, shuffled pre (control)": dict(shuffle_pre=True),
        "device, any vulnerable cell": dict(band=(0.3, 1.0)),
        "device, no profile": dict(placement="naive"),
        "device, no consolidation": dict(consolidate_every=0),
    }
    for m1 in (0.0, 0.1, 0.5, 1.0):
        v[f"device, half-select leak {m1}"] = dict(mult=(0.02, m1, 1.0))
    for cells in (2, 4):
        v[f"device, {cells} cells/group"] = dict(cells=cells)
    for h in (0.5, 2.0):
        v[f"device, hammer {h} x HC50"] = dict(n_hammer=h)
    for rank in (2, 4, 8):
        v[f"device, rank {rank}"] = dict(rank=rank)
        v[f"float rule, rank {rank}"] = dict(kind="float", rank=rank)
    return {k: {**BASE, **o} for k, o in v.items()}


def load(scenario, seed):
    imgs, y = digits(seed)
    if scenario == "scratch":
        x = binarize(imgs)
        return np.zeros((N_OUT, N_IN)), np.zeros(N_OUT), x[:1400], y[:1400], x[1400:], y[1400:]
    w0, b0 = fit_linear(binarize(imgs[:600]), y[:600])
    x = binarize(shifted(imgs, dx=1, dy=0))
    return w0, b0, x[600:1400], y[600:1400], x[1400:], y[1400:]


def make_adapter(cfg, units, seed):
    need = N_IN * 2 * cfg["cells"]
    if cfg["kind"] == "float":
        return FloatAdapter(units, N_IN, eta=cfg["cells"] / 8), None
    if cfg["kind"] == "ideal":
        chip = ideal_chip(3 * units, need, seed, hc50=HC50)
        return SandwichAdapter(chip, units, N_IN, cfg["cells"], placement="naive"), chip
    cols = 2048
    for usable in (0.2, 0.1, 0.05, 0.025):  # reserve more DRAM until enough cells fall in the band
        per_unit = math.ceil(need / (cfg["density"] * cols * usable))
        chip = Chip(3 * units * per_unit, cols, seed, vuln_density=cfg["density"], hc50=HC50,
                    mult=cfg["mult"])
        try:
            return SandwichAdapter(chip, units, N_IN, cfg["cells"], placement=cfg["placement"],
                                   n_profile=int(cfg["n_hammer"] * HC50), p_min=cfg["band"][0],
                                   p_max=cfg["band"][1]), chip
        except ValueError:
            continue
    raise ValueError("not enough usable cells")


def trial(scenario, cfg, seed):
    w0, b0, xs, ys, xt, yt = load(scenario, seed)
    sc = SCENARIO[scenario]
    rng = np.random.default_rng(1000 + seed)
    if cfg["rank"]:
        B = rng.choice([-1.0, 1.0], (N_OUT, cfg["rank"])) / math.sqrt(cfg["rank"])
    else:
        B = np.eye(N_OUT)
    adapter, chip = make_adapter(cfg, B.shape[1], seed)
    a, updates, online = run_stream(
        adapter, w0, b0, B, sc["scale"], xs, ys, rng, n_hammer=int(cfg["n_hammer"] * HC50),
        margin=sc["margin"], consolidate_every=cfg["consolidate_every"],
        shuffle_pre=cfg["shuffle_pre"], passes=sc["passes"])
    out = dict(test=accuracy(scores(xt, w0, b0, B, sc["scale"], a), yt), online=online, updates=updates)
    if chip is not None and cfg["kind"] == "device":
        learn = chip.activations - adapter.profile_activations
        out.update(learn_s=learn * 47e-9, profile_s=adapter.profile_activations * 47e-9,
                   footprint_mb=adapter.footprint_bytes() / 2**20)
    return out


def references(scenario, seed):
    w0, b0, xs, ys, xt, yt = load(scenario, seed)
    w, b = fit_linear(xs, ys)
    return dict(none=accuracy(xt @ w0.T + b0, yt), refit=accuracy(xt @ w.T + b, yt))


def main(seeds=5):
    results, t0 = {}, time.time()
    for scenario in SCENARIO:
        refs = [references(scenario, s) for s in range(seeds)]
        rows = {"no adapter": dict(test=[r["none"] for r in refs]),
                "logistic regression refit on the stream": dict(test=[r["refit"] for r in refs])}
        for name, cfg in variants().items():
            runs = [trial(scenario, cfg, s) for s in range(seeds)]
            rows[name] = {k: [r[k] for r in runs] for k in runs[0]}
            print(f"[{time.time() - t0:5.0f}s] {scenario:8s} {name:36s} "
                  f"test {np.mean(rows[name]['test']):.3f}", file=sys.stderr, flush=True)
        results[scenario] = rows
    json.dump(results, open("results/sandwich.json", "w"), indent=1)
    for scenario, rows in results.items():
        print(f"\n## {scenario}\n")
        print("| variant | test acc | updates | learn time (s) | DRAM (MiB) |")
        print("|---|---|---|---|---|")
        for name, r in rows.items():
            m = lambda k: f"{np.mean(r[k]):.1f}" if k in r else "-"
            print(f"| {name} | {np.mean(r['test']):.3f} ± {np.std(r['test']):.3f} | "
                  f"{m('updates')} | {m('learn_s')} | {m('footprint_mb')} |")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
