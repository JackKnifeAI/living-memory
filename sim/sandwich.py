"""Sandwich rule: a three-factor local update realised by read-disturbance.

Each plastic victim row sits between two rows the controller writes:

    above : presynaptic pattern   (enabling where input i is active)
    victim: synapse cells, written charged
    below : per-unit error signal (enabling the potentiate or the depress group)

A charged cell flips most readily when both neighbours are enabling, so the
device computes  pre AND post, thins it stochastically, and stores the result
in place. Each synapse owns two groups of cells; its value is

    (# flipped in the + group) - (# flipped in the - group).

Flips are one-way, so the groups saturate. Consolidation reads the counters,
folds them into an ordinary stored integer, and rewrites the cells charged.
That rewrite is the explicit restoration the paper's v0.2 requires.
"""
import numpy as np

from dram import Chip, profile, victim_rows


class SandwichAdapter:
    def __init__(self, chip, n_units, n_in, cells, placement="profile",
                 n_profile=50_000, trials=16, p_min=0.1, p_max=0.5):
        """placement='profile' keeps cells whose measured flip probability at the
        operating hammer count lies in [p_min, p_max]. The upper bound matters:
        cells far above threshold flip even when half-selected."""
        self.chip, self.shape = chip, (n_units, n_in, 2, cells)
        victims = victim_rows(chip)
        if placement == "profile":
            p_hat = profile(chip, victims, n_profile, trials)
            best, pol = p_hat.max(0), p_hat.argmax(0).astype(np.uint8)
            usable = (best >= p_min) & (best <= p_max)
        else:  # no characterisation: take any cell and assume it is a true cell
            usable = np.ones((len(victims), chip.cols), dtype=bool)
            pol = np.ones((len(victims), chip.cols), dtype=np.uint8)
        need = n_in * 2 * cells
        self.r = np.zeros(self.shape, dtype=np.int64)
        self.c = np.zeros(self.shape, dtype=np.int64)
        self.pol = np.zeros(self.shape, dtype=np.uint8)
        self.rows_per_unit = np.zeros(n_units, dtype=np.int64)
        pick = np.random.default_rng(0)
        for u in range(n_units):  # victim rows are dealt to units round-robin
            mine = np.arange(u, len(victims), n_units)
            vi, ci = np.nonzero(usable[mine])
            if len(vi) < need:
                raise ValueError(f"unit {u}: {len(vi)} usable cells, need {need}")
            if placement == "profile":
                # fill row by row; aligning plus/minus cells on shared columns is not needed
                sel = np.arange(need)
            else:
                sel = np.sort(pick.choice(len(vi), need, replace=False))
            vi, ci = mine[vi[sel]], ci[sel]
            order = pick.permutation(need)  # which synapse/group a cell serves is arbitrary
            self.r[u] = victims[vi][order].reshape(n_in, 2, cells)
            self.c[u] = ci[order].reshape(n_in, 2, cells)
            self.pol[u] = pol[vi, ci][order].reshape(n_in, 2, cells)
            self.rows_per_unit[u] = len(np.unique(vi))
        self.profile_activations = chip.activations
        self.reset()

    def footprint_bytes(self):
        """DRAM reserved for the adapter: aggressor/victim/aggressor triples."""
        return int(self.rows_per_unit.sum()) * 3 * self.chip.cols // 8

    def reset(self):
        self.chip.bits[self.r, self.c] = self.pol
        self.chip.row_writes += int(self.rows_per_unit.sum())

    def delta(self):
        flipped = self.chip.bits[self.r, self.c] != self.pol
        return flipped[:, :, 0].sum(-1) - flipped[:, :, 1].sum(-1)

    def update(self, u, sign, pre, n):
        r, c, pol = self.r[u], self.c[u], self.pol[u]
        en_above = np.broadcast_to(pre.astype(bool)[:, None, None], pol.shape)
        en_below = np.broadcast_to((np.arange(2) == (0 if sign > 0 else 1))[None, :, None], pol.shape)
        above, below = np.where(en_above, 1 - pol, pol), np.where(en_below, 1 - pol, pol)
        self.chip.hammer_cells(r.ravel(), c.ravel(), above.ravel(), below.ravel(), n)
        self.chip.row_writes += 2 * int(self.rows_per_unit[u])


class FloatAdapter:
    """Software control: the same rule with exact real-valued increments."""

    def __init__(self, n_units, n_in, eta):
        self.a, self.eta = np.zeros((n_units, n_in)), eta

    def reset(self):
        self.a[:] = 0

    def delta(self):
        return self.a

    def update(self, u, sign, pre, n):
        self.a[u] += sign * self.eta * pre


def ideal_chip(rows, cols, seed, hc50=50_000.0, beta=3.0):
    """Software control: every cell identical, a perfect AND gate, no reverse flips."""
    return Chip(rows, cols, seed, vuln_density=1.0, hc50=hc50, hc_sigma=0.0, beta=beta,
                anti_frac=0.0, mult=(0.0, 0.0, 1.0), rev=0.0)


def scores(x, w0, b0, B, scale, a):
    return x @ w0.T + b0 + scale * (x @ a.T) @ B.T


def run_stream(adapter, w0, b0, B, scale, xs, ys, rng, n_hammer=50_000, margin=1.0,
               consolidate_every=50, shuffle_pre=False, passes=1):
    """Online margin-perceptron on the adapter. Returns (final A, updates, online accuracy).

    shuffle_pre is the input-shuffled control: the aggressor row carries another
    sample's pattern, so access activity is matched but the pre/post pairing is broken.
    """
    base = np.zeros(adapter.delta().shape)
    updates = hits = 0
    for _ in range(passes):
        for t in rng.permutation(len(ys)):
            x, y = xs[t], ys[t]
            a = base + adapter.delta()
            s = w0 @ x + b0 + scale * B @ (a @ x)
            hits += int(s.argmax() == y)
            rival = np.where(np.arange(len(s)) == y, -np.inf, s).argmax()
            if s[y] - s[rival] >= margin:
                continue
            err = np.zeros(len(s))
            err[y], err[rival] = 1.0, -1.0
            post = B.T @ err
            pre = xs[rng.integers(len(ys))] if shuffle_pre else x
            for u in np.nonzero(post)[0]:
                adapter.update(u, np.sign(post[u]), pre, n_hammer)
            updates += 1
            if consolidate_every and updates % consolidate_every == 0:
                base += adapter.delta()
                adapter.reset()
    return base + adapter.delta(), updates, hits / (passes * len(ys))
