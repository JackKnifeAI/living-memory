"""Behavioural model of DRAM read-disturbance (RowHammer) for the Living Memory experiments.

This is a statistical model, not a circuit model. It encodes four published
tendencies and nothing else:

  1. Only a small fraction of cells are vulnerable at a given hammer count, the
     same cells flip repeatably, and their thresholds vary widely (Kim 2014, 2020).
  2. A cell flips from its charged state to its discharged state. True cells are
     charged at logical 1, anti cells at logical 0. Reverse flips are rare.
  3. A charged cell flips most readily when the cells directly above and below it
     hold the opposite value, and less readily otherwise (SoK 2201.02986 sec. 2.2;
     Ji et al., Pinpoint Rowhammer, 2019).
  4. Ordinary refresh does not undo a flip. Only an explicit rewrite does.

Every number below is a free parameter of the simulation, not a measurement.
"""
import numpy as np

T_ACT_NS = 47.0  # rough time per row activation; used only for wall-clock estimates


class Chip:
    def __init__(self, rows, cols, seed, vuln_density=0.01, hc50=50_000.0, hc_sigma=0.6,
                 beta=3.0, anti_frac=0.25, mult=(0.02, 0.25, 1.0), rev=0.001):
        """mult[k] scales the flip hazard when k of the two vertical neighbours
        hold the opposite value. rev scales the hazard of a discharged cell
        flipping back (the 'most, but not all' in the literature)."""
        rng = np.random.default_rng(seed)
        self.rows, self.cols = rows, cols
        self.beta, self.rev = beta, rev
        self.mult = np.asarray(mult, dtype=np.float64)
        vulnerable = rng.random((rows, cols)) < vuln_density
        hc = hc50 * np.exp(hc_sigma * rng.standard_normal((rows, cols)))
        self.hc = np.where(vulnerable, hc, np.inf).astype(np.float32)
        # cell polarity is laid out per row: charged value 1 = true cell, 0 = anti cell
        self.charged = np.repeat((rng.random((rows, 1)) >= anti_frac).astype(np.uint8), cols, axis=1)
        self.bits = np.zeros((rows, cols), dtype=np.uint8)
        self.rng = np.random.default_rng(seed + 7919)  # sampling noise, separate from identity
        self.activations = 0
        self.row_writes = 0

    # -- ordinary, non-disturbing accesses ---------------------------------
    def write(self, rows, bits):
        self.bits[rows] = bits
        self.row_writes += np.size(rows)

    def read(self, rows):
        return self.bits[rows].copy()

    # -- disturbance --------------------------------------------------------
    def _flip_prob(self, hc, charged, v, above, below, n):
        k = (above != v).astype(np.int8) + (below != v).astype(np.int8)
        hazard = (n / hc.astype(np.float64)) ** self.beta * self.mult[k]
        hazard = np.where(v == charged, hazard, hazard * self.rev)
        return -np.expm1(-hazard)

    def hammer(self, victims, n):
        """Double-sided hammer: activate rows v-1 and v+1 n times each.
        Their stored data are the aggressor patterns. Returns the flip mask."""
        victims = np.atleast_1d(victims)
        i, c = np.nonzero(np.isfinite(self.hc[victims]))  # only vulnerable cells can flip
        r = victims[i]
        v = self.bits[r, c]
        p = self._flip_prob(self.hc[r, c], self.charged[r, c], v,
                            self.bits[r - 1, c], self.bits[r + 1, c], n)
        hit = self.rng.random(p.shape) < p
        flips = np.zeros((len(victims), self.cols), dtype=np.uint8)
        flips[i[hit], c[hit]] = 1
        self.bits[r[hit], c[hit]] ^= 1
        self.activations += 2 * n * len(victims)
        return flips

    def hammer_cells(self, r, c, above, below, n):
        """Same physics as hammer(), restricted to the listed cells, with the
        aggressor bits for those columns supplied directly. Used when a
        controller writes the aggressor rows and only some cells are read back.
        Activations are charged for every distinct victim row touched."""
        v = self.bits[r, c]
        p = self._flip_prob(self.hc[r, c], self.charged[r, c], v, above, below, n)
        flips = (self.rng.random(p.shape) < p).astype(np.uint8)
        self.bits[r, c] = v ^ flips
        self.activations += 2 * n * len(np.unique(r))
        return flips

    def seconds(self):
        return self.activations * T_ACT_NS * 1e-9


def victim_rows(chip):
    """Rows laid out as aggressor / victim / aggressor triples."""
    return np.arange(1, chip.rows - 1, 3)


def profile(chip, victims, n, trials):
    """Estimate per-cell flip probability under the worst-case pattern for each
    stored polarity. Returns p_hat[polarity, victim, col]. The cost of the
    campaign lands on the chip's counters like any other use."""
    p_hat = np.zeros((2, len(victims), chip.cols))
    for pol in (0, 1):
        vict = np.full((len(victims), chip.cols), pol, dtype=np.uint8)
        for _ in range(trials):
            chip.write(victims, vict)
            chip.write(victims - 1, 1 - vict)
            chip.write(victims + 1, 1 - vict)
            p_hat[pol] += chip.hammer(victims, n)
    return p_hat / trials


# -- quantised value decoding (paper v0.2, sec. 2.3) -------------------------
def to_bits(q, nbits=8):
    """Two's-complement integers -> bits, least significant first, appended as a last axis."""
    u = np.asarray(q).astype(np.int64) & ((1 << nbits) - 1)
    return np.ascontiguousarray(((u[..., None] >> np.arange(nbits)) & 1).astype(np.uint8))


def from_bits(bits, signed=True):
    nbits = bits.shape[-1]
    u = (bits.astype(np.int64) << np.arange(nbits)).sum(-1)
    return np.where(u >= 1 << (nbits - 1), u - (1 << nbits), u) if signed else u
