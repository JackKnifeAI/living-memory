"""Stage A checks from the paper (v0.2 sec. 6.1). Run as a script: python test_sim.py"""
import numpy as np

from dram import Chip, from_bits, profile, to_bits, victim_rows
from sandwich import SandwichAdapter, ideal_chip


def sure_chip(seed=0, **kw):
    """Every cell vulnerable with the same threshold, so outcomes are easy to predict."""
    return Chip(9, 256, seed, **{**dict(vuln_density=1.0, hc50=1000.0, hc_sigma=0.0, anti_frac=0.0), **kw})


def stripe(chip, victim_bit):
    chip.bits[:] = 1 - victim_bit
    chip.bits[victim_rows(chip)] = victim_bit


def test_twos_complement():
    q = np.arange(-128, 128)
    assert (from_bits(to_bits(q)) == q).all()
    assert (from_bits(to_bits(q), signed=False) == q % 256).all()
    b = to_bits(np.array([5]))
    b[0, 7] ^= 1  # a sign-bit flip moves the value by -128, not +128
    assert from_bits(b)[0] == 5 - 128


def test_zero_disturbance():
    chip = sure_chip()
    stripe(chip, 1)
    assert chip.hammer(victim_rows(chip), 0).sum() == 0
    immune = sure_chip(vuln_density=0.0)
    stripe(immune, 1)
    assert immune.hammer(victim_rows(immune), 10**9).sum() == 0


def test_direction():
    true_cells = sure_chip(rev=0.0)
    stripe(true_cells, 0)  # discharged true cells cannot flip
    assert true_cells.hammer(victim_rows(true_cells), 10**6).sum() == 0
    stripe(true_cells, 1)
    assert true_cells.hammer(victim_rows(true_cells), 10**6).all()
    anti = sure_chip(rev=0.0, anti_frac=1.0)
    stripe(anti, 1)
    assert anti.hammer(victim_rows(anti), 10**6).sum() == 0
    stripe(anti, 0)
    assert anti.hammer(victim_rows(anti), 10**6).all()


def test_neighbour_gate():
    rates = []
    for above, below in ((1, 1), (0, 1), (0, 0)):  # 0, 1, 2 opposing neighbours of a stored 1
        chip, total = sure_chip(mult=(0.02, 0.25, 1.0)), 0
        v = victim_rows(chip)
        for _ in range(200):
            chip.bits[v], chip.bits[v - 1], chip.bits[v + 1] = 1, above, below
            total += chip.hammer(v, 700).sum()
        rates.append(total)
    assert rates[0] < rates[1] < rates[2]


def test_refresh_is_not_restore():
    chip = sure_chip()
    stripe(chip, 1)
    v = victim_rows(chip)
    chip.hammer(v, 10**6)
    assert (chip.read(v) == 0).all()  # nothing in the model returns the old value...
    assert chip.hammer(v, 0).sum() == 0 and (chip.read(v) == 0).all()
    chip.write(v, np.ones((len(v), chip.cols), dtype=np.uint8))  # ...except an explicit rewrite
    assert (chip.read(v) == 1).all()


def test_repeatable_and_chip_specific():
    a, a2, b = (Chip(30, 2048, s, vuln_density=0.05) for s in (1, 1, 2))
    pa, pa2, pb = (profile(c, victim_rows(c), 100_000, 8)[1].ravel() for c in (a, a2, b))
    assert np.corrcoef(pa, pa2)[0, 1] > 0.9
    assert abs(np.corrcoef(pa, pb)[0, 1]) < 0.1


def test_paper_counterexamples():
    A, B = np.array([[0.5, 2], [0, 0.5]]), np.array([[0.5, 0], [2, 0.5]])
    radius = lambda m: np.abs(np.linalg.eigvals(m)).max()
    assert radius(A) < 1 and radius(B) < 1 and radius(A @ B) > 1  # v0.2 sec. 4.3
    assert np.linalg.matrix_rank(0.1 * np.eye(3)) == 3            # v0.2 sec. 4.4


def test_sandwich_rule():
    chip = ideal_chip(3, 64, 0, hc50=1000.0)
    ad = SandwichAdapter(chip, 1, 4, 8, placement="naive")
    pre = np.array([1, 0, 1, 0])
    assert not ad.delta().any()
    ad.update(0, +1, pre, 10**6)
    assert (ad.delta()[0] == [8, 0, 8, 0]).all()   # potentiates only where pre is active
    ad.update(0, -1, pre, 10**6)
    assert (ad.delta()[0] == [0, 0, 0, 0]).all()   # one-way cells: + and - groups both spent
    ad.update(0, +1, pre, 10**6)
    assert (ad.delta()[0] == [0, 0, 0, 0]).all()   # saturated until consolidated
    ad.reset()
    ad.update(0, -1, 1 - pre, 10**6)
    assert (ad.delta()[0] == [0, -8, 0, -8]).all()


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
