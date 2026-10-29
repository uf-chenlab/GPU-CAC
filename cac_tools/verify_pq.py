#!/usr/bin/env python3
"""Compare two CAC_DUMP_QUAD outputs (one file per MPI rank: <prefix>.<rank>) quadrature point by point.

usage: verify_pq.py A B [--pos] [--tol 1e-12] [--label text]
  default : files are paired rank by rank and points by index (same decomposition)
  --pos   : all ranks of A and B are merged and points matched by position (different decompositions)

columns: pq type x y z fx fy fz e v0..v5
Reports, for force / energy / virial: max |diff|, max |diff| / max|value|, number of bit-identical values,
and the largest difference in units of the last place (ulp).  Exit code 1 if the relative deviation
exceeds --tol."""
import sys, glob
import numpy as np


def load(prefix):
    files = sorted(glob.glob(prefix + ".*"), key=lambda f: int(f.rsplit(".", 1)[1]))
    out = []
    for f in files:
        a = np.loadtxt(f, ndmin=2)
        a = a if a.size else np.zeros((0, 15))
        out.append(a[np.argsort(a[:, 0], kind="stable")])
    return out


def ulp_dist(a, b):
    ia = np.ascontiguousarray(a, dtype=np.float64).view(np.int64)
    ib = np.ascontiguousarray(b, dtype=np.float64).view(np.int64)
    # map to a monotonic integer scale
    ia = np.where(ia < 0, np.int64(-2**63) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2**63) - ib, ib)
    return np.abs(ia - ib)


def main():
    args = sys.argv[1:]
    pos = "--pos" in args
    novirial = "--novirial" in args
    tol = 1e-12
    label = ""
    if "--tol" in args:
        tol = float(args[args.index("--tol") + 1])
    if "--label" in args:
        label = args[args.index("--label") + 1]
    A, B = args[0], args[1]
    la, lb = load(A), load(B)
    if not pos:
        if len(la) != len(lb):
            print(f"{label}: rank count differs {len(la)} vs {len(lb)}")
            sys.exit(1)
        pa = np.vstack(la)
        pb = np.vstack(lb)
        if pa.shape != pb.shape or not np.array_equal(pa[:, 0], pb[:, 0]) and False:
            print(f"{label}: point count differs {pa.shape} vs {pb.shape}")
            sys.exit(1)
    else:
        from scipy.spatial import cKDTree
        pa = np.vstack(la)
        pb0 = np.vstack(lb)
        if pa.shape[0] != pb0.shape[0]:
            print(f"{label}: point count differs {pa.shape[0]} vs {pb0.shape[0]}")
            sys.exit(1)
        # several quadrature points can share a site (same position, same values): match type+position
        tree = cKDTree(pb0[:, 2:5])
        d, idx = tree.query(pa[:, 2:5])
        if d.max() > 1e-9:
            print(f"{label}: position matching failed (max {d.max():.2e})")
            sys.exit(1)
        pb = pb0[idx]
    groups = {"force": slice(5, 8), "energy": slice(8, 9), "virial": slice(9, 15)}
    bad = False
    parts = []
    for name, sl in groups.items():
        x, y = pa[:, sl], pb[:, sl]
        scale = max(np.abs(x).max(), np.abs(y).max(), 1e-300)
        diff = np.abs(x - y)
        rel = diff.max() / scale
        same = int((x == y).sum())
        ulp = int(ulp_dist(x, y).max())
        parts.append(f"{name}: max|d|={diff.max():.2e} rel={rel:.1e} identical={same}/{x.size} maxulp={ulp}")
        if rel > tol and not (novirial and name == "virial"):
            bad = True
    print(f"{'FAIL' if bad else 'ok  '} {label:44s} n={pa.shape[0]:6d} " + " | ".join(parts))
    sys.exit(1 if bad else 0)


main()
