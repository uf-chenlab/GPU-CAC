#!/usr/bin/env python3
"""Match CAC quadrature points (CAC_DUMP_QUAD output) to atoms of the fully
expanded atomistic run (dump custom id type x y z fx fy fz pe) and compare."""
import sys, glob
import numpy as np
from scipy.spatial import cKDTree

q = np.vstack([np.loadtxt(f, ndmin=2) for f in sorted(glob.glob(sys.argv[1] + ".*"))])
lines = open(sys.argv[2]).read().splitlines()
k = lines.index(next(l for l in lines if l.startswith("ITEM: ATOMS")))
a = np.loadtxt(lines[k + 1:], ndmin=2)
tree = cKDTree(a[:, 2:5])
d, idx = tree.query(q[:, 2:5])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e30
ok = d < tol
if not ok.all():
    print(f"ignoring {int((~ok).sum())} of {len(ok)} points with position mismatch >= {tol:g}")
q, d, idx = q[ok], d[ok], idx[ok]
fq, fa = q[:, 5:8], a[idx, 5:8]
eq, ea = q[:, 8], a[idx, 8]
err = np.linalg.norm(fq - fa, axis=1)
scale = np.linalg.norm(fa, axis=1).mean()
print(f"quad points: {len(q)}  max position mismatch: {d.max():.2e} A")
print(f"force: max |dF| = {err.max():.3e} eV/A, mean |F| = {scale:.3e}, max rel = {err.max()/scale:.2e}")
print(f"site energy: max |dE| = {np.abs(eq-ea).max():.3e} eV")
iw = np.argmax(err)
print("worst:", q[iw, :8], "vs", fa[iw])
