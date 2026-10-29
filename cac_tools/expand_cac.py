#!/usr/bin/env python3
"""Expand a CAC data file (atom_style cac) into a fully atomistic LAMMPS data
file (atom_style atomic): every virtual atom of every Eight_Node element is
interpolated at its isoparametric cell centre, s_i = -1 + (2i+1)/n."""
import sys
import numpy as np


def q8(s, t, w):
    sp, sm, tp, tm, wp, wm = (1 + s) / 8, (1 - s) / 8, 1 + t, 1 - t, 1 + w, 1 - w
    return np.array([sm * tm * wm, sp * tm * wm, sp * tp * wm, sm * tp * wm,
                     sm * tm * wp, sp * tm * wp, sp * tp * wp, sm * tp * wp])


def main(src, dst):
    lines = open(src).read().splitlines()
    box, masses, ntypes = [], [], 1
    i = 0
    while i < len(lines):
        w = lines[i].split()
        if len(w) >= 4 and w[2].endswith("lo"):
            box.append((float(w[0]), float(w[1])))
        if len(w) == 3 and w[1] == "atom" and w[2] == "types":
            ntypes = int(w[0])
        if lines[i].strip() == "Masses":
            j = i + 1
            while len(masses) < ntypes:
                ww = lines[j].split()
                if len(ww) >= 2: masses.append(ww[1])
                j += 1
        if lines[i].strip() == "CAC Elements":
            i += 1
            break
        i += 1
    atoms = []
    while i < len(lines):
        w = lines[i].split()
        i += 1
        if len(w) != 6 or not w[1][0].isalpha():
            continue
        kind, npoly, n = w[1], int(w[2]), [int(w[3]), int(w[4]), int(w[5])]
        if kind == "Atom":
            ww = lines[i].split(); i += 1
            atoms.append((int(ww[2]), [float(v) for v in ww[3:6]]))
            continue
        nodes = np.zeros((npoly, 8, 3)); ptype = [0] * npoly
        for _ in range(8 * npoly):
            ww = lines[i].split(); i += 1
            node, poly = int(ww[0]) - 1, int(ww[1]) - 1
            ptype[poly] = int(ww[2])
            nodes[poly, node] = [float(v) for v in ww[3:6]]
        for a in range(n[0]):
            for b in range(n[1]):
                for c in range(n[2]):
                    N = q8(-1 + (2 * a + 1) / n[0], -1 + (2 * b + 1) / n[1], -1 + (2 * c + 1) / n[2])
                    for p in range(npoly):
                        atoms.append((ptype[p], N @ nodes[p]))
    with open(dst, "w") as fh:
        fh.write(f"expanded from {src}\n\n{len(atoms)} atoms\n{ntypes} atom types\n\n")
        for (lo, hi), nm in zip(box, "xyz"):
            fh.write(f"{lo:.10f} {hi:.10f} {nm}lo {nm}hi\n")
        fh.write("\nMasses\n\n")
        for t, m in enumerate(masses, 1):
            fh.write(f"{t} {m}\n")
        fh.write("\nAtoms # atomic\n\n")
        for k, (t, x) in enumerate(atoms, 1):
            fh.write(f"{k} {t} {x[0]:.17g} {x[1]:.17g} {x[2]:.17g}\n")
    print(f"{dst}: {len(atoms)} atoms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
