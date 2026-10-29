#!/usr/bin/env python3
"""Generate CAC data files (atom_style cac) for a crystal made of a coarse
element region surrounding an atomistic core.

The crystal is split into blocks of `block` x `block` x `block` unit cells.
Blocks whose centre lies inside the atomistic core are written as atoms,
the rest as Eight_Node elements (one poly per basis atom).

Element convention (matches USER-CAC): lattice sites sit at unit-cell
centres in isoparametric space, s_i = -1 + (2i+1)/n, so the 8 nodes of each
poly lie half a lattice vector outside the outermost sites.
"""
import argparse
import numpy as np

LATTICES = {
    # name: (cell vectors in units of a (c for z where noted), basis fractions, types)
    "fcc": (np.eye(3), np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]), [1, 1, 1, 1]),
    "diamond": (np.eye(3), np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                                     [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75],
                                     [0.25, 0.75, 0.75]]), [1] * 8),
}

CORNERS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def wurtzite_ortho(a, c, u=0.382):
    """Orthorhombic 8-atom wurtzite cell (a, sqrt(3)a, c); type 1 = cation, 2 = anion."""
    A = np.diag([a, np.sqrt(3) * a, c])
    cat = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 1 / 6, 0.5], [0, 2 / 3, 0.5]]
    frac = [p for p in cat] + [[p[0], p[1], p[2] + u] for p in cat]
    return A, np.array(frac) % 1.0, [1, 1, 1, 1, 2, 2, 2, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", default="fcc", choices=["fcc", "diamond", "wurtzite"])
    ap.add_argument("--a", type=float, default=3.615)
    ap.add_argument("--c", type=float, default=None, help="c lattice constant (wurtzite)")
    ap.add_argument("--blocks", type=int, nargs=3, default=[6, 6, 6], help="number of blocks per dim")
    ap.add_argument("--block", type=int, default=4, help="unit cells per block edge")
    ap.add_argument("--core", type=float, default=0.34,
                    help="atomistic core half-width as a fraction of the box (0 = no atoms, >=0.5 = all atoms)")
    ap.add_argument("--masses", type=float, nargs="+", default=[63.546])
    ap.add_argument("--periodic", action="store_true", help="write box edges at the lattice period")
    ap.add_argument("--jitter", type=float, default=0.0, help="random displacement (A) of atoms and nodes")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--block2", type=int, default=0, help="finer element size (cells per edge, must divide --block) for blocks with x > --split")
    ap.add_argument("--split", type=float, default=0.0, help="box fraction x above which the finer elements are used")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    if args.lattice == "wurtzite":
        A, basis, types = wurtzite_ortho(args.a, args.c or args.a * 1.6)
    else:
        cell, basis, types = LATTICES[args.lattice]
        A = cell * args.a
    rng = np.random.default_rng(args.seed)
    nb = np.array(args.blocks)
    m = args.block
    ncell = nb * m
    origin = -0.5 * np.ones(3) @ A * 0  # lattice origin at 0
    centre = 0.5 * ncell  # in cells

    def site(cidx, b):
        return (np.asarray(cidx, float) + basis[b]) @ A

    ntypes = max(types)
    entries = []  # (kind, payload)
    for bi in range(nb[0]):
        for bj in range(nb[1]):
            for bk in range(nb[2]):
                c0 = np.array([bi, bj, bk]) * m
                bc = (c0 + 0.5 * m) / ncell - 0.5  # block centre in box fractions
                atomistic = np.all(np.abs(bc) < args.core)
                if atomistic:
                    for i in range(m):
                        for j in range(m):
                            for k in range(m):
                                for b in range(len(basis)):
                                    x = site(c0 + [i, j, k], b) + rng.normal(0, args.jitter, 3) * (args.jitter > 0)
                                    entries.append(("atom", (types[b], x)))
                else:
                    fine = args.block2 > 0 and bc[0] > args.split
                    msz = args.block2 if fine else m
                    sub = (m // msz) if fine else 1
                    for si in range(sub):
                        for sj in range(sub):
                            for sk in range(sub):
                                c1 = c0 + np.array([si, sj, sk]) * msz
                                polys = []
                                for b in range(len(basis)):
                                    nodes = []
                                    for cn in CORNERS:
                                        f = c1 - 0.5 + np.array(cn) * msz  # cells
                                        x = (f + basis[b]) @ A + rng.normal(0, args.jitter, 3) * (args.jitter > 0)
                                        nodes.append(x)
                                    polys.append((types[b], nodes))
                                entries.append(("elem", polys, msz))

    lo = (-0.5 * np.ones(3)) @ A
    hi = (ncell - 0.5) @ A
    if not args.periodic:
        pad = 0.5 * np.abs(np.diag(A))
        lo, hi = lo - pad, hi + pad
    with open(args.out, "w") as fh:
        fh.write(f"CAC model generated by make_cac_model.py ({args.lattice})\n\n")
        fh.write(f"{len(entries)} cac elements\n{ntypes} atom types\n\n")
        for d, nm in enumerate("xyz"):
            fh.write(f"{lo[d]:.8f} {hi[d]:.8f} {nm}lo {nm}hi\n")
        fh.write("\nMasses\n\n")
        for t in range(ntypes):
            fh.write(f"{t + 1} {args.masses[min(t, len(args.masses) - 1)]}\n")
        fh.write("\nCAC Elements\n\n")
        for eid, ent in enumerate(entries, 1):
            kind, pl = ent[0], ent[1]
            if kind == "atom":
                t, x = pl
                fh.write(f"{eid} Atom 1 1 1 1\n1 1 {t} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f}\n")
            else:
                fh.write(f"{eid} Eight_Node {len(pl)} {ent[2]} {ent[2]} {ent[2]}\n")
                for p, (t, nodes) in enumerate(pl, 1):
                    for n, x in enumerate(nodes, 1):
                        fh.write(f"{n} {p} {t} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f}\n")
    na = sum(1 for e in entries if e[0] == "atom")
    ne = len(entries) - na
    nequiv = na + sum(len(basis) * e[2] ** 3 for e in entries if e[0] == "elem")
    print(f"{args.out}: {na} atoms, {ne} elements, {nequiv} atom-equivalents")


if __name__ == "__main__":
    main()
