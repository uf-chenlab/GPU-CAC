#!/usr/bin/env python3
"""Markdown comparison tables for the meeting report (baseline vs final)."""
import re, os, sys
R = "/blue/ypchen/ntaormina/nep/cac_kokkos/runs"

def get(d, st, m):
    def one(kind):
        try:
            t = open(f"{R}/{d}/o_{st}_{m}_{kind}.out").read()
        except OSError:
            return None
        x = re.search(r"Loop time of ([0-9.e+-]+) on \d+ procs for (\d+) steps", t)
        return (float(x.group(1)), int(x.group(2))) if x else None
    a, b = one("static"), one("rebuild")
    if not a or not b:
        return None
    step = 1e3 * a[0] / a[1]
    reb = 1e3 * (b[0] / b[1]) - step
    return step, max(reb, 0.0)

styles = [("eam", "eamkk", "EAM (Cu_u3)"), ("eamal", "eamalkk", "EAM/alloy"), ("eamfs", "eamfskk", "EAM/fs"),
          ("lj", "ljkk", "LJ 5.0"), ("sw", "swkk", "Stillinger-Weber"), ("tersoff", "terskk", "Tersoff")]
models = [("atoms", "all atoms"), ("elem4", "all elements"), ("mix4", "atoms + elements")]

print("### Classical potentials, 13.8k-16.4k atom-equivalents, one rank\n")
print("`step` = ms per MD step, `rebuild` = ms per neighbour-list rebuild (everything that happens when LAMMPS rebuilds the lists). "
      "base = start of this work, cpu1/cpu16 = new code with 1 / 16 OpenMP threads, gpu = NVIDIA L4 (fp64) + 16 CPU threads.\n")
print("| potential | model | step base | step cpu1 | step cpu16 | step gpu | rebuild base | rebuild cpu1 | rebuild cpu16 | rebuild gpu |")
print("|---|---|---|---|---|---|---|---|---|---|")
for st, kk, name in styles:
    for m, mname in models:
        b = get("classic_base", st, m)
        c1 = get("H_cpu1", st, m)
        c16 = get("H_cpu16", st, m)
        g = get("H_gpu", kk, m)
        if not (b and c1 and c16 and g):
            continue
        print(f"| {name} | {mname} | {b[0]:.1f} | {c1[0]:.1f} | {c16[0]:.2f} | {g[0]:.2f} | {b[1]:.0f} | {c1[1]:.1f} | {c16[1]:.1f} | {g[1]:.1f} |")

print("\n### Speed-up factors (base / new)\n")
print("| potential | model | step: cpu1 | step: cpu16 | step: gpu | rebuild: cpu1 | rebuild: cpu16 | rebuild: gpu |")
print("|---|---|---|---|---|---|---|---|")
for st, kk, name in styles:
    for m, mname in models:
        b = get("classic_base", st, m)
        c1 = get("H_cpu1", st, m)
        c16 = get("H_cpu16", st, m)
        g = get("H_gpu", kk, m)
        if not (b and c1 and c16 and g):
            continue
        f = lambda x, y: f"{x / max(y, 1e-9):.1f}x"
        print(f"| {name} | {mname} | {f(b[0], c1[0])} | {f(b[0], c16[0])} | {f(b[0], g[0])} | {f(b[1], c1[1])} | {f(b[1], c16[1])} | {f(b[1], g[1])} |")

print("\n### One million atom-equivalents (1,048,576 Cu / 884,736 Si), CPU 16 threads vs L4 GPU\n")
print("`all elements` = 512 (Cu) / 216 (Si) elements of 8x8x8 cells; `mixed` = 131,072 (Cu) / 32,768 (Si) atoms in the centre + elements.\n")
print("| potential | model | step cpu16 (ms) | step gpu (ms) | rebuild cpu16 (ms) | rebuild gpu (ms) |")
print("|---|---|---|---|---|---|")
for st, kk, name in styles:
    for m, mname in (("big_e", "all elements"), ("big_m", "mixed")):
        c = get("H_big_cpu16", st, m)
        g = get("H_big_gpu", kk, m)
        if c and g:
            print(f"| {name} | {mname} | {c[0]:.1f} | {g[0]:.1f} | {c[1]:.0f} | {g[1]:.0f} |")
