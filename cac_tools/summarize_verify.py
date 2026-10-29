#!/usr/bin/env python3
"""Aggregate the verification logs: worst-case deviation per style and check category."""
import re, sys, collections
files = sys.argv[1:]
# category -> style -> {force, energy, virial}: max rel ; counts
ALL = ["sw", "ters", "eam", "eamal", "eamfs", "lj", "nep", "swkk", "terskk", "eamkk", "eamalkk", "eamfskk", "ljkk", "nepkk"]
agg = collections.defaultdict(lambda: collections.defaultdict(lambda: {"force": 0.0, "energy": 0.0, "virial": 0.0, "n": 0, "ulp": 0, "fail": 0}))
thermo = collections.defaultdict(float)
fullatom = collections.defaultdict(lambda: {"f": 0.0, "e": 0.0, "n": 0})
press = collections.defaultdict(list)
for f in files:
    for line in open(f, errors="ignore"):
        m = re.match(r"^(ok  |FAIL) +(\w+)/(\w+) ([ABDE]) .*?n= *(\d+) force: .*?rel=([0-9.e+-]+) .*?maxulp=(\d+) \| energy: .*?rel=([0-9.e+-]+) .*? \| virial: .*?rel=([0-9.e+-]+)", line)
        if m:
            st, mdl, cat = m.group(2), m.group(3), m.group(4)
            d = agg[cat][st]
            d["n"] += 1
            d["force"] = max(d["force"], float(m.group(6))); d["energy"] = max(d["energy"], float(m.group(8))); d["virial"] = max(d["virial"], float(m.group(9)))
            if m.group(1) == "FAIL": d["fail"] += 1
            continue
        m = re.match(r"^(ok  |FAIL) +(\w+)/(\w+) vs full-atom LAMMPS: force: max \|dF\| = ([0-9.e+-]+) .*max rel = ([0-9.e+-]+) site energy: max \|dE\| = ([0-9.e+-]+)", line)
        if m:
            d = fullatom[m.group(2)]; d["n"] += 1; d["f"] = max(d["f"], float(m.group(5))); d["e"] = max(d["e"], float(m.group(6)))
            continue
        m = re.match(r"^(ok  |note) +(\w+)/(\w+) C2 virial pressure .*\(rel ([0-9.e+-]+)\)", line)
        if m: press[(m.group(2), m.group(3))].append((m.group(1).strip(), float(m.group(4))))
        m = re.match(r"^(ok  |FAIL) +(\w+)/(\w+) ([ABDE]) .*thermo .*max rel diff ([0-9.e+-]+)", line)
        if m: thermo[(m.group(4), m.group(2))] = max(thermo[(m.group(4), m.group(2))], float(m.group(5)))
names = {"E": "GPU style vs the same potential on the CPU (new path), 1 rank", "A": "original vs new (same ranks)", "B": "new N ranks/T threads vs 1/1", "D": "10 MD steps, rebuild each step, original vs new"}
for cat in "ABDE":
    print(f"\n{cat}: {names[cat]}   (worst case over models/ranks/threads; relative to the largest value of each quantity)")
    print("  style    #checks  force      energy     virial     thermo(PE,T[,P])  fails")
    for st in ALL:
        d = agg[cat].get(st)
        if d: print(f"  {st:8s} {d['n']:6d}  {d['force']:.1e}  {d['energy']:.1e}  {d['virial']:.1e}  {thermo.get((cat, st), float('nan')):.1e}          {d['fail']}")
print("\nC: new path vs full-atom LAMMPS (every quadrature point = one atom; relative to mean |F|)")
print("  style    #models  max rel force error   max site-energy error (eV)")
for st in ALL:
    d = fullatom.get(st)
    if d: print(f"  {st:8s} {d['n']:6d}   {d['f']:.1e}              {d['e']:.1e}")
print("\nC2: global virial pressure vs full-atom LAMMPS")
for k, v in sorted(press.items()):
    print("  %-8s %-3s %s" % (k[0], k[1], ", ".join(f"{s}:{r:.1e}" for s, r in v)))
