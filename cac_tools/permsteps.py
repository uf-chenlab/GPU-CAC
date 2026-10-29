import re, sys, os
def get(f):
    try: t = open(f).read()
    except OSError: return None
    m = re.search(r"Loop time of ([0-9.e+-]+) on \d+ procs for (\d+) steps", t)
    return (float(m.group(1)), int(m.group(2))) if m else None
dirs = sys.argv[1:]
styles = "eam eamal eamfs lj sw tersoff eamkk eamalkk eamfskk ljkk swkk terskk nep nepkk nepkk32".split()
models = ["atoms", "elem4", "elem8", "mix4", "big_e", "big_m"]
print("ms per MD step | ms per neighbor rebuild (loop time of steps with rebuild each step, minus the step cost)")
print("%-9s %-6s" % ("style", "model") + "".join("  %-18s" % os.path.basename(d.rstrip("/")) for d in dirs))
for st in styles:
    for m in models:
        row = "%-9s %-6s" % (st, m); any_ = False
        for d in dirs:
            a = get(f"{d}/o_{st}_{m}_static.out"); b = get(f"{d}/o_{st}_{m}_rebuild.out")
            if a and b:
                step = 1e3 * a[0] / a[1]
                reb = 1e3 * (b[0] / b[1]) - step
                row += "  %7.2f | %7.1f   " % (step, reb); any_ = True
            else:
                row += "  %-18s" % "-"
        if any_: print(row)
