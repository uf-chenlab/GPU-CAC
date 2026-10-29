import sys, re, os, glob
dirs = sys.argv[1:]
def loop(f):
    try:
        t = open(f).read()
    except OSError:
        return None
    m = re.search(r"Loop time of ([0-9.e+-]+) on", t)
    n = re.search(r"^Neigh\s*\|\s*[0-9.e+-]+\s*\|\s*([0-9.e+-]+)", t, re.M)
    return (float(m.group(1)) if m else None, float(n.group(1)) if n else 0.0)
styles = [x for x in "eam eamal eamfs lj sw tersoff eamkk eamalkk eamfskk ljkk swkk terskk nep nepkk nepkk32".split()]
models = ["atoms", "elem4", "elem8", "mix4"]
print("%-8s %-6s" % ("style", "model") + "".join("  %-26s" % os.path.basename(d.rstrip('/')) for d in dirs))
print(" " * 16 + "".join("  %-12s%-14s" % ("static(5)", "rebuild(3)") for d in dirs))
for st in styles:
    for m in models:
        row = "%-8s %-6s" % (st, m)
        for d in dirs:
            a = loop(f"{d}/o_{st}_{m}_static.out"); b = loop(f"{d}/o_{st}_{m}_rebuild.out")
            if a is None and b is None: row += "  %-26s" % "-"; continue
            row += "  %-12s%-14s" % ("%.4f" % a[0] if a and a[0] else "-", "%.4f" % b[0] if b and b[0] else "-")
        print(row)
