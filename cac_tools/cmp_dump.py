#!/usr/bin/env python3
"""max relative difference between the numbers of two text files with identical layout"""
import sys
a = open(sys.argv[1]).read().split(); b = open(sys.argv[2]).read().split()
if len(a) != len(b):
    print("different length"); sys.exit(1)
m = 0.0; scale = 1e-300
for x, y in zip(a, b):
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        continue
    m = max(m, abs(fx - fy)); scale = max(scale, abs(fx), abs(fy))
print("max|d|=%.2e rel=%.1e" % (m, m / scale))
