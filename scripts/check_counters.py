#!/usr/bin/env python3
"""Check a counters.json against the GPU's own counter list before profiling.

  python3 check_counters.py counters.json mi300x_rocm72_counters.txt

Checks, per pass:
  - every counter exists in the `rocprofv3 --list-avail` output (names differ
    between GPU generations: gfx90a TCC_EA_* vs gfx942 TCC_EA0_*)
  - the list is for the expected architecture (default gfx942)
  - per-block counts stay within the per-pass limits that worked on gfx942:
    SQ <= 8, TCC <= 4, TCP <= 4, GRBM <= 2 (derived metrics without a block
    prefix, e.g. MeanOccupancyPerCU, are not counted; rocprofv3 itself is the
    final judge -- a pass over a limit fails fast)
  - every counter referenced in "derived" formulas is collected by some pass
Exit status 1 if anything is wrong.
"""
import argparse
import collections
import json
import re
import sys

LIMITS = {"SQ": 8, "TCC": 4, "TCP": 4, "GRBM": 2, "TA": 2, "TD": 2, "SPI": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("counters_json")
    ap.add_argument("list_avail_txt")
    ap.add_argument("--arch", default="gfx942")
    a = ap.parse_args()

    text = open(a.list_avail_txt, errors="replace").read()
    archs = sorted(set(re.findall(r"^Name\s*:\s*(gfx\w+)", text, re.M)))
    avail = set(re.findall(r"^Counter_Name\s*:\s*(\S+)", text, re.M))
    cfg = json.load(open(a.counters_json))
    bad = 0

    print(f"architectures in list: {archs}; {len(avail)} counters")
    if archs != [a.arch]:
        print(f"  ERROR: expected only {a.arch} -- was the list generated on the right node?")
        bad += 1

    used = set()
    for p in cfg["passes"]:
        cs = p["counters"]
        used.update(cs)
        missing = [c for c in cs if c not in avail]
        # block = name prefix; *_sum counters are sums over channels of one
        # hardware counter, so they still use one slot of their block
        blocks = collections.Counter(c.split("_")[0] for c in cs if c.split("_")[0] in LIMITS)
        over = {b: n for b, n in blocks.items() if b in LIMITS and n > LIMITS[b]}
        flag = "ok" if not missing and not over else "ERROR"
        print(f"  {p['name']:22s} {len(cs):2d} counters  blocks={dict(blocks)}  {flag}")
        if missing:
            print(f"      not on this GPU: {missing}")
        if over:
            print(f"      over per-pass limit: {over} (limits {LIMITS})")
        bad += bool(missing) + bool(over)

    refs = set(re.findall(r"\b(?:SQ|TCC|TCP|GRBM|TA|TD|SPI)_[A-Za-z0-9_]+", " ".join(cfg.get("derived", {}).values())))
    lost = sorted(r for r in refs if r not in used and not r.endswith("_X"))   # _X = placeholder
    if lost:
        print(f"  ERROR: derived formulas use counters no pass collects: {lost}")
        bad += 1
    print("OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
