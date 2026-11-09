import json, math, os, collections
P = "/Users/sahilmulki/Autonumerics/benchmark/results"
S = os.path.join(P, "benchop-step10")

def gu(h):
    fe = h.get("functional_errs") or {}
    tu = h.get("functional_tol_used") or {}
    ft = h.get("functional_tol")
    if not fe: return None
    return max(v / (tu.get(k) or ft) for k, v in fe.items())

def c0_arm(files):
    """best-of-N over draws, plus the draws themselves (for distinct-answer counts)."""
    out = collections.defaultdict(list)
    for f in files:
        for r in json.load(open(f))["results"]:
            g = gu(r["verify"])
            if g is not None: out[r["id"]].append(g)
    return out

def an_arm(files):
    """pool best = min gate units over the pool's plans (recomputed, not read)."""
    out = {}
    for f in files:
        for r in json.load(open(f))["results"]:
            h = r.get("harness_by_plan") or {}
            pool = {k: gu(v["harness"]) for k, v in (h.get("plans") or {}).items()
                    if gu(v.get("harness") or {}) is not None}
            if pool: out[r["id"]] = pool
    return out

opus10 = c0_arm([os.path.join(S, f) for f in ("s1.json","s2.json","s3.json","s4.json")])
son3   = c0_arm([os.path.join(P, "baselines/oneshot_benchop_c0_sonnet.json")])
opus3  = c0_arm([os.path.join(P, "baselines/oneshot_benchop_c0_opus_naked3.json")])
an1    = an_arm([os.path.join(S, f) for f in ("p_regrade123.json","p_regrade4.json",
                                              "p_b0506.json","p_b0708.json","p_b0910.json")])
an2    = an_arm([os.path.join(P, "pipeline/benchop_sonnet.json")])
an3    = an_arm([os.path.join(P, "pipeline/benchop_sonnet_other9_run2.json"),
                 os.path.join(P, "pipeline/benchop_b03_sonnet_run4.json")])

ids = sorted(opus3)
cols = [("C0 Opus x10", opus10, "c0"), ("C0 Sonnet x3", son3, "c0"), ("C0 Opus x3", opus3, "c0"),
        ("AN run1", an1, "an"), ("AN run2 Sonnet", an2, "an"), ("AN run3 Sonnet", an3, "an")]
print(f"{'id':4s} " + " ".join(f"{n:>15s}" for n, _, _ in cols))
best = {n: [] for n, _, _ in cols}
for i in ids:
    cells = []
    for n, d, kind in cols:
        v = d.get(i)
        if not v: cells.append("    --"); continue
        vals = list(v.values()) if kind == "an" else v
        b = min(vals); best[n].append(b)
        cells.append(f"{b:.4f} (n={len(vals)})")
    print(f"{i:4s} " + " ".join(f"{c:>15s}" for c in cells))
gm = lambda a: math.exp(sum(map(math.log, a)) / len(a))
print()
for n, _, _ in cols:
    print(f"geomean {n:16s} = {gm(best[n]):.4f}   (problems: {len(best[n])})")

print("\n--- distinct answers per problem (rounded to 10 sig figs) ---")
for n, d, kind in cols:
    row = {}
    for i in ids:
        v = d.get(i)
        if not v: continue
        vals = list(v.values()) if kind == "an" else v
        row[i] = f"{len(set(round(x,10) for x in vals))}/{len(vals)}"
    print(f"{n:16s} " + " ".join(f"{i}:{row.get(i,'--')}" for i in ids))
