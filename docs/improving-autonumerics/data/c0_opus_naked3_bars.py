import json, math, os
P = "/Users/sahilmulki/Autonumerics/benchmark/results"

def gu_and_bar(h):
    fe = h.get("functional_errs") or {}
    tu = h.get("functional_tol_used") or {}
    ft = h.get("functional_tol")
    if not fe: return None, None
    tol = lambda k: tu.get(k) or ft
    name = max(fe, key=lambda k: fe[k] / tol(k))
    g = fe[name] / tol(name)
    lv = ((h.get("functional_levels") or {}).get("values") or {}).get(name)
    bar = None
    if lv and lv.get("graded") is not None:
        r = (h["functional_levels"]["num_paths_graded"] /
             h["functional_levels"]["num_paths_coarse"])
        se = abs(lv["graded"] - lv["coarse"]) / math.sqrt(r - 1)
        truth = abs(h["functional_abs_errs"][name] / fe[name])  # |v_truth|
        bar = se / truth / tol(name)
    return g, bar

out = {}
d = json.load(open(os.path.join(P, "baselines/oneshot_benchop_c0_opus_naked3.json")))
for r in d["results"]:
    out.setdefault(r["id"], []).append(gu_and_bar(r["verify"]))
pipe = {}
for f in ["pipeline/benchop_sonnet_other9_run2.json", "pipeline/benchop_b03_sonnet_run4.json"]:
    for r in json.load(open(os.path.join(P, f)))["results"]:
        h = r["harness_by_plan"]
        pipe[r["id"]] = gu_and_bar(h["plans"][h["winner"]]["harness"])
print(f"{'id':4s} {'C0 bars (3 draws)':>34s} | {'AN ship gu':>10s} {'AN bar':>8s}")
for k in sorted(out):
    print(f"{k:4s} " + " ".join(f"{b:10.4f}" if b is not None else "        na" for _, b in out[k]) +
          f" | {pipe[k][0]:10.4f} {pipe[k][1]:8.4f}" if pipe[k][1] is not None else f"{k} na")
