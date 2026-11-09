import json, math, statistics as st, os
ROOT = "/Users/sahilmulki/Autonumerics"
P = os.path.join(ROOT, "benchmark/results")

def gu(h):
    fe = h.get("functional_errs") or {}
    tu = h.get("functional_tol_used") or {}
    ft = h.get("functional_tol")
    if not fe: return None
    return max(v / (tu.get(k) or ft) for k, v in fe.items())

# --- C0 @ Opus, naked, 3 draws
c0 = {}
d = json.load(open(os.path.join(P, "baselines/oneshot_benchop_c0_opus_naked3.json")))
for r in d["results"]:
    c0.setdefault(r["id"], []).append(gu(r["verify"]))

# --- latest Sonnet pipeline: other9 run2 (2026-11-07) + B03 run4 (2026-11-05)
pipe = {}
for f in ["pipeline/benchop_sonnet_other9_run2.json", "pipeline/benchop_b03_sonnet_run4.json"]:
    d = json.load(open(os.path.join(P, f)))
    for r in d["results"]:
        h = r.get("harness_by_plan") or {}
        pool = {k: gu(v["harness"]) for k, v in (h.get("plans") or {}).items()
                if (v.get("harness") or {}).get("functional_errs")}
        pipe[r["id"]] = dict(slug=r["slug"], verdict=r["verdict"],
                             ship=h.get("winner_harness_err"), winner=h.get("winner"),
                             best=h.get("best_harness_err"), best_plan=h.get("best_plan_by_harness"),
                             wrong=h.get("wrong_plan_won"), pool=pool,
                             status=(r.get("run") or {}).get("status"))

gm = lambda a: math.exp(sum(map(math.log, a)) / len(a))
rows = []
for pid in sorted(c0):
    v = c0[pid]; p = pipe[pid]
    rows.append(dict(id=pid, slug=p["slug"], draws=v, best3=min(v), med=st.median(v),
                     ship=p["ship"], pbest=p["best"], npool=len(p["pool"]),
                     wrong=p["wrong"], verdict=p["verdict"], status=p["status"],
                     winner=p["winner"], best_plan=p["best_plan"]))

print(f"{'id':4s} {'draw1':>8s} {'draw2':>8s} {'draw3':>8s} {'med':>8s} {'best3':>8s} {'AN ship':>8s} {'AN pool':>8s} {'med/ship':>9s} {'b3/pbest':>9s}")
for r in rows:
    print(f"{r['id']:4s} " + " ".join(f"{x:8.4f}" for x in r["draws"]) +
          f" {r['med']:8.4f} {r['best3']:8.4f} {r['ship']:8.4f} {r['pbest']:8.4f}"
          f" {r['med']/r['ship']:9.3f} {r['best3']/r['pbest']:9.3f}")

for label, key in [("median draw", "med"), ("best-of-3", "best3")]:
    print(f"\ngeomean C0@Opus {label:12s} = {gm([r[key] for r in rows]):.4f} gate units")
print(f"geomean AN@Sonnet shipped    = {gm([r['ship'] for r in rows]):.4f} gate units")
print(f"geomean AN@Sonnet pool best  = {gm([r['pbest'] for r in rows]):.4f} gate units")
print("\nwins (lower = better), median draw vs AN ship:",
      sum(r['med'] < r['ship'] for r in rows), "C0 /", sum(r['med'] > r['ship'] for r in rows), "AN")
print("wins, best-of-3 vs AN pool best:",
      sum(r['best3'] < r['pbest'] for r in rows), "C0 /", sum(r['best3'] > r['pbest'] for r in rows), "AN")
print("\nselection: wrong_plan_won =", {r['id']: r['wrong'] for r in rows})
print("verdict/status:", {r['id']: (r['verdict'], r['status']) for r in rows})
print("pool sizes:", {r['id']: r['npool'] for r in rows})
# rows[] holds every number the markdown table quotes; dump it if you want the raw form.
