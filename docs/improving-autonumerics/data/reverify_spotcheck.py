import json, os, sys
HERE = "/Users/sahilmulki/Autonumerics/benchmark"
sys.path.insert(0, HERE)
import problems as P, verify as V

CASES = [
    ("AN run 3, B04 best plan", "sde_heston_dnt",
     "workspace/sde_heston_dnt/plans/1-exact-chi2-bridge", 0.0254911863),
    ("Sonnet x3, B04 rep3", "sde_heston_dnt",
     "workspace_oneshot/sonnet/sde_heston_dnt/c0-manual/rep3", 0.06032867563731055),
]

def gu(h):
    fe = h["functional_errs"]; tu = h.get("functional_tol_used") or {}; ft = h.get("functional_tol")
    return max(v / (tu.get(k) or ft) for k, v in fe.items())

for label, slug, pd, recorded in CASES:
    prob = P.by_slug(slug)
    out = V.verify_problem(prob, os.path.join("/Users/sahilmulki/Autonumerics/workspace", slug),
                           plan_dir=os.path.abspath(pd), sandbox=True,
                           timeout_s=600, mem_mb=4096, work_precision=False)
    now = gu(out)
    print(f"{label}")
    print(f"   recomputed now : {now!r}")
    print(f"   recorded       : {recorded!r}")
    print(f"   bitwise equal  : {now == recorded}")
    print(f"   graded values  : {[out['functional_levels']['values'][k]['graded'] for k in sorted(out['functional_levels']['values'])]}")
