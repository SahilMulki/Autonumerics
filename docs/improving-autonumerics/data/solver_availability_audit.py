import json, os, collections
R="/Users/sahilmulki/Autonumerics"; P=R+"/benchmark/results"; S=P+"/benchop-step10"
SNAP=P+"/pipeline/archive_sonnet_other9_2026-11-05/workspace_snapshot"
def gu(h):
    fe=h.get("functional_errs") or {}; tu=h.get("functional_tol_used") or {}; ft=h.get("functional_tol")
    return max(v/(tu.get(k) or ft) for k,v in fe.items()) if fe else None

def c0_best(files, model):
    o={}
    for f in files:
        for r in json.load(open(f))["results"]:
            g=gu(r["verify"])
            if g is None: continue
            if r["id"] not in o or g<o[r["id"]][0]:
                o[r["id"]]=(g, f"workspace_oneshot/{model}/{r['slug']}/c0-{r['manuals']}/rep{r['rep']}")
    return o
def an_best(files, root):
    o={}
    for f in files:
        for r in json.load(open(f))["results"]:
            pl={k:gu(v["harness"]) for k,v in (r.get("harness_by_plan",{}).get("plans") or {}).items() if gu(v.get("harness") or {}) is not None}
            if not pl: continue
            k=min(pl,key=pl.get)
            o[r["id"]]=(pl[k], os.path.join(root, r["slug"], "plans", k))
    return o
arms={
 "Opus 1-shot best-of-10": c0_best([f"{S}/s{i}.json" for i in (1,2,3,4)], "opus"),
 "Sonnet 1-shot best-of-3": c0_best([f"{P}/baselines/oneshot_benchop_c0_sonnet.json"], "sonnet"),
 "Opus 1-shot best-of-3": c0_best([f"{P}/baselines/oneshot_benchop_c0_opus_naked3.json"], "opus"),
 "AN run 1 pool best": an_best([f"{S}/{n}.json" for n in ("p_regrade123","p_regrade4","p_b0506","p_b0708","p_b0910")], "workspace"),
 "AN run 2 pool best": an_best([f"{P}/pipeline/benchop_sonnet.json"], SNAP),
 "AN run 3 pool best": an_best([f"{P}/pipeline/benchop_sonnet_other9_run2.json", f"{P}/pipeline/benchop_b03_sonnet_run4.json"], R+"/workspace"),
}
for name,d in arms.items():
    have=[]; miss=[]
    for pid,(g,path) in sorted(d.items()):
        p = path if os.path.isabs(path) else os.path.join(R,path)
        (have if os.path.exists(os.path.join(p,"solver.py")) else miss).append(pid)
    print(f"{name:26s} solver on disk: {len(have)}/{len(d)}   missing: {miss or 'none'}")
