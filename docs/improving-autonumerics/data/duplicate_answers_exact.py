import json, os
P="/Users/sahilmulki/Autonumerics/benchmark/results"; S=os.path.join(P,"benchop-step10")
def gu(h):
    fe=h.get("functional_errs") or {}; tu=h.get("functional_tol_used") or {}; ft=h.get("functional_tol")
    return max(v/(tu.get(k) or ft) for k,v in fe.items()) if fe else None
def c0(files,pid):
    for f in files:
        for r in json.load(open(f))["results"]:
            if r["id"]==pid: yield gu(r["verify"])
def an(files,pid):
    for f in files:
        for r in json.load(open(f))["results"]:
            if r["id"]==pid:
                for k,v in (r["harness_by_plan"].get("plans") or {}).items(): yield gu(v["harness"])
AN1=[os.path.join(S,f) for f in ("p_regrade123.json","p_regrade4.json","p_b0506.json","p_b0708.json","p_b0910.json")]
O10=[os.path.join(S,f) for f in ("s1.json","s2.json","s3.json","s4.json")]
for pid in ("B04","B05","B01","B09"):
    arms={"Opus10":min(c0(O10,pid)), "Sonnet3":min(c0([P+"/baselines/oneshot_benchop_c0_sonnet.json"],pid)),
          "Opus3":min(c0([P+"/baselines/oneshot_benchop_c0_opus_naked3.json"],pid)), "AN1":min(an(AN1,pid))}
    print(pid, {k:f"{v:.17g}" for k,v in arms.items()})
    vals=list(arms.values())
    print("   exact equalities:", [(a,b) for i,(a,x) in enumerate(arms.items()) for b,y in list(arms.items())[i+1:] if x==y])
