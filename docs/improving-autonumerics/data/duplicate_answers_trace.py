import json, os
P = "/Users/sahilmulki/Autonumerics/benchmark/results"; S = os.path.join(P, "benchop-step10")
def gu(h):
    fe=h.get("functional_errs") or {}; tu=h.get("functional_tol_used") or {}; ft=h.get("functional_tol")
    return max(v/(tu.get(k) or ft) for k,v in fe.items()) if fe else None

def c0(files, pid):
    for f in files:
        for r in json.load(open(f))["results"]:
            if r["id"]!=pid: continue
            h=r["verify"]; lv=(h.get("functional_levels") or {}).get("values") or {}
            yield (r.get("manuals"), r.get("rep"), gu(h),
                   {k: round(v["graded"],12) for k,v in lv.items()},
                   (h.get("run") or {}).get("num_paths"), (h.get("run") or {}).get("dt"), (h.get("run") or {}).get("seed"))
def an(files, pid):
    for f in files:
        for r in json.load(open(f))["results"]:
            if r["id"]!=pid: continue
            for name,v in (r["harness_by_plan"].get("plans") or {}).items():
                h=v["harness"]; lv=(h.get("functional_levels") or {}).get("values") or {}
                yield (name, None, gu(h), {k: round(x["graded"],12) for k,x in lv.items()},
                       (h.get("run") or {}).get("num_paths"), (h.get("run") or {}).get("dt"), (h.get("run") or {}).get("seed"))

for pid in ("B04","B05","B09","B01"):
    print("="*100); print(pid)
    for label, src in [("C0 Opus x10", c0([os.path.join(S,f) for f in ("s1.json","s2.json","s3.json","s4.json")], pid)),
                       ("C0 Sonnet x3", c0([os.path.join(P,"baselines/oneshot_benchop_c0_sonnet.json")], pid)),
                       ("C0 Opus x3",  c0([os.path.join(P,"baselines/oneshot_benchop_c0_opus_naked3.json")], pid)),
                       ("AN run1", an([os.path.join(S,f) for f in ("p_regrade123.json","p_regrade4.json","p_b0506.json","p_b0708.json","p_b0910.json")], pid))]:
        for tag, rep, g, vals, npth, dt, seed in src:
            print(f"  {label:13s} {str(tag):34s} rep={rep} gu={g:.6f} paths={npth} dt={dt} seed={seed} {vals}")
