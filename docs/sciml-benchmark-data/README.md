# SciML adoption — measurement evidence

Scripts that reproduce the measured claims in
[plan-sciml-benchmark.md](../plan-sciml-benchmark.md) §2a, §2b, §3f, §5a and §10a. The first two are
standalone (numpy only, no repo imports) and need no Julia; the rest import the benchmark.

| Script | Reproduces | Run |
|---|---|---|
| `verify_closed_forms.py` | §2a's confirmation that each transcribed closed form **solves** its stated Itô SDE: a common-random-number Milstein ladder reads strong order ≈ 1 on S01–S05, and **exactly 0.0** on S06–S08, which is finding 1 | `uv run python docs/sciml-benchmark-data/verify_closed_forms.py` |
| `stiffquad_degeneracy.py` | §2a.1's table: the std of the stiff-quadratic exact terminal state across 2×10⁵ paths, for the library's own `p = (1,1)` and the notebook's three sets | `uv run python docs/sciml-benchmark-data/stiffquad_degeneracy.py` |
| `calibrate_sciml_suite.py` | §2b's calibration: every shipped entry against four solvers one decision apart — a correct Euler–Maruyama, the one-decision-wrong variant, the best constant, and the transcribed closed form — run through the **real sandbox** on the real entries. Prints each gate's margin on both sides and the narrowest separation | `uv run python docs/sciml-benchmark-data/calibrate_sciml_suite.py` |

| `emit_harvest_spec.py` | writes `benchmark/references/sciml/harvest_spec.json` from `sciml_problems()` — the horizon, step grid, path count, seed, save grid and gate of each entry, plus sample points of this repo's closed forms for the Julia side to check the library's own `analytic` against. It exists so the comparator runs cannot drift off the problems the suite actually grades; the coefficients deliberately stay in `SDEProblemLibrary` | `uv run python docs/sciml-benchmark-data/emit_harvest_spec.py` |
| `comparator_tables.py` | every table in §3f, read out of `comparators.json` rather than transcribed: the measured stage counts, work to reach each entry's gate, the matched-work and matched-seconds comparisons, work to reach a fixed relative error, and the measured disagreement between the two cost axes | `uv run python docs/sciml-benchmark-data/comparator_tables.py benchmark/references/sciml/comparators.json` |

| `ks_gate_resolution.py` | §10a's diagnosis of the first pipeline run: it re-runs the kernel's own terminal-law KS test (`verifylib/kernel/sde/laws.ks_standardized`, same standardization, same `1.3581/√n` bar, same n = 50 000 and `dt`) on four solvers sharing one Brownian path — the run's three plans plus the **pathwise-exact** map. At seed 42 it reproduces the run's three statistics to 13 digits, which is what licenses the fourth row: a solver with *zero* error fails that gate, and fails it worse than the Euler–Maruyama that passed. Then nine seeds, and the mechanism — the statistic divides by an estimated standard deviation whose own relative error is 1.58% on this law, and reads it | `uv run python docs/sciml-benchmark-data/ks_gate_resolution.py` (~5 min) |
| `audit_plan_claims.py` | re-derives the 145 numbers §3f, §9a and §10a–10m quote from `comparators.json`, the two baseline result files, the step-10 runs (all three results files — `sciml_probe.json`, `sciml_rest/sciml_probe_rest.json` and `sciml_m09/sciml_probe_m09.json`, the last superseding the first's `RUN_ERROR` row for the same slug — plus the plan `SOLUTION.md`s for the fields the record does not carry) and `measurements/ks_gate_resolution.json` and `measurements/m05_cross_plan_compare.json`, and checks the plan still says each one. It is not a test — it reads gitignored result files — but it is what to run before trusting a figure quoted out of that document. It exists because it caught a real error: an over-broad line rewrite had replaced a row of §2b's measured save-grid table with a row of the step table, and nothing else would have noticed | `uv run python docs/sciml-benchmark-data/audit_plan_claims.py` |

`measurements/m05_cross_plan_compare.json` is §10m's route-(iii) measurement: `cli.py compare`
on M05's two decisive plan pairs at 1000, 20 000 and 200 000 paths. It is recorded rather than
re-run by the audit because it is the one claim in §10m that is not free — twelve sandbox solves
per row — and because the higher counts cannot be reached from the CLI at all (`--N=` is the PDE
grid level and is silently ignored on the SDE path, §7a defect 11), so they were driven through
`cli._compare_sde` with the threshold overridden. The finding is that the pairs read
INCONCLUSIVE at the graded count and **AGREE** at both higher ones: more sampling confirms the
plans agree in moment space instead of separating them, because M05's 622× is pathwise.

`measurements/ks_gate_resolution.json` is `ks_gate_resolution.py`'s output, kept so the audit can hold §10a's prose to it: the per-seed cells live under a `workspace/` that every later pipeline run overwrites, so the reproduction is the durable record, not the run.

`measurements/em_anchor_{julia,python}.json` are the two halves of the §5a anchor as measured
(julia 1.11.9 single-threaded on apple-m1; python 3.14.7 / numpy 2.4.6). The scripts that produce
them live beside the harvest, at `benchmark/references/sciml/em_anchor.{jl,py}`, because they are
comparator artefacts rather than evidence for a claim.

The first two are seeded, so the numbers in §2a are the numbers they print. The third runs solvers through the sandbox and takes a few minutes; its `rel err` column is deterministic because the suite's increments come from a fixed seed, so the §2b tables are reproducible too.

A note on its verdict column: the harness is told the plan claimed a pass, so a solver that fails its gate reads `OVERCLAIM` rather than `FAIL` — that is the pipeline verdict for a claimed pass that did not verify. The baseline path, which has no self-score, reads `FAIL` for the same run.

## Sources read in step 2

Fetched raw, never through a rendered page or a summarising agent:

| Source | Pin |
|---|---|
| `lib/SDEProblemLibrary/src/SDEProblemLibrary.jl` | `SciML/DiffEqProblemLibrary.jl`, tag `SDEProblemLibrary-v1.2.4` → commit `d77baf216b3a77e591a9aabd10032090053b84f1`; sha256 `58adc9b105d5c40fedf17ffd99ad403c160e4f905c2694437073ad5f4ac84dbb` |
| `benchmarks/NonStiffSDE/{BasicSDEWorkPrecision,BasicSDEWeakWorkPrecision,HighOrderWeakSDEWorkPrecision,LotkaVolterraSDE}.jmd` | `SciML/SciMLBenchmarks.jl` `master`, read 2026-11-10 |
| `benchmarks/StiffSDE/{QuadraticStiffness,StochasticHeat}.jmd` | same |
| `benchmarks/{NonStiffSDE,StiffSDE}/{Project,Manifest}.toml` | same — the version set §3a now inherits rather than chooses |
| `SciMLBase/src/solutions/rode_solutions.jl` | `SciML/SciMLBase.jl` `master`, read 2026-11-10 — the `:l2` vs `:final` definitions behind §2a.2 |

The library file is **unchanged** between the pinned tag and what the current notebooks resolve to
(`SDEProblemLibrary = 1.2.4` in both benchmark manifests), so the formulas and the published
comparisons refer to the same equations.
