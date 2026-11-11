# SciMLBenchmarks comparator artefacts

Everything in this directory exists so that the `sciml` suite's comparison against
`StochasticDiffEq.jl` can be re-run and audited. See `docs/plan-sciml-benchmark.md`
§1a (why nothing published could be cited), §3 (the pinned environment) and §5 (the
work axis and the timing anchor).

**`comparators.json` and `harvest_spec.json` are answer-key-adjacent.** The first holds
the reference error values this suite's entries are graded near and the schemes they are
compared against; the second holds every gate plus M09's three exact moments. Both are
covered by `pipeline-settings.json`'s `Read(./benchmark/**)` deny rules and matched by
name in `verifylib/leakage.py`, for the reason that module's docstring gives: a plan does
not have to quote a path to have been told what it is being measured against.

| File | What it is |
| ---- | ---------- |
| `nonstiff/{Project,Manifest}.toml` | the upstream `benchmarks/NonStiffSDE` environment, **inherited verbatim** — not re-resolved. Reproduces all five SDE notebooks. |
| `stiff/{Project,Manifest}.toml` | the upstream `benchmarks/StiffSDE` environment, likewise verbatim. Its closure is a strict subset of the non-stiff one with zero version disagreements; it is kept because its *direct* deps differ. |
| `harvest_spec.json` | generated from `benchmark/problems.py` by `docs/sciml-benchmark-data/emit_harvest_spec.py`. Carries the configuration — horizon, step grid, path count, seed, save grid, gate — plus sample points of this repo's closed forms for `harvest.jl` to check the library's own `analytic` against. **Not** the coefficients: those stay in `SDEProblemLibrary`. |
| `harvest.jl` | runs the published fixed-step schemes on the nine problems and writes `comparators.json`. Single-threaded, warm-up discarded, fixed step only, stage counts measured rather than recalled. |
| `comparators.json` | the output: one row per (problem, scheme, `dt`) with error, work and time, plus the environment, the analytic cross-checks, the notebook-reproduction log and any upstream defects hit. |
| `em_anchor.jl`, `em_anchor.py` | the same Euler–Maruyama in both stacks, on two problem shapes, so that a seconds column can be divided by a measured language factor instead of compared raw (§5). |

## Reproducing

```bash
curl -fsSL https://install.julialang.org | sh      # juliaup
juliaup add 1.11.9 && juliaup default 1.11.9

mkdir -p ~/sciml-comparators/nonstiff && cd ~/sciml-comparators/nonstiff
cp <repo>/benchmark/references/sciml/nonstiff/*.toml .
JULIA_NUM_THREADS=1 julia +1.11.9 --project=. -e 'using Pkg; Pkg.instantiate()'

cd <repo>
uv run python docs/sciml-benchmark-data/emit_harvest_spec.py
JULIA_NUM_THREADS=1 julia +1.11.9 --project=~/sciml-comparators/nonstiff \
    benchmark/references/sciml/harvest.jl

JULIA_NUM_THREADS=1 julia +1.11.9 benchmark/references/sciml/em_anchor.jl \
    --out /tmp/em_anchor_julia.json
uv run python benchmark/references/sciml/em_anchor.py \
    --out /tmp/em_anchor_python.json --julia /tmp/em_anchor_julia.json
```

The project is built **outside** the repository working tree on purpose: it is
third-party code, and the download rules in this repo treat it as untrusted data.
Only the two resolved `.toml` pairs come back in.
