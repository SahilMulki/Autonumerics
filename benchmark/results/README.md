# benchmark/results

Run artifacts. Almost all of it is generated and gitignored, but this directory is an **evidence
store, not scratch space**: several files here are the measurement behind a claim made somewhere
else in the repo, cited by name. A tidy-up that deleted them would leave those claims
unsupported, so the layout below exists to make it obvious what each file is and what still
depends on it.

**Nothing here should be deleted to save space.** Some of these runs cost hours of real API usage
and cannot be regenerated as they were — the model, the prompts and the harness have all changed
since. If something must go, check first whether a document cites it (`grep -rn '<filename>' docs/
benchmark/design/`).

## Layout

The organizing rule: **the top level is what the tools read and write by default**; everything
beneath it is a finished artifact, grouped by what produced it.

```
results/
├── README.md                 this file                              [committed]
├── REPORT.md                 the headline report                    [committed]
├── matched_attempts.json     frozen input to the matched baseline   [committed]
├── results.json              the live pipeline results file
├── COMPARISON.md             the live head-to-head report
├── logs/                     per-problem transcripts + per-run settings files
├── pipeline/                 completed C2 pipeline sweeps and their reports
├── baselines/                completed C0/C1 sweeps, their selections and summaries
├── studies/                  targeted measurement runs cited by a plan
├── superseded/               void, clobbered and pre-fix files — kept as the record
└── snapshots/                frozen read-only copies
```

### Committed files

Three files are tracked; the rest of the directory is ignored by
[`.gitignore`](../../.gitignore) (the `benchmark/results/*` rule, with one negation per committed
file). The negations must name files **at this level** — git does not descend into an ignored
directory, so a file inside `pipeline/` or `studies/` cannot be un-ignored without restructuring
the rule. That is the main reason the three committed files stay at the top.

| File | Why it is in the repo |
|---|---|
| `REPORT.md` | so the numbers quoted in the root README are checkable without re-running anything |
| `matched_attempts.json` | it is a frozen **input**, not an output — the per-problem attempt counts the matched baseline is granted as reps, read from `workspace/` on 2026-09-26 and frozen so the comparison is reproducible |
| `README.md` | this file |

### Top level — the live files

| File | Written by | State |
|---|---|---|
| `results.json` | `run.py` (crash-safe, appended per problem); read by `report.py` and `compare.py` | the **August 28-problem main-suite run** (2026-08-29) |
| `REPORT.md` | `run.py` at the end of a sweep, or `report.py --out` | generated 2026-09-25 from `pipeline/no_closed_form_v2.json` — the 3-problem second wave, **not** from `results.json` |
| `COMPARISON.md` | `compare.py` | 2026-07-29, 36 checkable problems, and a **pipeline column only** — no baseline has been run against it |
| `logs/` | `run.py` | see below |

> **Two footguns worth knowing.**
>
> `report.py` defaults to `--results results/results.json`, but the committed `REPORT.md` was
> generated from `pipeline/no_closed_form_v2.json`. A bare `uv run python benchmark/report.py`
> therefore **overwrites the committed report with the older August run**. Pass `--results` and
> `--out` explicitly.
>
> `compare.py` and `oneshot.py` default to `results/oneshot_results.json`, **which does not
> exist** — the live baseline file was clobbered on 2026-09-27 and never rebuilt. The surviving
> copies are in `superseded/`. This is why `COMPARISON.md` has only one column.

### logs/

67 files, flat — `run.py` writes `logs/{slug}.log`, it does not make a directory per run.

- **56** per-problem conductor transcripts, `{slug}.log`, one per problem the pipeline has run.
- **7** hand-named sweep logs: `c0_full.log`, `c0_retry.log`, `c2a.log`, `c2b_burgers.log`,
  `c2b_sonnet.log`, `ncf_c0_opus.log`, `ncf_c0_smoke.log`.
- **4** `{slug}.settings.json` — the per-run isolated settings file `run.isolated_settings` writes
  beside the transcript, denying reads of every *other* problem's workspace. Kept because it is
  the record of what a given run was actually allowed to see.

### pipeline/

Completed C2 sweeps, kept aside from the live `results.json`.

| File | What it is |
|---|---|
| `pipeline_sonnet.json` | the Sonnet-pinned C2b sweep — **35** problems, 2026-08-12. Byte-identical to `snapshots/c2b_sonnet_35of50_2026-08-13.json`, which is the frozen copy |
| `no_closed_form.json` | the first no-closed-form wave — **4** problems, 2026-09-19. The input to [`docs/no-closed-form-findings.md`](../../docs/no-closed-form-findings.md) |
| `NO_CLOSED_FORM_REPORT.md` | the report rendered from it |
| `no_closed_form_v2.json` | the second wave — **3** problems (KdV, Gross–Pitaevskii, double-well), 2026-09-26. **This is what the committed `REPORT.md` was generated from** |

### baselines/

Completed C0/C1 one-shot sweeps, the selections made from them, and the flat summaries.

| File | What it is |
|---|---|
| `oneshot_no_closed_form.json` | the C0@Opus matched sweep on the no-closed-form suite — **44** units, 2026-09-27 |
| `oneshot_no_closed_form.selection.json` | `baseline_select.py`'s choice per problem, made with the conductor's own ranking key |
| `oneshot_ncf_smoke.json` | the cheap plumbing smoke test — **2** units, 2026-09-26 |
| `oneshot_ncf_smoke.selection.json` | its selection output |
| `c0_summary.tsv` | per-problem C0 pass rates, naked vs manual, with average cost / output tokens / wall time (2026-08-10) |
| `c2b_summary.tsv` | per-problem C2b verdicts with self-score, verified score, errors and observed order (2026-08-17) |

### studies/

Targeted runs made to answer one question, each cited by a document. These are not benchmark
sweeps.

| File | What it measured |
|---|---|
| `sde_baseline_2026-10-04.json` | the before-state for the SDE verification work — per-plan recorded values at a named commit. Cited by [`docs/plan-sde-verification-gaps.md`](../../docs/plan-sde-verification-gaps.md) |
| `sde_dynkin_diagnostic_2026-10-04.json` | 10 Dynkin-residual runs across plans — order, z-score, bar, path count |
| `sde_dynkin_seed_study_2026-10-04.json` | the same diagnostic across 11 seeds, which is how the residual's seed sensitivity was established. Cited twice by the SDE plan |
| `sde_suite_baseline_2026-10-14.json` | a **test-suite** timing baseline, not a solver run: per-file durations, totals, deselected and skipped detail for the SDE tests |
| `spec_ks_limitrun.json` | a captured `problem_spec.json` (21 fields) from a Kuramoto–Sivashinsky run that hit a limit |
| `spec_schrodinger_crashrun.json` | a captured `problem_spec.json` (18 fields) from a Schrödinger run that crashed |

### superseded/

Files that are **not** the current answer, kept because they are the record of what went wrong.
`benchmark/design/BASELINE_COMPARISON_PLAN.md` cites several of them by name.

| File | Why it is kept |
|---|---|
| `oneshot_results.json.pre-maxturns-bak` | the August C0/C1 sweep as it stood before the turn-cap change — **492 units, 2.1 MB** |
| `oneshot_results.json.pre-retry-bak` | the same sweep before the retry change — **504 units, 2.0 MB** |
| `oneshot_results.json.CLOBBERED-2026-09-27` | the live baseline file as it was when a run with no `--results-json` overwrote it: **44** units where there had been hundreds. The accident this directory's naming convention exists to prevent |
| `oneshot_no_closed_form.void-turncap2-2026-09-27.json` | the **void** C0@Opus run — 44 units, of which 36 produced no scorable solver because Opus behaves like an agent under a one-shot harness. Kept as the evidence for that finding, not as data |
| `results.stale-jul16.json` | the July pipeline results file, 37 problems, moved aside rather than overwritten |
| `pipeline_sonnet.json.pre-decontam-bak` | the C2b sweep before the decontamination fix — 28 problems |
| `pipeline_sonnet.opus-planners-invalid.bak.json` | a 5-problem C2b run invalidated by Opus planners being used where Sonnet was intended |

> The two `pre-*-bak` files are the **largest baseline datasets in the tree** — 492 and 504 units
> against 44 in the clobbered live file. "Backup" in the name undersells them: the Aug-7 baseline
> was deliberately never re-measured, so these are the most complete C0/C1 data that exists here.
> Do not delete them as redundant copies.

### snapshots/

Frozen point-in-time copies, kept deliberately. The two files are **read-only** (`r--r--r--`), so
a stray write cannot touch them.

| Entry | What it is |
|---|---|
| `c2b_sonnet_35of50_2026-08-13.json` | the C2b sweep frozen at 35 of 50 problems. Byte-identical to `pipeline/pipeline_sonnet.json` |
| `c2b_summary_35of50_2026-08-13.tsv` | its summary table |
| `ch_workspace_snapshot_2133/` | a whole pipeline workspace preserved: `problem.md`, `problem_spec.json`, `STATE.md` and four plan directories. This is the Cahn–Hilliard coarsening run in which the formulator `cat`-ed its manufactured-solution twin's spec and used it as a template — the cross-workspace leak that `run.isolated_settings` was written to close. It is the only artifact of that leak, and unlike the other two it has no tracked test fixture |

## Where the old flat paths went

This directory was flat until 2026-10-17. Documents written before then — in particular the
recorded commands and run log in `benchmark/design/BASELINE_COMPARISON_PLAN.md` — cite the old
paths. Those transcripts are deliberately **not** rewritten, because they record commands that
were really run; use this table to resolve them.

| Recorded as `results/…` | Now at |
|---|---|
| `no_closed_form.json`, `no_closed_form_v2.json`, `NO_CLOSED_FORM_REPORT.md`, `pipeline_sonnet.json` | `pipeline/` |
| `oneshot_no_closed_form*.json`, `oneshot_ncf_smoke*.json`, `c0_summary.tsv`, `c2b_summary.tsv` | `baselines/` |
| `sde_*.json`, `spec_*.json` | `studies/` |
| `results.stale-jul16.json`, `oneshot_results.json.*`, `*.void-*`, `pipeline_sonnet.json.pre-decontam-bak`, `pipeline_sonnet.opus-planners-invalid.bak.json` | `superseded/` |
| `archive/*` | `snapshots/` |
| `results.json`, `REPORT.md`, `COMPARISON.md`, `matched_attempts.json`, `logs/` | unchanged |

## Naming conventions

Worth following, because the clobbering accident above is what happens when a run is given no
explicit output path.

- **Always pass `--results-json`** for anything that is not the main sweep. `run.py` derives
  `logs/` and `REPORT.md` from the directory of the file you name, so pointing it at
  `results/pipeline/<name>.json` keeps a sweep's whole output together.
- `*.selection.json` — the output of `baseline_select.py` for the run of the same stem.
- `<name>.pre-<change>-bak` — a copy taken immediately before a behaviour change, so the
  before-state survives the fix.
- `<name>.void-<reason>-<date>.json` — a run that completed but must not be quoted, with the
  reason in the name.
- `<name>.CLOBBERED-<date>` — a file an unguarded run overwrote.
- `<name>_<date>.json` — a one-off study; the date is part of the identity, since a plan cites it.
