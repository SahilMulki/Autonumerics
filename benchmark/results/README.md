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
| `sciml_m01_probe.json` | one pipeline run on M01, scheme-matched arm, taken as a cost probe before committing the other eight — 19.6 min, `VERIFIED_PASS`, and `wrong_plan_won: True`. Not a sweep; cited by [`docs/plan-sciml-benchmark.md`](../../../docs/plan-sciml-benchmark.md) §10a |
| `notable_n02_smoke.json` | the notable-suite smoke test ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) step 7, read in §8b): N02 in the **scheme-matched** arm, 2026-11-15, 242 s (resumed from `STATE.md` after the usage limit), `UNDERCLAIM` — kernel MC-inconclusive at the brief's 2 000 paths, harness strong error 1.6% |
| `notable_n02_smoke_unrestricted.json` | the same in the **unrestricted** arm, 595 s, `VERIFIED_PASS` — formulator evaluated at 20 000 paths, all three plans certified; `ledger_spec_defects` flags the path-count mismatch |
| `notable_9a_n01_before_kernel_fix.json` | step 9a's first unit ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8c): N01, 2026-11-15, 543 s, **`FAIL`** with three passing plans in the pool — the kernel did not yet pass `dW_sub`, so the ranker's tie default crowned the Euler control. Kept as measured; superseded by the re-run after the fixes |
| `notable_9a_scheme_matched.json` | step 9a, 2026-11-15: N01 `RUN_ERROR` (the conductor deleted its own plans block from STATE.md; the winner passes the harness at 0.85%) and N03 `VERIFIED_PASS`, 314 s — [`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8d |
| `notable_9a_n01_rerun.json` | N01's third attempt, 2026-11-16, 473 s, `FAIL`: the kernel measured the right order and the ranker tied the pool and shipped the Euler control (§8d). Workspace kept as `workspace-n01-rerun-tie/` |
| `notable_9e_rep1.json` | **the post-fix re-run** ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8f): N04 and N06 in the **scheme-matched** arm, 2026-11-19, one rep each, both `VERIFIED_PASS` (764 s / 1 792 s). The first pipeline run on this suite that records **cost** — $5.09 and $7.42, against the C0 baseline's $0.12–0.25 per rep. Both instrument fixes fired and both measured accurately: the Dynkin escalation reversed a false failure on N06 plan 1 (failed on three levels, passed on four) and abstained on the Euler control, and the bias retry repaired two plans carrying 3.82× and 3.55× of their variance tolerance in one cycle. **Neither fix is why the numbers moved**: N04 improved 2.9× with the instruction never firing (its winner certified on cycle 1, so the early exit ended the pool) and N06, where it did fire, came out slightly worse than run 1 — run-to-run scheme variance is the larger term, and one rep cannot separate them |
| `notable_9e_rep2.json` | **N06's second post-fix rep** ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8g): `scheme-matched`, 2026-11-19, `VERIFIED_PASS`, $12.24, 2 168.7 s, 6 iterations. The best N06 result on record (0.131 of gate) and the first with `wrong_plan_won: false`. **The first directly attributable repair on this suite**: the bias diagnostic read 3.54x of tolerance in removable variance bias, and the winner replaced a first-order backward-Euler drift substep with the exact logistic flow -- variance 33.9% -> 0.26%, mean 7.6% -> 0.92%. Two counterweights in the same run: plan 1 took the instruction, measured TR-BDF2 *worse* at the graded step and correctly kept its fitted theta; plan 3 rewrote itself into Strang splitting and declared `scheme_family: strang-splitting`, which `SDE_FAMILIES` does not carry, so the kernel read it as null -- the retry *generates* unrecognised families. Dynkin escalated twice more (plan 1 abstained at z = 18.41; plan 3 passed thin at z = 2.19 and D9 seed replication took the median over three seeds to `False`), the agreement gate returned its first cross-family AGREE on N06, and a formulator spec defect capped the pool at 9 for the third run running |
| `notable_9e_n08_rule6.json` | **N08, the held-out test of the two new manual rules** ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8j): `scheme-matched`, 2026-11-21, `UNDERCLAIM` (harness `verified_score: 10`, pipeline claimed 9), 39.3 min. N08 is named in neither rule 0c nor rule 6, and both apply to it -- a `V**1.5` diffusion for the Lamperti route, and a `1/V` that *is* a CIR process with exact moments for the moment-matching one. **The pipeline lands at 0.0057 of tolerance against C0's 0.2417, 42.2x better**, on `E[1/V]` 0.0016% / `Var[1/V]` 0.057% against C0's 0.801% / 2.417%; `wrong_plan_won: false` and the winner is `1-lamperti-strang-split`, which is rule 6's prescription almost verbatim. **It does not credit the rule**: no plan cites the manual, and §8i's N06 run reached three Lamperti/Strang plans with rule 6 *removed from disk*, so the capability predates it -- plan 2's own history ("was a Milstein noise sub-step scoring 3 in round 1") makes the bias retry a competing sufficient cause in this very run. The counterfactual that settles it is `notable_9e_n08_pinned.json`, below -- and it settles it against the rule. Both seed instruments were inert as predicted: N08 declares no analytic moments, so `seed_study` returns `not_applicable` and accuracy resolves through the functional branch instead |
| `notable_9e_n06_phase2b_live.json` | **N06 with both seed instruments on** ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8i): `scheme-matched`, 2026-11-21, `VERIFIED_PASS`, 44.4 min, `--seed-study 16` and `AUTONUMERICS_ACCURACY_SEEDS=16`, with `sde_manual.md` pinned to its pre-rule-6 commit `df2001c` so the run does not measure a rule written from this entry. **All three plans certified at 10**, so `score` could not break the tie -- and accuracy resolved pool-wide for the first time here, so `decided_by: sqi` picked the plan the harness independently ranks best on both the graded seed (0.0726) and the 16-seed median (0.0483), at `wrong_plan_won: false`. Stripping the three study fields back out of the same blocks ships `3-explicit-milstein-clipped` instead, whose harness median is 0.1986 -- **4.1x worse** -- which is what the instrument is worth on this pool. Phase 2b did **not** fire (`polish_cycle: not_run`): all three plans repaired themselves on the in-loop bias retry while still eligible, so the window it covers never opened. Two costs recorded: the retry collapsed the pool onto one family (`single_family_pool: true`, no cross-family agreement break), and the run would have scored three plans at 3 had `strang-splitting` not been added to `SDE_FAMILIES` first |
| `notable_9e_n04_rep2.json` | **N04's second post-fix rep** ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8g): `scheme-matched`, 2026-11-20, `VERIFIED_PASS`, $5.33, 809.5 s. Reproduces rep 1 **to all 17 digits** under a different plan name -- `--fresh` was passed and the plan text differs (`Y = arcsin(2X-1)` vs `Y = 2*arcsin(sqrt X)`), the same Lamperti transform up to a shift of pi/2, so the pipeline rediscovered the same scheme and the fixed seed returned the same sample. Zero within-condition spread. The winner certified on iteration 1 again, but its `moment_bias_actionable` is `False` and the bias is measured at +0.05%/+0.02% -- the early exit forfeited nothing, and §8f's reading of the winner-exit window as the binding constraint is not supported. This is the run that exposed the **seed-resolution finding** (§8g): re-grading both sides at 64 seeds collapses N04's reported 7.5x gap to 1.08x, reverses N03's, and shows seed 42 is the *worst of 120 draws* for the pipeline's own N04 scheme |
| `notable_9a_v2.json` | **step 9a complete** ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8e): the five order-bearing entries in the **scheme-matched** arm, 2026-11-16, one rep — N01, N03, N04, N06, N07 all `VERIFIED_PASS` (1 116 / 148 / 465 / 603 / 863 s). N07's first launch is in the same file as a `RUN_ERROR` with `graded_incomplete: true` — the machine slept on battery, the network dropped, and the monotonic-clock timeout took 11 281 s of wall clock to fire. Read against the C0 baseline the verdicts separate nothing (15/15 vs 5/5) and the accuracy does: N04 3.6% vs 0.17% and N06 2.2% vs 0.55% on the variance, both from the noise sub-step of an otherwise-correct scheme. Graded **before** the two instrument fixes §8e records, so its kernel scores are pre-fix: three Dynkin failures in it (N07 plan 1, N06 plans 2 and 3) are false, and the N06/N07 winners are the wrong plans |

### baselines/

Completed C0/C1 one-shot sweeps, the selections made from them, and the flat summaries.

| File | What it is |
|---|---|
| `oneshot_no_closed_form.json` | the C0@Opus matched sweep on the no-closed-form suite — **44** units, 2026-09-27 |
| `oneshot_no_closed_form.selection.json` | `baseline_select.py`'s choice per problem, made with the conductor's own ranking key |
| `oneshot_ncf_smoke.json` | the cheap plumbing smoke test — **2** units, 2026-09-26 |
| `oneshot_ncf_smoke.selection.json` | its selection output |
| `oneshot_sciml_scheme_matched.json` | the C0@Opus baseline on the sciml suite, **scheme-matched** arm — **9** units, naked only, `--c0-max-turns 4`, 2026-11-10. Zero of eight degenerate ladders |
| `oneshot_sciml_unrestricted.json` | the same nine in the **unrestricted** arm, where evaluating the closed form is a permitted answer — **8 of 8** degenerate ladders at 5e-17 to 1.5e-15. The two files are the *same problems with different briefs*; `compare.py` refuses to put them in one column, and the arm is recorded on every record |
| `oneshot_notable_scheme_matched.json` | the C0@Opus baseline on the **notable** suite ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) step 8, read in §8a), **scheme-matched** arm — 9 problems × 3 reps = 27 units, 2026-11-14, naked, `--c0-max-turns 4`, **27/27 `PASS`**, $3.70, 10.6 min. Every record carries `wall_s`, `cost_usd`, `input_tokens` and `output_tokens` under `run` |
| `oneshot_notable_unrestricted.json` | the same baseline in the **unrestricted** arm — 27 units, **26/27 `PASS`**, $2.90, 11.3 min. The one `FAIL` (N06 rep 1) is a sign error in the Itô correction of an otherwise-correct logit-chart splitting; flipping the sign reproduces the reference (§8a) |
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
  Since 2026-10-21 this is **enforced rather than remembered**: `run.py` and `oneshot.py`
  both refuse a selection that reaches outside the `main` suite unless they are given a
  path, and the refusal names one that follows this convention. `--list` is unaffected,
  and a main-suite run — including a one-problem smoke test — is exactly as it was.
  (`benchmark/setup.py:guard_results_path`, plan-benchop-slv.md §6 step 9.)
- `*.selection.json` — the output of `baseline_select.py` for the run of the same stem.
- `<name>.pre-<change>-bak` — a copy taken immediately before a behaviour change, so the
  before-state survives the fix.
- `<name>.void-<reason>-<date>.json` — a run that completed but must not be quoted, with the
  reason in the name.
- `<name>.CLOBBERED-<date>` — a file an unguarded run overwrote.
- `<name>_<date>.json` — a one-off study; the date is part of the identity, since a plan cites it.
| `notable_9e_n08_pinned.json` | **The N08 counterfactual: rule 6 withdrawn** ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8k): `scheme-matched`, 2026-11-21, `RUN_ERROR`, 40.0 min of a 2400 s budget. `sde_manual.md` pinned to `df2001c` -- 399 lines, **zero occurrences of `Lamperti`** -- with everything else held; the run recorded `agent_file_hashes["references/sde_manual.md"] = fa237e15277663d8` against the rule-6 run's `d56974dc2594f147`, so the pin is on the record rather than asserted. **Without the rule the pipeline still ships `reciprocal-chart-strang-splitting` at 0.00728 of tolerance, 33.2x past C0** -- a Strang composition in `Y = 1/V`, a *different* chart from the rule-6 run's `V**-0.5`, from a manual that never states the move. No plan cites the manual. **The attribution claim is therefore measured and negative**; the capability claim stands. Two caveats: the harness grade is sound (`verify.status: ok`, `verified_score: 10`, both functionals level-consistent) but `best_plan_source: fallback` with `best_plan_tied: true`, so 0.00728 is a floor on the arm rather than its selected value. The `RUN_ERROR` is a harness-pairing error, not the mid-run laptop sleep: the kill landed at `wall_seconds` exactly 2400.0 against `wall_clock_seconds` 21495.2, so the monotonic-active timer spent no budget during sleep, and the real cause is that the command omitted `--timeout` and took the 2400 s default where the run it pairs against had 3600 s and used 2337.5 s. Incidentally the pool is **family-diverse** here -- `strang-splitting`/`implicit`/`tamed-milstein` -- so the agreement gate did real work and capped all three plans at 8 on two genuine disagreements |
| `notable_9e_n02_n05_n09.json` | **Step 9b: the last three head-to-head entries** ([`docs/plan-notable-sde.md`](../../docs/plan-notable-sde.md) §8l): `scheme-matched`, 2026-11-21, 3600 s budget, 17.5 + 16.9 + 17.1 min, all three `completed` with no timeout. **All three beat C0 and all three graded `passed: true` at `verified_score: 10`**, closing the head-to-head at 9 of 9: N02 `0.005279` vs C0 `0.162280` (**30.7x**, `UNDERCLAIM`), N05 `0.039974` vs `0.195372` (**4.89x**, `UNDERCLAIM`), N09 `0.039203` vs `0.143391` (**3.66x**, `VERIFIED_PASS`). `wrong_plan_won: false` on all three, and where a per-plan error exists the ranker took the harness-best plan outright. **Two kernel defects found in one run**: the Dynkin generator check is diffusion-only, so on N05's jump diffusion it fires identically on every correct solver (plans 2 and 3 scored 3, plan 1 scored 9 on `dynkin unresolved`, and no solver change can reach it without falsifying the compensated drift) -- recorded as a gap bounding the run rather than worked around; and `dynkin.py:203` reads a top-level `initial_condition` while N02's spec carried `verification.initial_condition`, voiding `dynkin_ok` pool-wide with `check-spec` exiting 0 throughout. Also: Phase 2b declined for the first time on a genuinely eligible certified winner (`moment_bias_frac_tol` 4.5e-13 -- no removable bias), N09's winner was separated from a same-score certified sibling by `sqi` at 0.432 vs 0.423 while that sibling's harness error is **7.7x worse**, and `harness_by_plan` carries no error metric at all for a pathwise ground truth, so `wrong_plan_won` on N02 reduces to the score ladder |
