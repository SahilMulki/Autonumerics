# Notable-suite winning solvers, as reported

The winning plan's `solver.py` and `SOLUTION.md` for the notable SDE suite entries in
`docs/paper-results-review.md` §3c, copied out of `workspace/` on 2026-11-22 before a
frozen re-run cleared it. `workspace/` is gitignored and holds exactly one copy of each
plan directory, so this is the only durable record of the code behind those numbers.

| id | slug | winning plan | solver.py mtime |
|---|---|---|---|
| N02 | `sde_kubo_oscillator` | `2-stratonovich-heun` | 2026-11-21 13:51 |
| N03 | `sde_aitsahalia_rate` | `1-lamperti-trapezoidal` | 2026-11-16 02:10 |
| N04 | `sde_wright_fisher_mutation` | `1-lamperti-reflected-euler` | 2026-11-20 18:38 |
| N05 | `sde_merton_jump_call` | `1-jump-adapted-milstein-control-variate` | 2026-11-21 14:08 |
| N07 | `sde_langevin_baoab` | `2-stochastic-heun` | 2026-11-16 13:55 |
| N09 | `sde_geometric_asian` | `1-log-exact-bridge-conditioned` | 2026-11-21 17:26 |

Each mtime agrees with the run that reported the entry, which is the check that makes
these the artifacts behind the published figures rather than a later run's.

## Three entries are missing, and that is the finding

`N01`, `N06` and `N08` are **not** here. A plan directory is keyed by
`workspace/<slug>/plans/<plan>`, and the arm is part of the workspace path but the *run*
is not -- so a second run of the same problem in the same arm writes over the first one's
plans. It happened three times:

- **N08** (42.21x, the suite's largest reported margin). The `notable_9e_n08_rule6.json`
  run of 2026-11-21 00:54 won with `1-lamperti-strang-split`. The `n08_pinned` run at
  06:16 the same day -- the deliberate rule-6-absent counterfactual of
  `plan-notable-sde.md` §8k -- then overwrote the directory. What survives on disk is
  `1-log-coordinate-tamed-milstein`, which §8k names as the *pinned* run's winner (33.2x).
- **N06.** Recorded winner `1-logit-strang-splitting`; on disk are
  `1-lamperti-split-step-backward-euler`, `2-drift-implicit-milstein` and
  `3-explicit-milstein-clipped`.
- **N01.** Recorded winner `2-levy-area-milstein` from the 2026-11-16 run. The frozen
  re-run of 2026-11-22, stopped after its first problem, produced a plan of the same name
  and overwrote the file at 02:25.

`benchmark/run.py` already carries the comment for the cross-*arm* version of this hole
("`oneshot.py` learned this on SciML: two arms keyed on the same artifact directory, and
the second run wrote its solver.py over the first's"). Keying the workspace on the arm did
not close the same-arm case.

The three are recoverable only by re-running, which the frozen re-run of the whole suite
does -- writing all nine at one revision, in one invocation, with artifacts that match
their records.

---

## Superseded by the frozen run

The frozen re-run happened on 2026-11-23/26 and is in `../notable-frozen-v5/`: all nine
winners at one revision, in one invocation, each solver's mtime inside the window of the
run that reported it. The three entries missing here (N01, N06, N08) are recovered there.

This folder is kept as the record of the artifacts behind `docs/paper-results-review.md`
§3c, which the frozen run's numbers replace, and of the overwrite hole described above --
which `benchmark/run.py` still has.
