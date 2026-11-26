# Notable SDE suite, frozen run

The whole nine-problem notable suite, run at one revision in one logical invocation, with
the winning solver for every entry. This is the run `../notable-sde-artifacts/README.md`
said was needed: that folder holds six of nine winners because a same-arm re-run
overwrites a plan directory, and N01, N06 and N08 had been lost that way.

- **Code**: tag `notable-frozen-v4`, commit `0435bf2`. (The tag's number trails the data
  file's by one; they are the same revision.)
- **Data**: `notable_frozen_v5.json`, copied here because `benchmark/results/*` is
  gitignored, so the repo otherwise keeps no record of it. `REPORT.md` is the generated
  report for the same file.
- **Arm**: `scheme-matched`. **Models**: `mixed:opus+sonnet` (agent defaults, no
  `--model` override).
- **Cost**: $44.32, 2.0 h of pipeline wall time, nine problems, three plans each.

Each winner's `solver.py` mtime falls inside the start/end window of the run that reported
it -- checked for all nine -- which is what makes these the artifacts behind the numbers
rather than a later run's.

## Outcome

`verified-pass 5 | underclaim 4 | overclaim 0 | fail 0 | blow-up 0 | run-error 0`, and
`wrong_plan_won: false` on all nine: the ranker never shipped a plan the harness grades
worse than a sibling by more than the sampling noise on the deciding statistic.

| id | slug | winning plan | verdict | kernel | harness |
|---|---|---|---|---|---|
| N01 | `sde_clark_cameron` | `1-levy-area-milstein-drift-corrected` | UNDERCLAIM | 9 | 10/10 |
| N02 | `sde_kubo_oscillator` | `2-stochastic-heun` | UNDERCLAIM | 9 | 10/10 |
| N03 | `sde_aitsahalia_rate` | `1-lamperti-trapezoidal` | VERIFIED_PASS | 10, surrogate | 10/10 |
| N04 | `sde_wright_fisher_mutation` | `1-beta-moment-matched` | VERIFIED_PASS | 10 | 10/10 |
| N05 | `sde_merton_jump_call` | `1-milstein-cv` | VERIFIED_PASS | 10 | 10/10 |
| N06 | `sde_sis_epidemic` | `1-logit-strang-heun` | VERIFIED_PASS | 10, surrogate | 10/10 |
| N07 | `sde_langevin_baoab` | `2-stochastic-trapezoidal` | UNDERCLAIM | 9 | 10/10 |
| N08 | `sde_three_halves_vol` | `3-log-implicit-euler` | UNDERCLAIM | 8 | 10/10 |
| N09 | `sde_geometric_asian` | `1-milstein-bridge-cv` | VERIFIED_PASS | 10 | 10/10 |

**No entry overclaims, and all four underclaims pass the external harness at 10/10.** The
four are not pipeline errors; each is a refusal to certify:

- **N01, N02** -- `strong_gate_not_measured`. The problem declares a pathwise strong-error
  gate; the kernel's strong error is `crn.strong_error`'s self-convergence estimate, which
  ranks the pool but is not calibrated to decide a gate, so the gate goes unevaluated.
- **N07** -- the generator (Dynkin) check ran and could not reach a verdict: the
  residual's own convergence rate was unreadable.
- **N08** -- `cross_plan_disagreement`. Two plans differ by 0.234 against a 0.1 threshold,
  so no winner may be certified until a third family or a refinement breaks the tie.

## Against the one-shot Opus baseline

`../../benchmark/results/baselines/oneshot_notable_scheme_matched.json`, three reps per
problem, same arm, same harness. Both sides are scored as *worst graded metric divided by
its own gate*, so the column is dimensionless and 1.0 is the pass line. A gap smaller than
`harness_error_margin` -- the Monte-Carlo standard error of the deciding statistic -- is
not a difference this measurement can resolve, and is recorded as a tie.

| id | pipeline | Opus (median) | gap | margin | ratio | call |
|---|---|---|---|---|---|---|
| N01 | 0.22338 | 0.22338 | +0.00000 | 0.00499 | 1.00x | tie |
| N02 | 0.00264 | 0.08114 | -0.07850 | 0.00004 | **30.74x** | **win** |
| N03 | 0.04424 | 0.01236 | +0.03188 | 0.06704 | 0.28x | tie |
| N04 | 0.01691 | 0.01691 | -0.00000 | 0.03727 | 1.00x | tie |
| N05 | 0.37179 | 0.19537 | +0.17642 | 0.21171 | 0.53x | tie |
| N06 | 0.07265 | 0.05505 | +0.01760 | 0.04994 | 0.76x | tie |
| N07 | 0.13505 | 0.10858 | +0.02646 | 0.02577 | 0.80x | loss |
| N08 | 0.01252 | 0.24165 | -0.22914 | 0.00519 | **19.31x** | **win** |
| N09 | 0.28662 | 0.14339 | +0.14322 | 0.23962 | 0.50x | tie |

**2 wins, 6 ties, 1 loss.** Opus is bit-reproducible across its three reps on every entry
but N03, where two reps give 1.236e-3 and one gives 8.163e-3; the median is used.

Three things about this table need saying wherever it is cited.

1. **The aggregate is carried by two entries.** The geometric mean of the ratio column is
   **1.44x better** across all nine; drop N02 and N08 and it is **1.56x worse**. On the
   other seven the pipeline ties twice at the identical number and sits behind on the
   point estimate five times -- inside the noise every time, but consistently in the same
   direction. Anyone can compute this, so it belongs in the claim rather than beneath it.
2. **N07's loss is marginal**: the gap is 0.02646 against a margin of 0.02577, 1.03x the
   bar. One more rep either way could make it a tie.
3. **The margins here are single-sided.** Each is the bar on the pipeline's own
   measurement; a pipeline-vs-Opus comparison carries that noise on both sides, so the
   honest tie bands are roughly sqrt(2) wider than shown. That only moves entries toward
   "tie", never away from it.

N08 is the entry worth reading twice. The pipeline wins 19.31x; it is also the only
baseline entry where Opus does not reach 10 on the harness (`verified_score 8`, all three
reps) while the pipeline's winner passes 10/10. And the pipeline's own kernel scores it
**8** and refuses to certify, on a cross-plan disagreement. The largest accuracy win in
the suite is one the system declined to claim.

## Known defects under these rows

Two kernel bugs are currently worked around at the spec level rather than fixed, and both
sit under published entries:

- **N03 and N06 are both certified on `provenance: surrogate`**, "a surrogate reference
  passed its trust guard and the moments match it" -- and on `sde_aitsahalia_rate` the
  Kolmogorov truncation guard's 1.5x widening crosses `X = 0`, the shift comes back NaN,
  and the guard reports `surrogate_ok: true` on it. NaN comparisons are false, so a guard
  written as "fail if the shift exceeds the tolerance" passes everything. Two of the five
  verified passes rest on that guard.
- **N05** needed a spec workaround for the `generator_nd` d=1 broadcast bug (a dict
  generator with bare `"X"` test functions broadcasts to `(M, M)` and times the Dynkin
  check out pool-wide).
