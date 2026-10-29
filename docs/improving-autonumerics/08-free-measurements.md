# 08 — The free measurements, run

**Status:** measured 2026-10-29, on this machine, from the artefacts in `workspace/` and
`benchmark/results/benchop-step10/`. No pipeline calls, no credits, no conductor. Every number
below is reproducible from the scripts in §6.

[06](06-retest-protocol.md) §1 said to run four offline measurements before implementing anything,
because two of them could disconfirm the plan before a line of code was written. They have been
run. **One of them did.**

The short version: [F0](03-fix-selection.md#f0)/[F2](03-fix-selection.md#f2) — wire the graded
price into the kernel — is necessary and **is not sufficient**, and the folder as written would
have shipped it and found the ranking unchanged. The quality index does not fail to read accuracy.
It reads it, saturates, and then ranks on wall-clock time.

---

## §1 What was run

| measurement | source | cost | outcome |
|---|---|---|---|
| Seed replication | [06](06-retest-protocol.md) §1 | 22 solves, ~70 min CPU | **done** — the loss is real and seed-robust at ~1.7× |
| Estimator validation | [06](06-retest-protocol.md) §1 step 3 | free, offline | **done** — the raw estimate orders plans 7/10; the index built on it scores 2/10 |
| [F11](05-fix-loop-and-harness.md#f11) | B01's two plan dirs | free, offline | **answered** — and it generalises past B01 |
| [F15](05-fix-loop-and-harness.md#f15) | B02's three plan dirs | free, offline | **answered** — and B02 is worse than recorded |

A fifth result fell out of the second and is the most consequential thing in this document: §2.

---

## §2 The quality index is anti-correlated with accuracy {#sqi-inverted}

**This is new, it was measured by running the shipped ranker on the shipped metrics blocks, and it
changes the priority order of the whole folder.**

### What `verifylib/cli.py rank` prints today

```
$ uv run python -m verifylib.cli rank workspace/sde_heston_dnt
plan                         score  cert  provenance     sqi      acc      ord      str      res     cost  solve_s
1-euler-maruyama               9.0    no  analytic     1.000    1.000        -    1.000        -    1.000   57.888
2-drift-implicit-variance-leg  9.0    no  analytic     0.976    1.000        -    1.000        -    0.845   58.915
3-qe-exact-variance            9.0    no  analytic     0.846    1.000        -    1.000        -    0.000   64.866

best: 1-euler-maruyama
note: no plan's <metrics> block carries the input for order -- the index is the mean of the rest.
```

Three plans, identical `score`, identical `provenance`, identical `structure`, identical
`accuracy`. The index spreads them 1.000 / 0.976 / 0.846, and **every bit of that spread is
`cost`**. The winner is the fastest plan by 7 seconds out of 65. The plan it put last,
`3-qe-exact-variance`, is the most accurate of the three.

### Three of the five components are dead or constant

Measured across all eleven B01–B04 plans:

| component | weight | measured | why |
|---|---|---|---|
| `accuracy` | 0.35 | **1.000 on 11 of 11** | saturated — see below |
| `order` | 0.25 | **None on 11 of 11** | `ranking.order` requires `design_order`; nothing sets it on an SDE |
| `structure` | 0.20 | 0.085 – 1.000 | live |
| `consistency` | 0.10 | **None on 11 of 11** | reads `dynkin_z`; Dynkin is `unavailable` at BENCHOP's path counts |
| `cost` | 0.10 | 0.000 – 1.000 | live, and pool-relative |

`sqi` renormalises over what was measurable, which is the right rule. The consequence here is that
the live index is

> **SQI = ⅔ · structure + ⅓ · cost**

on every BENCHOP plan. A component declared at a tenth of the index is deciding a third of it.

### `accuracy` saturates at exactly 1.0

[`ranking.headroom_decades`](../../verifylib/ranking.py#L155) snaps the error up to its own error
bar — `err = max(err, uncertainty(summary))`, where `uncertainty` is `ci_mult × mc_se_rel`, so
2 × the sampling standard error. [`achievable_decades`](../../verifylib/ranking.py#L169) then uses
**the same half-width** as the ceiling. So any plan whose `estimated_rel_error` is inside twice its
own Monte-Carlo noise gets `headroom_fraction = log10(tol/half_width) / log10(tol/half_width)`,
which is 1.0 exactly.

Measured, `estimated_rel_error ÷ mc_se_rel`:

| | plan | est_err | mc_se_rel | ratio | accuracy |
|---|---|---|---|---|---|
| B01 | `1-euler-maruyama-absorbing` | 3.68e-03 | 2.17e-03 | 1.70 | 1.000 |
| B01 | `2-exact-conditional-besq0` | 7.07e-04 | 2.49e-03 | 0.28 | 1.000 |
| B02 | `1-euler-maruyama-absorbed` | 7.52e-03 | 2.64e-01 | **0.03** | 1.000 |
| B02 | `2-exact-vol-em-absorbed` | 7.91e-03 | 1.81e-01 | **0.04** | 1.000 |
| B02 | `3-aks-decorrelated-em` | 7.37e-03 | 2.84e-01 | **0.03** | 1.000 |
| B03 | `1-full-truncation-log-euler` | 2.55e-03 | 3.66e-03 | 0.70 | 1.000 |
| B03 | `2-drift-implicit-variance` | 1.32e-03 | 3.63e-03 | 0.36 | 1.000 |
| B03 | `3-qe-exact-variance` | 3.26e-03 | 3.65e-03 | 0.89 | 1.000 |
| B04 | `1-euler-maruyama` | 7.53e-03 | 8.15e-03 | 0.92 | 1.000 |
| B04 | `2-drift-implicit-variance-leg` | 9.82e-03 | 8.12e-03 | 1.21 | 1.000 |
| B04 | `3-qe-exact-variance` | 4.95e-03 | 8.25e-03 | 0.60 | 1.000 |

Every ratio is below the 2.0 bar. Eleven of eleven snap, and the snap lands them all on the
ceiling.

The snap itself is correct and [`uncertainty`](../../verifylib/ranking.py) argues for it well:
two plans whose errors differ by less than the combined half-width differ by sampling noise, and
ranking them apart is reading the seed. **What is wrong is where the tie lands.** Declaring the
tie at the component's *maximum* means accuracy can never push any plan down, so its 0.35 of the
weight contributes nothing to the ordering and is redistributed, by renormalisation, onto
structure and cost.

### `ranking.order` has never fired, on anything

`ranking.order` needs `design_order` in the metrics block.

- The only assignment in the repo is [`verifylib/kernel/driver.py:103`](../../verifylib/kernel/driver.py#L103),
  `m["summary"]["design_order"] = m["plan"]["spatial_order"]`, which is in the **PDE** driver.
  `verifylib/kernel/sde/driver.py` never sets it. The SDE path writes `order_floor`, which
  `ranking.order` does not read as the design rate.
- Across **all 88 `<metrics>` blocks in `workspace/`** — 23 SDE, 65 PDE — `design_order` appears in
  **zero**, and `ranking.order()` returns `None` for **all 88**.
- 22 plans declare `spatial_order` in frontmatter, all PDE, all from the no-closed-form pools,
  all evaluated before `design_order` landed in d5bb447 (2026-10-02). For those, the CLI's
  "Re-run `evaluate` to refresh" would work.

For an SDE it would not. The note is wrong on the SDE side, and it is the only warning the system
gives that a quarter of the index is missing.

### The index is not uninformative — it is inverted

| | most accurate plan | is it the slowest? | SQI picks | shipped / best (seed 42) |
|---|---|---|---|---|
| B01 | `2-exact-conditional-besq0` | **yes** | `1-euler-maruyama-absorbing` | 9.98× |
| B02 | `3-aks-decorrelated-em` | **yes** | `1-euler-maruyama-absorbed` | 1.26× |
| B03 | `3-qe-exact-variance` | **yes** | `2-drift-implicit-variance` | 1.03× |
| B04 | `3-qe-exact-variance` | **yes** | `1-euler-maruyama` | 2.89× |

**In four pools of four the most accurate plan is the slowest, and in four of four the index ranks
it last or near-last.** Within-pool pairs where the higher SQI is the more accurate plan: **2 of 10**
at seed 42, and **8 of 30** pooled over three seeds (binomial p = 0.008 against a coin flip). The
index is not failing to measure quality; it is measuring something that runs the other way.

This is [02](02-root-causes.md)'s M2 — cost decides ties — confirmed numerically and worse than
M2 stated. M2 had cost as the fifth lexicographic key after the index. It is also a third of the
index.

### And the information the index destroys was already there

The raw `estimated_rel_error` the kernel already computes, read straight, without the snap:

| | raw estimator picks | truth | |
|---|---|---|---|
| B01 | `2-exact-conditional-besq0` | `2-exact-conditional-besq0` | ✓ |
| B02 | `3-aks-decorrelated-em` | `3-aks-decorrelated-em` | ✓ |
| B03 | `2-drift-implicit-variance` | `3-qe-exact-variance` | ✗ |
| B04 | `3-qe-exact-variance` | `3-qe-exact-variance` | ✓ |

**3 of 4 pools, and 7 of 10 pairwise** — against the index's 0 of 4 and 2 of 10, computed from the
same numbers. This is [06](06-retest-protocol.md) §1 step 3's estimator-validation gate, and the
estimator passes it: it separates B04's `1-euler-maruyama` (0.1741) from `3-qe-exact-variance`
(0.0603), which §1 named as the test it had to pass.

It is a moment error standing in for a price error, so it should not work this well and the
agreement may not survive on problems where the two diverge. The point stands regardless: **the
kernel is not short of signal. The index is throwing signal away.**

---

## §3 Seed replication: the loss is real, and smaller than seed 42 says {#seeds}

Every B01–B04 plan re-solved at seeds 101 and 202 at the problem's own `(num_paths, dt)` contract,
graded by the harness's own gate. 22 solves. Seed 42 is the graded record.

| | plan | seed 42 | seed 101 | seed 202 | mean | CV |
|---|---|---|---|---|---|---|
| B01 | `1-euler-maruyama-absorbing` ★ | 0.2584 | 0.4982 | 0.3586 | 0.3718 | 32% |
| B01 | `2-exact-conditional-besq0` | 0.0259 | 0.2478 | 0.1101 | **0.1279** | 88% |
| B02 | `1-euler-maruyama-absorbed` ★ | 0.9594 | 0.6103 | **1.1380** | 0.9026 | 30% |
| B02 | `2-exact-vol-em-absorbed` | 0.9098 | **1.0963** | **1.0118** | 1.0060 | 9% |
| B02 | `3-aks-decorrelated-em` | 0.7592 | 0.3938 | 0.8863 | **0.6798** | 38% |
| B03 | `1-full-truncation-log-euler` | 0.5803 | 0.1325 | 0.1322 | **0.2817** | 92% |
| B03 | `2-drift-implicit-variance` ★ | 0.4606 | 0.8019 | 0.3916 | 0.5514 | 40% |
| B03 | `3-qe-exact-variance` | 0.4478 | 0.2951 | 0.3141 | 0.3523 | 24% |
| B04 | `1-euler-maruyama` ★ | 0.1741 | 0.1357 | 0.1017 | 0.1372 | 26% |
| B04 | `2-drift-implicit-variance-leg` | 0.2246 | 0.0124 | 0.4725 | 0.2365 | 97% |
| B04 | `3-qe-exact-variance` | 0.0603 | 0.2202 | 0.0978 | **0.1261** | 66% |

★ shipped. **Bold** in the body: the pool's best at that seed, or a value above 1.0.

**Findings.**

1. **Per-plan error carries 9–97% relative spread across seeds, median ~35%.** Any claim of the
   form "plan A is 1.4× better than plan B" taken from one seed is not supported. [01](01-what-was-measured.md)
   §3's single-seed per-plan figures should be read with this attached.

2. **The selection loss survives.** Geometric mean shipped ÷ best-in-pool, per seed:
   **2.47× (42) / 3.79× (101) / 1.89× (202)**. Computed on the three-seed *means* — the right
   number for "what would better selection buy" — it is **1.69×**:

   | | shipped (mean) | best (mean) | ratio |
   |---|---|---|---|
   | B01 | 0.3718 | 0.1279 `2-exact-conditional-besq0` | 2.91× |
   | B02 | 0.9026 | 0.6798 `3-aks-decorrelated-em` | 1.33× |
   | B03 | 0.5514 | 0.2817 `1-full-truncation-log-euler` | 1.96× |
   | B04 | 0.1372 | 0.1261 `3-qe-exact-variance` | 1.09× |

   So the folder's two competing figures were both wrong in the expected directions: the raw
   **2.18×** overstated it, and [07](07-review.md)'s noise-floored **1.33×** understated it.
   **1.69× is the measured answer on the four problems where it can be measured**, and the
   pre-registered target in [06](06-retest-protocol.md) §6 should be restated against it.

3. **The best plan is seed-stable in half the pools.** B01 (`2-exact-conditional-besq0`) and B02
   (`3-aks-decorrelated-em`) win at all three seeds; B03 and B04 change winner with the seed. So
   in two of four pools the pipeline reliably picks a worse plan, and in two it is choosing among
   plans that are genuinely close.

4. **B04's seed-42 record is misleading in the pipeline's favour and against it at once.** QE
   beat EM 0.0603 to 0.1741 at seed 42 — 2.9× — but on three-seed means they are 0.1261 to 0.1372,
   a 9% difference inside the noise. [01](01-what-was-measured.md) §3 and [02](02-root-causes.md)
   should not lean on B04 as a selection failure; it is a near-tie the ranker got slightly wrong.

5. **[F6](04-fix-planning.md#f6)'s 104× does not appear anywhere in this data.** The largest
   three-seed mean ratio on the suite is 2.91×. F6's mechanism argument stands on its own; the
   magnitude claim has no support here.

---

## §4 F11, answered: the order check punishes exact *sampling* {#f11-answered}

[F11](05-fix-loop-and-harness.md#f11) asked why B01's more accurate plan scored one point lower.
The answer is in the metrics blocks, no re-run required, and it is not a bug in the sense F11
expected.

```
1-euler-maruyama-absorbing    score 9   observed_order  0.4852   order_check true    scheme_family euler-maruyama
2-exact-conditional-besq0     score 8   observed_order -0.0251   order_check true    scheme_family other
```

Both against `order_floor: 0.5000` with a two-sided ±0.4 band. The first clears it; the second
does not, and [`score.rubric_sde`](../../verifylib/kernel/score.py#L522) drops a converged, clean,
invariant-passing plan from 9 to 7 — here 8 after `ceiling` — with the reason the kernel writes
itself:

> *the moments match the closed form, but the CRN order study reads a strong order of -0.025: the
> solver does not consume the supplied Brownian increments (manual §18), so nothing about its
> discretization was measured and it cannot be certified however well its moments match*

**And it is true, and it is not a defect.** The exact-conditional BESQ(0) step samples
`S' = ½·v·Γ(N)`, `N ~ Poisson(2S/v)`. That transition is exact — it carries *no* discretisation
bias in S — and it is not a function of a single Gaussian draw, so it cannot consume the supplied
increment. The CRN ladder shares only the Brownian path, the auxiliary Poisson and Gamma draws are
independent across levels, the pathwise differences do not shrink, and the measured order is zero.

The plan said so **before it was implemented**, in its own SOLUTION.md:

> *In the CRN order study the S component therefore contributes a pathwise difference that does not
> shrink with `dt`, and the measured strong order is expected to come back **near zero** […] a
> measured order near zero below the 0.5 floor — the rubric's 8 row applies […] **This caps the
> plan at 8.***

It predicted its own penalty, took it deliberately in exchange for removing S's discretisation
bias, and was then ranked below the plan that kept the bias. Nothing downstream can read that
declaration.

### The repo has already fixed the sibling case and not this one

`crn.abstention` has three escape hatches, and one of them is exactly this shape —
[`test_an_exactly_integrating_scheme_abstains_instead_of_failing`](../../verifylib/tests/test_kernel_sde_path_b.py#L663)
covers a scheme whose CRN differences collapse to **roundoff** (`s0 = 1.9e-16`), which abstains and
costs nothing. The distinction that matters:

| | differences | abstention | outcome |
|---|---|---|---|
| exact **integration** (`dX = dW` under EM; B02's lognormal σ leg) | → roundoff | fires, `roundoff_floor` | free |
| exact **sampling** (BESQ(0), non-central χ², Poisson-mixed) | → O(1), CRN broken | **does not fire** | hard fail |

Both mean "there is no discretisation error here to measure". The first is detected because the
differences vanish; the second is invisible because the differences are *large*, and the estimator
is confidently wrong — B01's reading is `-0.0251 ± 2×0.01261`, a tight interval around a
meaningless number, so neither the standard-error hatch nor the band-width hatch fires either.

**This is a new fix, not an open question.** It needs a declared exemption the kernel can read —
the plan already writes the fact in prose — and its home is beside the existing roundoff test.
Everything that follows from it is real: every exact-transition scheme in the SDE literature
(Broadie–Kaya, QE with an exact variance leg, BESQ, non-central χ²) is in this class, and those
are the schemes BENCHOP is testing.

### It is not confined to B01

The same metrics across B03 and B04 show the measurement is weak even when it passes:

| | plan | observed_order | floor | verdict |
|---|---|---|---|---|
| B03 | `1-full-truncation-log-euler` | 0.2986 | 0.5 | pass (band ±0.4) |
| B03 | `2-drift-implicit-variance` | 0.2934 | 0.5 | pass |
| B03 | `3-qe-exact-variance` | 0.2463 | 0.5 | pass |
| B04 | `1-euler-maruyama` | 0.3112 | 0.5 | pass |
| B04 | `2-drift-implicit-variance-leg` | 0.2345 | 0.5 | pass |
| B04 | `3-qe-exact-variance` | 0.2605 | 0.5 | pass |

Six plans, three distinct scheme families, every reading in 0.23–0.31 against a floor of 0.5 —
systematically ~half the declared order, discriminating nothing, and passing only because the band
is ±0.4 wide. On these problems the order study is not measuring the scheme. Which makes
§2's finding — that `ranking.order` is `None` anyway — less costly than it looks, and makes
[F6](04-fix-planning.md#f6)'s plan to *add* `FAMILY_ORDER_REQUIRED` entries a change that should
be justified against this table first.

---

## §5 F15, answered: B02 is an unresolved measurement, not a bad pool {#f15-answered}

[F15](05-fix-loop-and-harness.md#f15) called B02 "the closest thing here to a failure". The seed
replication says that was an understatement.

**B02's shipped plan fails the external BENCHOP gate at 1 of 3 seeds** (1.1380 at seed 202), and
`2-exact-vol-em-absorbed` fails at **2 of 3** (1.0963, 1.0118). The recorded 0.9594 pass was the
seed. This is the only problem on the suite where that is true, and the graded record shows it as
a clean `UNDERCLAIM`.

### The pool knew, and recorded it, and nothing acted

```
1-euler-maruyama-absorbed   est_err 7.521e-03   mc_se_rel 0.2635   resolved false   order_check unresolved
2-exact-vol-em-absorbed     est_err 7.912e-03   mc_se_rel 0.1811   resolved false   order_check unresolved
3-aks-decorrelated-em       est_err 7.371e-03   mc_se_rel 0.2837   resolved false   order_check unresolved
```

The Monte-Carlo standard error on the graded moment is **18–28% relative**, and the reported error
estimate is **25–38× smaller than it**. `resolved: false` is recorded on all three. The agreement
gate returned `INCONCLUSIVE (exit 3)`. Every signal the pipeline has says *this problem was not
measured at this path count* — and nothing in the loop converts "unresolved" into "raise
`num_paths`". All three plans plateaued at iter 2 with the evaluator finding "no defect".

That is the diagnosis, and it is not about scheme quality: **B02 is an unresolved-measurement
failure that no gate turns into an action.** The right fix is not another scheme in the pool, it
is a rule that `resolved: false` on the graded quantity blocks `state: stopped`.

### The `test_review` failure is a true positive — against the metrics block, not the prose

[06](06-retest-protocol.md) §7a recorded `test_review::test_no_false_positives_on_archived_reviews`
flagging `2-exact-vol-em-absorbed` for `observed_order 0.235`. Resolved:

- The order study is **per component**. Component 0 (S, the EM-stepped leg) measured
  **0.235 ± 0.1305**, inside the two-sided band → pass. Component 1 (σ, the exact lognormal leg)
  **abstained** — "both pathwise differences are at the roundoff scale of the terminal values
  (1.24e-15 against a floor of 5.42e-13)", the signature of an exact step.
- [`_worst_component`](../../verifylib/kernel/sde/driver.py) ranks fail < abstain < pass and binds
  to the abstaining component. Correct for the *verdict*: `order_check: unresolved`, which costs
  nothing.
- But the scalar `observed_order` carried into the metrics block is then **the abstaining
  component's number**, `-1.0772` — roundoff noise, not an order. `3-aks-decorrelated-em` reports
  the identical `-1.0772`, which confirms it: two different schemes cannot share an order to four
  decimals.

So the prose is right and the frontmatter is wrong. A single scalar cannot represent a
per-component study, and the component it picks is the one with nothing to report. The fix is to
publish `None` (or the binding *measured* component) when the binding component abstained.

**This is the same root cause as §4** in its other disguise: the order study has no representation
for "this component has no discretisation error", and what it reports instead is noise that either
fails the floor (B01) or is published as if it were a measurement (B02).

---

## §6 Reproducing this

Scripts, written outside the repo:

```
/private/tmp/claude-502/-Users-sahilmulki-Autonumerics/acf236b5-504c-4b87-955f-a814f873ce5c/scratchpad/free/driver.py    # one (plan, seed) -> harness_error
/private/tmp/claude-502/-Users-sahilmulki-Autonumerics/acf236b5-504c-4b87-955f-a814f873ce5c/scratchpad/free/seeds.py     # the 22-solve sweep, 3 at a time
/private/tmp/claude-502/-Users-sahilmulki-Autonumerics/acf236b5-504c-4b87-955f-a814f873ce5c/scratchpad/free/analyse.py   # §3's tables
/private/tmp/claude-502/-Users-sahilmulki-Autonumerics/acf236b5-504c-4b87-955f-a814f873ce5c/scratchpad/free/seeds.jsonl  # the 22 results
```

`driver.py` imports each `solver.py` with `sys.modules["plan_solver"] = mod` registered — the line
[`benchmark/runner.py:157`](../../benchmark/runner.py#L157) is missing, and whose absence voided
B07 ([F8](05-fix-loop-and-harness.md#f8)). It grades with the harness's own
`benchmark.verify._functional_tols`, and it reproduces the graded record exactly: B04's
`1-euler-maruyama` at seed 42 comes back **0.17408353947545374** against the recorded
**0.1740835394754518**.

§2's component table is `uv run python -m verifylib.cli rank workspace/<slug>` on the four pools.
The 88-block census is a walk over `workspace/*/plans/*/SOLUTION.md` calling `ranking.order` on
each parsed block.

**The scratchpad is session-scoped**, so all four files are also copied to
`benchmark/results/benchop-step10/free/`, beside the graded JSONs. That directory is gitignored
(`.gitignore:27`) like the rest of step 10 — **back it up outside the repo**; the 22 solves cost
~70 minutes of CPU to regenerate and the credits that produced the plans are spent.

---

## §7 What this changes

| fix | was | now |
|---|---|---|
| [F0](03-fix-selection.md#f0)/[F2](03-fix-selection.md#f2) | the fix | **necessary, not sufficient** — a price error inside its own MC noise will saturate `accuracy` exactly as the moment error does |
| **new, highest priority** | — | **unsaturate `accuracy`**: the 2×-noise snap must declare a tie *without* placing it at the component's ceiling |
| **new, high priority** | — | **`ranking.order` is dead on every SDE and on all 88 blocks on disk.** Either give the SDE path a `design_order` or drop the component's weight honestly |
| [F4](03-fix-selection.md#f4) (cost) | one of several | **the live index is ⅔ structure + ⅓ cost.** Cost is not a tiebreaker, it is a third of the decision |
| [F11](05-fix-loop-and-harness.md#f11) | "a measurement, not a fix" | **answered, and promoted to a fix**: exact *sampling* breaks CRN and is hard-failed where exact *integration* abstains for free |
| [F15](05-fix-loop-and-harness.md#f15) | "undiagnosed" | **answered**: B02 fails the gate at 1 of 3 seeds; `resolved: false` never blocks `stopped`; the `-1.0772` is an abstaining component's noise published as a measurement |
| [F13](05-fix-loop-and-harness.md#f13) | "refuse an estimate below its own noise" | **already implemented in `ranking`**, and it is the saturation. Re-scope it to the *reporting* layer, where B02's 25–38× gap is still unflagged |
| [F6](04-fix-planning.md#f6) | 104× | **no support in this data** (largest mean ratio 2.91×), and §4's table argues against adding order requirements before the order study is fixed |
| [06](06-retest-protocol.md) §6 target | 2.18× raw / 1.33× floored | **1.69×**, measured on three-seed means |
| [01](01-what-was-measured.md) §3 | single-seed per-plan errors | attach the 9–97% CV; B04 is a near-tie, not a 2.9× selection failure |
