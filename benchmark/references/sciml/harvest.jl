# harvest.jl -- run StochasticDiffEq.jl's published fixed-step schemes on the nine
# problems of the `sciml` suite and write `comparators.json`.
#
# WHY THIS EXISTS (plan-sciml-benchmark.md §1a, §3c): the three published
# SciMLBenchmarks pages are plots. They print almost no numbers, the only printed
# timings are up to 99.98% compilation, and the runs used 128 threads on hardware
# nobody else has. None of that can be cited, so everything is re-measured locally at
# pinned versions and written down as numbers.
#
# WHAT IT IS NOT: this file does not define any problem's coefficients. Those come from
# `SDEProblemLibrary` at the pinned version, which is their authority. The horizons,
# step grids, path counts and seeds come from `harvest_spec.json`, which is generated
# from `benchmark/problems.py`. The two halves are tied together by `analytic_checks`:
# sample points of this repo's own closed forms that the library's `analytic` is
# checked against before a single scheme is run. If the Python entry and the Julia
# problem are not the same problem, the run aborts instead of producing a table that
# merely looks like a comparison.
#
# CONVENTIONS, all three forced by what the published pages got wrong (§3b):
#   * single-threaded -- run with JULIA_NUM_THREADS=1; the thread count is recorded.
#   * no compilation in the clock -- every timed cell is warmed up once, then timed
#     over 5 repetitions and the MINIMUM is reported.
#   * fixed step only -- §11: the benchmark contract fixes `dt`, so adaptive results
#     are outside the comparison and are not run.
#
# WORK AXIS (§5, H4): `evals = paths * steps * stages`, where `stages` is MEASURED by
# counting drift and diffusion evaluations per step rather than read off a paper. The
# measurement differences two step counts, so solver start-up does not leak in.
#
# Usage, from the pinned project:
#   JULIA_NUM_THREADS=1 julia +1.11.9 --project=/path/to/nonstiff \
#       benchmark/references/sciml/harvest.jl [--out comparators.json] [--only M01,M09]

using StochasticDiffEq
using SDEProblemLibrary
using Random
using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# A minimal JSON reader and writer.
#
# The pinned environment is inherited verbatim from the upstream benchmark manifests
# (§3a) and contains no JSON package. Adding one would rewrite `Manifest.toml` and
# break the one property that makes the comparator run reproducible, so the ~90 lines
# below are the price of leaving the manifest alone. They cover exactly the subset of
# JSON this script reads and writes, and `--selftest` round-trips them.
# ---------------------------------------------------------------------------

mutable struct JParser
    s::String
    i::Int
end

function jskip!(p::JParser)
    while p.i <= lastindex(p.s) && isspace(p.s[p.i])
        p.i = nextind(p.s, p.i)
    end
end

function jvalue(p::JParser)
    jskip!(p)
    c = p.s[p.i]
    if c == '{'
        return jobject(p)
    elseif c == '['
        return jarray(p)
    elseif c == '"'
        return jstring(p)
    elseif startswith(SubString(p.s, p.i), "true")
        p.i += 4
        return true
    elseif startswith(SubString(p.s, p.i), "false")
        p.i += 5
        return false
    elseif startswith(SubString(p.s, p.i), "null")
        p.i += 4
        return nothing
    else
        return jnumber(p)
    end
end

function jobject(p::JParser)
    out = Dict{String, Any}()
    p.i = nextind(p.s, p.i)  # past '{'
    jskip!(p)
    if p.s[p.i] == '}'
        p.i = nextind(p.s, p.i)
        return out
    end
    while true
        jskip!(p)
        k = jstring(p)
        jskip!(p)
        @assert p.s[p.i] == ':' "expected ':' at $(p.i)"
        p.i = nextind(p.s, p.i)
        out[k] = jvalue(p)
        jskip!(p)
        c = p.s[p.i]
        p.i = nextind(p.s, p.i)
        c == ',' && continue
        @assert c == '}' "expected ',' or '}' at $(p.i)"
        break
    end
    return out
end

function jarray(p::JParser)
    out = Any[]
    p.i = nextind(p.s, p.i)  # past '['
    jskip!(p)
    if p.s[p.i] == ']'
        p.i = nextind(p.s, p.i)
        return out
    end
    while true
        push!(out, jvalue(p))
        jskip!(p)
        c = p.s[p.i]
        p.i = nextind(p.s, p.i)
        c == ',' && continue
        @assert c == ']' "expected ',' or ']' at $(p.i)"
        break
    end
    return out
end

function jstring(p::JParser)
    @assert p.s[p.i] == '"' "expected string at $(p.i)"
    p.i = nextind(p.s, p.i)
    buf = IOBuffer()
    while true
        c = p.s[p.i]
        p.i = nextind(p.s, p.i)
        if c == '"'
            break
        elseif c == '\\'
            e = p.s[p.i]
            p.i = nextind(p.s, p.i)
            if e == 'n'; print(buf, '\n')
            elseif e == 't'; print(buf, '\t')
            elseif e == 'r'; print(buf, '\r')
            elseif e == 'b'; print(buf, '\b')
            elseif e == 'f'; print(buf, '\f')
            elseif e == 'u'
                hex = SubString(p.s, p.i, p.i + 3)
                p.i += 4
                print(buf, Char(parse(UInt16, hex, base = 16)))
            else
                print(buf, e)
            end
        else
            print(buf, c)
        end
    end
    return String(take!(buf))
end

function jnumber(p::JParser)
    j = p.i
    while j <= lastindex(p.s) && (isdigit(p.s[j]) || p.s[j] in ('-', '+', '.', 'e', 'E'))
        j = nextind(p.s, j)
    end
    txt = SubString(p.s, p.i, prevind(p.s, j))
    p.i = j
    if occursin('.', txt) || occursin('e', txt) || occursin('E', txt)
        return parse(Float64, txt)
    end
    return parse(Int, txt)
end

json_parse(txt::AbstractString) = jvalue(JParser(String(txt), 1))

jesc(s::AbstractString) = replace(replace(replace(s, '\\' => "\\\\"), '"' => "\\\""),
    '\n' => "\\n")

function json_write(io::IO, v, indent::Int = 0)
    pad = " " ^ indent
    if v isa AbstractDict
        isempty(v) && return print(io, "{}")
        println(io, "{")
        ks = collect(keys(v))
        for (n, k) in enumerate(ks)
            print(io, pad, " ", "\"", jesc(String(k)), "\": ")
            json_write(io, v[k], indent + 1)
            println(io, n == length(ks) ? "" : ",")
        end
        print(io, pad, "}")
    elseif v isa AbstractVector
        isempty(v) && return print(io, "[]")
        println(io, "[")
        for (n, x) in enumerate(v)
            print(io, pad, " ")
            json_write(io, x, indent + 1)
            println(io, n == length(v) ? "" : ",")
        end
        print(io, pad, "]")
    elseif v === nothing
        print(io, "null")
    elseif v isa Bool
        print(io, v ? "true" : "false")
    elseif v isa AbstractString || v isa Symbol
        print(io, "\"", jesc(String(v)), "\"")
    elseif v isa Integer
        print(io, v)
    elseif v isa AbstractFloat
        if !isfinite(v)
            # JSON has no Inf/NaN. A non-finite number here means a scheme blew up,
            # which is a result worth keeping, so it is written as a string and the
            # reader is expected to notice.
            print(io, "\"", isnan(v) ? "NaN" : (v > 0 ? "Inf" : "-Inf"), "\"")
        else
            print(io, repr(v))
        end
    else
        print(io, "\"", jesc(string(v)), "\"")
    end
    return nothing
end

function json_string(v)
    io = IOBuffer()
    json_write(io, v)
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# The comparator sets, in the published order (§1).
#
# A scheme appears only where the published pages apply it: the SRA family needs
# additive noise and the SRI family diagonal noise, so running the additive set on a
# diagonal problem would produce a column of errors that says nothing about either
# library. `EM` and `RKMil` appear everywhere because the pages use them as the
# baseline everywhere, and because `EM` is also this repo's own anchor (§5).
# ---------------------------------------------------------------------------

const SCHEME_SETS = Dict{String, Vector{String}}(
    "diagonal" => ["EM", "RKMil", "SRI", "SRIW1", "SRIW2", "SOSRI", "SOSRI2"],
    "additive" => ["EM", "RKMil", "SRIW1", "SRA1", "SRA2", "SRA3", "SOSRA", "SOSRA2"],
    "stiff" => ["EM", "RKMil", "SRIW1", "ImplicitEM", "ImplicitRKMil"],
    "weak" => ["EM", "SimplifiedEM", "DRI1", "PL1WM", "RDI1WM", "RDI2WM"],
)

const ALGS = Dict{String, Any}(
    "EM" => () -> EM(),
    "RKMil" => () -> RKMil(),
    "SimplifiedEM" => () -> SimplifiedEM(),
    "SRI" => () -> SRI(),
    "SRIW1" => () -> SRIW1(),
    "SRIW2" => () -> SRIW2(),
    "SOSRI" => () -> SOSRI(),
    "SOSRI2" => () -> SOSRI2(),
    "SRA1" => () -> SRA1(),
    "SRA2" => () -> SRA2(),
    "SRA3" => () -> SRA3(),
    "SOSRA" => () -> SOSRA(),
    "SOSRA2" => () -> SOSRA2(),
    "ImplicitEM" => () -> ImplicitEM(),
    "ImplicitRKMil" => () -> ImplicitRKMil(),
    "DRI1" => () -> DRI1(),
    "PL1WM" => () -> PL1WM(),
    "RDI1WM" => () -> RDI1WM(),
    "RDI2WM" => () -> RDI2WM(),
)

# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------

# The weak page's problem is defined inline in the notebook, not in the library
# (§2a.5), so it is transcribed here -- the only coefficients in this file.
function f_noncommutative!(du, u, p, t)
    du[1] = -273 // 512 * u[1]
    du[2] = -1 // 160 * u[1] - (-785 // 512 + sqrt(2) / 8) * u[2]
    return nothing
end

function g_noncommutative!(du, u, p, t)
    du[1, 1] = 1 // 4 * u[1]
    du[1, 2] = 1 // 16 * u[1]
    du[2, 1] = (1 - 2 * sqrt(2)) / 4 * u[1]
    du[2, 2] = 1 // 10 * u[1] + 1 // 16 * u[2]
    return nothing
end

function build_problem(spec)
    name = spec["julia_problem"]
    T = Float64(spec["T"])
    if name == "noncommutative_inline"
        return SDEProblem(f_noncommutative!, g_noncommutative!, [1.0, 1.0], (0.0, T),
            noise_rate_prototype = zeros(2, 2))
    end
    base = getfield(SDEProblemLibrary, Symbol(split(name, '.')[end]))
    if haskey(spec, "p") && spec["p"] !== nothing
        pv = Tuple(Float64.(spec["p"]))
        return remake(base, tspan = (0.0, T), p = pv)
    end
    return remake(base, tspan = (0.0, T))
end

"""Check the library's own `analytic` against this repo's closed form at the sample
points `harvest_spec.json` carries. This is the join between the two stacks: it is
what makes the comparator numbers numbers *about the same problem*."""
function check_analytic(prob, spec; rtol = 1e-10)
    checks = get(spec, "analytic_checks", Any[])
    isempty(checks) && return (0, 0, 0.0)
    an = prob.f.analytic
    an === nothing && error("$(spec["id"]): library problem carries no analytic")
    u0 = prob.u0
    nok = 0
    worst = 0.0
    for c in checks
        t = Float64(c["t"]); W = Float64(c["W"]); want = Float64(c["x"])
        got = Float64(first(an(u0, prob.p, t, W)))
        rel = abs(got - want) / max(abs(want), 1e-300)
        worst = max(worst, rel)
        rel <= rtol && (nok += 1)
    end
    return (nok, length(checks), worst)
end

# ---------------------------------------------------------------------------
# Measured stage count (H4, §5)
#
# `stages` is the number of (drift, diffusion) evaluation rounds a scheme spends per
# step -- 1 for Euler-Maruyama, 2 for RKMil, more for the Rossler schemes. It is
# MEASURED here rather than read off a paper, by counting evaluations over 4 steps and
# over 8 steps and differencing: start-up evaluations cancel, so what is left is the
# per-step cost. The two known anchors (EM = 1, RKMil = 2) are what this is checked
# against; an implicit scheme has no fixed count at all, because its nonlinear solve
# iterates, and the measured number is reported for it with that caveat.
# ---------------------------------------------------------------------------

function counting_problem(prob, nf::Ref{Int}, ng::Ref{Int})
    f0 = prob.f.f
    g0 = prob.g
    if StochasticDiffEq.SciMLBase.isinplace(prob)
        fc = (du, u, p, t) -> (nf[] += 1; f0(du, u, p, t))
        gc = (du, u, p, t) -> (ng[] += 1; g0(du, u, p, t))
        return SDEProblem(fc, gc, prob.u0, prob.tspan, prob.p;
            noise_rate_prototype = prob.noise_rate_prototype)
    end
    fc = (u, p, t) -> (nf[] += 1; f0(u, p, t))
    gc = (u, p, t) -> (ng[] += 1; g0(u, p, t))
    return SDEProblem(fc, gc, prob.u0, prob.tspan, prob.p)
end

function measure_stages(prob, algname, dt)
    counts = Int[]
    fs = Int[]
    gs = Int[]
    for nsteps in (4, 8)
        nf = Ref(0); ng = Ref(0)
        cp = counting_problem(prob, nf, ng)
        t0 = prob.tspan[1]
        cp = remake(cp, tspan = (t0, t0 + nsteps * dt))
        try
            solve(cp, ALGS[algname](); dt = dt, adaptive = false,
                save_everystep = false, save_start = false, seed = 20261110,
                maxiters = 10^8)
        catch e
            return (nothing, nothing, nothing, "stage count failed: $(typeof(e))")
        end
        push!(fs, nf[]); push!(gs, ng[]); push!(counts, nsteps)
    end
    fper = (fs[2] - fs[1]) / (counts[2] - counts[1])
    gper = (gs[2] - gs[1]) / (counts[2] - counts[1])
    return (fper, gper, max(fper, gper), nothing)
end

# ---------------------------------------------------------------------------
# One (problem, scheme, dt) cell
# ---------------------------------------------------------------------------

"""Strong error on the pathwise problems.

Three error numbers are recorded, not one, because the published pages and this repo's
gate do not measure the same thing (§2a.2):

  * `strong_err_rel_final` -- the l2-over-paths terminal error divided by the l2 norm
    of the exact terminal states. THIS is the number the `sciml` suite gates on, and
    the only one of the three that is comparable across problems.
  * `strong_err_abs_final` -- the same numerator, undivided.
  * `sde_err_final_abs` -- `sol.errors[:final]`, computed by SciMLBase from the
    library's own `analytic`. Recorded as an independent cross-check of the two
    numbers above: it is the same quantity arrived at by the other stack, so a
    disagreement would mean this script is reading the wrong field.

SciMLBenchmarks' own `error_estimate = :l2` -- an ABSOLUTE error over a timeseries
(§2a.2) -- is NOT recorded, and the reason is an upstream defect this run reproduced:
`saveat` with three or more save points on an SDE problem that carries an `analytic`
raises `BoundsError` inside `SciMLBase.calculate_solution_errors!`
(`rode_solutions.jl:221`), because that function indexes the saved noise `sol.W.W` with
the solution's save index and `saveat` desynchronises the two. `calculate_error = false`
does not suppress it. One and two save points are fine; three is not. See
`upstream_defects` in the output.

Paths are driven by a fixed seed vector, so every scheme at every step size on a given
problem sees the same Brownian paths: differences between columns are differences
between schemes.
"""
function strong_cell(prob, algname, dt, npaths, seeds)
    alg = ALGS[algname]()
    num = fill(NaN, npaths)
    exact = fill(NaN, npaths)
    l2sum = 0.0
    nl2 = 0
    nfail = 0
    errmsg = nothing
    for i in 1:npaths
        sol = nothing
        try
            sol = solve(prob, alg; dt = dt, adaptive = false,
                save_everystep = false, seed = seeds[i], maxiters = 10^9)
        catch e
            nfail += 1
            errmsg === nothing && (errmsg = "$(typeof(e))")
            continue
        end
        if isempty(sol.u)
            nfail += 1
            continue
        end
        num[i] = Float64(first(sol.u[end]))
        if sol.u_analytic !== nothing && !isempty(sol.u_analytic)
            exact[i] = Float64(first(sol.u_analytic[end]))
        end
        if sol.errors !== nothing && haskey(sol.errors, :final)
            v = Float64(sol.errors[:final])
            if isfinite(v)
                l2sum += v ^ 2
                nl2 += 1
            end
        end
    end
    ok = findall(i -> isfinite(exact[i]), 1:npaths)
    if isempty(ok)
        return Dict{String, Any}("status" => nfail == npaths ? "solver_error" : "no_truth",
            "paths_failed" => nfail, "error" => errmsg)
    end
    d2 = 0.0
    e2 = 0.0
    nbad = 0
    for i in ok
        if !isfinite(num[i])
            nbad += 1
            continue
        end
        d2 += (num[i] - exact[i])^2
        e2 += exact[i]^2
    end
    nfin = length(ok) - nbad
    rel = nbad > 0 ? Inf : sqrt(d2 / nfin) / max(sqrt(e2 / nfin), 1e-300)
    return Dict{String, Any}(
        "status" => nbad > 0 ? "nonfinite" : "ok",
        "paths_failed" => nfail,
        "paths_nonfinite" => nbad,
        "strong_err_rel_final" => rel,
        "strong_err_abs_final" => nbad > 0 ? Inf : sqrt(d2 / nfin),
        "sde_err_final_abs" => nl2 > 0 ? sqrt(l2sum / nl2) : nothing,
        "exact_rms_final" => sqrt(e2 / max(nfin, 1)),
    )
end

"""Weak error on the non-commutative problem (M09).

All three of the suite's observables are recorded, not only SciML's `E[X1(T)^2]`: that
one decouples from the channel structure entirely (the second-moment system's first row
is exactly `[-1, 0, 0]`, plan §2b finding 5), so a scheme that collapses the two noise
channels is invisible in it. The Monte-Carlo standard error of each is recorded beside
it, because at 1e5 paths it is the standard error and not the scheme that sets what is
resolvable here (§10 prediction 4).
"""
function weak_cell(prob, algname, dt, npaths, seeds, truth)
    alg = ALGS[algname]()
    names = ["second_moment_x1", "cross_moment_x1x2", "second_moment_x2"]
    s = zeros(3)
    q = zeros(3)
    n = 0
    nfail = 0
    errmsg = nothing
    for i in 1:npaths
        sol = nothing
        try
            sol = solve(prob, alg; dt = dt, adaptive = false, save_everystep = false,
                save_start = false, seed = seeds[i], maxiters = 10^8)
        catch e
            nfail += 1
            errmsg === nothing && (errmsg = "$(typeof(e))")
            continue
        end
        u = sol.u[end]
        v = (u[1]^2, u[1] * u[2], u[2]^2)
        if !all(isfinite, v)
            nfail += 1
            continue
        end
        for k in 1:3
            s[k] += v[k]
            q[k] += v[k]^2
        end
        n += 1
    end
    if n == 0
        return Dict{String, Any}("status" => "solver_error", "paths_failed" => nfail,
            "error" => errmsg)
    end
    out = Dict{String, Any}("status" => "ok", "paths_failed" => nfail, "paths_used" => n)
    for (k, nm) in enumerate(names)
        m = s[k] / n
        var = max(q[k] / n - m^2, 0.0)
        se = sqrt(var / n)
        tr = Float64(truth[nm])
        out["mean_$nm"] = m
        out["stderr_$nm"] = se
        out["weak_err_rel_$nm"] = abs(m - tr) / abs(tr)
        out["weak_err_sigmas_$nm"] = se > 0 ? abs(m - tr) / se : nothing
    end
    return out
end

"""Minimum of five timed repetitions after one discarded warm-up (§3b).

Timed on `paths_timed` paths rather than all of them: the quantity the anchor of §5
needs is time per path-step, the full-path error run already dominates the wall clock,
and multiplying the most expensive cells by six to gain a third significant figure on a
number that is only ever used as a ratio would not pay.

CAVEAT, measured rather than assumed: at small step counts this number is dominated by
the per-path `solve` setup, not by stepping. Euler-Maruyama on a scalar problem measures
~215 ns per path-step at 25 steps and far less at 15 625, because the ~5 us of `solve`
overhead is amortised over more steps. Any ratio taken from these timings -- the §5
anchor above all -- must be taken at the largest step count available, or it is a
comparison of two languages' call overhead."""
function time_cell(prob, algname, dt, paths_timed, seeds; reps = 5)
    alg = ALGS[algname]()
    function sweep()
        for i in 1:paths_timed
            solve(prob, alg; dt = dt, adaptive = false, save_everystep = false,
                seed = seeds[i], maxiters = 10^9)
        end
        return nothing
    end
    try
        sweep()   # discarded: this is the call that pays for compilation
    catch
        return nothing
    end
    best = Inf
    for _ in 1:reps
        t0 = time_ns()
        sweep()
        best = min(best, (time_ns() - t0) / 1e9)
    end
    return best
end

# ---------------------------------------------------------------------------
# Environment capture (§3b): every number above is meaningless without it.
# ---------------------------------------------------------------------------

function environment()
    pkgs = Dict{String, Any}()
    try
        @eval using Pkg
        for (_, info) in Base.invokelatest(Pkg.dependencies)
            if info.is_direct_dep && info.version !== nothing
                pkgs[info.name] = string(info.version)
            end
        end
        # The four packages the comparison actually rests on are recorded even when
        # they are indirect deps of the inherited project.
        for (_, info) in Base.invokelatest(Pkg.dependencies)
            if info.name in ("StochasticDiffEq", "SDEProblemLibrary",
                "DiffEqNoiseProcess", "DiffEqDevTools", "SciMLBase") &&
               info.version !== nothing
                pkgs[info.name] = string(info.version)
            end
        end
    catch e
        pkgs["_error"] = "$(typeof(e))"
    end
    return Dict{String, Any}(
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "blas_threads" => try
            LinearAlgebra.BLAS.get_num_threads()
        catch
            nothing
        end,
        "cpu" => string(Sys.CPU_NAME),
        "cpu_threads" => Sys.CPU_THREADS,
        "machine" => string(Sys.MACHINE),
        "total_memory_gb" => round(Sys.total_memory() / 2^30, digits = 1),
        "project" => Base.active_project(),
        "pkg_versions" => pkgs,
        "harvested_at" => string(round(Int, time())),
    )
end

"""Parse the per-cell log of a notebook reproduction run into the record.

§3c: *"the script must also record, per cell, whether it reproduced the published
result or hit the `BoundsError` of §1a.2"*. The reproduction is a separate run (the
notebooks are Weave documents, not functions), so its log is folded in here rather than
re-executed, and `comparators.json` stays the single artefact."""
function notebook_log(paths)
    out = Any[]
    for path in paths
        isempty(path) && continue
        if !isfile(path)
            push!(out, Dict{String, Any}("log" => path, "status" => "missing"))
            continue
        end
        file = basename(path)
        cells = Any[]
        pending = nothing
        for line in eachline(path)
            if startswith(line, "FILE ")
                file = split(line)[2]
            elseif startswith(line, "CELL ")
                pending !== nothing && push!(cells, pending)
                f = split(line)
                st = f[5]
                pending = Dict{String, Any}("cell" => parse(Int, f[2]),
                    "jmd_line" => parse(Int, f[4]), "status" => st,
                    "first_line" => strip(split(line, "| ", limit = 2)[end]))
            elseif startswith(strip(line), "ERR ") && pending !== nothing
                pending["error"] = strip(line)[5:end]
            end
        end
        pending !== nothing && push!(cells, pending)
        push!(out, Dict{String, Any}("log" => path, "notebook" => file,
            "status" => "read", "cells" => cells))
    end
    return out
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

function parse_args(args)
    here = dirname(@__FILE__)
    o = Dict{String, Any}("spec" => joinpath(here, "harvest_spec.json"),
        "out" => joinpath(here, "comparators.json"),
        "only" => String[], "notebook_logs" => String[],
        "paths_timed" => 200, "reps" => 5)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--spec"
            o["spec"] = args[i + 1]; i += 2
        elseif a == "--out"
            o["out"] = args[i + 1]; i += 2
        elseif a == "--only"
            o["only"] = String.(split(args[i + 1], ',')); i += 2
        elseif a == "--notebook-logs"
            o["notebook_logs"] = String.(split(args[i + 1], ',')); i += 2
        elseif a == "--paths-timed"
            o["paths_timed"] = parse(Int, args[i + 1]); i += 2
        elseif a == "--reps"
            o["reps"] = parse(Int, args[i + 1]); i += 2
        else
            error("unknown argument $a")
        end
    end
    return o
end

function main(args)
    opt = parse_args(args)
    spec = json_parse(read(opt["spec"], String))
    env = environment()
    @printf("harvest.jl  julia %s  threads=%d  cpu=%s\n", env["julia_version"],
        env["threads"], env["cpu"])
    for k in ("StochasticDiffEq", "SDEProblemLibrary", "DiffEqNoiseProcess")
        @printf("  %-20s %s\n", k, get(env["pkg_versions"], k, "?"))
    end

    rows = Any[]
    checks = Any[]
    for ps in spec["problems"]
        pid = ps["id"]
        if !isempty(opt["only"]) && !(pid in opt["only"])
            continue
        end
        prob = build_problem(ps)
        T = Float64(ps["T"])
        npaths = Int(ps["num_paths"])
        kind = ps["kind"]

        nok, ntot, worst = check_analytic(prob, ps)
        push!(checks, Dict{String, Any}("id" => pid, "checks_passed" => nok,
            "checks_total" => ntot, "worst_rel" => worst))
        @printf("\n%s  %s  kind=%s  T=%g  paths=%d  analytic-check %d/%d (worst rel %.2e)\n",
            pid, ps["sciml_key"], kind, T, npaths, nok, ntot, worst)
        if ntot > 0 && nok < ntot
            error("$pid: the library's analytic disagrees with the suite's closed " *
                  "form (worst relative difference $worst). Refusing to harvest: the " *
                  "comparator numbers would not be about the same problem.")
        end

        # One fixed seed vector per problem: every scheme and every step size on this
        # problem sees the same Brownian paths.
        rng = Random.MersenneTwister(Int(ps["seed"]))
        seeds = rand(rng, UInt32, npaths)
        paths_timed = min(npaths, Int(opt["paths_timed"]))
        schemes = SCHEME_SETS[ps["scheme_set"]]

        for algname in schemes
            sres = measure_stages(prob, algname, Float64(ps["graded_dt"]))
            fper, gper, stages, serr = sres
            for dt in Float64.(ps["dts"])
                steps = round(Int, T / dt)
                row = Dict{String, Any}(
                    "problem" => pid, "slug" => ps["slug"],
                    "sciml_key" => ps["sciml_key"], "scheme" => algname,
                    "kind" => kind, "T" => T, "dt" => dt, "steps" => steps,
                    "paths" => npaths, "seed" => Int(ps["seed"]),
                    "f_evals_per_step" => fper, "g_evals_per_step" => gper,
                    "stages" => stages, "stages_error" => serr,
                    "evals" => stages === nothing ? nothing :
                               round(Int, npaths * steps * stages),
                    "adaptive" => false,
                )
                cell = if kind == "pathwise"
                    strong_cell(prob, algname, dt, npaths, seeds)
                else
                    weak_cell(prob, algname, dt, npaths, seeds, ps["moment_truth"])
                end
                merge!(row, cell)
                secs = time_cell(prob, algname, dt, paths_timed, seeds;
                    reps = Int(opt["reps"]))
                row["seconds_min"] = secs
                row["timing_reps"] = Int(opt["reps"])
                row["paths_timed"] = paths_timed
                row["ns_per_path_step"] = secs === nothing ? nothing :
                                          secs * 1e9 / (paths_timed * steps)
                push!(rows, row)
                if kind == "pathwise"
                    @printf("  %-14s dt=%-10g steps=%-7d rel=%-11.4g l2abs=%-11s %s\n",
                        algname, dt, steps,
                        get(row, "strong_err_rel_final", NaN),
                        string(get(row, "sde_err_final_abs", nothing)),
                        row["status"])
                else
                    @printf("  %-14s dt=%-10g steps=%-7d relE[x1^2]=%-11.4g sig=%-8s %s\n",
                        algname, dt, steps,
                        get(row, "weak_err_rel_second_moment_x1", NaN),
                        string(get(row, "weak_err_sigmas_second_moment_x1", nothing)),
                        row["status"])
                end
                flush(stdout)
            end
        end
    end

    out = Dict{String, Any}(
        "_comment" => "Generated by benchmark/references/sciml/harvest.jl. " *
                      "ANSWER-KEY-ADJACENT: contains the reference error values the " *
                      "sciml suite is graded near and the schemes it is compared " *
                      "against. Denied to the pipeline by pipeline-settings.json and " *
                      "matched by verifylib/leakage.py.",
        "environment" => env,
        "spec" => opt["spec"],
        "analytic_cross_checks" => checks,
        "upstream_defects" => Any[
            Dict{String, Any}(
                "what" => "saveat with >= 3 save points on an SDEProblem carrying an " *
                          "`analytic` raises BoundsError in " *
                          "SciMLBase.calculate_solution_errors! (rode_solutions.jl:221)",
                "mechanism" => "that function indexes the saved noise sol.W.W with the " *
                               "solution's save index; saveat desynchronises the two. " *
                               "One or two save points work, three do not, and " *
                               "calculate_error = false does not suppress it.",
                "versions" => Dict{String, Any}(
                    "SciMLBase" => get(env["pkg_versions"], "SciMLBase", "?"),
                    "StochasticDiffEq" => get(env["pkg_versions"], "StochasticDiffEq", "?"),
                    "julia" => env["julia_version"]),
                "consequence_here" => "no timeseries :l2 column; the terminal relative " *
                                      "error that the suite gates on is unaffected",
                "relation_to_published_failure" => "same error shape as the four " *
                    "BoundsError cells of plan §1a.2 (an n-element vector indexed at " *
                    "n+1); see notebook_reproduction for whether those cells still fail",
            ),
        ],
        "notebook_reproduction" => notebook_log(opt["notebook_logs"]),
        "rows" => rows,
    )
    open(opt["out"], "w") do io
        json_write(io, out)
        println(io)
    end
    @printf("\nwrote %s  (%d rows)\n", opt["out"], length(rows))
    return nothing
end

main(ARGS)
