# em_anchor.jl -- the Julia half of the Euler-Maruyama anchor (plan §5, step 4).
#
# WHY AN ANCHOR EXISTS AT ALL. The comparator table's primary axis is work --
# `paths x steps x stages` -- precisely because a numpy solver and a compiled Julia
# solver cannot share a seconds axis without an argument. But seconds are what anyone
# running either system actually pays, so they are reported too, divided by a measured
# language-and-implementation factor: the ratio of these two files' times on the SAME
# Euler-Maruyama, same problem, same step, same path count, single-threaded.
#
# WHAT IS IN THE CLOCK. Only the stepping loop. The increments are drawn once, before
# the clock starts, in both stacks -- the benchmark contract hands a solver its `dW`
# (H1), so stepping is the work being compared, and a random-number generator's speed
# is a different measurement. Each stack is written the way that stack is normally
# written, which is the whole point: numpy vectorised across paths, Julia looping one
# trajectory at a time, as `StochasticDiffEq.jl` itself does.
#
# TWO SHAPES, NOT ONE (§5). A scalar many-path problem and a many-state few-path one
# load the two languages differently and the plan predicts the factors differ by more
# than 3x. Measuring once and quoting it everywhere would be the error this guards.
#
# Needs no packages: plain Julia, so it runs under any project.
# Usage: JULIA_NUM_THREADS=1 julia +1.11.9 em_anchor.jl [--out em_anchor_julia.json]
#        [--reps 5]

function step_scalar!(x, dW, a, b, dt)
    paths, nt = size(dW)
    @inbounds for p in 1:paths
        xp = x[p]
        for i in 1:nt
            xp += a * xp * dt + b * xp * dW[p, i]
        end
        x[p] = xp
    end
    return x
end

# Four independent trajectories advanced together, and the reason this variant exists.
#
# The loop above is *latency*-bound, not compute- or memory-bound: `xp += a*xp*dt +
# b*xp*dW` is a serially dependent chain of multiply-adds, so each step waits on the
# previous one's result -- about 4.6 ns on this machine, which is four FMA latencies and
# nothing to do with how fast Julia is. Interleaving four chains breaks the dependency
# and the same arithmetic runs at 1.76 ns.
#
# That matters because numpy has no choice but to vectorise across paths, which hides
# the same latency, so comparing numpy against the *single-chain* loop measures a
# bottleneck one stack was forced out of and the other was not. Both are reported: the
# trajectory-at-a-time number is what `StochasticDiffEq.jl` actually does, and this one
# is the ceiling a Julia author could reach on the same shape.
function step_scalar_x4!(x, dW, a, b, dt)
    paths, nt = size(dW)
    @inbounds for p in 1:4:(paths - 3)
        x1, x2, x3, x4 = x[p], x[p + 1], x[p + 2], x[p + 3]
        for i in 1:nt
            x1 += a * x1 * dt + b * x1 * dW[p, i]
            x2 += a * x2 * dt + b * x2 * dW[p + 1, i]
            x3 += a * x3 * dt + b * x3 * dW[p + 2, i]
            x4 += a * x4 * dt + b * x4 * dW[p + 3, i]
        end
        x[p], x[p + 1], x[p + 2], x[p + 3] = x1, x2, x3, x4
    end
    return x
end

function step_system!(x, dW, a, b, dt)
    paths, nt, d = size(dW)
    @inbounds for p in 1:paths
        for k in 1:d
            xk = x[p, k]
            ak = a[k]
            bk = b[k]
            for i in 1:nt
                xk += ak * xk * dt + bk * xk * dW[p, i, k]
            end
            x[p, k] = xk
        end
    end
    return x
end

"""min of `reps` timed repetitions after one discarded warm-up (§3b)."""
function timed(f, reps)
    f()
    best = Inf
    for _ in 1:reps
        t0 = time_ns()
        f()
        best = min(best, (time_ns() - t0) / 1e9)
    end
    return best
end

function main(args)
    out = "em_anchor_julia.json"
    reps = 5
    i = 1
    while i <= length(args)
        if args[i] == "--out"
            out = args[i + 1]; i += 2
        elseif args[i] == "--reps"
            reps = parse(Int, args[i + 1]); i += 2
        else
            error("unknown argument $(args[i])")
        end
    end

    T = 1.0
    nt = 15625            # M01's finest declared level: 5^-6 over [0, 1]
    dt = T / nt
    sq = sqrt(dt)
    rows = []

    # --- shape A: scalar, many paths (M01's own coefficients) -------------------
    paths = 1000
    dWa = randn(paths, nt) .* sq
    xa = fill(0.5, paths)
    sa = timed(() -> step_scalar!(copy(xa), dWa, 1.01, 0.87, dt), reps)
    push!(rows, Dict{String, Any}("shape" => "scalar_many_paths", "d" => 1,
        "paths" => paths, "steps" => nt, "seconds_min" => sa,
        "ns_per_path_step" => sa * 1e9 / (paths * nt)))

    s4 = timed(() -> step_scalar_x4!(copy(xa), dWa, 1.01, 0.87, dt), reps)
    push!(rows, Dict{String, Any}("shape" => "scalar_many_paths_interleaved4", "d" => 1,
        "paths" => paths, "steps" => nt, "seconds_min" => s4,
        "ns_per_path_step" => s4 * 1e9 / (paths * nt)))

    # --- shape B: many states, few paths ---------------------------------------
    d = 8
    pathsb = 10
    dWb = randn(pathsb, nt, d) .* sq
    xb = fill(0.5, pathsb, d)
    av = collect(range(0.8, 1.2; length = d))
    bv = collect(range(0.5, 0.9; length = d))
    sb = timed(() -> step_system!(copy(xb), dWb, av, bv, dt), reps)
    push!(rows, Dict{String, Any}("shape" => "many_states_few_paths", "d" => d,
        "paths" => pathsb, "steps" => nt, "seconds_min" => sb,
        "ns_per_path_step" => sb * 1e9 / (pathsb * nt * d)))

    io = IOBuffer()
    print(io, "{\n \"stack\": \"julia\",\n \"julia_version\": \"", VERSION, "\",")
    print(io, "\n \"threads\": ", Threads.nthreads(), ",")
    print(io, "\n \"cpu\": \"", Sys.CPU_NAME, "\",\n \"machine\": \"", Sys.MACHINE, "\",")
    print(io, "\n \"reps\": ", reps, ",\n \"rows\": [")
    for (n, r) in enumerate(rows)
        print(io, "\n  {\"shape\": \"", r["shape"], "\", \"d\": ", r["d"],
            ", \"paths\": ", r["paths"], ", \"steps\": ", r["steps"],
            ", \"seconds_min\": ", r["seconds_min"],
            ", \"ns_per_path_step\": ", r["ns_per_path_step"], "}",
            n == length(rows) ? "" : ",")
    end
    print(io, "\n ]\n}\n")
    write(out, String(take!(io)))

    println("em_anchor.jl  julia $(VERSION)  threads=$(Threads.nthreads())  " *
            "cpu=$(Sys.CPU_NAME)  reps=$reps")
    for r in rows
        println("  $(rpad(r["shape"], 22)) d=$(r["d"]) paths=$(r["paths"]) " *
                "steps=$(r["steps"])  $(round(r["seconds_min"], digits = 4)) s  " *
                "$(round(r["ns_per_path_step"], digits = 2)) ns/path-step")
    end
    println("wrote $out")
    return nothing
end

main(ARGS)
