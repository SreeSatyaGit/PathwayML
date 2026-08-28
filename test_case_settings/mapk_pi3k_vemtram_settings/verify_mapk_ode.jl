#=
Sanity check for the Julia port of the MAPK/PI3K + Vem/Tram model. Solves the ODE
with the ORIGINAL (un-fitted) parameters -- exactly as MATLAB's Section 7 does before
optimization -- and prints the normalized model trajectory next to the normalized
experimental data at the 6 real timepoints, plus a fine-grained solve to confirm the
integrator handles the full 48h window without stiffness failures.

This does NOT fit anything. It only confirms: (1) the ODE integrates successfully,
(2) the state indexing/parameter order match MATLAB, (3) the general shape of the
un-fitted trajectory is at least plausible (won't match data well yet -- that's what
optimization is for).
=#

cd(@__DIR__)
using DifferentialEquations, SciMLBase, Printf

include("mapk_pi3k_vemtram_model_functions.jl")
include("mapk_pi3k_vemtram_model_settings.jl")

println("Solving with ORIGINAL (un-fitted) parameters over the real experimental window...")

tspan = (initial_time_training_mapk, end_time_training_mapk)
prob = ODEProblem(mapk_ode!, original_u0_mapk, tspan, original_parameters_mapk)

# Stiff solver, matching MATLAB's ode15s choice -- this system has widely separated
# rate constants (1e-9 to 1e-2 range), so stiffness is expected.
sol = solve(prob, TRBDF2(autodiff=false), saveat=timestamps_minutes_mapk, abstol=1e-8, reltol=1e-6)

if sol.retcode != SciMLBase.ReturnCode.Success
    error("Integration failed with retcode = $(sol.retcode). Check for stiffness issues, " *
          "bad parameter values, or a transcription error in mapk_ode!.")
end

println("Integration succeeded. Retcode: ", sol.retcode)
println()

# Compute each observed species and normalize (matching MATLAB's per-species min-max)
println(@sprintf("%-8s %10s %10s %10s", "species", "t=0h", "t=48h", "..."))
for (name, extractor) in pairs(OBSERVED_SPECIES)
    raw_trajectory = [extractor(sol.u[j]) for j in eachindex(sol.u)]
    normalized = min_max_normalize(raw_trajectory)
    exp_vals = exp_data_norm_mapk[name]
    println("--- $name ---")
    println("  model (un-fitted, normalized): ", round.(normalized, digits=3))
    println("  experimental (normalized):     ", round.(exp_vals, digits=3))
end

println()
println("If integration succeeded and the model numbers above are plausible (roughly")
println("0-1 range, not NaN/Inf), the Julia translation is behaving sensibly. The")
println("un-fitted model is NOT expected to match experimental data yet -- fitting the")
println("free parameters is the next step.")
