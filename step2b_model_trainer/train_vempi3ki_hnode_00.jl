#=
Step 2b: full training of the Vem+PI3K HNODE model, using hyperparameters selected by
Step 2a (extract_best_trial_vempi3ki.jl). Same structure as the Vem+Tram Step 2b (best-
checkpoint tracking across training, 10 parallel NN re-initializations), adapted for this
condition's 9 species, absent Trametinib, and the extra fittable PI3K IC50 parameter.

TODO: after running Step 2a and extract_best_trial_vempi3ki.jl, fill in the
"TUNED HYPERPARAMETERS" block below.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, SciMLBase, LinearAlgebra, Random, DataFrames, CSV, Statistics, Printf, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux

include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_hnode_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_settings.jl")
include("../test_case_settings/mapk_pi3k_vempi3ki_settings/mapk_pi3k_vempi3ki_hnode_functions.jl")
include("../test_case_settings/mapk_pi3k_vempi3ki_settings/mapk_pi3k_vempi3ki_model_settings.jl")

folder_name = "res_vempi3ki_hnode"
if !isdir(folder_name)
    mkdir(folder_name)
end
result_name_string = "vempi3ki_hnode_00.jld"

integrator = TRBDF2(autodiff=true)
abstol = 1e-8
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

################################### TUNED HYPERPARAMETERS (TODO: fill in from Step 2a) ###################################
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
    Lux.Dense(2, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

learning_rate_adam = 0.01           # TODO: replace with tuned value
l2_reg_weight = 1e-4                # TODO: replace with tuned value
pretrained_ode_pars_mech = copy(original_parameters_mapk[free_idx_hnode_mapk])  # TODO: replace with
                                                                                  # extracted p1..p60
pretrained_pi3ki_ic50_scale = 1.0   # TODO: replace with extracted pi3ki_ic50_scale
###########################################################################################################################

rng = Random.default_rng()
Random.seed!(rng, 0)

n_mech_params = length(free_idx_hnode_mapk)
n_total_params = n_mech_params + 1
tspan = (initial_time_training_vempi3ki, end_time_training_vempi3ki)

fixed_values_vempi3ki = copy(fixed_values_mapk)
fixed_values_vempi3ki[1] = 0.0   # Trametinib off

const species_order = (:pEGFR, :pCRAF, :pMEK, :pERK, :DUSP, :pAKT, :IGF1R, :pHer2, :pHer3)
species_extractors = OBSERVED_SPECIES_VEMPI3KI

min_max_normalize_for_loss(v) = (v .- minimum(v)) ./ (maximum(v) - minimum(v) + 1e-6)

function weighted_loss(sol)
    if sol.retcode != SciMLBase.ReturnCode.Success || length(sol.t) != length(timestamps_minutes_vempi3ki)
        return Inf
    end
    total = 0.0
    for (i, name) in enumerate(species_order)
        extractor = species_extractors[name]
        raw = [extractor(sol.u[j]) for j in eachindex(sol.u)]
        model_norm = min_max_normalize_for_loss(raw)
        exp_norm = exp_data_norm_vempi3ki[name]
        total += species_weights_vempi3ki[i] * sum(abs2, model_norm .- exp_norm)
    end
    return total
end

results = []

function train_uode_model(seed, iterator)
    initial_time = time()

    rng_local = StableRNG(seed)
    local_approximating_neural_network = deepcopy(approximating_neural_network)
    p_net, st = Lux.setup(rng_local, local_approximating_neural_network)

    hnode_derivative_function = get_uode_model_function_vempi3ki(
        approximating_neural_network, st, pretrained_ode_pars_mech,
        free_idx_hnode_mapk, fixed_idx_mapk, fixed_values_vempi3ki,
        pi3ki_conc_ref, pi3ki_hill_n_ref, pi3ki_ic50_nominal
    )
    initial_ode_par = vcat(ones(n_mech_params), pretrained_pi3ki_ic50_scale)
    prob = ODEProblem{true}(hnode_derivative_function, original_u0_mapk, tspan)

    function unstable_check(dt, u, p, t)
        return any(abs.(u) .> 1e7)
    end

    function loss_fn(θ)
        par = θ.p
        local sol
        try
            sol = solve(prob, integrator; p=par, saveat=timestamps_minutes_vempi3ki,
                        abstol=abstol, reltol=reltol, sensealg=sensealg,
                        unstable_check=unstable_check, verbose=false)
        catch
            return 1e6
        end
        data_loss = weighted_loss(sol)
        if !isfinite(data_loss)
            return 1e6
        end
        reg_loss = l2_reg_weight * sum(abs2, θ.p.p_net)
        return data_loss + reg_loss
    end

    function callback(θ, l, training_epochs, training_costs, best_training_parameters)
        epoch = extrema(training_epochs)[2] + 1
        println("Epoch " * string(epoch) * " -- cost: " * string(l) * " seed " * string(seed))
        flush(stdout)

        if time() - initial_time > 60 * 3.5 * 60
            println("Too slow optimization")
            return true
        end

        training_epochs[epoch] = epoch
        training_costs[epoch] = l

        if epoch == 1 || l < minimum(training_costs[1:(epoch-1)])
            best_training_parameters[1] = deepcopy(θ)
        end

        if epoch > 200 && minimum(training_costs[(epoch-5):(epoch)]) > 1e5
            println("Stuck in non-integrability region")
            return true
        end

        return false
    end

    adtype = Optimization.AutoZygote()

    training_epochs = zeros(Int, 50000)
    training_costs = zeros(50000)

    p_net_ca = ComponentArray(p_net)
    ode_par_ca = ComponentArray(initial_ode_par)
    p = ComponentArray{eltype(p_net_ca)}()
    p = ComponentArray(p; p_net=p_net_ca)
    p = ComponentArray(p; ode_par=ode_par_ca)
    starting_point_in = ComponentVector{Float64}(p=p)

    best_training_parameters = [starting_point_in]

    optf = Optimization.OptimizationFunction((x, p) -> loss_fn(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, starting_point_in)
    opt = OptimizationOptimisers.Adam(learning_rate_adam)

    res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, best_training_parameters), maxiters=10000)

    println("********** Starting L-BFGS refinement **********")
    try
        optprob2 = remake(optprob, u0=best_training_parameters[1])
        res = Optimization.solve(optprob2, Optim.LBFGS(), callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, best_training_parameters), maxiters=2000, allow_f_increases=true)
    catch
        println("L-BFGS failed")
    end

    best_parameterization = best_training_parameters[1]
    final_training_cost = loss_fn(best_parameterization)

    # convert mechanistic scale factors -> absolute; PI3K IC50 scale -> absolute IC50
    best_parameterization.p.ode_par[1:n_mech_params] = best_parameterization.p.ode_par[1:n_mech_params] .* pretrained_ode_pars_mech
    best_parameterization.p.ode_par[n_mech_params+1] = best_parameterization.p.ode_par[n_mech_params+1] * pi3ki_ic50_nominal

    result = (
        parameters_training=best_parameterization.p,
        net_status=st,
        final_training_cost=final_training_cost,
        status="success"
    )

    return result
end

multiseeds = 10

lock_results = ReentrantLock()
Threads.@threads for iterator in 1:multiseeds
    try
        random_seed = abs(rand(rng, Int))
        result = train_uode_model(random_seed, iterator)
        lock(lock_results)
        push!(results, result)
        unlock(lock_results)
    catch ex
        showerror(stdout, ex)
        lock(lock_results)
        push!(results, (status="failed",))
        unlock(lock_results)
    end
end

successful = filter(r -> r.status == "success", results)
if !isempty(successful)
    best = successful[argmin([r.final_training_cost for r in successful])]
    println("Best replicate training cost: ", best.final_training_cost)
end

serialize(folder_name * "/" * result_name_string, results)
