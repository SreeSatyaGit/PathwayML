#=
Step 2b: full training of the Vem+Tram HNODE model, noise level 0.0, using the
hyperparameters selected by Step 2a (tpe_vem_tram_00.jl).

TODO: after running Step 2a, copy the best trial's hyperparameters from
`step2a_hyperparameter_tuning/results_vem_tram/vem_tram_00.jld` into the
"TUNED HYPERPARAMETERS" block below, replacing the placeholder values.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics, Printf, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux

result_name_string = "vem_tram_00.jld"
folder_name = "res_vem_tram"
if !isdir(folder_name)
    mkdir(folder_name)
end
error_level = "e0.0"

include("../test_case_settings/vem_tram_model_settings/vem_tram_model_functions.jl")
include("../test_case_settings/vem_tram_model_settings/vem_tram_model_settings.jl")
column_names = ["t", "s1", "s2", "s3", "s4", "s5"]

integrator = TRBDF2(autodiff=true)
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

ode_data = deserialize("../datasets/e0.0/data/ode_data_vem_tram.jld")
ode_data_sd = deserialize("../datasets/e0.0/data/ode_data_std_vem_tram.jld")
solution_dataframe = deserialize("../datasets/e0.0/data/pert_df_vem_tram.jld")
solution_sd_dataframe = deserialize("../datasets/e0.0/data/pert_df_sd_vem_tram.jld")

################################### TUNED HYPERPARAMETERS (TODO: fill in from Step 2a) ###################################
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
    Lux.Dense(3, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform)
)

learning_rate_adam = 0.01          # TODO: replace with tuned value
pretrained_ode_pars = copy(original_parameters)  # TODO: replace with the p1..p10 estimates from Step 2a
group_size = 4                     # TODO: replace with tuned ms_group_size
continuity_term = 1.0              # TODO: replace with tuned ms_continuity_term
###########################################################################################################################

rng = Random.default_rng()
Random.seed!(rng, 0)

#############################################################################################################################
#################################### TRAINING VALIDATION SPLIT ##############################################################
shuffled_positions = shuffle(2:size(solution_dataframe)[1])
first_validation = rand(2:5)
validation_mask = [(first_validation + k * 5) for k in 0:3]
training_mask = [j for j in 1:size(solution_dataframe)[1] if !(j in validation_mask)]
training_mask = sort(training_mask)
validation_mask = sort(validation_mask)

original_ode_data = deepcopy(ode_data)
original_ode_data_sd = deepcopy(ode_data_sd)
original_solution_dataframe = deepcopy(solution_dataframe)
original_solution_sd_dataframe = deepcopy(solution_sd_dataframe)

ode_data = original_ode_data[:, training_mask]
ode_data_sd = original_ode_data_sd[:, training_mask]
solution_dataframe = original_solution_dataframe[training_mask, :]
solution_sd_dataframe = original_solution_sd_dataframe[training_mask, :]

validation_ode_data = original_ode_data[:, validation_mask]
validation_ode_data_sd = original_ode_data_sd[:, validation_mask]
validation_solution_dataframe = original_solution_dataframe[validation_mask, :]
validation_solution_sd_dataframe = original_solution_sd_dataframe[validation_mask, :]

normalization_factor = maximum(original_ode_data, dims=2) - minimum(original_ode_data, dims=2)
normalization_factor_training = repeat(normalization_factor, 1, size(ode_data)[2])
normalization_factor_validation = repeat(normalization_factor, 1, size(validation_ode_data)[2])

tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)
tspan = (initial_time_training, end_time_training)

results = []

function train_uode_model(seed, iterator)
    initial_time = time()

    rng_local = StableRNG(seed)
    local_approximating_neural_network = deepcopy(approximating_neural_network)
    p_net, st = Lux.setup(rng_local, local_approximating_neural_network)

    uode_derivative_function = get_uode_model_function(approximating_neural_network, st, pretrained_ode_pars)
    initial_ode_pars = ones(length(pretrained_ode_pars))
    prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

    ranges = DiffEqFlux.group_ranges(datasize, group_size)

    function loss_function(data, deviation, pred)
        original_cost = sum(abs2.(data .- pred) ./ abs2.(deviation))
        return 1 / size(data, 2) * sum(original_cost)
    end

    function loss_multiple_shooting(θ)
        p = θ.p
        tsteps = tmp_steps
        prob = prob_uode_pred
        initial_point_parameters = θ.u0

        function unstable_check(dt, u, p, t)
            return any(abs.(u) .> 1e7)
        end

        sols = [
            solve(
                remake(prob; p=p, tspan=(tsteps[first(rg)], tsteps[last(rg)]), u0=initial_point_parameters[:, first(rg)]),
                integrator; saveat=tsteps[rg], reltol=reltol, abstol=abstol,
                sensealg=sensealg, unstable_check=unstable_check, verbose=false
            ) for rg in ranges
        ]

        for i in 1:length(sols)
            if size(Array(sols[i]))[2] != length(ranges[i])
                return Inf
            end
        end

        group_predictions = Array.(sols)
        loss = 0
        for (i, rg) in enumerate(ranges)
            u = ode_data[:, rg]
            std = normalization_factor_training[:, rg]
            û = group_predictions[i]
            loss += loss_function(u, std, û)
        end

        for (i, rg) in enumerate(ranges)
            i == 1 && continue
            u0 = group_predictions[i-1][:, end]
            u1 = group_predictions[i][:, 1]
            loss += continuity_term * sum(abs2, u0 - u1)
        end

        return loss
    end

    function callback(θ, l, training_epochs, training_costs, best_training_parameters)
        epoch = extrema(training_epochs)[2] + 1
        println("Epoch " * string(epoch) * " -- cost: " * string(l) * " seed " * string(seed))

        if time() - initial_time > 60 * 3.5 * 60
            println("Too slow optimization")
            return true
        end

        training_epochs[epoch] = epoch
        training_costs[epoch] = l

        if epoch == 1 || l < minimum(training_costs[1:(epoch-1)])
            best_training_parameters[1] = deepcopy(θ)
        end

        if epoch > 200 && minimum(training_costs[(epoch-5):(epoch)]) > 10^6
            println("Stuck in non-integrability region")
            return true
        end

        return false
    end

    function validation_loss_function(θ)
        loss = 0.0
        try
            validation_df = validation_solution_dataframe
            max_time = extrema(validation_df.t)[2]
            par = θ.p
            init_par = θ.u0
            prob = remake(prob_uode_pred; p=par, tspan=(0, max_time), u0=init_par[:, 1])

            function unstable_check(dt, u, p, t)
                return any(abs.(u) .> 1e7)
            end

            solutions = solve(prob, integrator, p=par, saveat=validation_df.t, abstol=abstol, reltol=reltol,
                sensealg=sensealg, unstable_check=unstable_check, verbose=false)
            x = Array(solutions)
            if size(x)[2] != size(validation_df, 1)
                return Inf
            end
            loss = 1 / size(validation_df, 1) * sum(abs2.(Array(validation_df[:, 2:end])' .- x) ./ abs2.(normalization_factor_validation))
        catch
            loss = Inf
        end
        return loss
    end

    adtype = Optimization.AutoZygote()

    training_epochs = zeros(Int, 50000)
    training_costs = zeros(50000)

    p_net = ComponentArray(p_net)
    ode_par = ComponentArray(initial_ode_pars)
    p = ComponentArray{eltype(p_net)}()
    p = ComponentArray(p; p_net)
    p = ComponentArray(p; ode_par)
    u0 = deepcopy(ode_data)
    starting_point_in = ComponentVector{Float64}(p=p, u0=u0)

    best_training_parameters = [starting_point_in]

    optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, starting_point_in)
    opt = OptimizationOptimisers.Adam(learning_rate_adam)

    res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, best_training_parameters), maxiters=10000)

    println("********** Starting L-BFGS refinement (noiseless data only) **********")
    try
        optprob2 = remake(optprob, u0=best_training_parameters[1])
        res = Optimization.solve(optprob2, Optim.LBFGS(), callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, best_training_parameters), maxiters=5000, allow_f_increases=true)
    catch
        println("L-BFGS failed")
    end

    best_parameterization = best_training_parameters[1]
    validation_resulting_cost = validation_loss_function(best_parameterization)

    best_parameterization.p.ode_par = best_parameterization.p.ode_par .* pretrained_ode_pars
    initial_values_to_save = best_parameterization.u0

    result = (
        parameters_training=best_parameterization.p,
        initial_state_training=initial_values_to_save,
        net_status=st,
        validation_resulting_cost=validation_resulting_cost,
        status="success"
    )

    return result
end

multiseeds = 10  # paper default: 10 NN re-initializations, keep the best validation loss

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

serialize(folder_name * "/" * result_name_string, results)
