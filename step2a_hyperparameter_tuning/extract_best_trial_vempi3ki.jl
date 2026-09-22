#=
Run this AFTER tpe_vempi3ki_hnode_00.jl finishes. Loads the serialized Optuna study, finds
the best trial (scanning all trials directly, not trusting study.best_trial), and prints
exactly what to paste into step2b_model_trainer/train_vempi3ki_hnode_00.jl's "TUNED
HYPERPARAMETERS" block.
=#

cd(@__DIR__)
using Serialization, PyCall, ComponentArrays

result = deserialize("results_vempi3ki_hnode/vempi3ki_hnode_00.jld")
study = result.study

all_trials = study.trials
println("Total trials recorded: ", length(all_trials))

trial_info = [(number=t.number, value=t.value, state=string(t.state)) for t in all_trials]
finite_trials = filter(t -> t.value !== nothing && isfinite(t.value), trial_info)
println("Trials with a finite recorded value: ", length(finite_trials), " / ", length(trial_info))

if isempty(finite_trials)
    error("No trials have a finite value.")
end

sorted_trials = sort(finite_trials, by = t -> t.value)
println()
println("Top 5 trials by lowest loss:")
for t in sorted_trials[1:min(5, end)]
    println("  trial ", t.number, "  loss = ", t.value, "  state = ", t.state)
end
println()

best_trial_number = sorted_trials[1].number
println("Using trial ", best_trial_number, " (loss = ", sorted_trials[1].value, ") as the best trial.")
println()

best_trial = first(t for t in all_trials if t.number == best_trial_number)
params = best_trial.params

println("=== Architecture ===")
println("num_hidden_layers = ", params["num_hidden_layers"])
width = 2^params["num_hidden_nodes"]
println("num_hidden_nodes  = ", params["num_hidden_nodes"], "  (width = ", width, ")")
println()

println("=== Optimizer settings ===")
println("learning_rate_adam = ", params["learning_rate_adam"])
println("l2_reg_weight       = ", params["l2_reg_weight"])
println()

println("=== PI3K inhibitor (fittable) ===")
println("pi3ki_ic50_scale = ", params["pi3ki_ic50_scale"])
println("  (effective IC50 = ", params["pi3ki_ic50_scale"], " x nominal 4.6e-9 M = ", params["pi3ki_ic50_scale"] * 4.6e-9, " M)")
println()

n_mech_params = 60
println("=== Mechanistic parameter starting values (paste as pretrained_ode_pars_mech) ===")
p_values = [params["p$i"] for i in 1:n_mech_params]
println("pretrained_ode_pars_mech = [")
for (i, v) in enumerate(p_values)
    print("    ", v)
    print(i < n_mech_params ? ",\n" : "\n")
end
println("]")
println()

println("=== Reminder: build the NN architecture in train_vempi3ki_hnode_00.jl to match ===")
nlayers = params["num_hidden_layers"]
println("Lux.Chain(")
println("    Lux.Dense(2, $width, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),")
for _ in 1:(nlayers - 1)
    println("    Lux.Dense($width, $width, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),")
end
println("    Lux.Dense($width, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),")
println(")")
