#=
Generates in-silico training data for the Vem+Tram MAPK/PI3K test case, following the
same recipe as HNODECB: solve the full mechanistic model (including the crosstalk term
we're pretending not to know) to get ground truth, then add 0% and 5% Gaussian noise
(scaled by each timeseries' min-max range) to produce two datasets: e0.0/ and e0.05/.

If you have REAL experimental data instead of synthetic data, skip this script entirely
and just drop your data into the same serialized format Step 2a expects
(ode_data, ode_data_sd, pert_df, pert_df_sd -- see step2a script for exact shapes).
=#

cd(@__DIR__)

using ComponentArrays, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Statistics, Printf
using StableRNGs

noise_magnitudes = [0.0, 0.05]
column_names = ["t", "s1", "s2", "s3", "s4", "s5"]  # t, pRAF, pMEK, pERK, pPI3K, pAKT
rng = Random.default_rng()

include("non_perturbed_dataset_generator.jl")
include("../test_case_settings/vem_tram_model_settings/vem_tram_model_functions.jl")
include("../test_case_settings/vem_tram_model_settings/vem_tram_model_settings.jl")

# ground_truth_function expects [mechanistic params..., crosstalk param] as p
full_ground_truth_parameters = vcat(original_parameters, crosstalk_akt_raf)

integrator = TRBDF2(autodiff=false)  # MAPK/PI3K crosstalk models are often stiff; switch to
                                      # Vern7() if you observe the solver taking very few steps

for noise_magnitude in noise_magnitudes

    Random.seed!(rng, 1)

    println("Generating vem_tram dataset with noise level " * string(noise_magnitude) * "...")

    tspan = (initial_time_training, end_time_training)
    tsteps = range(tspan[1], tspan[2], length=21)  # TODO: match your real sampling frequency
    stepsize = (tspan[2] - tspan[1]) / (21 - 1)

    solution_dataframe = generate_non_perturbed_training_set(
        ground_truth_function, original_u0, full_ground_truth_parameters, tspan, tsteps;
        column_names=column_names, integrator=integrator
    )
    solution_matrix = Array(solution_dataframe[:, :])

    ode_data_pure = transpose(solution_matrix[:, 2:end])
    max_variation = maximum(ode_data_pure, dims=2) - minimum(ode_data_pure, dims=2)
    ode_data = max.(ode_data_pure .+ (noise_magnitude * max_variation) .* randn(rng, eltype(ode_data_pure), size(ode_data_pure)), 0.0)
    ode_data_std = reshape(repeat(noise_magnitude * max_variation, outer=21), size(ode_data))

    perturbed_dataframe = DataFrame(transpose(ode_data), names(solution_dataframe)[2:end])
    perturbed_dataframe[!, :t] = tsteps
    perturbed_dataframe = perturbed_dataframe[:, [end; 1:end-1]]

    perturbed_dataframe_sd = DataFrame(transpose(ode_data_std), names(solution_dataframe)[2:end])
    perturbed_dataframe_sd[!, :t] = tsteps
    perturbed_dataframe_sd = perturbed_dataframe_sd[:, [end; 1:end-1]]

    folder_name = "e" * string(noise_magnitude) * "/"
    if !isdir(folder_name)
        mkdir(folder_name)
    end
    data_folder_name = folder_name * "data/"
    if !isdir(data_folder_name)
        mkdir(data_folder_name)
    end

    serialize(data_folder_name * "ode_data_vem_tram.jld", ode_data)
    serialize(data_folder_name * "ode_data_std_vem_tram.jld", ode_data_std)
    serialize(data_folder_name * "pert_df_vem_tram.jld", perturbed_dataframe)
    serialize(data_folder_name * "pert_df_sd_vem_tram.jld", perturbed_dataframe_sd)
end
