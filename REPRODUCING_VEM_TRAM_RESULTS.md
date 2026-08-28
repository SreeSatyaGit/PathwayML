# Reproducing the Vemurafenib + Trametinib HNODE Results

This document is a complete, self-contained guide to reproducing the fitted MAPK/PI3K
Hybrid Neural ODE (HNODE) model for the Vemurafenib + Trametinib condition, from a clean
HPC account to the final identifiability analysis, confidence intervals, and fit plots.

**Repository:** `PathwayML` (adapted from the HNODECB pipeline, Giampiccolo et al. 2024,
[https://doi.org/10.1038/s41540-024-00460-3](https://doi.org/10.1038/s41540-024-00460-3),
original code: [https://github.com/cosbi-research/HNODECB](https://github.com/cosbi-research/HNODECB))

---

## 1. What this pipeline produces

A 68-state mechanistic ODE model of the MAPK/PI3K signaling pathway under Vemurafenib
(BRAF inhibitor) + Trametinib (MEK inhibitor) combination treatment, where one
mechanistically-uncertain term — **paradoxical RAF activation** (the well-documented but
poorly quantified phenomenon where RAF inhibitors paradoxically activate the other RAF
protomer in a dimer) — is replaced by a small neural network. The pipeline:

1. Fits the remaining 60 free mechanistic parameters and the neural network jointly against
   real experimental time-course data (6 timepoints: 0, 1, 4, 8, 24, 48h; 12 observed
   phospho-protein/species readouts).
2. Assesses which of the 60 mechanistic parameters are actually **identifiable** given this
   data — and critically, whether the neural network is silently compensating for any of
   them (Hessian-based local identifiability analysis).
3. Estimates confidence intervals for the identifiable parameters (Fisher Information
   Matrix).
4. Exports the fitted trajectory for plotting.

**Result summary from the reference run:** best training cost 15.55 (out of 100
hyperparameter-search trials); 5 of 60 mechanistic parameters classified as identifiable
(`kDuspStop`, `k43b1`, `kpCraf`, `kErkPhosPcraf`, `kErkInbEgfr`); the neural network's
contribution to non-identifiability was negligible for all 60 parameters, indicating the
non-identifiability reflects genuine data scarcity (72 data points constraining 60+
parameters) rather than the NN absorbing mechanistic effects.

---

## 2. Repository layout

```
PathwayML/
├── Project.toml, Manifest.toml          # Julia environment (20 direct dependencies)
├── venv/                                 # Python venv (optuna, for PyCall)
├── test_case_settings/
│   └── mapk_pi3k_vemtram_settings/
│       ├── mapk_pi3k_vemtram_model_functions.jl    # Full mechanistic ODE (68 states)
│       ├── mapk_pi3k_vemtram_model_settings.jl     # Real params, ICs, data, bounds
│       ├── mapk_pi3k_vemtram_hnode_functions.jl    # HNODE variant (paradox term -> NN)
│       ├── verify_mapk_ode.jl                      # Sanity check: mechanistic-only
│       └── verify_mapk_hnode.jl                    # Sanity check: HNODE plumbing
├── step2a_hyperparameter_tuning/
│   ├── tpe_mapk_hnode_00.jl              # TPE/Optuna hyperparameter search (100 trials)
│   ├── extract_best_trial.jl             # Pulls the best trial's hyperparameters
│   └── submit_step2a.sh                  # Slurm batch submission
├── step2b_model_trainer/
│   ├── train_mapk_hnode_00.jl            # Full training (Adam+L-BFGS, 10 replicates)
│   ├── export_fit_for_plotting.jl        # CSV export for plotting
│   └── submit_step2b.sh                  # Slurm batch submission
├── step3_parameters_identifiability/
│   └── mapk_identifiability_00.jl        # Hessian null-space identifiability analysis
└── step4_confidence_intervals/
    └── mapk_fisher_CI_00.jl              # FIM-based CIs for identifiable parameters
```

---

## 3. Environment setup

### 3.1 Julia

Install Julia **1.9.1** via `juliaup` (matches the pinned `Manifest.toml`):

```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes
juliaup add 1.9.1
juliaup default 1.9.1
```

If `julia` isn't found immediately after install in the *same* terminal session, either
`source ~/.bashrc` or open a new session — `juliaup` adds itself to shell startup files,
which only takes effect for new shells.

### 3.2 Python virtual environment (for PyCall/Optuna)

Step 2a's Bayesian hyperparameter search uses Optuna's Python implementation, called from
Julia via `PyCall`.

```bash
cd PathwayML
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install optuna
deactivate
```

Point `PyCall` at this venv and build it (only needs to happen once):

```bash
julia --project=. -e '
ENV["PYTHON"] = "'"$(pwd)"'/venv/bin/python3"
using Pkg
Pkg.add("PyCall")
Pkg.build("PyCall")
'
```

### 3.3 Julia project dependencies

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Project.toml` is intentionally pruned to the 20 packages actually used by this pipeline
(dropped from the ~47 in the original HNODECB `Project.toml`, which included plotting
backends, structural-identifiability tooling, and R interop unused here — several of which
have system-library dependencies, e.g. `RCall` needs R, `Plots`/`StatsPlots`/`PlotlyJS`
need Qt6/GLFW/OpenGL, that routinely fail to build on headless HPC nodes):

```toml
CSV, ComponentArrays, DataFrames, Dates, DiffEqFlux, DifferentialEquations,
LinearAlgebra, Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers,
Pkg, Printf, PyCall, Random, SciMLBase, SciMLSensitivity, Serialization,
StableRNGs, Statistics, Zygote
```

### 3.4 Verify the environment

```bash
julia --project=. -e '
using DifferentialEquations, Lux, SciMLSensitivity, DiffEqFlux, SciMLBase, PyCall
println("Julia packages OK")
pyimport("optuna")
println("optuna OK")
'
```

Both lines should print without error before proceeding.

---

## 4. HPC-specific notes (if applicable)

- **Run long jobs as dedicated Slurm batch jobs (`sbatch`), not inside an interactive
  session** (e.g. an Open OnDemand VSCode/Jupyter session). Interactive sessions can reap
  background/`nohup`'d processes on browser disconnect or session recycling, independent of
  the underlying Slurm job's actual walltime or memory limits. Submission scripts for both
  training steps are included (`submit_step2a.sh`, `submit_step2b.sh`).
- **Step 2a** (`tpe_mapk_hnode_00.jl`) is single-threaded — no `-t` flag needed.
- **Step 2b** (`train_mapk_hnode_00.jl`) trains 10 NN re-initializations in parallel via
  `Threads.@threads` — launch with `julia -t 10` (or match `--cpus-per-task`).
- Expect **Zygote's first gradient compilation** in a fresh Julia process to take several
  minutes before any real output appears — this is normal, one-time-per-process overhead,
  not a hang.
- Redirect output to a log file and monitor with `tail -f`, rather than relying on a
  live-attached terminal for multi-hour runs.

---

## 5. Step-by-step reproduction

### Step 0 — Verify the mechanistic model translation

Before trusting anything built on top of it, confirm the Julia port of the MATLAB model
integrates correctly:

```bash
julia --project=. test_case_settings/mapk_pi3k_vemtram_settings/verify_mapk_ode.jl
julia --project=. test_case_settings/mapk_pi3k_vemtram_settings/verify_mapk_hnode.jl
```

Both should report successful integration (`retcode = Success`) with plausible-range
values. Neither script fits anything — they only confirm the ODE system and the HNODE
plumbing (untrained NN wired into the paradox-activation term) are structurally correct.

### Step 1 — Generate/confirm data

The real experimental dataset (6 timepoints × 12 species) is hard-coded in
`mapk_pi3k_vemtram_model_settings.jl` (`exp_data_raw_mapk`) — no separate data-generation
step is needed for this real-data case (unlike the synthetic-data test cases in the
original HNODECB pipeline).

### Step 2a — Hyperparameter tuning (TPE/Optuna)

Searches 100 trials over: NN architecture (1–3 layers, width 4–16), Adam learning rate,
L2 regularization weight, and starting values for the 60 free mechanistic parameters
(within `±1.5×` of literature nominal values — see §6 for why this range matters).

```bash
sbatch step2a_hyperparameter_tuning/submit_step2a.sh
squeue -u $USER                              # monitor
tail -f slurm_step2a_<jobid>.log
```

Each trial trains for up to 300 epochs (Adam) or until a 2-minute-per-trial wall-clock cap
or a "stuck" (non-improving) cutoff. The best checkpoint *within* each trial's training run
is tracked and reported — not simply wherever the optimizer ended up after 300 epochs,
which may have since drifted (see §6, "Best-checkpoint tracking").

**Reference run:** ~100 trials in ~2–3 hours at ~1.5–5 min/trial (pace varies with sampled
architecture size).

Once finished, extract the winning hyperparameters:

```bash
julia --project=. step2a_hyperparameter_tuning/extract_best_trial.jl
```

This scans every trial's recorded value directly (rather than trusting Optuna's
`study.best_trial` convenience accessor — see §6) and prints the winning architecture,
learning rate, L2 weight, and all 60 starting parameter values, formatted for direct
pasting into Step 2b.

**Reference result:** trial 57, loss 36.22, architecture = 2 hidden layers × width 4.

### Step 2b — Full training

Paste the extracted hyperparameters into the `TUNED HYPERPARAMETERS` block near the top of
`train_mapk_hnode_00.jl`, then submit:

```bash
sbatch step2b_model_trainer/submit_step2b.sh
squeue -u $USER
tail -f slurm_step2b_<jobid>.log
```

Trains 10 NN re-initializations in parallel (Adam, up to 10,000 epochs, followed by L-BFGS
refinement), and reports the lowest-cost replicate. Results are serialized to
`step2b_model_trainer/res_mapk_hnode/mapk_hnode_00.jld`.

**Reference result:** best replicate training cost 15.55.

### Step 3 — Local identifiability analysis

```bash
julia --project=. step3_parameters_identifiability/mapk_identifiability_00.jl
```

Computes the Gauss-Newton Hessian of the trajectory-sensitivity function across all 6
timepoints × 12 observed species, finds its near-null eigenspace, and reports each
mechanistic parameter's projection onto it — split into mechanistic vs. neural-network
contribution. Involves 72 `Zygote.jacobian` calls through the full ODE solve; expect several
minutes of runtime. Output: `step3_parameters_identifiability/results/identifiability_summary_mapk_00.csv`.

**Note the NN architecture in this script (and Step 4) is hard-coded** to match whichever
trial Step 2a selected as best. If you re-run Step 2a and get a different winning trial,
update the `Lux.Chain(...)` definitions in both Step 3 and Step 4 to match.

### Step 4 — Confidence intervals

```bash
julia --project=. step4_confidence_intervals/mapk_fisher_CI_00.jl
```

Reads Step 3's output and computes Fisher Information Matrix based 95% confidence intervals
**only** for the parameters classified as identifiable — CIs for non-identifiable
parameters aren't statistically meaningful and are intentionally omitted. Uses an assumed
0.5% measurement-uncertainty fraction (your real data has no reported per-point error bars;
adjust `measurement_uncertainty_fraction` in the script if you have a better estimate).
Output: `step4_confidence_intervals/results/mapk_fisher_CI_00.csv`.

**Reference result:** 3 of 5 identifiable parameters (`kDuspStop`, `k43b1`, `kErkPhosPcraf`)
have tight, physically sensible (positive) CIs. `kErkInbEgfr`'s CI spans zero (identifiable
but not significantly different from no-effect). `kpCraf`'s CI is tight but **negative**,
which is not physically valid for a mass-action rate constant — see §6 for why, and the
recommended fix if this matters for your use.

### Plotting

```bash
julia --project=. step2b_model_trainer/export_fit_for_plotting.jl
```

Simulates the fitted model on a fine 200-point grid (0–48h) and at the 6 real timepoints,
exporting both to CSV — deliberately with no plotting library involved, since `Plots.jl`
etc. aren't viable on this HPC's headless nodes (see §6). Plot the resulting
`model_fit_fine.csv` / `model_vs_data_6pt.csv` locally, or via any external tool.

---

## 6. Key implementation notes and known pitfalls

These are worth understanding before modifying the pipeline, since they weren't obvious
during development and cost significant debugging time.

**`sol.retcode == :Success` silently always evaluates to `false`.** Current `SciMLBase`
represents `retcode` as a `ReturnCode` enum (`SciMLBase.ReturnCode.T`), not a `Symbol`.
Comparing it against the `Symbol` literal `:Success` compiles and runs without error but
is *never* true, regardless of actual solve outcome — this silently broke every
success/failure check in the pipeline until diagnosed. Fixed throughout by comparing
against `SciMLBase.ReturnCode.Success` directly (the `successful_retcode()` convenience
helper is not available in this pinned SciMLBase version). `SciMLBase` must be an explicit
direct dependency (`using SciMLBase`) and listed in `Project.toml` — it's not re-exported
by `using DifferentialEquations` in this version.

**Zygote and `NamedTuple`/`Tuple` iteration inside differentiated code.** `Zygote.jacobian`
requires its function to return a plain `Array`. Iterating directly over `OBSERVED_SPECIES`
(a `NamedTuple` of extractor closures) inside a function being differentiated causes
Zygote's AD tracing to collapse the result back into a `NamedTuple` or `Tuple` rather than
a plain array, which `jacobian()` rejects. Fixed by writing the 12 species extractions as
an explicit array literal (`extract_species(state) = [state[3], state[16]+state[18]+..., ...]`)
with no closures or iteration at all inside differentiated functions — array literals are
always genuine `Vector`s regardless of what AD machinery surrounds them.

**Best-checkpoint tracking in Step 2a.** The first working version of `tpe_mapk_hnode_00.jl`
reported `final_cost = loss_fn(res.u)` — i.e. wherever Adam happened to be after all 300
epochs (or `Inf` if flagged "stuck"), not the best point reached *during* training. Over
300 unconstrained gradient steps on this sensitive, stiff, 60+-parameter system, many
trials achieved a good loss partway through and then drifted into instability by epoch 300,
silently discarding the good result. Fixed by tracking `best_theta`/`best_cost` throughout
the callback (mirroring the pattern Step 2b already used) and reporting/saving that instead.

**Per-trial exception isolation.** A single trial throwing an uncaught exception (e.g. a
`DomainError` from the adjoint quadrature integrand going non-finite at an isolated
timepoint — occurs in roughly 2–3% of trials, an occasional `QuadratureAdjoint` fragility
on stiff systems, not a pipeline bug) used to crash the entire multi-hour 100-trial run.
Fixed by wrapping each trial in `try`/`catch` in the outer loop, logging the caught error,
and continuing to the next trial.

**stdout buffering.** Julia buffers `println` output more aggressively than expected,
especially over SSH/non-interactive sessions — killed or crashed processes can lose
recently-printed-but-unflushed lines entirely, including the actual error message
explaining a crash. All diagnostic and per-epoch prints in the training scripts call
`flush(stdout)` immediately after printing for this reason.

**Parameter search bounds for Step 2a.** The paper's default `nominal/50` to `nominal×50`
search range (calibrated for their small, 1–9-parameter synthetic test cases) is far too
wide to sample independently across 60 parameters simultaneously on a system this large and
stiff — the joint probability that all 60 land somewhere numerically integrable
approaches zero. Tightened to `nominal/1.5` to `nominal×1.5` for this model
(`hnode_tuning_bound_factor_low`/`high` in the settings file); widen only incrementally
and only after confirming stability at a given range.

**`kpCraf`'s negative confidence interval.** Adam's gradient descent had no constraint
preventing this rate constant from crossing zero during optimization, and found a
local optimum with a negative value that fits the data well numerically despite not being
physically valid for a mass-action rate constant. The parameter is genuinely *identifiable*
(the data pins its value down tightly) — identifiability and physical validity are
different questions. If this matters for your downstream use, the correct fix is
reparametrizing Step 2b to optimize `log(kpCraf)` (guaranteeing positivity when exponentiated
back) rather than `kpCraf` directly, and re-running Step 2b (not Step 2a).

**No plotting packages.** `Plots.jl`, `StatsPlots.jl`, `PlotlyJS.jl`, `Cairo.jl`,
`Fontconfig.jl`, `Gadfly.jl`, `PyPlot.jl`, and `RCall.jl` were all removed from
`Project.toml` — all either need R or GUI/graphics system libraries (Qt6, GLFW, OpenGL,
fontconfig) unavailable on this HPC's headless compute nodes. Visualization is handled by
exporting simulation results to CSV and plotting externally.

---

## 7. Expected wall-clock time (reference hardware: 20-core Slurm allocation, 60GB RAM)

| Step | Typical duration |
|---|---|
| Environment setup | 15–30 min (one-time) |
| Step 0 (verification) | < 1 min each |
| Step 2a (100 trials) | 2–5 hours |
| Step 2b (10 replicates, parallel) | 1–2 hours |
| Step 3 (identifiability) | 5–15 min |
| Step 4 (confidence intervals) | 5–15 min |
| Export for plotting | < 1 min |

---

## 8. Extending this pipeline

To adapt this for a different drug combination (e.g. adding a PI3K inhibitor), at minimum:

1. Add a new mechanistic inhibition term (Hill-type, mirroring the existing Trametinib
   term) to `mapk_pi3k_vemtram_model_functions.jl` and the HNODE variant.
2. Note that parameters governing the PI3K/AKT/mTOR submodule
   (`k_PIP2_to_PIP3`, `k_PI3K_recruit`, `kMTOR_Feedback`, and most of that branch) were
   classified as **non-identifiable** by this fit — the Vem+Tram data barely constrains
   that part of the model, since Trametinib doesn't touch PI3K directly. Predictions
   involving that submodule should be treated as exploratory hypotheses, not validated
   forecasts, unless refit against data that actually perturbs PI3K.
