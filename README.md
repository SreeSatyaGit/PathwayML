# PathwayML — Vemurafenib + Trametinib MAPK/PI3K crosstalk (HNODE)

This repo is a working scaffold for applying the HNODE parameter-estimation and
identifiability-analysis pipeline (Giampiccolo et al., 2024,
https://doi.org/10.1038/s41540-024-00460-3, code: https://github.com/cosbi-research/HNODECB)
to a MAPK/PI3K crosstalk model under Vemurafenib (BRAF inhibitor) + Trametinib
(MEK inhibitor) treatment.

**Everything here is a runnable template with placeholder biology.** The pipeline
machinery (multiple shooting, TPE hyperparameter search, Hessian-based identifiability,
FIM confidence intervals) is complete and adapted from HNODECB. The model itself
(equations, parameter values, dosing, initial conditions) needs your real values — every
placeholder is marked `TODO`.

## What's assumed / needs replacing

- **State variables** (5): `pRAF, pMEK, pERK, pPI3K, pAKT` — normalized activation
  fractions in [0, 1]. If your real model has more species (receptor level, scaffolds,
  feedback inhibitors like DUSPs, etc.), you'll need to extend the state vector and
  every script that hardcodes `n_mech_params`, `column_names`, and NN input indices.
- **The "unknown" term**: AKT-mediated bypass reactivation of RAF (`k_cross * pAKT * (1 - pRAF)`),
  replaced by a neural network taking `[pRAF, pERK, pAKT]` as input. This is a real,
  well-documented resistance mechanism to BRAF/MEK inhibition, but the specific kinetic
  form here is illustrative — pick whichever term in *your* model you're least confident
  about the functional form of.
- **All kinetic parameters, IC50s, dosing regimen, and initial conditions** in
  `vem_tram_model_settings.jl` are placeholders (order-of-magnitude guesses, not fit to
  any real data).
- **Data**: `dataset_generator.jl` produces *synthetic* data from the mechanistic model so
  you can validate the whole pipeline against a known ground truth first (same self-test
  strategy the HNODECB paper uses for its three test cases). Swap in your real
  experimental time-course data before drawing scientific conclusions — see the note at
  the top of that script for the expected serialized format.

## Directory structure

```
test_case_settings/vem_tram_model_settings/
    vem_tram_model_functions.jl   # mechanistic ODE + HNODE variant
    vem_tram_model_settings.jl    # parameters, ICs, dosing, timespan
datasets/
    dataset_generator.jl          # generates e0.0/ and e0.05/ synthetic datasets
step2a_hyperparameter_tuning/
    tpe_vem_tram_00.jl            # Bayesian (TPE) search over NN arch + mechanistic
                                   # parameter starting points + multiple-shooting settings
step2b_model_trainer/
    train_vem_tram_00.jl          # full training with tuned hyperparameters (Adam + L-BFGS)
step3_parameters_identifiability/
    vem_tram_identifiability_00.jl # Hessian null-space / sloppy-direction analysis
step4_confidence_intervals/
    vem_tram_fisher_CI_00.jl      # FIM-based CIs for identifiable parameters
```

## How to run (in order)

1. Instantiate the Julia environment:
   ```
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```
   You'll also need Python + `optuna` importable via `PyCall` (Step 2a uses it for TPE).

2. Fill in your real equations/parameters/data in `test_case_settings/vem_tram_model_settings/`.

3. Generate (or replace with real) data:
   ```
   julia --project=. datasets/dataset_generator.jl
   ```

4. Run hyperparameter tuning (Step 2a) — this is the most expensive step:
   ```
   julia --project=. step2a_hyperparameter_tuning/tpe_vem_tram_00.jl
   ```
   Inspect `results_vem_tram/vem_tram_00.jld` for the best trial's hyperparameters.

5. Copy the best hyperparameters into `train_vem_tram_00.jl`'s "TUNED HYPERPARAMETERS"
   block, then run full training (Step 2b):
   ```
   julia --project=. step2b_model_trainer/train_vem_tram_00.jl
   ```

6. Run identifiability analysis (Step 3) on the best trained replicate:
   ```
   julia --project=. step3_parameters_identifiability/vem_tram_identifiability_00.jl
   ```
   Check `plots/identifiability_summary_vem_tram_00.csv` for which parameters are
   identifiable and whether the neural network is compensating for any of them.

7. Estimate confidence intervals (Step 4) for the identifiable parameters only:
   ```
   julia --project=. step4_confidence_intervals/vem_tram_fisher_CI_00.jl
   ```

## Before trusting any real result

- Confirm stiffness assumptions: the scripts default to `TRBDF2` (stiff solver). If your
  real system is non-stiff, switch to `Vern7()` + `InterpolatingAdjoint` for faster training.
- Re-run Steps 2a–4 on synthetic noiseless data first and confirm the pipeline recovers
  your known ground-truth parameters before trusting results on real/noisy data.
- The 0.05 noise-level variants (`_05` suffix in HNODECB) aren't scaffolded here yet —
  duplicate the `_00` scripts and point them at `e0.05/` data once the noiseless case works.
