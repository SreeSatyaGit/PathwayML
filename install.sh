#!/usr/bin/env bash
# Installation script for running the PathwayML HNODE scaffold on an HPC cluster (no sudo).
#
# IMPORTANT: run this on your cluster's LOGIN node, not a compute node. Julia's package
# manager and pip both need outbound internet access to download packages, which compute
# nodes on most HPC clusters do not have. Once everything is installed, you can submit
# the actual pipeline runs (Steps 2a/2b/3/4) as batch jobs on compute nodes as usual.
set -e

# Project root -- adjust this if your project lives somewhere else.
PROJECT_DIR="/home/nandivada.s/vanaja_lab/satya/PathwayML"
VENV_DIR="$PROJECT_DIR/venv"

echo "=== 1. Check system prerequisites (no sudo -- HPC cluster) ==="
# HPC systems almost always already provide curl, git, and a C compiler system-wide or
# via environment modules (Lmod/module system), rather than through apt/yum.
# Check what's available first:
command -v curl  >/dev/null && echo "curl found"  || echo "curl NOT found -- see note below"
command -v git   >/dev/null && echo "git found"   || echo "git NOT found -- see note below"
command -v gcc   >/dev/null && echo "gcc found"   || echo "gcc NOT found -- see note below"
command -v python3 >/dev/null && echo "python3 found ($(python3 --version))" || echo "python3 NOT found -- see note below"

# If any of the above are missing, your cluster likely provides them as modules. Uncomment
# and adjust the lines below to match what `module avail` shows on your system, e.g.:
# module load gcc
# module load python/3.10
# module load git
#
# If modules aren't used on your cluster and something is still missing, ask your HPC
# admins how to get it (spack, conda, or a support ticket) -- do not attempt to install
# system packages yourself without sudo.

echo "=== 2. Install Julia via juliaup, pinned to 1.9.1 ==="
curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="$HOME/.juliaup/bin:$PATH"   # so this script can use juliaup right away
juliaup add 1.9.1
juliaup default 1.9.1

echo "=== 3. Python virtual environment + optuna (for PyCall / Step 2a) ==="
mkdir -p "$PROJECT_DIR"
if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "python3's venv module isn't available. Try 'module load python/<version>' for a"
  echo "python module that includes venv, or use 'conda create' instead if your cluster"
  echo "provides conda/miniconda as a module."
  exit 1
fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install optuna
deactivate

echo "=== 4. Point PyCall at that Python and build it ==="
julia -e "
ENV[\"PYTHON\"] = \"$VENV_DIR/bin/python3\"
using Pkg
Pkg.add(\"PyCall\")
Pkg.build(\"PyCall\")
"

echo "=== 5. Get the project files ==="
echo "If you haven't already: unzip PathwayML_scaffold.zip into $PROJECT_DIR"
echo "or: git clone https://github.com/SreeSatyaGit/PathwayML.git $PROJECT_DIR"
echo "The rest of this script assumes the Julia project files (Project.toml etc.) are directly inside $PROJECT_DIR."

echo "=== 6. Instantiate the Julia project (installs pinned SciML packages) ==="
cd "$PROJECT_DIR"
julia --project=. -e 'using Pkg; Pkg.instantiate()'

echo "=== 7. Sanity check ==="
julia --project=. -e '
using DifferentialEquations, Lux, SciMLSensitivity, DiffEqFlux, PyCall
println("All Julia packages loaded OK")
pyimport("optuna")
println("optuna import OK")
'

echo "=== Done. Environment is ready. ==="
echo "Venv location: $VENV_DIR"
echo "Next: cd $PROJECT_DIR && julia --project=. datasets/dataset_generator.jl"