#!/bin/bash
#SBATCH --job-name=w_factor_comparison
#SBATCH --output=logs/w_factor_comparison_%j.log
#SBATCH --error=logs/w_factor_comparison_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# w_factor Comparison Analysis
# Compares each base simulation against its 6 w_factor variants
# Generates one plot per base showing impact of w_factor on error norms

echo "[INFO] ===== w_factor Comparison Analysis ====="
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Started: $(date)"
echo ""

# Setup
SCRIPT_DIR="$HOME/uguca_project/source/postprocessing/ppscripts"
DATA_DIR="$HOME/uguca_project/data"
EXP_CSV="$SCRIPT_DIR/data_experiments/McKlaskey_Slip.csv"
PLOT_DIR="$SCRIPT_DIR/plots/w_factor_comparison"

cd "$SCRIPT_DIR" || exit 1

# Activate Python environment
source "$HOME/python_env/bin/activate"

# Create output directories
mkdir -p "$PLOT_DIR"
mkdir -p logs

echo "[INFO] Configuration:"
echo "  Script dir:    $SCRIPT_DIR"
echo "  Data dir:      $DATA_DIR"
echo "  Experimental:  $EXP_CSV"
echo "  Plot output:   $PLOT_DIR"
echo ""

# Run comparison (use precomputed error results)
echo "[INFO] Starting w_factor comparison analysis from existing error results..."
echo ""

python compare_w_factor_vs_base.py \
    --from-results \
    --error-dir "$SCRIPT_DIR/plots/error_plots" \
    --plot-dir "$PLOT_DIR"

EXIT_CODE=$?

echo ""
echo "[INFO] Finished: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[OK] w_factor comparison completed successfully"
    echo "[INFO] Results saved to: $PLOT_DIR"
else
    echo "[ERROR] w_factor comparison failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
