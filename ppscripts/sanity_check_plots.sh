#!/bin/bash
# Sanity Check Script for Plot Generation
# Run this first to verify paths and test one simulation

echo "=================================="
echo "SANITY CHECK: Plot Generation"
echo "=================================="

# Paths
DATA_DIR="/gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts/data"
EXP_DIR="/gpfs01/home/pmyjm22/uguca_project/source/postprocessing/ppscripts/data_experiments"
EXP_CSV="$EXP_DIR/FS01-043-4MPa-RP-1_Event_1_displacement.csv"

echo ""
echo "1. Checking paths..."
echo "   Data directory: $DATA_DIR"
if [ -d "$DATA_DIR" ]; then
    echo "   ✓ Data directory exists"
else
    echo "   ✗ Data directory NOT FOUND"
    exit 1
fi

echo "   Experiment directory: $EXP_DIR"
if [ -d "$EXP_DIR" ]; then
    echo "   ✓ Experiment directory exists"
else
    echo "   ✗ Experiment directory NOT FOUND"
    exit 1
fi

echo "   Experiment CSV: $EXP_CSV"
if [ -f "$EXP_CSV" ]; then
    echo "   ✓ Experiment CSV exists"
else
    echo "   ✗ Experiment CSV NOT FOUND"
    exit 1
fi

echo ""
echo "2. Testing processed simulation data..."
TEST_SIM="gabs_fine_124-datamanager-files"
if [ -d "$DATA_DIR/$TEST_SIM" ]; then
    echo "   ✓ Test simulation exists: $TEST_SIM"
else
    echo "   ✗ Test simulation NOT FOUND: $TEST_SIM"
    echo "   Available simulations:"
    ls -d $DATA_DIR/*/ | head -10
    exit 1
fi

echo ""
echo "3. Running test error calculation..."
# Strip the -datamanager-files suffix for Python scripts
TEST_SIM_NAME="${TEST_SIM%-datamanager-files}"
python calculate_slip_error_norms.py $TEST_SIM_NAME $EXP_CSV --wdir $DATA_DIR --group interface

if [ $? -eq 0 ]; then
    echo ""
    echo "   ✓ Error calculation successful"
else
    echo ""
    echo "   ✗ Error calculation FAILED"
    exit 1
fi

echo ""
echo "4. Running test waterfall plot..."
python waterfall_and_space_time_plot.py $TEST_SIM_NAME interface auto $EXP_CSV

if [ $? -eq 0 ]; then
    echo ""
    echo "   ✓ Waterfall plot successful"
else
    echo ""
    echo "   ✗ Waterfall plot FAILED"
    exit 1
fi

echo ""
echo "=================================="
echo "✓ SANITY CHECK PASSED"
echo "=================================="
echo ""
echo "All systems operational. Ready to submit batch jobs."
