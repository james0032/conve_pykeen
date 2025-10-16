#!/bin/bash

# Pipeline script for running in Kubernetes pod
# No SLURM configuration needed

# Exit on error
set -e

# Print commands as they execute
set -x

# Parse command line arguments
START_STEP=1
END_STEP=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-step)
            START_STEP="$2"
            shift 2
            ;;
        --end-step)
            END_STEP="$2"
            shift 2
            ;;
        --step)
            START_STEP="$2"
            END_STEP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --start-step N    Start from step N (1-5, default: 1)"
            echo "  --end-step N      End at step N (1-5, default: 5)"
            echo "  --step N          Run only step N (equivalent to --start-step N --end-step N)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Steps:"
            echo "  1. Create ROBOKOP subgraph"
            echo "  2. Prepare node and relation dictionaries"
            echo "  3. Extract test set"
            echo "  4. Split train/validation sets"
            echo "  5. Preprocess for PyKEEN"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run all steps (1-5)"
            echo "  $0 --start-step 3           # Run steps 3-5"
            echo "  $0 --step 3                 # Run only step 3"
            echo "  $0 --start-step 2 --end-step 4  # Run steps 2-4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate step numbers
if [ "$START_STEP" -lt 1 ] || [ "$START_STEP" -gt 5 ]; then
    echo "Error: start-step must be between 1 and 5"
    exit 1
fi

if [ "$END_STEP" -lt 1 ] || [ "$END_STEP" -gt 5 ]; then
    echo "Error: end-step must be between 1 and 5"
    exit 1
fi

if [ "$START_STEP" -gt "$END_STEP" ]; then
    echo "Error: start-step must be <= end-step"
    exit 1
fi

# Configuration variables
BASE_DIR="/workspace"
WORK_DIR="${BASE_DIR}/conve_pykeen"
DATA_DIR="${BASE_DIR}/data/robokop"

# Input files
NODE_FILE="${DATA_DIR}/nodes.jsonl"
EDGES_FILE="${DATA_DIR}/edges.jsonl"

# Style configuration
STYLE="CGGD_alltreat"
OUTPUT_BASE="${DATA_DIR}/${STYLE}"

# Pipeline parameters
# Based on observed data distribution:
#   1-to-1: ~1.6%, 1-to-N: ~11%, N-to-1: ~5.3%, N-to-M: ~82%
# Strategy: Focus test set on dominant N-to-M category while maintaining representation
ONE_TO_ONE_PCT=0.005   # 1.5% - use all available (only 1.58% exist)
ONE_TO_N_PCT=0.02      # 2% - reasonable sample from 11% of data
N_TO_ONE_PCT=0.01      # 1% - reasonable sample from 5.3% of data
MANY_TO_MANY_PCT=0.065 # 6.5% - majority from dominant category (82% of data)
TRAIN_RATIO=0.9        # 90% train, 10% valid
RANDOM_SEED=42
# Total test set: ~10% of treats edges

# Logging
LOG_LEVEL="INFO"

echo "================================================================================"
echo "ROBOKOP Knowledge Graph Subgraph Preprocessing Pipeline (Pod)"
echo "================================================================================"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: ${WORK_DIR}"
echo "Running steps: ${START_STEP} to ${END_STEP}"
echo "================================================================================"

# Change to working directory
cd ${WORK_DIR}

# Print Python and package versions
echo ""
echo "Python environment:"
python --version
pip list | grep -E "(torch|pykeen|jsonlines|pandas|numpy)" || echo "Package list not available"
echo ""

# Validate input files exist (only for step 1)
if [ "$START_STEP" -le 1 ]; then
    echo "Validating input files..."
    if [ ! -f "${NODE_FILE}" ]; then
        echo "ERROR: Node file not found: ${NODE_FILE}"
        exit 1
    fi

    if [ ! -f "${EDGES_FILE}" ]; then
        echo "ERROR: Edges file not found: ${EDGES_FILE}"
        exit 1
    fi

    echo "✓ Input files validated"
    echo ""
fi

# Create output directory
mkdir -p ${OUTPUT_BASE}

# Initialize timing
TOTAL_TIME=0

# ============================================================================
# STEP 1: Create ROBOKOP subgraph
# ============================================================================
if [ "$START_STEP" -le 1 ] && [ "$END_STEP" -ge 1 ]; then
    echo "================================================================================"
    echo "STEP 1/5: Creating ROBOKOP subgraph"
    echo "================================================================================"
    echo "Input:"
    echo "  Nodes: ${NODE_FILE}"
    echo "  Edges: ${EDGES_FILE}"
    echo "  Style: ${STYLE}"
    echo "Output directory: ${OUTPUT_BASE}"
    echo ""

    STEP1_START=$(date +%s)

    python src/create_robokop_subgraph.py \
        --style ${STYLE} \
        --node-file ${NODE_FILE} \
        --edges-file ${EDGES_FILE} \
        --outdir ${OUTPUT_BASE} \
        --log-level ${LOG_LEVEL}

    STEP1_END=$(date +%s)
    STEP1_TIME=$((STEP1_END - STEP1_START))
    TOTAL_TIME=$((TOTAL_TIME + STEP1_TIME))

    echo ""
    echo "✓ Step 1 completed in ${STEP1_TIME} seconds"
    echo "Output files:"
    ls -lh ${OUTPUT_BASE}/rotorobo.txt ${OUTPUT_BASE}/edge_map.json
    echo ""
else
    echo "Skipping Step 1 (Create ROBOKOP subgraph)"
    echo ""
    STEP1_TIME=0
fi

# ============================================================================
# STEP 2: Prepare node and relation dictionaries
# ============================================================================
if [ "$START_STEP" -le 2 ] && [ "$END_STEP" -ge 2 ]; then
    echo "================================================================================"
    echo "STEP 2/5: Preparing node and relation dictionaries"
    echo "================================================================================"
    echo "Input directory: ${OUTPUT_BASE}"
    echo ""

    STEP2_START=$(date +%s)

    python src/prepare_dict.py \
        --input-dir ${OUTPUT_BASE} \
        --log-level ${LOG_LEVEL}

    STEP2_END=$(date +%s)
    STEP2_TIME=$((STEP2_END - STEP2_START))
    TOTAL_TIME=$((TOTAL_TIME + STEP2_TIME))

    echo ""
    echo "✓ Step 2 completed in ${STEP2_TIME} seconds"
    echo "Output files:"
    ls -lh ${OUTPUT_BASE}/node_dict.txt ${OUTPUT_BASE}/rel_dict.txt
    echo ""
else
    echo "Skipping Step 2 (Prepare dictionaries)"
    echo ""
    STEP2_TIME=0
fi

# ============================================================================
# STEP 3: Extract test set (treats edges with stratified sampling)
# ============================================================================
if [ "$START_STEP" -le 3 ] && [ "$END_STEP" -ge 3 ]; then
    echo "================================================================================"
    echo "STEP 3/5: Extracting test set"
    echo "================================================================================"
    echo "Sampling strategy:"
    echo "  1-to-1: ${ONE_TO_ONE_PCT} (1.5% - use all available)"
    echo "  1-to-N: ${ONE_TO_N_PCT} (2%)"
    echo "  N-to-1: ${N_TO_ONE_PCT} (1%)"
    echo "  N-to-M: ${MANY_TO_MANY_PCT} (6.5% - dominant category)"
    echo "  Total: ~10% for test set"
    echo "  Random seed: ${RANDOM_SEED}"
    echo ""

    STEP3_START=$(date +%s)

    python src/make_test.py \
        --input-dir ${OUTPUT_BASE} \
        --one-to-one-pct ${ONE_TO_ONE_PCT} \
        --one-to-n-pct ${ONE_TO_N_PCT} \
        --n-to-one-pct ${N_TO_ONE_PCT} \
        --many-to-many-pct ${MANY_TO_MANY_PCT} \
        --seed ${RANDOM_SEED} \
        --log-level ${LOG_LEVEL}

    STEP3_END=$(date +%s)
    STEP3_TIME=$((STEP3_END - STEP3_START))
    TOTAL_TIME=$((TOTAL_TIME + STEP3_TIME))

    echo ""
    echo "✓ Step 3 completed in ${STEP3_TIME} seconds"
    echo "Output files:"
    ls -lh ${OUTPUT_BASE}/test.txt ${OUTPUT_BASE}/train_candidates.txt ${OUTPUT_BASE}/test_statistics.json
    echo ""
else
    echo "Skipping Step 3 (Extract test set)"
    echo ""
    STEP3_TIME=0
fi

# ============================================================================
# STEP 4: Split remaining edges into train/valid
# ============================================================================
if [ "$START_STEP" -le 4 ] && [ "$END_STEP" -ge 4 ]; then
    echo "================================================================================"
    echo "STEP 4/5: Splitting train/validation sets"
    echo "================================================================================"
    echo "Split ratio: ${TRAIN_RATIO} train / $(awk "BEGIN {print 1 - ${TRAIN_RATIO}}") valid"
    echo "Random seed: ${RANDOM_SEED}"
    echo ""

    STEP4_START=$(date +%s)

    python src/train_valid_split.py \
        --input-dir ${OUTPUT_BASE} \
        --train-ratio ${TRAIN_RATIO} \
        --seed ${RANDOM_SEED} \
        --log-level ${LOG_LEVEL}

    STEP4_END=$(date +%s)
    STEP4_TIME=$((STEP4_END - STEP4_START))
    TOTAL_TIME=$((TOTAL_TIME + STEP4_TIME))

    echo ""
    echo "✓ Step 4 completed in ${STEP4_TIME} seconds"
    echo "Output files:"
    ls -lh ${OUTPUT_BASE}/train.txt ${OUTPUT_BASE}/valid.txt ${OUTPUT_BASE}/split_statistics.json
    echo ""
else
    echo "Skipping Step 4 (Train/valid split)"
    echo ""
    STEP4_TIME=0
fi

# ============================================================================
# STEP 5: Preprocess for PyKEEN
# ============================================================================
if [ "$START_STEP" -le 5 ] && [ "$END_STEP" -ge 5 ]; then
    echo "================================================================================"
    echo "STEP 5/5: Preprocessing for PyKEEN"
    echo "================================================================================"

    PROCESSED_DIR="${OUTPUT_BASE}/processed"
    mkdir -p ${PROCESSED_DIR}

    echo "Output directory: ${PROCESSED_DIR}"
    echo ""

    STEP5_START=$(date +%s)

    python preprocess.py \
        --train ${OUTPUT_BASE}/train.txt \
        --valid ${OUTPUT_BASE}/valid.txt \
        --test ${OUTPUT_BASE}/test.txt \
        --node-dict ${OUTPUT_BASE}/node_dict.txt \
        --rel-dict ${OUTPUT_BASE}/rel_dict.txt \
        --edge-map ${OUTPUT_BASE}/edge_map.json \
        --output-dir ${PROCESSED_DIR} \
        --no-validate

    STEP5_END=$(date +%s)
    STEP5_TIME=$((STEP5_END - STEP5_START))
    TOTAL_TIME=$((TOTAL_TIME + STEP5_TIME))

    echo ""
    echo "✓ Step 5 completed in ${STEP5_TIME} seconds"
    echo "Output files:"
    ls -lh ${PROCESSED_DIR}/
    echo ""
else
    echo "Skipping Step 5 (Preprocess for PyKEEN)"
    echo ""
    STEP5_TIME=0
fi

# ============================================================================
# Pipeline Summary
# ============================================================================
echo "================================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo "Steps executed: ${START_STEP} to ${END_STEP}"
echo "Timing summary:"
if [ "$STEP1_TIME" -gt 0 ]; then
    echo "  Step 1 (Create subgraph):       ${STEP1_TIME}s"
fi
if [ "$STEP2_TIME" -gt 0 ]; then
    echo "  Step 2 (Prepare dictionaries):  ${STEP2_TIME}s"
fi
if [ "$STEP3_TIME" -gt 0 ]; then
    echo "  Step 3 (Extract test set):      ${STEP3_TIME}s"
fi
if [ "$STEP4_TIME" -gt 0 ]; then
    echo "  Step 4 (Train/valid split):     ${STEP4_TIME}s"
fi
if [ "$STEP5_TIME" -gt 0 ]; then
    echo "  Step 5 (Preprocess for PyKEEN): ${STEP5_TIME}s"
fi
echo "  Total time:                      ${TOTAL_TIME}s ($((TOTAL_TIME / 60)) minutes)"
echo ""
echo "Output directory: ${OUTPUT_BASE}"
if [ "$END_STEP" -ge 5 ]; then
    echo "Processed data: ${PROCESSED_DIR}"
fi
echo ""

# Optional: Print dataset statistics (only if all steps completed)
if [ "$START_STEP" -le 5 ] && [ "$END_STEP" -ge 5 ]; then
    echo "Dataset Statistics:"
    echo "-------------------------------------------------------------------------------"
    echo "Train set:"
    wc -l ${PROCESSED_DIR}/train.txt
    echo "Valid set:"
    wc -l ${PROCESSED_DIR}/valid.txt
    echo "Test set:"
    wc -l ${PROCESSED_DIR}/test.txt
    echo ""
    echo "Node dictionary:"
    wc -l ${OUTPUT_BASE}/node_dict.txt
    echo "Relation dictionary:"
    wc -l ${OUTPUT_BASE}/rel_dict.txt
    echo "================================================================================"
    echo ""
    echo "Ready for ConvE training!"
    echo "To start training, run:"
    echo "  python train.py --data-dir ${PROCESSED_DIR}"
fi

echo ""
echo "End time: $(date)"
echo "================================================================================"

exit 0
