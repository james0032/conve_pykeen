#!/bin/bash -l

#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-6-1
#SBATCH --nodes=1 --ntasks=24
#SBATCH --mem=600G --gres=gpu:1 --time=10-0:00
#SBATCH --job-name=ROBOKOPKG_subgraph_preprocess
#SBATCH --output=/projects/aixb/jchung/everycure/influence_estimate/sjob_logs/pipeline_%j.out
#SBATCH --mail-user=jchung@renci.org
#SBATCH --mail-type=ALL

# Exit on error
set -e

# Print commands as they execute
set -x

# Configuration variables
BASE_DIR="/projects/aixb/jchung/everycure"
WORK_DIR="${BASE_DIR}/git/conve_pykeen"
DATA_DIR="${BASE_DIR}/influence_estimate/robokop"

# Input files
NODE_FILE="${DATA_DIR}/nodes.jsonl"
EDGES_FILE="${DATA_DIR}/edges.jsonl"

# Style configuration
STYLE="CGGD_alltreat"
OUTPUT_BASE="${DATA_DIR}/${STYLE}"

# Pipeline parameters
# Note: Adjust these based on your data distribution
# If 1-to-1 edges are scarce (<5%), the script will use all available and warn you
ONE_TO_ONE_PCT=0.015   # 1.5% - use all available 1-to-1 edges
ONE_TO_N_PCT=0.03      # 3% - increase to compensate
N_TO_ONE_PCT=0.03      # 3% - increase to compensate
MANY_TO_MANY_PCT=0.025 # 2.5% - sample from N-to-M edges
TRAIN_RATIO=0.9        # 90% train, 10% valid
RANDOM_SEED=42

# Logging
LOG_LEVEL="INFO"

# Python environment
source /home/jchung/environments/conve_pykeen/.venv/bin/activate

echo "================================================================================"
echo "ROBOKOP Knowledge Graph Subgraph Preprocessing Pipeline"
echo "================================================================================"
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Working directory: ${WORK_DIR}"
echo "================================================================================"

# Change to working directory
cd ${WORK_DIR}

# Print Python and package versions
echo ""
echo "Python environment:"
python --version
# pip list | grep -E "(torch|pykeen|jsonlines|pandas|numpy)"
echo ""

# Validate input files exist
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

# Create output directory
mkdir -p ${OUTPUT_BASE}

# ============================================================================
# STEP 1: Create ROBOKOP subgraph
# ============================================================================
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

echo ""
echo "✓ Step 1 completed in ${STEP1_TIME} seconds"
echo "Output files:"
ls -lh ${OUTPUT_BASE}/rotorobo.txt ${OUTPUT_BASE}/edge_map.json
echo ""

# ============================================================================
# STEP 2: Prepare node and relation dictionaries
# ============================================================================
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

echo ""
echo "✓ Step 2 completed in ${STEP2_TIME} seconds"
echo "Output files:"
ls -lh ${OUTPUT_BASE}/node_dict.txt ${OUTPUT_BASE}/rel_dict.txt
echo ""

# ============================================================================
# STEP 3: Extract test set (treats edges with stratified sampling)
# ============================================================================
echo "================================================================================"
echo "STEP 3/5: Extracting test set"
echo "================================================================================"
echo "Sampling strategy:"
echo "  1-to-1: ${ONE_TO_ONE_PCT} (1.5%)"
echo "  1-to-N: ${ONE_TO_N_PCT} (3%)"
echo "  N-to-1: ${N_TO_ONE_PCT} (3%)"
echo "  N-to-M: ${MANY_TO_MANY_PCT} (2.5%)"
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

echo ""
echo "✓ Step 3 completed in ${STEP3_TIME} seconds"
echo "Output files:"
ls -lh ${OUTPUT_BASE}/test.txt ${OUTPUT_BASE}/train_candidates.txt ${OUTPUT_BASE}/test_statistics.json
echo ""

# ============================================================================
# STEP 4: Split remaining edges into train/valid
# ============================================================================
echo "================================================================================"
echo "STEP 4/5: Splitting train/validation sets"
echo "================================================================================"
echo "Split ratio: ${TRAIN_RATIO} train / $((1 - TRAIN_RATIO)) valid"
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

echo ""
echo "✓ Step 4 completed in ${STEP4_TIME} seconds"
echo "Output files:"
ls -lh ${OUTPUT_BASE}/train.txt ${OUTPUT_BASE}/valid.txt ${OUTPUT_BASE}/split_statistics.json
echo ""

# ============================================================================
# STEP 5: Preprocess for PyKEEN
# ============================================================================
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

echo ""
echo "✓ Step 5 completed in ${STEP5_TIME} seconds"
echo "Output files:"
ls -lh ${PROCESSED_DIR}/
echo ""

# ============================================================================
# Pipeline Summary
# ============================================================================
TOTAL_TIME=$((STEP1_TIME + STEP2_TIME + STEP3_TIME + STEP4_TIME + STEP5_TIME))

echo "================================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo "Timing summary:"
echo "  Step 1 (Create subgraph):       ${STEP1_TIME}s"
echo "  Step 2 (Prepare dictionaries):  ${STEP2_TIME}s"
echo "  Step 3 (Extract test set):      ${STEP3_TIME}s"
echo "  Step 4 (Train/valid split):     ${STEP4_TIME}s"
echo "  Step 5 (Preprocess for PyKEEN): ${STEP5_TIME}s"
echo "  Total time:                      ${TOTAL_TIME}s ($((TOTAL_TIME / 60)) minutes)"
echo ""
echo "Output directory: ${OUTPUT_BASE}"
echo "Processed data: ${PROCESSED_DIR}"
echo ""
echo "Final output files:"
echo "  - Dictionary files: node_dict.txt, rel_dict.txt"
echo "  - Data splits: train.txt, valid.txt, test.txt"
echo "  - Processed data: ${PROCESSED_DIR}/"
echo "  - Statistics: test_statistics.json, split_statistics.json"
echo ""
echo "End time: $(date)"
echo "================================================================================"

# Optional: Print dataset statistics
echo ""
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
echo ""

exit 0
