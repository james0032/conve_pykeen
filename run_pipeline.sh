#!/bin/bash

# Complete pipeline script for ConvE training
# Usage: ./run_pipeline.sh <data_dir> <output_dir>

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <data_dir> <output_dir>"
    echo ""
    echo "  data_dir: Directory containing:"
    echo "    - train.tsv, valid.tsv, test.tsv (triple files)"
    echo "    - node_dict (entity to index mapping)"
    echo "    - rel_dict (relation to index mapping)"
    echo "    - edge_map.json (optional)"
    echo ""
    echo "  output_dir: Directory for outputs (will be created)"
    echo ""
    echo "Example:"
    echo "  ./run_pipeline.sh ./data/raw ./output"
    exit 1
fi

DATA_DIR=$1
OUTPUT_DIR=$2
PROCESSED_DIR="${OUTPUT_DIR}/processed"
MODEL_DIR="${OUTPUT_DIR}/model"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found: ${DATA_DIR}${NC}"
    exit 1
fi

# Check required files
REQUIRED_FILES=("train.tsv" "valid.tsv" "test.tsv" "node_dict" "rel_dict")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${DATA_DIR}/${file}" ]; then
        echo -e "${RED}Error: Required file not found: ${DATA_DIR}/${file}${NC}"
        exit 1
    fi
done

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}ConvE Training Pipeline${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Step 1: Preprocess data
echo -e "${YELLOW}[Step 1/3] Preprocessing data...${NC}"
python preprocess.py \
    --train "${DATA_DIR}/train.tsv" \
    --valid "${DATA_DIR}/valid.tsv" \
    --test "${DATA_DIR}/test.tsv" \
    --node-dict "${DATA_DIR}/node_dict" \
    --rel-dict "${DATA_DIR}/rel_dict" \
    --edge-map "${DATA_DIR}/edge_map.json" \
    --output-dir "${PROCESSED_DIR}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Preprocessing failed!${NC}"
    exit 1
fi
echo -e "${GREEN}Preprocessing completed!${NC}"
echo ""

# Step 2: Train model
echo -e "${YELLOW}[Step 2/3] Training model...${NC}"
python train.py \
    --train "${PROCESSED_DIR}/train.txt" \
    --valid "${PROCESSED_DIR}/valid.txt" \
    --test "${PROCESSED_DIR}/test.txt" \
    --entity-to-id "${PROCESSED_DIR}/train_entity_to_id.tsv" \
    --relation-to-id "${PROCESSED_DIR}/train_relation_to_id.tsv" \
    --output-dir "${MODEL_DIR}" \
    --num-epochs 100 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --embedding-dim 200 \
    --embedding-height 10 \
    --embedding-width 20

if [ $? -ne 0 ]; then
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi
echo -e "${GREEN}Training completed!${NC}"
echo ""

# Step 3: Run examples
echo -e "${YELLOW}[Step 3/3] Running examples...${NC}"
python example.py \
    "${MODEL_DIR}" \
    "${PROCESSED_DIR}/train.txt" \
    "${PROCESSED_DIR}/test.txt"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Examples failed (non-critical)${NC}"
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  - Model: ${MODEL_DIR}/trained_model.pkl"
echo "  - Test results: ${MODEL_DIR}/test_results.json"
echo "  - Test results (CSV): ${MODEL_DIR}/test_results.csv"
echo "  - Configuration: ${MODEL_DIR}/config.json"
echo ""
echo "Next steps:"
echo "  1. Review test results: cat ${MODEL_DIR}/test_results.json"
echo "  2. Make predictions: python predict.py --model-dir ${MODEL_DIR} --query <head> <relation>"
echo "  3. Run examples: python example.py ${MODEL_DIR}"
