#!/bin/bash

# ======================================================================
# Script Name: run_analyze.sh
# Description: Runs the 'analyze' mode of vector.py to extract style vectors.
#
# Usage:
# 1. Ensure you have the necessary LoRA models and a dataset file.
# 2. Run this script with the required paths as arguments.
#
#    bash run_analyze.sh --model-path <path> --formal-lora <path> \
#                        --informal-lora <path> --dataset <path> \
#                        --output <path> --target-layers <layers>
# ======================================================================

# --- Default Configuration ---
# You can change the default values here
METHOD="pca_denoise"
TARGET_LAYERS="1-27"
BATCH_SIZE=16
USE_INSTRUCT=true # Set to 'false' to disable chat/instruction template

# --- Help Message ---
usage() {
    echo "Usage: $0 --model-path <path> --formal-lora <path> --informal-lora <path> --dataset <path> --output <path> [OPTIONS]"
    echo
    echo "Required Arguments:"
    echo "  --model-path      Path to the base Hugging Face model."
    echo "  --formal-lora     Path to the formal-style LoRA adapter."
    echo "  --informal-lora   Path to the informal-style LoRA adapter."
    echo "  --dataset         Path to the dataset JSONL file."
    echo "  --output          Path to save the output style vector file (.pt)."
    echo
    echo "Optional Arguments:"
    echo "  --target-layers   Layers to analyze (default: \"${TARGET_LAYERS}\")."
    echo "  --method          Analysis method [pca_denoise, all, etc.] (default: ${METHOD})."
    echo "  --batch-size      Batch size for processing (default: ${BATCH_SIZE})."
    echo "  --no-instruct     Disable the use of the chat/instruction template."
    echo "  --help            Display this help message."
    echo
    echo "Example:"
    echo "  bash run_analyze.sh --model-path /models/Qwen-7B --formal-lora /loras/formal --informal-lora /loras/informal --dataset data.jsonl --output vectors.pt --target-layers \"18-23\""
}

# --- Parse Command-Line Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-path) BASE_MODEL_PATH="$2"; shift ;;
        --formal-lora) FORMAL_LORA_PATH="$2"; shift ;;
        --informal-lora) INFORMAL_LORA_PATH="$2"; shift ;;
        --dataset) DATASET_FILE="$2"; shift ;;
        --output) OUTPUT_PATH="$2"; shift ;;
        --target-layers) TARGET_LAYERS="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --no-instruct) USE_INSTRUCT=false ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
    esac
    shift
done

# --- Validate Required Arguments ---
if [ -z "${BASE_MODEL_PATH-}" ] || [ -z "${FORMAL_LORA_PATH-}" ] || [ -z "${INFORMAL_LORA_PATH-}" ] || [ -z "${DATASET_FILE-}" ] || [ -z "${OUTPUT_PATH-}" ]; then
    echo "Error: Missing one or more required arguments."
    usage
    exit 1
fi

# --- Prepare Flags ---
INSTRUCT_FLAG=""
if [ "$USE_INSTRUCT" = true ]; then
    INSTRUCT_FLAG="--instruct"
fi

# --- Run Python Script ---
echo "--- Starting Style Vector Extraction (analyze mode) ---"
echo "Model: ${BASE_MODEL_PATH}"
echo "Formal LoRA: ${FORMAL_LORA_PATH}"
echo "Informal LoRA: ${INFORMAL_LORA_PATH}"
echo "Dataset: ${DATASET_FILE}"
echo "Output: ${OUTPUT_PATH}"
echo "Layers: ${TARGET_LAYERS}"
echo "Method: ${METHOD}"
echo "-----------------------------------------------------"

python vector.py analyze \
    --base_model_path "$BASE_MODEL_PATH" \
    --target_layers "$TARGET_LAYERS" \
    --formal_lora_path "$FORMAL_LORA_PATH" \
    --informal_lora_path "$INFORMAL_LORA_PATH" \
    --dataset_name "$DATASET_FILE" \
    --output_path "$OUTPUT_PATH" \
    --method "$METHOD" \
    --batch_size "$BATCH_SIZE" \
    $INSTRUCT_FLAG

echo "--- Style vector extraction complete! File saved to $OUTPUT_PATH ---"