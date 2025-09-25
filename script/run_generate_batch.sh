#!/bin/bash

# ======================================================================
# Script Name: run_generate_batch.sh
# Description: Runs the 'generate_batch' mode of vector.py to perform
#              controllable text generation on a dataset.
#
# Usage:
# 1. Ensure you have run an analysis script (e.g., run_analyze.sh) to
#    generate a style vector file (.pt).
# 2. Run this script with the required paths and parameters as arguments.
#
#    bash run_generate_batch.sh --model-path <path> --vector-path <path> \
#                               --input-file <path> --output-file <path> \
#                               --layers "19-23" --alpha -5.0
# ======================================================================

# --- Default Configuration ---
# You can change the default values here
GENERATION_BATCH_SIZE=8
USE_INSTRUCT=true
COMPUTE_PERPLEXITY=false

# --- Help Message ---
usage() {
    echo "Usage: $0 --model-path <path> --vector-path <path> --input-file <path> --output-file <path> --layers <layers> --alpha <alpha> [OPTIONS]"
    echo
    echo "Required Arguments:"
    echo "  --model-path      Path to the base Hugging Face model."
    echo "  --vector-path     Path to the style vector activations file (.pt)."
    echo "  --input-file      Path to the input JSONL file containing prompts."
    echo "  --output-file     Path to save the generated output JSONL file."
    echo "  --layers          Target layers for intervention (e.g., \"19-23\")."
    echo "  --alpha           The alpha value (style strength) for generation."
    echo
    echo "Optional Arguments:"
    echo "  --batch-size      Batch size for generation (default: ${GENERATION_BATCH_SIZE})."
    echo "  --no-instruct     Disable the use of the chat/instruction template."
    echo "  --compute-perplexity  Enable computation of perplexity for generated text."
    echo "  --help            Display this help message."
    echo
    echo "Example:"
    echo "  bash run_generate_batch.sh --model-path /models/Qwen-7B --vector-path vec.pt --input-file prompts.jsonl --output-file results.jsonl --layers \"19-23\" --alpha -4.0"
}

# --- Parse Command-Line Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-path) BASE_MODEL_PATH="$2"; shift ;;
        --vector-path) ACTIVATIONS_FILE="$2"; shift ;;
        --input-file) INPUT_JSONL_FILE="$2"; shift ;;
        --output-file) OUTPUT_JSONL_FILE="$2"; shift ;;
        --layers) TARGET_LAYERS="$2"; shift ;;
        --alpha) ALPHA="$2"; shift ;;
        --batch-size) GENERATION_BATCH_SIZE="$2"; shift ;;
        --no-instruct) USE_INSTRUCT=false ;;
        --compute-perplexity) COMPUTE_PERPLEXITY=true ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
    esac
    shift
done

# --- Validate Required Arguments ---
if [ -z "${BASE_MODEL_PATH-}" ] || [ -z "${ACTIVATIONS_FILE-}" ] || [ -z "${INPUT_JSONL_FILE-}" ] || [ -z "${OUTPUT_JSONL_FILE-}" ] || [ -z "${TARGET_LAYERS-}" ] || [ -z "${ALPHA-}" ]; then
    echo "Error: Missing one or more required arguments."
    usage
    exit 1
fi

# --- Prepare Flags ---
INSTRUCT_FLAG=""
if [ "$USE_INSTRUCT" = true ]; then
    INSTRUCT_FLAG="--instruct"
fi

PERPLEXITY_FLAG=""
if [ "$COMPUTE_PERPLEXITY" = true ]; then
    PERPLEXITY_FLAG="--compute_perplexity"
fi

# --- Run Python Script ---
echo "--- Starting Batch Style Generation (generate_batch mode) ---"
echo "Model: ${BASE_MODEL_PATH}"
echo "Vector File: ${ACTIVATIONS_FILE}"
echo "Input Prompts: ${INPUT_JSONL_FILE}"
echo "Output File: ${OUTPUT_JSONL_FILE}"
echo "Target Layers: ${TARGET_LAYERS}"
echo "Alpha: ${ALPHA}"
echo "------------------------------------------------------------"

python vector.py generate_batch \
    --base_model_path "$BASE_MODEL_PATH" \
    --target_layers "$TARGET_LAYERS" \
    --activations_path "$ACTIVATIONS_FILE" \
    --input_jsonl "$INPUT_JSONL_FILE" \
    --output_jsonl "$OUTPUT_JSONL_FILE" \
    --alpha "$ALPHA" \
    --generation_batch_size "$GENERATION_BATCH_SIZE" \
    --max_new_tokens 512 \
    $INSTRUCT_FLAG \
    $PERPLEXITY_FLAG

echo "--- Batch generation complete! Results saved to $OUTPUT_JSONL_FILE ---"