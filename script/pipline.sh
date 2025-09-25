#!/bin/bash

# ==============================================================================
# BASH SCRIPT: Full Workflow for Generation, GPT Scoring, and Analysis
#
# DESCRIPTION:
# This script automates a three-step pipeline:
#   1. Batch Generation: Generates text from prompts using a style vector.
#   2. GPT Scoring:      Scores the generated text using the OpenAI API.
#   3. Analysis:         Calculates and prints statistics from the scores.
#
# The EXPERIMENT_NAME is automatically generated based on the prefix, target
# layers, and alpha value to ensure unique output files for each run.
#
# USAGE:
# 1. Set your OpenAI API credentials in your terminal:
#    export OPENAI_API_KEY="sk-..."
#    export OPENAI_BASE_URL="https://api.openai.com/v1" (optional)
# 2. Grant execute permissions to the script: chmod +x run_pipeline.sh
# 3. Run the script with the desired arguments. See --help for options.
# ==============================================================================

# --- Safe Scripting Settings ---
# set -e: Exit immediately if a command exits with a non-zero status.
# set -u: Treat unset variables as an error when substituting.
# set -o pipefail: The return value of a pipeline is the status of the last
#                  command to exit with a non-zero status.
set -euo pipefail

# ========================== Default Configuration ==========================

# --- Experiment Identifier ---
EXPERIMENT_PREFIX="style_vector_exp"

# --- Generation Parameters ---
TARGET_LAYERS="18-23"
GENERATION_BATCH_SIZE=8
DO_SAMPLE="false"
TEMPERATURE=0.7
TOP_P=0.9
REPETITION_PENALTY=1.2

# --- Paths ---
# Default names for the python utility scripts
GENERATION_SCRIPT="vector.py"
EVALUATION_SCRIPT="evaluate_with_gpt_v2.py"
ANALYSIS_SCRIPT="analyze_results_v2.py"
RESULTS_DIR="./results"

# ========================== Argument Parsing ==========================

# --- Help Message ---
usage() {
    echo "Usage: $0 --model-path <path> --activations-path <path> --input-prompts <path> --alpha <value> [OPTIONS]"
    echo
    echo "Required Arguments:"
    echo "  --model-path          Path to the base Hugging Face model."
    echo "  --activations-path    Path to the style vector activations file (.pt)."
    echo "  --input-prompts       Path to the input prompts JSONL file."
    echo "  --alpha               The alpha value (style strength) for generation."
    echo
    echo "Optional Arguments:"
    echo "  --prefix              Prefix for the experiment name (default: ${EXPERIMENT_PREFIX})."
    echo "  --target-layers       Layers to intervene on (default: ${TARGET_LAYERS})."
    echo "  --batch-size          Batch size for generation (default: ${GENERATION_BATCH_SIZE})."
    echo "  --help                Display this help message."
    echo
    echo "Example:"
    echo "  ./run_pipeline.sh --model-path /data/LLMs/Qwen2.5-7B --activations-path ./vectors/formal.pt --input-prompts ./prompts.jsonl --alpha -5.0"
}

# --- Parse Command-Line Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-path) BASE_MODEL_PATH="$2"; shift ;;
        --activations-path) ACTIVATIONS_PATH="$2"; shift ;;
        --input-prompts) INPUT_PROMPTS_JSONL="$2"; shift ;;
        --alpha) ALPHA_VALUE="$2"; shift ;;
        --prefix) EXPERIMENT_PREFIX="$2"; shift ;;
        --target-layers) TARGET_LAYERS="$2"; shift ;;
        --batch-size) GENERATION_BATCH_SIZE="$2"; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
    esac
    shift
done

# --- Validate Required Arguments ---
if [ -z "${BASE_MODEL_PATH-}" ] || [ -z "${ACTIVATIONS_PATH-}" ] || [ -z "${INPUT_PROMPTS_JSONL-}" ] || [ -z "${ALPHA_VALUE-}" ]; then
    echo "Error: Missing one or more required arguments."
    usage
    exit 1
fi

# ========================== Script Body ==========================

# --- Step 0: Preparation ---

# Get the directory where this script is located to find other scripts
SCRIPTS_DIR=$(dirname "$0")

# Automatically generate a unique experiment name
# Example: style_vector_exp_layers_18-23_alpha_-5.0
EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_layers_${TARGET_LAYERS}_alpha_${ALPHA_VALUE}"

echo "🚀 === Starting Workflow: ${EXPERIMENT_NAME} ==="

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"

# Define paths for intermediate and final files
GENERATED_OUTPUT_FILE="${RESULTS_DIR}/generated_${EXPERIMENT_NAME}.jsonl"
SCORED_OUTPUT_FILE="${RESULTS_DIR}/scored_${EXPERIMENT_NAME}.jsonl"
ANALYSIS_OUTPUT_FILE="${RESULTS_DIR}/analysis_${EXPERIMENT_NAME}.txt"


# --- Step 1: Batch Generate Text ---
echo -e "\n[1/3] 🏃 Starting batch generation..."
echo "      - Output file: ${GENERATED_OUTPUT_FILE}"

# Skip generation if the output file already exists
if [ -f "${GENERATED_OUTPUT_FILE}" ]; then
    echo "      - ⚠️ Output file already exists. Skipping generation step."
else
    # Dynamically build the --do_sample argument
    do_sample_arg=""
    if [ "$DO_SAMPLE" = "true" ]; then
        do_sample_arg="--do_sample"
    fi

    python "${SCRIPTS_DIR}/${GENERATION_SCRIPT}" generate_batch \
        --base_model_path "${BASE_MODEL_PATH}" \
        --activations_path "${ACTIVATIONS_PATH}" \
        --input_jsonl "${INPUT_PROMPTS_JSONL}" \
        --output_jsonl "${GENERATED_OUTPUT_FILE}" \
        --target_layers "${TARGET_LAYERS}" \
        --alpha "${ALPHA_VALUE}" \
        --generation_batch_size "${GENERATION_BATCH_SIZE}" \
        ${do_sample_arg} \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --repetition_penalty "${REPETITION_PENALTY}"

    echo "      - ✅ Generation complete."
fi


# --- Step 2: Score with GPT ---
echo -e "\n[2/3] 🤖 Starting scoring with GPT..."
echo "      - Input file: ${GENERATED_OUTPUT_FILE}"
echo "      - Output file: ${SCORED_OUTPUT_FILE}"

# The scoring script automatically handles resuming, so we always run it.
python "${SCRIPTS_DIR}/${EVALUATION_SCRIPT}" \
    --input_file "${GENERATED_OUTPUT_FILE}" \
    --output_file "${SCORED_OUTPUT_FILE}"

echo "      - ✅ Scoring complete."


# --- Step 3: Analyze Scoring Results ---
echo -e "\n[3/3] 📊 Analyzing average scores..."
echo "      - Input file: ${SCORED_OUTPUT_FILE}"

# Redirect the analysis output to a text file for a clean record
python "${SCRIPTS_DIR}/${ANALYSIS_SCRIPT}" \
    --input_file "${SCORED_OUTPUT_FILE}" | tee "${ANALYSIS_OUTPUT_FILE}"

echo "      - ✅ Analysis complete. Results saved to ${ANALYSIS_OUTPUT_FILE}"

# --- End of Workflow ---
echo -e "\n🎉 === Workflow '${EXPERIMENT_NAME}' has completed successfully! ==="
echo "All intermediate files are stored in the '${RESULTS_DIR}' directory."