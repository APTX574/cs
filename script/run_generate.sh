#!/bin/bash

# ======================================================================
# Script Name: run_generate.sh
# Description: Runs the 'generate' mode of vector.py for interactive,
#              style-controlled text generation.
#
# Usage:
# 1. Ensure you have already run an analysis script (e.g., run_analyze.sh)
#    to generate a style vector file (.pt).
# 2. Run this script with the required paths as arguments:
#
#    bash run_generate.sh /path/to/your/model /path/to/your/vector.pt "18-23"
#
# Arguments:
#   $1: Path to the base model.
#   $2: Path to the pre-computed activations/style vector file (.pt).
#   $3: Target layers to apply the steering vector to (e.g., "18-23").
# ======================================================================

# --- Configuration ---
# Set to true to use the chat/instruction template, false to use raw text.
USE_INSTRUCT=true

# --- Argument Validation ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: bash run_generate.sh <BASE_MODEL_PATH> <ACTIVATIONS_FILE_PATH> <TARGET_LAYERS>"
    echo "Example: bash run_generate.sh ./models/llama-3-8b ./vectors/formal_style.pt \"18-23\""
    exit 1
fi

# --- Assign Arguments to Variables ---
BASE_MODEL_PATH="$1"
ACTIVATIONS_FILE="$2"
TARGET_LAYERS="$3"

# --- Prepare Flags ---
INSTRUCT_FLAG=""
if [ "$USE_INSTRUCT" = true ]; then
    INSTRUCT_FLAG="--instruct"
fi

# --- Run Python Script ---
echo "--- Starting Interactive Style Generation (generate mode) ---"
echo "Model Path: $BASE_MODEL_PATH"
echo "Vector File: $ACTIVATIONS_FILE"
echo "Target Layers: $TARGET_LAYERS"
echo "-------------------------------------------------------------"

python vector.py generate \
    --base_model_path "$BASE_MODEL_PATH" \
    --activations_paths "$ACTIVATIONS_FILE" \
    --target_layers "$TARGET_LAYERS" \
    $INSTRUCT_FLAG

echo "--- Interactive generation session has ended ---"