#!/bin/bash

# This script runs Depth Anything V2 and works around the Python script's
# hardcoded model path by creating a symbolic link to the correct weights file.

# --- Argument Parsing ---
INPUT_VIDEO=""
OUTPUT_DIR=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --input_video)
            INPUT_VIDEO="$2"; shift; shift;;
        --output_dir)
            OUTPUT_DIR="$2"; shift; shift;;
        *)
            echo "❌ Unknown parameter passed: $1"; exit 1;;
    esac
done

# --- Validation ---
if [ -z "$INPUT_VIDEO" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "❌ Error: Both --input_video and --output_dir arguments are required."
    echo "Usage: $0 --input_video <path_to_video> --output_dir <path_to_output>"
    exit 1
fi

# --- Configuration ---
PYTHON_SCRIPT="/coc/flash7/yliu3735/workspace/Inpaint/depth_anything/run_video.py"
# The correct, full path to your model weights file
MODEL_WEIGHTS_PATH="/coc/flash7/yliu3735/workspace/Inpaint/weights/depth_anything_v2_vitl.pth"
# The path the Python script *expects* the weights to be at
EXPECTED_WEIGHTS_PATH="checkpoints/depth_anything_v2_vitl.pth"

# --- Execution ---
echo "🚀 Starting Depth Anything V2 process..."

# --- 💡 WORKAROUND ---
# The Python script has a hardcoded path. We create the directory and a
# symbolic link (a shortcut) to your actual weights file.
echo "🔧 Setting up symbolic link for model weights..."
mkdir -p "$(dirname "$EXPECTED_WEIGHTS_PATH")"
ln -sf "$MODEL_WEIGHTS_PATH" "$EXPECTED_WEIGHTS_PATH"
# 'ln -sf' creates a symbolic link, overwriting if it already exists.

echo "========================================"
echo "▶️  Input Video: $INPUT_VIDEO"
echo "📂 Output Directory: $OUTPUT_DIR"
echo "🤖 Model: v2-large (vitl encoder)"
echo "========================================"

# Create the output directory if it doesn't already exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script. It will now find the model via the symbolic link.
python "$PYTHON_SCRIPT" \
    --encoder vitl \
    --video-path "$INPUT_VIDEO" \
    --outdir "$OUTPUT_DIR"

echo "✅ Processing complete! Output saved in $OUTPUT_DIR"