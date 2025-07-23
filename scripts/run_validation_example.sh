#!/bin/bash

# Example script showing how to use the validation script
# This script demonstrates validating a model from a checkpoint

echo "=== Model Validation Example ==="
echo "This script shows how to validate a trained model from a checkpoint"
echo ""

# Configuration file path
CONFIG_FILE="confs/validate/example_validation.yaml"

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "Please create the configuration file first."
    exit 1
fi

echo "📋 Configuration file: $CONFIG_FILE"

# Example checkpoint path (you would replace this with your actual checkpoint)
EXAMPLE_CHECKPOINT="models/example_model/checkpoint-1000"

echo "🔍 Example checkpoint path: $EXAMPLE_CHECKPOINT"
echo ""

# Check if checkpoint exists
if [ ! -d "$EXAMPLE_CHECKPOINT" ]; then
    echo "⚠️  Example checkpoint not found: $EXAMPLE_CHECKPOINT"
    echo "This is expected for the example. You would replace this with your actual checkpoint path."
    echo ""
    echo "To run validation with your own checkpoint, use:"
    echo ""
    echo "python src/tactic_gen/validate_model.py \\"
    echo "    --config $CONFIG_FILE \\"
    echo "    --checkpoint /path/to/your/checkpoint \\"
    echo "    --output_dir validation_results \\"
    echo "    --num_eval_examples 1000"
    echo ""
    exit 0
fi

# Output directory for results
OUTPUT_DIR="validation_results"

echo "📊 Output directory: $OUTPUT_DIR"
echo ""

# Run validation (commented out since checkpoint doesn't exist)
echo "🚀 Running validation..."
echo ""

# Uncomment the following lines when you have a real checkpoint:
# python src/tactic_gen/validate_model.py \
#     --config "$CONFIG_FILE" \
#     --checkpoint "$EXAMPLE_CHECKPOINT" \
#     --output_dir "$OUTPUT_DIR" \
#     --num_eval_examples 1000

echo "✅ Validation script is ready to use!"
echo ""
echo "To run validation with your own model:"
echo "1. Update the configuration file: $CONFIG_FILE"
echo "2. Replace the checkpoint path with your actual checkpoint"
echo "3. Run the validation command above"
echo ""
echo "For more information, see: src/tactic_gen/README_validation.md" 