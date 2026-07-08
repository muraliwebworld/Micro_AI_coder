#!/bin/bash
# Quick start script for Hugging Face dataset pipeline

set -e

PROJECT_DIR="/Users/muralidharanramasamy/Micro_AI_coder"
cd "$PROJECT_DIR"

print_header() {
    echo -e "\n════════════════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════════════════════\n"
}

print_success() {
    echo "✅  $1"
}

print_step() {
    echo "➤  $1"
}

print_error() {
    echo "❌  $1"
}

# Check if HF dataset exists
if [ ! -f "datasets/data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl" ]; then
    print_error "Hugging Face dataset not found at datasets/data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl"
    print_step "Please download and place the dataset there, then run this script again."
    exit 1
fi

print_header "HUGGING FACE DATASET PIPELINE"

# Step 1: Analyze Dataset
print_step "Step 1: Analyzing Hugging Face dataset..."
python3 scripts/analyze_hf_dataset.py
print_success "Dataset analysis complete"

# Step 2: Train Model
print_step "Step 2: Training model on Hugging Face data..."
print_step "(This may take 10-30 minutes on GPU, 1-3 hours on CPU)"
read -p "Continue with training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 phase2_training/v2_train_model_huggingface.py
    print_success "Model training complete"
else
    print_step "Skipping training"
fi

# Step 3: Test Inference
print_step "Step 3: Testing inference..."
read -p "Start interactive inference agent? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 phase4_agent/micro_ai_coder_agent_huggingface.py
else
    print_step "You can start the agent later with:"
    echo "   python3 phase4_agent/micro_ai_coder_agent_huggingface.py"
fi

print_header "🎉 PIPELINE COMPLETE!"
echo "Your Hugging Face-trained model is ready to use!"
echo ""
echo "📁 Output locations:"
echo "   Model:      models/tiny_code_model_huggingface.pt"
echo "   Config:     models/model_config_huggingface.json"
echo "   Log:        logs/training_log_huggingface.jsonl"
echo "   Components: outputs/*_hf/"
echo ""
echo "📖 Documentation: HUGGINGFACE_SETUP_GUIDE.md"
