#!/bin/bash
# CPU-Optimized Pipeline for Mac Mini M2 (8GB RAM)

set -e

PROJECT_DIR="/Users/muralidharanramasamy/Micro_AI_coder"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅  $1${NC}"
}

print_step() {
    echo -e "${YELLOW}➤${NC}  $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌  $1${NC}"
}

# Check if HF dataset exists
if [ ! -f "datasets/data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl" ]; then
    print_error "Hugging Face dataset not found!"
    print_step "Please download and place it at: datasets/data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl"
    exit 1
fi

print_header "🚀 CPU-OPTIMIZED HUGGING FACE PIPELINE (M2 Mac 8GB RAM)"

print_warning "This is optimized for your Mac Mini M2 with 8GB RAM"
print_warning "Expected training time: 20-45 minutes"
print_warning "Your Mac will use CPU swap - this is normal\n"

# Step 1: Analyze Dataset
print_step "Step 1/3: Analyzing Hugging Face dataset..."
python3 scripts/analyze_hf_dataset.py
print_success "Dataset analysis complete\n"

# Step 2: Train Model
print_step "Step 2/3: Training CPU-optimized model..."
print_warning "(You can interrupt with Ctrl+C - checkpoint will be saved)\n"

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 phase2_training/v2_train_model_huggingface_cpu.py
    print_success "Training complete\n"
else
    print_step "Skipping training - you can run it later:"
    echo "   python3 phase2_training/v2_train_model_huggingface_cpu.py"
    exit 0
fi

# Step 3: Test Inference
print_step "Step 3/3: Testing inference agent..."

read -p "Start interactive inference? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 phase4_agent/micro_ai_coder_agent_huggingface_cpu.py
else
    print_step "You can start the agent later with:"
    echo "   python3 phase4_agent/micro_ai_coder_agent_huggingface_cpu.py"
fi

print_header "🎉 PIPELINE COMPLETE!"
echo "Your CPU-trained model is ready!\n"
echo -e "${GREEN}📁 Output Locations:${NC}"
echo "   Model:      models/tiny_code_model_huggingface_cpu.pt"
echo "   Config:     models/model_config_huggingface_cpu.json"
echo "   Log:        logs/training_log_huggingface_cpu.jsonl"
echo "   Components: outputs/*_cpu/"
echo ""
echo -e "${GREEN}📖 Documentation:${NC}"
echo "   Setup Guide: HUGGINGFACE_SETUP_GUIDE.md"
echo "   CPU Guide:   CPU_TRAINING_GUIDE.md"
echo ""
echo -e "${GREEN}🧪 Test the Model:${NC}"
echo "   Single generation: python3 phase4_agent/micro_ai_coder_agent_huggingface_cpu.py"
echo "   Compare models:    python3 scripts/compare_models.py"
echo ""
