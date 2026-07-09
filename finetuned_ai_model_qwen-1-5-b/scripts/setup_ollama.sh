#!/bin/bash

# Setup Ollama for Qwen2.5-Coder-1.5B model
# Run this script on your M2 Mac mini after GGUF conversion

set -e

echo "=================================================="
echo "🚀 Ollama Setup for Qwen2.5-Coder-1.5B"
echo "=================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not installed"
    echo ""
    echo "💡 Install with: brew install ollama"
    exit 1
fi

echo "✅ Ollama found"

# Check if model file exists
MODEL_PATH="models/qwen_1_5b_reactjs_merged.Q4_K_M.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    echo ""
    echo "💡 Make sure to:"
    echo "   1. Download merged model from Google Drive"
    echo "   2. Convert to GGUF format"
    echo "   3. Place GGUF file at: $MODEL_PATH"
    exit 1
fi

echo "✅ Model file found: $MODEL_PATH"

# Check if Modelfile exists
MODELFILE="models/Modelfile_qwen_1_5b"
if [ ! -f "$MODELFILE" ]; then
    echo "❌ Modelfile not found: $MODELFILE"
    exit 1
fi

echo "✅ Modelfile found: $MODELFILE"

# Get absolute paths
MODEL_PATH=$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")
MODELFILE=$(cd "$(dirname "$MODELFILE")" && pwd)/$(basename "$MODELFILE")

echo ""
echo "📝 Preparing Modelfile..."

# Update Modelfile with correct model path
sed -i '' "s|FROM qwen_1_5b_reactjs_merged.Q4_K_M.gguf|FROM $(echo "$MODEL_PATH" | sed 's|/|\\\/|g')|g" "$MODELFILE"

echo "✅ Modelfile updated"

# Check if Ollama is running
echo ""
echo "🔍 Checking Ollama server..."
if pgrep -xq "ollama"; then
    echo "✅ Ollama server is running"
else
    echo "ℹ️  Ollama server is not running"
    echo ""
    echo "💡 Start Ollama with:"
    echo "   ollama serve &"
    echo ""
    echo "   Then run this script again"
    exit 0
fi

# Register model
echo ""
echo "📦 Registering model with Ollama..."
echo "   (This may take 1-2 minutes)"

MODEL_NAME="micro-ai-coder-1-5b:latest"
ollama create "$MODEL_NAME" -f "$MODELFILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model registered successfully!"
    echo ""
    echo "🧪 Test the model:"
    echo "   ollama run $MODEL_NAME \"Write a React button component\""
    echo ""
    echo "📊 List all models:"
    echo "   ollama list"
    echo ""
    echo "🚀 Run agent:"
    echo "   python phase4_agent/micro_ai_coder_agent_1_5b.py"
else
    echo "❌ Model registration failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
