#!/usr/bin/env python3
"""
Inference Script for Fine-Tuned Qwen2.5-Coder-1.5B via Ollama
Local testing on M2 Mac
"""

import requests
import json
import logging
from typing import Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ollama Configuration
BASE_URL = "http://localhost:11434"
MODEL_NAME = "micro-ai-coder-1-5b:latest"
GENERATION_URL = f"{BASE_URL}/api/generate"

# Generation Parameters
TEMPERATURE = 0.3
TOP_K = 40
TOP_P = 0.9
MAX_TOKENS = 256


def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    logger.info("🔍 Checking Ollama connection...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info("   ✅ Ollama is running")
            
            # Check if our model is available
            model_names = [m.get('name') for m in models]
            if any(MODEL_NAME in name for name in model_names):
                logger.info(f"   ✅ Model '{MODEL_NAME}' found")
                return True
            else:
                logger.warning(f"   ⚠️  Model '{MODEL_NAME}' not found")
                logger.info(f"   Available models: {model_names}")
                return False
        else:
            logger.error(f"   ❌ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("   ❌ Cannot connect to Ollama at http://localhost:11434")
        logger.info("   Start Ollama with: ollama serve")
        return False
    except Exception as e:
        logger.error(f"   ❌ Error: {e}")
        return False


def validate_react_syntax(code: str) -> bool:
    """Validate if generated code looks like valid React"""
    # Check for balanced brackets
    if code.count('{') != code.count('}'):
        return False
    if code.count('[') != code.count(']'):
        return False
    if code.count('(') != code.count(')'):
        return False
    
    # Check for import/export
    has_import_export = ('import' in code or 'export' in code) or 'React' in code
    
    # Check for JSX
    has_jsx = '<' in code and '>' in code and '/' in code
    
    return has_import_export and has_jsx


def generate_code(prompt: str) -> Optional[dict]:
    """Generate React code using fine-tuned model"""
    logger.info(f"\n📝 Prompt: {prompt}")
    logger.info("-" * 60)
    
    # Format prompt
    full_prompt = f"Prompt: {prompt}\n\nCode:\n"
    
    try:
        # Call Ollama API
        response = requests.post(
            GENERATION_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "num_predict": MAX_TOKENS,
            },
            timeout=60,
        )
        
        if response.status_code != 200:
            logger.error(f"❌ API error: {response.status_code}")
            return None
        
        # Extract generated code
        result = response.json()
        full_response = result.get('response', '')
        generated_code = full_response.split("Code:\n")[-1].strip()
        
        # Validate
        is_valid = validate_react_syntax(generated_code)
        validation_status = "✅ Valid" if is_valid else "⚠️  Needs review"
        
        logger.info(f"\n{validation_status} React Code:")
        logger.info(generated_code[:300])
        if len(generated_code) > 300:
            logger.info("...")
        
        return {
            "prompt": prompt,
            "code": generated_code,
            "valid": is_valid,
            "timestamp": datetime.now().isoformat(),
        }
    
    except requests.exceptions.Timeout:
        logger.error("❌ Request timeout (model may be loading)")
        return None
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None


def single_generation():
    """Single code generation mode"""
    logger.info("\n" + "=" * 60)
    logger.info("React Component Generator - Single Mode")
    logger.info("=" * 60)
    
    prompt = input("\n📝 Enter prompt (or 'quit' to exit): ").strip()
    
    if prompt.lower() == 'quit':
        return False
    
    if not prompt:
        logger.warning("⚠️  Empty prompt")
        return True
    
    result = generate_code(prompt)
    
    if result:
        # Save to log
        with open("./outputs/generation_log.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        logger.info("   ✅ Saved to outputs/generation_log.jsonl")
    
    return True


def batch_generation(prompts: list):
    """Batch code generation"""
    logger.info("\n" + "=" * 60)
    logger.info("React Component Generator - Batch Mode")
    logger.info(f"Generating {len(prompts)} components...")
    logger.info("=" * 60)
    
    results = []
    valid_count = 0
    
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\n[{i}/{len(prompts)}] {prompt}")
        result = generate_code(prompt)
        
        if result:
            results.append(result)
            if result['valid']:
                valid_count += 1
        
        logger.info("")
    
    # Summary
    success_rate = (valid_count / len(results) * 100) if results else 0
    logger.info("\n" + "=" * 60)
    logger.info("📊 Batch Generation Summary")
    logger.info("=" * 60)
    logger.info(f"Generated: {len(results)}/{len(prompts)}")
    logger.info(f"Valid: {valid_count}/{len(results)} ({success_rate:.1f}%)")
    
    # Save results
    output_file = f"./outputs/batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Results saved to: {output_file}")
    
    return results


def interactive_menu():
    """Interactive menu"""
    test_prompts = [
        "Create a React button component with onClick handler",
        "Write a React counter component with state",
        "Generate a React form with input fields",
        "Create a React card component with props",
        "Write a React list component that renders an array",
    ]
    
    while True:
        logger.info("\n" + "=" * 60)
        logger.info("React Component Generator")
        logger.info("=" * 60)
        logger.info("1. Single generation")
        logger.info("2. Batch generation (test prompts)")
        logger.info("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            if not single_generation():
                break
        elif choice == "2":
            batch_generation(test_prompts)
        elif choice == "3":
            logger.info("\n👋 Goodbye!")
            break
        else:
            logger.warning("⚠️  Invalid option")


def main():
    """Main inference interface"""
    logger.info("=" * 60)
    logger.info("Qwen2.5-Coder-1.5B Inference - M2 Mac")
    logger.info("=" * 60)
    
    # Check connection
    if not check_ollama_connection():
        logger.error("\n❌ Cannot proceed without Ollama")
        return
    
    # Interactive menu
    try:
        interactive_menu()
    except KeyboardInterrupt:
        logger.info("\n\n👋 Interrupted by user")


if __name__ == "__main__":
    main()
