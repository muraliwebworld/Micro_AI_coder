#!/usr/bin/env python3
"""
PHASE 3: INFERENCE WITH FINE-TUNED MODEL
Updated to use the fine-tuned Qwen2.5-Coder model via Ollama
"""
import requests
import json
import time
import re
from pathlib import Path
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:11434"
GENERATION_URL = f"{BASE_URL}/api/generate"
INFERENCE_LOG = Path(__file__).parent.parent.parent / "logs" / "generation_results_finetuned.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"

# Use fine-tuned model
MODEL_NAME = "micro-ai-coder-v2:latest"

INFERENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def print_header(text): 
    print("\n" + "="*80 + f"\n{text}\n" + "="*80)

def print_step(text): 
    print(f"\n➤ {text}")

def print_success(text): 
    print(f"✅ {text}")

def print_warning(text): 
    print(f"⚠️  {text}")

def print_error(text): 
    print(f"❌ {text}")

def check_ollama_connection():
    """Verify Ollama is running with fine-tuned model"""
    print_step("Checking Ollama connection...")
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            tags = response.json().get("models", [])
            model_names = [m.get("name", "") for m in tags]
            
            if any(MODEL_NAME in m for m in model_names):
                print_success(f"Ollama running with {MODEL_NAME} loaded")
                return True
            else:
                print_error(f"Model {MODEL_NAME} not found in Ollama")
                print(f"Available models: {model_names}")
                return False
    except Exception as e:
        print_error(f"Cannot connect to Ollama: {str(e)}")
        print("Make sure Ollama is running: ollama serve")
        return False

def validate_react_component(code):
    """Validate React component syntax"""
    if len(code) < 50:
        return False, "Code too short"
    
    has_import = 'import' in code
    has_export = 'export' in code
    has_jsx = '<' in code and '>' in code
    
    # Check bracket balance
    if not (code.count('{') == code.count('}')):
        return False, "Unbalanced braces"
    if not (code.count('[') == code.count(']')):
        return False, "Unbalanced brackets"
    if not (code.count('(') == code.count(')')):
        return False, "Unbalanced parentheses"
    
    if not (has_import and has_export and has_jsx):
        missing = []
        if not has_import: missing.append("import")
        if not has_export: missing.append("export")
        if not has_jsx: missing.append("JSX")
        return False, f"Missing: {', '.join(missing)}"
    
    return True, ""

def generate_react_component(prompt):
    """Generate React component using fine-tuned model"""
    print_step(f"Generating React component...")
    
    enhanced_prompt = f"""Generate a complete, self-contained React component for:
{prompt}

Requirements:
- Use React hooks (useState, useEffect, etc.)
- Include proper imports at the top
- Export as default
- Include JSX with proper structure
- Make it functional and practical
- Valid JavaScript/JSX syntax
- NO explanations, just the code"""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": enhanced_prompt,
        "stream": False,
        "temperature": 0.3,
        "top_k": 40,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(GENERATION_URL, json=payload, timeout=30)
        if response.status_code == 200:
            code = response.json().get("response", "").strip()
            
            # Clean up markdown if present
            code = re.sub(r'^```[a-zA-Z]*\s*\n?', '', code, flags=re.MULTILINE)
            code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
            code = code.strip()
            
            is_valid, reason = validate_react_component(code)
            if is_valid:
                print_success(f"Generated component ({len(code)} chars)")
                return code, True
            else:
                print_warning(f"Validation failed: {reason}")
                return code, False
        else:
            print_error(f"API Error {response.status_code}")
            return None, False
    except Exception as e:
        print_error(f"Exception: {str(e)}")
        return None, False

def generate_explanation(prompt, code):
    """Generate explanation for the component"""
    print_step(f"Generating explanation...")
    
    cot_prompt = f"""Given this React component requirement and implementation, provide a brief explanation.

Requirement: {prompt}

Code:
{code}

Provide a concise explanation (2-3 sentences) of what the component does."""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": cot_prompt,
        "stream": False,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(GENERATION_URL, json=payload, timeout=30)
        if response.status_code == 200:
            explanation = response.json().get("response", "").strip()
            print_success(f"Generated explanation ({len(explanation)} chars)")
            return explanation
        else:
            print_warning("Could not generate explanation")
            return f"React component for: {prompt}"
    except Exception as e:
        print_warning(f"Explanation generation failed: {str(e)}")
        return f"React component for: {prompt}"

def save_generation(prompt, code, explanation, success):
    """Log generation result"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "code_length": len(code) if code else 0,
        "explanation_length": len(explanation) if explanation else 0,
        "validation_success": success
    }
    
    with open(INFERENCE_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')

def generate_and_save_component(prompt):
    """Main function: generate component and save output"""
    print("\n" + "─"*80)
    print(f"Prompt: {prompt}")
    print("─"*80)
    
    # Generate code
    code, success = generate_react_component(prompt)
    
    if code:
        # Generate explanation
        explanation = generate_explanation(prompt, code)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = OUTPUT_DIR / timestamp
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save component
        component_file = output_subdir / "component.jsx"
        with open(component_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Save explanation
        explanation_file = output_subdir / "explanation.md"
        with open(explanation_file, 'w', encoding='utf-8') as f:
            f.write(f"# {prompt}\n\n{explanation}")
        
        # Save metadata
        metadata = {
            "prompt": prompt,
            "validation_success": success,
            "code_length": len(code),
            "timestamp": timestamp
        }
        metadata_file = output_subdir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Log
        save_generation(prompt, code, explanation, success)
        
        print_success(f"Component saved to {output_subdir}")
        return True
    else:
        save_generation(prompt, "", "", False)
        print_warning("Generation failed, no output saved")
        return False

def interactive_mode():
    """Interactive CLI for generating components"""
    print_header("🚀 REACT COMPONENT GENERATOR (Fine-Tuned Model)")
    
    while True:
        print("\n" + "─"*80)
        print("Describe a React component you want to generate:")
        print("Examples:")
        print("  • Login form with email/password validation")
        print("  • Todo list with add/delete/complete functionality")
        print("  • Counter with increment/decrement buttons")
        print("  • User profile card with avatar")
        print("  • Search bar with suggestions")
        print("\nEnter 'quit' to exit")
        print("\n➤ Component description: ", end="")
        
        prompt = input().strip()
        
        if prompt.lower() == 'quit':
            print_header("✅ GOODBYE!")
            break
        
        if not prompt:
            print_warning("No description provided")
            continue
        
        # Generate
        generate_and_save_component(prompt)
        
        # Rate limiting
        time.sleep(2)

def main():
    """Main entry point"""
    # Check Ollama connection
    if not check_ollama_connection():
        print("\n" + "="*80)
        print("ERROR: Cannot connect to Ollama")
        print("="*80)
        print("\nPlease:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Run: ollama serve")
        print(f"3. In another terminal, run: ollama create {MODEL_NAME} -f models/Modelfile_qwen_reactjs")
        print("4. Try again")
        return
    
    # Start interactive mode
    interactive_mode()
    
    print_header("✅ SESSION COMPLETE")
    print(f"\nGenerated components saved to: {OUTPUT_DIR}")
    print(f"Generation log: {INFERENCE_LOG}")

if __name__ == "__main__":
    main()
