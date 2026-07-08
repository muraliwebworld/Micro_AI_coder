#!/usr/bin/env python3
"""
PHASE 1: REACT COMPONENT DATASET CREATOR
Generates React component training datasets matching reactjs_projects_dataset.jsonl format.
Each dataset entry: {prompt, code, Complex_CoT}
"""
import requests
import json
import time
import sys
import re
from pathlib import Path
from datetime import datetime

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# Output to the same format as reactjs_projects_dataset.jsonl
OUTPUT_JSONL = DATASETS_DIR / "generated_projects_final.jsonl" 
OUTPUT_TXT = DATASETS_DIR / "generated_projects.txt"
MODEL_NAME = "qwen2.5-coder:3b"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

def print_header(text): print("\n" + "="*80 + f"\n{text}\n" + "="*80)
def print_step(text): print(f"\n➤ {text}")
def print_success(text): print(f"✅ {text}")
def print_warning(text): print(f"⚠️  {text}")
def print_error(text): print(f"❌ {text}")

def check_ollama_connection():
    print_step("Checking Ollama connection...")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            tags = response.json().get("models", [])
            model_names = [m.get("name", "") for m in tags]
            if any(MODEL_NAME in m for m in model_names):
                print_success(f"Ollama is running with {MODEL_NAME} loaded")
                return True
            else:
                print_error(f"Model {MODEL_NAME} not found in Ollama")
                return False
    except Exception as e:
        print_error(f"Cannot connect to Ollama: {str(e)}")
        return False

def get_component_prompt_from_user():
    print_header("🚀 MICRO AI CODER - REACT COMPONENT DATASET GENERATOR (Phase 1)")
    print("\nDescribe a React component you want to generate:")
    print("Examples:")
    print("  • Generate a React component for a login form")
    print("  • Create a React counter component with increment/decrement buttons")
    print("  • Write a React todo list component")
    print("  • Create a React form with validation")
    print("\n➤ Component description:", end="")
    prompt = input().strip()
    if not prompt:
        print_warning("No description provided. Using default example...")
        prompt = "Generate a React component for displaying user profile information"
    return prompt

def clean_code_output(code):
    """Remove Markdown code fences and clean JSX code"""
    # Remove opening fences like ```javascript, ```jsx, ```javascript etc.
    code = re.sub(r'^```[a-zA-Z]*\s*\n?', '', code, flags=re.MULTILINE)
    # Remove closing fences
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    return code.strip()

def validate_react_component(code):
    """Validate that code is a proper React component"""
    if len(code) < 100:
        return False, "Code too short for a React component (< 100 chars)"
    
    # Check for essential React patterns
    has_import = 'import' in code and ('React' in code or 'from' in code)
    has_export = 'export' in code
    has_jsx = '<' in code and '>' in code  # Basic JSX check
    has_function_or_class = 'function' in code or 'const' in code or 'class' in code
    
    if not (has_import and has_export and has_jsx and has_function_or_class):
        missing = []
        if not has_import: missing.append("React import")
        if not has_export: missing.append("export statement")
        if not has_jsx: missing.append("JSX (< > tags)")
        if not has_function_or_class: missing.append("function/const/class definition")
        return False, f"Missing: {', '.join(missing)}"
    
    return True, ""

def generate_react_component(prompt):
    """Generate a single React component from a prompt"""
    print_step(f"Generating React component...")
    
    # Enhance prompt for better React generation
    enhanced_prompt = f"""Generate a complete, self-contained React component that implements the following:
{prompt}

Requirements:
- Use React hooks (useState, useEffect, etc.)
- Include proper imports at the top
- Export as default
- Include JSX with proper structure
- Make it functional and practical
- NO explanations, just the code

Output only the React component code, no markdown backticks."""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": enhanced_prompt,
        "stream": False,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code == 200:
            code = response.json().get("response", "").strip()
            code = clean_code_output(code)
            
            is_valid, reason = validate_react_component(code)
            if is_valid:
                print_success(f"Generated component ({len(code)} chars)")
                return code
            else:
                print_warning(f"Generated code validation failed: {reason}")
                return None
        else:
            print_error(f"API Error {response.status_code}")
            return None
    except Exception as e:
        print_error(f"Exception: {str(e)}")
        return None

def generate_chain_of_thought(prompt, code):
    """Generate Chain-of-Thought explanation for the component"""
    print_step(f"Generating explanation...")
    
    cot_prompt = f"""Given this React component prompt and implementation, provide a brief technical explanation of how it works.

Prompt: {prompt}

Code:
{code}

Provide a concise explanation (2-3 sentences) of what the component does and how it works. Start with "The prompt asks for..." or "This component implements...".
No markdown, just plain text explanation."""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": cot_prompt,
        "stream": False,
        "temperature": 0.5
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
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

def save_to_jsonl(prompt, code, cot_explanation):
    """Save component to JSONL in the format of reactjs_projects_dataset.jsonl"""
    entry = {
        "prompt": prompt,
        "code": code,
        "Complex_CoT": cot_explanation
    }
    
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print_success(f"Saved to dataset: {OUTPUT_JSONL}")

def main():
    if not check_ollama_connection():
        sys.exit(1)
    
    # Clear file for fresh run (or ask user)
    print_header("🚀 MICRO AI CODER - REACT COMPONENT DATASET GENERATOR")
    
    if OUTPUT_JSONL.exists():
        print(f"\nDataset file exists: {OUTPUT_JSONL}")
        print("➤ Append to existing? (yes/no):", end="")
        if input().strip().lower() not in ["yes", "y"]:
            OUTPUT_JSONL.write_text("", encoding="utf-8")
    else:
        OUTPUT_JSONL.write_text("", encoding="utf-8")
    
    component_count = 0
    
    while True:
        print("\n" + "─"*80)
        prompt = get_component_prompt_from_user()
        
        # Generate component
        code = generate_react_component(prompt)
        
        if code:
            # Generate explanation
            cot_explanation = generate_chain_of_thought(prompt, code)
            
            # Save to dataset
            save_to_jsonl(prompt, code, cot_explanation)
            component_count += 1
            
            print_success(f"Component {component_count} added to dataset")
        else:
            print_warning("Skipping this component due to validation failure")
        
        print("\n➤ Generate another component? (yes/no):", end="")
        if input().strip().lower() not in ["yes", "y"]:
            break
        
        # Rate limiting
        time.sleep(2)
    
    print_header("✅ DATASET GENERATION COMPLETE!")
    print(f"Total components generated: {component_count}")
    print(f"Dataset saved to: {OUTPUT_JSONL}")
    print(f"\nNext step: python phase2_training/v2_train_model.py")

if __name__ == "__main__":
    main()