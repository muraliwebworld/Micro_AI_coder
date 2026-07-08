#!/usr/bin/env python3
"""
PHASE 1: DATASET CREATOR
Generates diverse full-stack training datasets from user prompts via local Ollama.
Target Skills: ReactJS, NodeJS, MySQL, PostgreSQL, WordPress, PHP
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

# FIXED: Match the filename expected by Phase 2
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

def get_project_prompt_from_user():
    print_header("🚀 MICRO AI CODER - DATASET GENERATOR (Phase 1)")
    print("\nEnter your project description (what should the app do?):")
    print("➤ Project description:", end="")
    project_description = input().strip()
    if not project_description:
        print_warning("No description provided. Using default example...")
        project_description = "Members Club with user registration, login, and member profiles"
    return project_description

def get_technologies():
    print("\nSelect technologies (comma-separated):")
    print("Available: react, node, express, php, mysql, postgresql, wordpress")
    print("➤ Technologies:", end="")
    techs_input = input().strip().lower()
    if not techs_input:
        return ["react", "node", "express", "mysql"]
    return [t.strip() for t in techs_input.split(",")]

def clean_code_output(code):
    """Robustly remove Markdown code fences using Regex"""
    # Remove opening fences like ```php, ```javascript, ```sql, etc.
    code = re.sub(r'^```[a-zA-Z]*\s*\n?', '', code, flags=re.MULTILINE)
    # Remove closing fences
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    return code.strip()

def validate_code_structure(code, file_type):
    """
    FIXED: Replaced flawed 'word_ratio' check. 
    Code naturally has repetitive keywords (const, function, div, SELECT).
    We now check for structural code elements instead.
    """
    if len(code) < 40:
        return False, "Code too short"
    
    has_newlines = code.count('\n') >= 3
    has_code_symbols = any(sym in code for sym in ['{', '}', ';', '=', '<', '>', 'function', 'class', 'SELECT', 'CREATE', 'import', 'require'])
    
    if not (has_newlines and has_code_symbols):
        return False, "Lacks typical code structure or newlines"
        
    return True, ""

def generate_multi_file_project(project_description, technologies):
    # ... [Keep your existing file_specs logic exactly as you had it] ...
    # (For brevity, assume the file_specs dictionary generation from your original code is here)
    
    # Determine file types based on technologies (Simplified for example, use your full logic)
    file_specs = []
    if "react" in technologies:
        file_specs.append({"name":"App.jsx","type":"react","description":"Main app component"})
    if "node" in technologies or "express" in technologies:
        file_specs.append({"name":"server.js","type":"node","description":"Express server setup"})
    if "php" in technologies:
        file_specs.append({"name":"index.php","type":"php","description":"Main PHP entry point"})
    if "mysql" in technologies or "postgresql" in technologies:
        file_specs.append({"name":"schema.sql","type":"sql","description":"Database schema"})

    print(f"\n📋 Project Structure: {len(file_specs)} files to generate")
    generated_project = []

    for idx, file_spec in enumerate(file_specs, 1):
        file_name = file_spec["name"]
        file_type = file_spec["type"]
        description = file_spec["description"]
        
        print(f"\n[{idx}/{len(file_specs)}] Generating {file_name}...")
        
        # Build prompt (Use your existing detailed prompts here)
        file_prompt = f"Create a {file_type} file named {file_name}. Context: {project_description}. Purpose: {description}. Output ONLY raw code, NO markdown backticks."
        
        payload = {"model": MODEL_NAME, "prompt": file_prompt, "stream": False, "temperature": 0.6}
        
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            if response.status_code == 200:
                generated_code = response.json().get("response", "").strip()
                
                # 1. Clean Markdown
                generated_code = clean_code_output(generated_code)
                
                # 2. Validate Structure (FIXED)
                is_valid, reason = validate_code_structure(generated_code, file_type)
                
                if is_valid:
                    generated_project.append({"filename": file_name, "code": generated_code})
                    print_success(f"Generated {file_name} ({len(generated_code)} chars)")
                else:
                    print_warning(f"Invalid structure for {file_name}: {reason}")
            else:
                print_error(f"Error {response.status_code}")
        except Exception as e:
            print_error(f"Exception: {str(e)}")
            
        if idx < len(file_specs): time.sleep(2) # Rate limiting

    return generated_project

def save_to_jsonl(project_name, project_description, technologies, files):
    # FIXED: Appending to the correct final filename
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        for file_info in files:
            entry = {
                "project": project_name,
                "type": "code",
                "file": file_info['filename'],
                "description": f"Part of {project_description}",
                "code": file_info['code']
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    if not check_ollama_connection(): sys.exit(1)
    
    # Clear files for fresh run
    OUTPUT_JSONL.write_text("", encoding="utf-8")
    OUTPUT_TXT.write_text("# MICRO AI CODER DATASET\n", encoding="utf-8")
    
    project_count = 0
    while True:
        project_description = get_project_prompt_from_user()
        technologies = get_technologies()
        project_name = f"Project_{project_count + 1}"
        
        files = generate_multi_file_project(project_description, technologies)
        if files:
            save_to_jsonl(project_name, project_description, technologies, files)
            print_success(f"Saved {len(files)} files to dataset.")
            project_count += 1
            
        print("\nGenerate another project? (yes/no):", end="")
        if input().strip().lower() not in ["yes", "y"]:
            break
            
    print_header("✅ DATASET GENERATION COMPLETE!")
    print(f"Ready for Phase 2: python phase2_training/v2_train_model.py")

if __name__ == "__main__":
    main()