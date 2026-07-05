#!/usr/bin/env python3
"""
PHASE 1: DATASET CREATOR
Generates diverse full-stack training datasets from user prompts via local Ollama.
Output: datasets/generated_projects.jsonl (training format), datasets/generated_projects.txt (human-readable)
"""

import requests
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_JSONL = DATASETS_DIR / "generated_projects.jsonl"
OUTPUT_TXT = DATASETS_DIR / "generated_projects.txt"
MODEL_NAME = "qwen2.5-coder:3b"

# Ensure datasets directory exists
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text)
    print("="*80)

def print_step(text):
    """Print step info"""
    print(f"\n➤ {text}")

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def check_ollama_connection():
    """Verify Ollama server is running and model is loaded"""
    print_step("Checking Ollama connection...")
    try:
        response = requests.get(f"{OLLAMA_URL.replace('/api/generate', '')}/api/tags", timeout=5)
        if response.status_code == 200:
            tags = response.json().get("models", [])
            model_names = [m.get("name", "") for m in tags]
            if any(MODEL_NAME in m for m in model_names):
                print_success(f"Ollama is running with {MODEL_NAME} loaded")
                return True
            else:
                print_error(f"Model {MODEL_NAME} not found in Ollama")
                print(f"Available models: {', '.join(model_names)}")
                return False
        else:
            print_error("Ollama server returned error")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to Ollama at {OLLAMA_URL}")
        print("Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return False

def get_project_prompt_from_user():
    """Get project description from user input"""
    print_header("🚀 MICRO AI CODER - DATASET GENERATOR (Phase 1)")
    print("\nEnter your project description (what should the app do?):")
    print("Examples:")
    print("  - 'Social media feed with posts and comments'")
    print("  - 'Todo list app with user authentication'")
    print("  - 'Product marketplace with shopping cart'")
    print("  - 'Task management system with teams'")
    print("\n➤ Project description: ", end="")
    
    project_description = input().strip()
    
    if not project_description:
        print_warning("No description provided. Using default example...")
        project_description = "Members Club with user registration, login, and member profiles"
    
    return project_description

def get_technologies():
    """Get technology stack from user"""
    print("\nSelect technologies to use (comma-separated):")
    print("Available: react, node, express, php, mysql, postgresql, wordpress")
    print("Default: react, node, express, mysql")
    print("\n➤ Technologies: ", end="")
    
    techs_input = input().strip().lower()
    
    if not techs_input:
        techs = ["react", "node", "express", "mysql"]
    else:
        techs = [t.strip() for t in techs_input.split(",")]
    
    return techs

def generate_multi_file_project(project_description, technologies):
    """Generate a complete multi-file project based on description and tech stack"""
    
    # Determine file types based on technologies
    file_specs = []
    
    if "react" in technologies:
        file_specs.extend([
            {"name": "HomePage.jsx", "type": "react", "description": "Home page component with overview and navigation"},
            {"name": "ListPage.jsx", "type": "react", "description": "List/grid view of items"},
            {"name": "DetailPage.jsx", "type": "react", "description": "Detail view for individual item with related info"},
            {"name": "FormPage.jsx", "type": "react", "description": "Form to create or edit item"},
            {"name": "App.jsx", "type": "react", "description": "Main app component with routing"}
        ])
    
    if "express" in technologies or "node" in technologies:
        file_specs.extend([
            {"name": "server.js", "type": "node", "description": "Express server setup with routes"},
            {"name": "routes.js", "type": "node", "description": "API route handlers and endpoints"},
            {"name": "middleware.js", "type": "node", "description": "Authentication and validation middleware"},
            {"name": "database.js", "type": "node", "description": "Database connection and setup"}
        ])
    
    if "php" in technologies and "wordpress" not in technologies:
        file_specs.extend([
            {"name": "index.php", "type": "php", "description": "Main entry point with routing"},
            {"name": "api.php", "type": "php", "description": "API endpoints and controllers"},
            {"name": "config.php", "type": "php", "description": "Configuration and database setup"}
        ])
    
    if "wordpress" in technologies:
        file_specs.extend([
            {"name": "plugin.php", "type": "php", "description": "WordPress plugin main file"},
            {"name": "wp-api-endpoint.php", "type": "php", "description": "WordPress REST API endpoints"},
            {"name": "admin-page.php", "type": "php", "description": "WordPress admin dashboard page"}
        ])
    
    if "mysql" in technologies or "postgresql" in technologies:
        db_type = "MySQL" if "mysql" in technologies else "PostgreSQL"
        file_specs.append({
            "name": "schema.sql",
            "type": "sql",
            "description": f"{db_type} database schema with tables and relationships"
        })
    
    print(f"\n{'─'*80}")
    print(f"📋 Project Structure: {len(file_specs)} files to generate")
    print(f"{'─'*80}")
    
    generated_project = []
    
    for idx, file_spec in enumerate(file_specs, 1):
        file_name = file_spec["name"]
        file_type = file_spec["type"]
        description = file_spec["description"]
        
        print(f"\n[{idx}/{len(file_specs)}] Generating {file_name}...")
        print(f"    📝 {description}")
        
        # Build specific prompt for this file
        if file_type == "react":
            file_prompt = f"""Create a complete React component file named {file_name}.
Project context: {project_description}
Purpose: {description}
Requirements:
- Use React hooks (useState, useEffect, useContext)
- Use functional components
- Include proper error handling
- Add comments explaining the code
- Make it production-ready
- Code should be 50-100 lines
- Do NOT include markdown backticks or explanation text
- Output ONLY the code"""
        
        elif file_type == "node":
            file_prompt = f"""Create a complete Node.js/Express file named {file_name}.
Project context: {project_description}
Purpose: {description}
Requirements:
- Use Express.js best practices
- Include error handling and logging
- Use async/await for database operations
- Add proper middleware setup
- Include comments explaining the code
- Code should be 50-100 lines
- Do NOT include markdown backticks or explanation text
- Output ONLY the code"""
        
        elif file_type == "php":
            file_prompt = f"""Create a complete PHP file named {file_name}.
Project context: {project_description}
Purpose: {description}
Requirements:
- Use modern PHP (7.4+)
- Include security best practices (sanitization, validation)
- Use prepared statements for database queries
- Add error handling
- Include comments explaining the code
- Code should be 50-100 lines
- Do NOT include markdown backticks or explanation text
- Output ONLY the code"""
        
        elif file_type == "sql":
            file_prompt = f"""Create complete SQL database schema file named {file_name}.
Project context: {project_description}
Purpose: {description}
Requirements:
- Create all necessary tables for the project
- Include primary keys, foreign keys, and indexes
- Add appropriate data types and constraints
- Include sample INSERT statements for test data
- Add comments explaining table structure
- Schema should be 50-100 lines
- Do NOT include markdown backticks or explanation text
- Output ONLY the SQL code"""
        
        payload = {
            "model": MODEL_NAME,
            "prompt": file_prompt,
            "stream": False,
            "temperature": 0.6
        }
        
        try:
            print(f"    ⏳ Querying Ollama...")
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            
            if response.status_code == 200:
                response_data = response.json()
                generated_code = response_data.get("response", "").strip()
                
                if generated_code:
                    # Clean up common markdown formatting
                    if generated_code.startswith("```"):
                        lines = generated_code.split("\n")
                        # Skip first line (```python, ```javascript, etc.)
                        generated_code = "\n".join(lines[1:])
                    if generated_code.endswith("```"):
                        lines = generated_code.split("\n")
                        # Remove last line (```)
                        generated_code = "\n".join(lines[:-1])
                    
                    generated_project.append({
                        "filename": file_name,
                        "code": generated_code
                    })
                    print(f"    ✅ Generated {file_name} ({len(generated_code)} chars)")
                else:
                    print_warning(f"Empty response for {file_name}")
            else:
                print_error(f"Error {response.status_code}")
                
        except requests.exceptions.Timeout:
            print_warning(f"Timeout generating {file_name}")
        except Exception as e:
            print_error(f"Exception: {str(e)}")
        
        # Rate limiting
        if idx < len(file_specs):
            time.sleep(3)
    
    return generated_project

def save_to_jsonl(project_name, project_description, technologies, files):
    """Save generated project to JSONL format (for training)"""
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        for file_info in files:
            entry = {
                "project": project_name,
                "type": "code",  # Can be 'code', 'schema', 'config', etc.
                "file": file_info['filename'],
                "description": f"Part of {project_description}",
                "code": file_info['code']
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def save_to_txt(project_name, project_description, technologies, files):
    """Save generated project to human-readable format"""
    with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
        # Write project header
        f.write(f"\n{'='*80}\n")
        f.write(f"# PROJECT: {project_name}\n")
        f.write(f"# Description: {project_description}\n")
        f.write(f"# Technologies: {', '.join(technologies).upper()}\n")
        f.write(f"# Files: {len(files)}\n")
        f.write(f"{'='*80}\n\n")
        
        # Write each file
        for file_info in files:
            f.write(f"\n--- FILE: {file_info['filename']} ---\n\n")
            f.write(file_info['code'])
            f.write(f"\n\n")

def main():
    """Main execution flow"""
    
    # Check Ollama connection
    if not check_ollama_connection():
        print_error("Cannot proceed without Ollama connection")
        sys.exit(1)
    
    # Initialize output files
    OUTPUT_JSONL.write_text("", encoding="utf-8")
    OUTPUT_TXT.write_text(
        "# MICRO AI CODER - FULL-STACK PROJECT TRAINING DATASET\n"
        "# Generated with tiktoken (cl100k_base) tokenization\n"
        "# Ready for Phase 2 training\n\n",
        encoding="utf-8"
    )
    
    project_count = 0
    total_files = 0
    total_chars = 0
    
    while True:
        # Get input from user
        project_description = get_project_prompt_from_user()
        technologies = get_technologies()
        
        project_name = f"Project_{project_count + 1}"
        
        print(f"\n✅ Generating project:")
        print(f"   Name: {project_name}")
        print(f"   Description: {project_description}")
        print(f"   Technologies: {', '.join(technologies)}")
        
        # Generate multi-file project
        files = generate_multi_file_project(project_description, technologies)
        
        if files:
            # Calculate stats
            file_count = len(files)
            char_count = sum(len(f['code']) for f in files)
            
            print(f"\n✅ Generated {file_count} files")
            
            # Save to both formats
            save_to_jsonl(project_name, project_description, technologies, files)
            save_to_txt(project_name, project_description, technologies, files)
            print_success(f"Saved to datasets/")
            
            total_files += file_count
            total_chars += char_count
        else:
            print_error(f"No files generated for {project_name}")
        
        project_count += 1
        
        # Ask if user wants to generate another project
        print(f"\n{'='*80}")
        print(f"📊 Progress: {project_count} project(s) generated, {total_files} file(s), {total_chars:,} character(s)")
        print(f"{'='*80}")
        print("\nGenerate another project? (yes/no): ", end="")
        
        continue_choice = input().strip().lower()
        if continue_choice not in ["yes", "y"]:
            break
    
    # Print final statistics
    try:
        with open(OUTPUT_TXT, 'r', encoding='utf-8') as f:
            txt_content = f.read()
            txt_chars = len(txt_content)
            txt_lines = len(txt_content.split('\n'))
        
        with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            jsonl_lines = len([line for line in f if line.strip()])
        
        print_header("✅ DATASET GENERATION COMPLETE!")
        print(f"📊 Dataset Statistics:")
        print(f"   Projects: {project_count}")
        print(f"   Total Files: {total_files}")
        print(f"   Total Characters: {total_chars:,}")
        print(f"   Text File Size: {txt_chars / 1024:.2f} KB")
        print(f"   JSONL Entries: {jsonl_lines}")
        print(f"\n📁 Outputs:")
        print(f"   - datasets/generated_projects.txt (human-readable)")
        print(f"   - datasets/generated_projects.jsonl (training format)")
        print(f"\n🎓 Ready for Phase 2 Training!")
        print(f"   Run: python phase2_training/v2_train_model.py")
        print("="*80 + "\n")
        
    except Exception as e:
        print_error(f"Error reading dataset: {str(e)}")

if __name__ == "__main__":
    main()
