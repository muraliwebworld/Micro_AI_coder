#!/usr/bin/env python3
"""
PHASE 4: MICRO AI CODER AGENT
Main orchestrator that accepts natural language prompts and generates complete
multi-file projects with organized output structure.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import re

# Import Phase 3 CodeGenerator
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_inference.v2_inference import CodeGenerator, load_model_and_config, DEVICE

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)

def print_step(text):
    print(f"\n➤ {text}")

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_log(text):
    print(f"📊 {text}")

# ============================================================================
# MICRO AI CODER AGENT
# ============================================================================

class MicroAICoderAgent:
    """
    Main agent that orchestrates multi-file code generation.
    Accepts natural language prompts and produces organized project structures.
    """
    
    def __init__(self):
        """Initialize agent with trained model"""
        print_step("Initializing Micro AI Coder Agent...")
        
        config, enc = load_model_and_config()
        self.generator = CodeGenerator(config, enc, device=DEVICE)
        
        self.current_project_dir = None
        print_success("Agent initialized and ready!")
    
    def create_project_directory(self):
        """Create timestamped project directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = OUTPUTS_DIR / timestamp
        
        # Create subdirectories
        (project_dir / "frontend").mkdir(parents=True, exist_ok=True)
        (project_dir / "backend").mkdir(parents=True, exist_ok=True)
        (project_dir / "config").mkdir(parents=True, exist_ok=True)
        
        self.current_project_dir = project_dir
        return project_dir
    
    def classify_backend_and_pages(self, user_prompt):
        """
        Classify backend type and extract page information
        Returns: dict with backend, database, pages, description
        """
        spec = self.generator.parse_user_prompt(user_prompt)
        
        return {
            'backend': spec['backend'],
            'database': spec['database'],
            'pages': spec['page_names'],
            'file_types': spec['file_types'],
            'prompt': user_prompt
        }
    
    def generate_project(self, user_prompt):
        """
        Generate a complete project structure with streaming output.
        Yields (file_path, code, stream) tuples.
        """
        # Create project directory
        project_dir = self.create_project_directory()
        
        # Classify and plan
        print_step(f"Analyzing prompt...")
        spec = self.classify_backend_and_pages(user_prompt)
        
        print_success(f"Backend: {spec['backend'].upper()}")
        print_success(f"Database: {spec['database'].upper()}")
        print_success(f"Pages: {', '.join(spec['pages'])}")
        
        # Generate files
        file_count = 0
        total_chars = 0
        
        print_header("GENERATING PROJECT")
        
        for file_name, code in self.generator.generate_multi_file_project(user_prompt):
            # Determine output directory
            if file_name.endswith('.jsx'):
                output_subdir = project_dir / "frontend"
            elif file_name.endswith('.js'):
                output_subdir = project_dir / "backend"
            elif file_name.endswith('.sql'):
                output_subdir = project_dir / "backend"
            elif file_name.endswith('.php'):
                output_subdir = project_dir / "backend"
            else:
                output_subdir = project_dir / "config"
            
            # Write file
            file_path = output_subdir / file_name
            file_path.write_text(code, encoding='utf-8')
            
            file_count += 1
            total_chars += len(code)
            
            # Display progress
            rel_path = file_path.relative_to(project_dir)
            print(f"✅ {rel_path} ({len(code)} chars)")
            
            yield (str(rel_path), code)
        
        # Create metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'prompt': user_prompt,
            'backend': spec['backend'],
            'database': spec['database'],
            'pages': spec['pages'],
            'total_files': file_count,
            'total_chars': total_chars,
            'project_dir': str(project_dir)
        }
        
        metadata_file = project_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        # Create manifest entry
        manifest_file = OUTPUTS_DIR / "manifest.json"
        manifest = {}
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        
        manifest[str(datetime.now())] = {
            'project_dir': str(project_dir.relative_to(OUTPUTS_DIR)),
            'prompt': user_prompt,
            'backend': spec['backend'],
            'files': file_count
        }
        
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        
        return {
            'project_dir': project_dir,
            'file_count': file_count,
            'total_chars': total_chars,
            'metadata': metadata
        }

# ============================================================================
# CLI INTERFACE
# ============================================================================

def interactive_agent():
    """Run agent in interactive mode"""
    print_header("🤖 MICRO AI CODER AGENT")
    print("Generate complete React + Full-Stack projects with AI")
    print("Powered by: Ollama (qwen2.5-coder:3b) + PyTorch + tiktoken")
    
    # Initialize agent
    agent = MicroAICoderAgent()
    
    while True:
        print("\n" + "="*80)
        print("MENU:")
        print("  1. Generate new project")
        print("  2. Get help with prompt format")
        print("  3. View output directory")
        print("  4. Exit")
        print("="*80)
        print("\n➤ Choice: ", end="")
        
        choice = input().strip()
        
        if choice == '1':
            print("\n" + "─"*80)
            print("Enter your project description:")
            print("Examples:")
            print("  - 'Create React app with homepage, about, contact pages and Express backend'")
            print("  - 'Build social media feed with posts, comments, and MySQL database'")
            print("  - 'Create e-commerce site with product page, cart, checkout, and WordPress backend'")
            print("─"*80)
            print("\n➤ Your prompt: ", end="")
            
            user_prompt = input().strip()
            
            if not user_prompt:
                print_error("Empty prompt, skipping...")
                continue
            
            print(f"\n🔄 Starting code generation...\n")
            
            try:
                # Generate project
                result = agent.generate_project(user_prompt)
                
                # Summary
                print_header("✅ PROJECT GENERATED SUCCESSFULLY")
                print(f"📁 Location: {result['project_dir']}")
                print(f"📊 Statistics:")
                print(f"   - Files: {result['file_count']}")
                print(f"   - Total Characters: {result['total_chars']:,}")
                print(f"   - Backend: {result['metadata']['backend'].upper()}")
                print(f"   - Database: {result['metadata']['database'].upper()}")
                print(f"\n📂 Structure:")
                print(f"   {result['project_dir'].name}/")
                print(f"   ├── frontend/     (React components)")
                print(f"   ├── backend/      (Express, SQL schemas)")
                print(f"   ├── config/       (Environment files)")
                print(f"   └── metadata.json (Project metadata)")
                print("="*80)
                
            except Exception as e:
                print_error(f"Error during generation: {str(e)}")
        
        elif choice == '2':
            print_header("PROMPT FORMAT GUIDE")
            print("""
Your prompt should specify:

1. FRAMEWORK:
   - 'React', 'React app', 'React frontend'

2. PAGES/COMPONENTS:
   - 'with homepage, about, contact pages'
   - 'pages: home, login, register, dashboard'
   - 'HomePage, LoginPage, etc.'

3. BACKEND:
   - 'Express backend', 'Node.js backend'
   - 'WordPress backend'
   - 'PHP backend'

4. DATABASE:
   - 'MySQL', 'PostgreSQL'
   - If not specified, defaults to MySQL

EXAMPLE PROMPTS:
  "Create a React app with homepage, about, contact, register, 
   and login pages using Express.js backend and MySQL database"
  
  "Build a social media feed app with React frontend, Node.js 
   Express API, and PostgreSQL database"
  
  "Make a WordPress site with user profiles and admin dashboard"

Note: Model will generate appropriate file types automatically:
  - React: .jsx components
  - Express/Node: .js files (server, routes, middleware)
  - Database: .sql schemas
  - WordPress: .php plugins
            """)
        
        elif choice == '3':
            print(f"\n📁 Output directory: {OUTPUTS_DIR}")
            print("\nRecent projects:")
            if OUTPUTS_DIR.exists():
                projects = sorted(OUTPUTS_DIR.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
                if projects:
                    for i, proj in enumerate(projects, 1):
                        try:
                            meta_file = proj / "metadata.json"
                            if meta_file.exists():
                                meta = json.loads(meta_file.read_text(encoding='utf-8'))
                                print(f"\n{i}. {proj.name}")
                                print(f"   Files: {meta['total_files']}")
                                print(f"   Backend: {meta['backend']}")
                                print(f"   Prompt: {meta['prompt'][:60]}...")
                        except:
                            pass
                else:
                    print("No projects generated yet")
            else:
                print("Output directory not found")
        
        elif choice == '4':
            print("\n✅ Thank you for using Micro AI Coder! 🚀")
            break
        
        else:
            print_error("Invalid choice, try again")

def main():
    """Main entry point"""
    try:
        interactive_agent()
    except KeyboardInterrupt:
        print("\n\n✅ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
