#!/usr/bin/env python3
"""
PHASE 3: INFERENCE & STREAMING
Provides streaming inference for multi-file code generation with real-time token output.
Input: Trained model + tiktoken tokenizer
Output: Generated code files + streaming tokens
"""

import torch
import torch.nn.functional as F
import json
import tiktoken
import re
from pathlib import Path
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODEL_PATH = MODELS_DIR / "tiny_code_model.pt"
CONFIG_PATH = MODELS_DIR / "model_config.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_config():
    """Load trained model and configuration"""
    print_step("Loading model and configuration...")
    
    if not CONFIG_PATH.exists() or not MODEL_PATH.exists():
        print_error("Model or config not found!")
        print("Run Phase 2 first: python phase2_training/v2_train_model.py")
        sys.exit(1)
    
    # Load config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print_success(f"Config loaded")
    print(f"  - Embedding: {config['n_embd']}")
    print(f"  - Heads: {config['n_head']}")
    print(f"  - Layers: {config['n_layer']}")
    print(f"  - Vocab: {config['vocab_size']:,}")
    
    # Load tokenizer
    enc = tiktoken.get_encoding(config['tokenizer'])
    print_success(f"Tokenizer loaded ({config['tokenizer']})")
    
    return config, enc

# ============================================================================
# CODE GENERATOR CLASS
# ============================================================================

class CodeGenerator:
    """Generate code with token-by-token streaming"""
    
    def __init__(self, config, enc, device=DEVICE):
        self.config = config
        self.enc = enc
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        from phase2_training.v2_train_model import TinyCodeModel
        
        model = TinyCodeModel(
            vocab_size=self.config['vocab_size'],
            n_embd=self.config['n_embd'],
            n_head=self.config['n_head'],
            n_layer=self.config['n_layer'],
            block_size=self.config['block_size'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        model.eval()
        self.model = model
        print_success("Model loaded")
    
    def parse_user_prompt(self, user_prompt):
        """
        Parse user prompt to extract:
        - React page names (HomePage, ContactPage, etc.)
        - Backend type (express, wordpress, mysql)
        - Database type (mysql, postgresql)
        """
        prompt_lower = user_prompt.lower()
        
        # File type keywords
        file_types = []
        if 'react' in prompt_lower:
            file_types.append('react')
        if 'express' in prompt_lower or 'node' in prompt_lower:
            file_types.append('node')
        if 'mysql' in prompt_lower or 'postgres' in prompt_lower:
            file_types.append('sql')
        if 'wordpress' in prompt_lower or 'php' in prompt_lower:
            file_types.append('php')
        
        # Default to react + express + mysql if not specified
        if not file_types:
            file_types = ['react', 'node', 'sql']
        
        # Extract page names
        page_patterns = [
            r'(\w+)\s+page',
            r'(\w+)\s+screen',
            r'pages[:\s]+([a-z0-9\s,]+)',
        ]
        
        page_names = set()
        for pattern in page_patterns:
            matches = re.findall(pattern, prompt_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                # Clean and capitalize
                name = match.strip().replace(',', '').replace('and', '').strip()
                if len(name) > 2:
                    page_names.add(name.capitalize())
        
        # Extract backend choice
        backend = 'express'
        if 'wordpress' in prompt_lower:
            backend = 'wordpress'
        elif 'php' in prompt_lower:
            backend = 'php'
        
        # Extract database
        database = 'mysql'
        if 'postgresql' in prompt_lower or 'postgres' in prompt_lower:
            database = 'postgresql'
        
        return {
            'file_types': file_types,
            'page_names': list(page_names) if page_names else ['HomePage'],
            'backend': backend,
            'database': database
        }
    
    def generate_token_stream(self, prompt, max_tokens=500, temperature=0.7, top_k=50):
        """
        Generate tokens one-by-one (streaming).
        Yields decoded tokens as they're generated.
        """
        # Encode prompt
        tokens = self.enc.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Get next token logits
                logits = self.model(x[:, -self.config['block_size']:])
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Decode token
                token_str = self.enc.decode([idx_next.item()])
                yield token_str
                
                # Append to sequence
                x = torch.cat((x, idx_next), dim=1)
                
                # Stop if we hit end-of-file marker
                if '<|file_end|>' in token_str:
                    break
    
    def generate_single_file(self, file_name, file_type, purpose, temperature=0.6):
        """Generate code for a single file with specific type"""
        
        if file_type == 'react':
            prompt = f"""Create a complete React component file named {file_name}.
Purpose: {purpose}
Requirements:
- Use React hooks (useState, useEffect)
- Use functional components
- Include error handling
- Add comments
- 50-100 lines
- Output ONLY code"""
        
        elif file_type == 'node':
            prompt = f"""Create Express.js code for {file_name}.
Purpose: {purpose}
Requirements:
- Use Express.js best practices
- Include error handling
- Use async/await
- Add middleware
- 50-100 lines
- Output ONLY code"""
        
        elif file_type == 'sql':
            prompt = f"""Create SQL database schema for {file_name}.
Purpose: {purpose}
Requirements:
- Create tables with proper structure
- Include primary/foreign keys
- Add indexes and constraints
- Include sample data
- 50-100 lines
- Output ONLY SQL"""
        
        elif file_type == 'php':
            prompt = f"""Create PHP code for {file_name}.
Purpose: {purpose}
Requirements:
- Use modern PHP (7.4+)
- Include security practices
- Use prepared statements
- Add error handling
- 50-100 lines
- Output ONLY code"""
        
        else:
            prompt = f"Generate code for {file_name}: {purpose}"
        
        # Generate with streaming
        code = ""
        for token in self.generate_token_stream(prompt, max_tokens=300, temperature=temperature):
            code += token
        
        return code.strip()
    
    def generate_multi_file_project(self, user_prompt):
        """
        Generate multiple files for a complete project.
        Yields (filename, code) tuples.
        """
        # Parse prompt
        spec = self.parse_user_prompt(user_prompt)
        file_types = spec['file_types']
        page_names = spec['page_names']
        backend = spec['backend']
        database = spec['database']
        
        files_to_generate = []
        
        # React components
        if 'react' in file_types:
            for page in page_names:
                # Capitalize properly
                page_clean = page.replace(' ', '').replace('_', '')
                file_name = f"{page_clean}Page.jsx" if 'Page' not in page_clean else f"{page_clean}.jsx"
                files_to_generate.append({
                    'name': file_name,
                    'type': 'react',
                    'purpose': f"{page} component for the application"
                })
            
            # Add App component
            files_to_generate.append({
                'name': 'App.jsx',
                'type': 'react',
                'purpose': 'Main App component with routing'
            })
        
        # Node.js/Express backend
        if 'node' in file_types:
            files_to_generate.extend([
                {'name': 'server.js', 'type': 'node', 'purpose': 'Express server setup'},
                {'name': 'routes.js', 'type': 'node', 'purpose': 'API route handlers'},
                {'name': 'middleware.js', 'type': 'node', 'purpose': 'Authentication middleware'},
            ])
        
        # Database schema
        if 'sql' in file_types:
            files_to_generate.append({
                'name': 'schema.sql',
                'type': 'sql',
                'purpose': f'{database.upper()} database schema'
            })
        
        # PHP/WordPress
        if 'php' in file_types and 'wordpress' in backend:
            files_to_generate.extend([
                {'name': 'plugin.php', 'type': 'php', 'purpose': 'WordPress plugin main file'},
                {'name': 'wp-api-endpoint.php', 'type': 'php', 'purpose': 'REST API endpoints'},
            ])
        
        # Generate each file
        for file_spec in files_to_generate:
            file_name = file_spec['name']
            file_type = file_spec['type']
            purpose = file_spec['purpose']
            
            code = self.generate_single_file(file_name, file_type, purpose)
            yield (file_name, code)

# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def interactive_menu(generator):
    """Interactive testing menu"""
    while True:
        print_header("PHASE 3: INFERENCE TESTING")
        print("\n1. Generate single file")
        print("2. Generate multi-file project")
        print("3. Streaming demo")
        print("4. Prompt parsing test")
        print("5. Exit")
        print("\n➤ Choice: ", end="")
        
        choice = input().strip()
        
        if choice == '1':
            print("\nFile types: react, node, sql, php")
            print("➤ File name: ", end="")
            file_name = input().strip()
            print("➤ File type: ", end="")
            file_type = input().strip()
            print("➤ Purpose: ", end="")
            purpose = input().strip()
            
            print("\n🔄 Generating code...")
            code = generator.generate_single_file(file_name, file_type, purpose)
            print(f"\n{'─'*80}")
            print(code[:500] + ("..." if len(code) > 500 else ""))
            print(f"{'─'*80}")
            print(f"✅ Generated {len(code)} characters")
        
        elif choice == '2':
            print("\n➤ Project prompt (e.g., 'React app with homepage and login'): ", end="")
            prompt = input().strip()
            
            output_dir = OUTPUTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n🔄 Generating project to {output_dir}/\n")
            
            for file_name, code in generator.generate_multi_file_project(prompt):
                print(f"📝 {file_name} ({len(code)} chars)")
                
                # Save file
                file_path = output_dir / file_name
                file_path.write_text(code, encoding='utf-8')
            
            print(f"\n✅ Project generated to {output_dir}/")
        
        elif choice == '3':
            print("\n➤ Prompt for streaming: ", end="")
            prompt = input().strip()
            
            print("\n🔄 Streaming tokens:\n")
            for token in generator.generate_token_stream(prompt, max_tokens=200):
                print(token, end='', flush=True)
            print("\n\n✅ Done!")
        
        elif choice == '4':
            print("\n➤ User prompt for parsing: ", end="")
            prompt = input().strip()
            
            spec = generator.parse_user_prompt(prompt)
            print(f"\nParsed specification:")
            print(f"  - File types: {spec['file_types']}")
            print(f"  - Pages: {spec['page_names']}")
            print(f"  - Backend: {spec['backend']}")
            print(f"  - Database: {spec['database']}")
        
        elif choice == '5':
            print("\n✅ Exiting...")
            break
        
        else:
            print("❌ Invalid choice")

def main():
    """Main entry point"""
    print_header("PHASE 3: INFERENCE & STREAMING")
    
    # Load model
    config, enc = load_model_and_config()
    print(f"✅ Ready to generate code!")
    
    # Create generator
    generator = CodeGenerator(config, enc)
    
    # Run interactive menu
    interactive_menu(generator)

if __name__ == "__main__":
    main()
