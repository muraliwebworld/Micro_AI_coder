#!/usr/bin/env python3
"""
PHASE 4: MICRO AI CODER AGENT
Main orchestrator that accepts natural language prompts and generates React components.
Simplified for single component generation instead of multi-file projects.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import re

# Import Phase 3 CodeGenerator
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_inference.v2_inference import CodeGenerator, load_model_and_config, DEVICE, generate_single_component, save_component

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
    Main agent that orchestrates React component generation.
    Accepts natural language prompts and produces individual components.
    """
    
    def __init__(self):
        """Initialize agent with trained model"""
        print_step("Initializing Micro AI Coder Agent...")
        
        config, enc = load_model_and_config()
        self.generator = CodeGenerator(config, enc, device=DEVICE)
        
        self.current_output_dir = None
        print_success("Agent initialized and ready!")
    
    def generate_component(self, user_prompt):
        """
        Generate a single React component from a prompt.
        Returns a dictionary with component metadata and code.
        """
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_output_dir = OUTPUTS_DIR / timestamp
        
        # Generate component
        print_step(f"Analyzing prompt...")
        
        print_success(f"Generating React component...")
        
        # Generate the component
        component_dict = generate_single_component(self.generator, user_prompt)
        
        # Save files
        save_component(component_dict, self.current_output_dir)
        
        return {
            'output_dir': self.current_output_dir,
            'prompt': user_prompt,
            'code_length': len(component_dict['code']),
            'explanation_length': len(component_dict['Complex_CoT']),
            'component': component_dict
        }

# ============================================================================
# CLI INTERFACE
# ============================================================================

def interactive_agent():
    """Run agent in interactive mode"""
    print_header("🤖 MICRO AI CODER AGENT - React Component Generator")
    print("Generate React components from natural language descriptions")
    print("Powered by: Trained PyTorch Model + tiktoken")
    
    # Initialize agent
    agent = MicroAICoderAgent()
    
    while True:
        print("\n" + "="*80)
        print("MENU:")
        print("  1️⃣  Generate a React component")
        print("  2️⃣  Generate multiple components")
        print("  3️⃣  Get help with prompt format")
        print("  4️⃣  View output directory")
        print("  5️⃣  Exit")
        print("="*80)
        print("\n➤ Enter choice (1-5): ", end="")
        
        choice = input().strip()
        
        if choice == '1':
            print("\n" + "─"*80)
            print("💬 Describe a React component (what should it do?):")
            print("─"*80)
            print("\nExamples:")
            print("  • Generate a React component for a login form")
            print("  • Create a React counter component with increment and decrement buttons")
            print("  • Write a React todo list component with add/remove functionality")
            print("  • Create a React form with email and password validation")
            print("  • Build a React product card component with image and rating")
            print("\n➤ Component description: ", end="")
            
            user_prompt = input().strip()
            
            if not user_prompt:
                print_error("Empty prompt, skipping...")
                continue
            
            print(f"\n🔄 Starting component generation...\n")
            
            try:
                # Generate component
                result = agent.generate_component(user_prompt)
                
                # Summary
                print_header("✅ COMPONENT GENERATED SUCCESSFULLY")
                print(f"📁 Location: {result['output_dir']}")
                print(f"📊 Statistics:")
                print(f"   - Component Code: {result['code_length']:,} characters")
                print(f"   - Explanation: {result['explanation_length']:,} characters")
                
                print(f"\n📝 Generated Files:")
                print(f"   ├── component.jsx (React component)")
                print(f"   ├── explanation.md (How it works)")
                print(f"   └── metadata.json (Metadata)")
                
                print(f"\n🎯 Component Preview (first 400 chars):")
                print("─" * 80)
                preview = result['component']['code'][:400]
                print(preview + ("..." if len(result['component']['code']) > 400 else ""))
                print("─" * 80)
                
                print(f"\n💡 Explanation Preview:")
                print("─" * 80)
                print(result['component']['Complex_CoT'][:300] + ("..." if len(result['component']['Complex_CoT']) > 300 else ""))
                print("─" * 80)
                
            except Exception as e:
                print_error(f"Generation failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        elif choice == '2':
            print("\n📋 Generate Multiple Components")
            print("➤ How many components? (1-10): ", end="")
            try:
                count = int(input().strip())
                count = max(1, min(10, count))
            except ValueError:
                count = 3
            
            prompts = []
            print(f"\nEnter {count} component prompts:")
            for i in range(count):
                print(f"\n  Component {i+1}:")
                print(f"  ➤ Description: ", end="")
                prompts.append(input().strip())
            
            print(f"\n🔄 Generating {count} components...\n")
            
            successful = 0
            failed = 0
            
            for idx, prompt in enumerate(prompts, 1):
                if not prompt:
                    print_log(f"[{idx}/{count}] ⏭️  Skipped (empty prompt)")
                    continue
                
                try:
                    result = agent.generate_component(prompt)
                    print_success(f"[{idx}/{count}] ✅ {prompt[:60]}")
                    print(f"            → {result['output_dir'].name}/")
                    successful += 1
                    
                except Exception as e:
                    print_error(f"[{idx}/{count}] ❌ {prompt[:60]}")
                    print(f"            Error: {str(e)}")
                    failed += 1
            
            print_header(f"📊 BATCH GENERATION COMPLETE")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print(f"⏭️  Skipped: {count - successful - failed}")
        
        elif choice == '3':
            print_header("💡 PROMPT FORMAT GUIDE")
            print("\n✨ Best Practices for Component Descriptions:\n")
            print("1. Be specific about functionality:")
            print("   ❌ 'Create a component'")
            print("   ✅ 'Create a React button component with click handler'")
            print()
            print("2. Include UI elements you want:")
            print("   ✅ 'Login form with email, password, and submit button'")
            print("   ✅ 'Card component with image, title, description, and price'")
            print()
            print("3. Mention state management needs:")
            print("   ✅ 'Counter component with increment/decrement buttons and display'")
            print("   ✅ 'Todo list with add item, remove item, and mark complete'")
            print()
            print("4. Specify styling preferences (optional):")
            print("   ✅ 'Button with Tailwind CSS styling'")
            print("   ✅ 'Card component using Bootstrap classes'")
            print()
            print("5. Include validation if needed:")
            print("   ✅ 'Form with email validation and required field checking'")
            print()
            print("Examples:")
            print("  • Generate a navbar component with home, about, and contact links")
            print("  • Create a star rating component with click functionality")
            print("  • Build a search bar with input field and search button")
            print("  • Write a modal dialog component with title, body, and action buttons")
        
        elif choice == '4':
            print_header("📁 OUTPUT DIRECTORY")
            print(f"\nGenerated components are saved in:")
            print(f"  {OUTPUTS_DIR}")
            print(f"\nEach component is in a timestamped directory:")
            print(f"  {OUTPUTS_DIR}/20260707_120000/")
            print(f"    ├── component.jsx (React component code)")
            print(f"    ├── explanation.md (How it works)")
            print(f"    └── metadata.json (Generation info)")
            
            # List recent outputs
            if OUTPUTS_DIR.exists():
                outputs = sorted(OUTPUTS_DIR.glob('*'))[-5:]
                if outputs:
                    print(f"\nRecent outputs:")
                    for output_dir in outputs:
                        if output_dir.is_dir():
                            print(f"  • {output_dir.name}/")
        
        elif choice == '5':
            print_success("Thank you for using Micro AI Coder Agent! Goodbye!")
            break
        
        else:
            print_error("Invalid choice. Please enter 1-5.")

def main():
    """Main entry point"""
    try:
        interactive_agent()
    except KeyboardInterrupt:
        print_success("\nExiting...")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
