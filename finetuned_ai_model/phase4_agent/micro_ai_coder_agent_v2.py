#!/usr/bin/env python3
"""
PHASE 4: AGENT ORCHESTRATOR - UPDATED FOR FINE-TUNED MODEL
Generates React components using the fine-tuned Qwen model via Ollama
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase3_inference.v3_inference_finetuned import (
    check_ollama_connection,
    generate_and_save_component,
    print_header,
    print_step,
    print_success,
    print_error,
    print_warning
)

def main_menu():
    """Display main menu"""
    print_header("🚀 MICRO AI CODER - FINE-TUNED MODEL AGENT")
    print("\nOptions:")
    print("  1. Generate a single React component")
    print("  2. Batch generate multiple components")
    print("  3. View generation logs")
    print("  4. Exit")
    print("\n➤ Select option (1-4): ", end="")
    
    choice = input().strip()
    return choice

def single_component_mode():
    """Generate a single component"""
    print_step("Single Component Generation")
    print("\nDescribe the React component you want:")
    print("➤ Prompt: ", end="")
    
    prompt = input().strip()
    if not prompt:
        print_warning("No prompt provided")
        return
    
    success = generate_and_save_component(prompt)
    if success:
        print_success("Component generated and saved!")
    else:
        print_error("Generation failed")

def batch_component_mode():
    """Generate multiple components"""
    print_step("Batch Component Generation")
    print("\nEnter component descriptions (one per line, empty line to finish):")
    
    prompts = []
    while True:
        prompt = input().strip()
        if not prompt:
            break
        prompts.append(prompt)
    
    if not prompts:
        print_warning("No prompts provided")
        return
    
    print_step(f"Generating {len(prompts)} components...")
    
    success_count = 0
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt}")
        success = generate_and_save_component(prompt)
        if success:
            success_count += 1
    
    print_header("✅ BATCH GENERATION COMPLETE")
    print(f"Success rate: {success_count}/{len(prompts)} ({100*success_count//len(prompts)}%)")

def view_logs():
    """View generation logs"""
    from phase3_inference.v3_inference_finetuned import INFERENCE_LOG
    
    if not INFERENCE_LOG.exists():
        print_warning("No generation logs found")
        return
    
    print_step("Recent Generation Logs")
    print()
    
    with open(INFERENCE_LOG) as f:
        lines = f.readlines()
    
    # Show last 10
    for line in lines[-10:]:
        import json
        try:
            log = json.loads(line)
            status = "✅" if log.get("validation_success") else "❌"
            print(f"{status} {log['timestamp']}: {log['prompt'][:50]}...")
        except:
            pass

def main():
    """Main agent loop"""
    # Check Ollama first
    print_step("Checking fine-tuned model...")
    if not check_ollama_connection():
        print_error("Cannot connect to fine-tuned model!")
        print("\nSetup required:")
        print("1. Download merged model from Google Drive")
        print("2. Convert to GGUF format")
        print("3. Register with Ollama: ollama create micro-ai-coder-v2:latest -f models/Modelfile_qwen_reactjs")
        return
    
    print_success("Fine-tuned model ready!")
    
    # Main menu loop
    while True:
        choice = main_menu()
        
        if choice == "1":
            single_component_mode()
        elif choice == "2":
            batch_component_mode()
        elif choice == "3":
            view_logs()
        elif choice == "4":
            print_header("✅ GOODBYE!")
            break
        else:
            print_error("Invalid option")

if __name__ == "__main__":
    main()
