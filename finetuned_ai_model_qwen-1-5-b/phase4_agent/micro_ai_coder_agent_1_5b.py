#!/usr/bin/env python3
"""
Micro AI Coder Agent v2 - Menu-driven orchestrator for React component generation.
Uses fine-tuned Qwen2.5-Coder-1.5B model via Ollama.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import requests

# Import from inference module
sys.path.insert(0, str(Path(__file__).parent.parent / "phase3_inference"))
from v3_inference_finetuned_1_5b import (
    generate_react_component,
    check_ollama_connection,
    list_available_models,
    save_generation,
    MODEL_NAME,
    OUTPUT_DIR,
    LOG_DIR,
)

LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_menu():
    """Display main menu."""
    print("\n📋 Menu:")
    print("  1. Generate single React component")
    print("  2. Batch generate multiple components")
    print("  3. View generation logs")
    print("  4. Exit")
    print()


def single_generation():
    """Generate a single React component."""
    print_header("Single Component Generation")
    prompt = input("\n🎯 Enter prompt (or 'back' to return): ").strip()

    if prompt.lower() == "back":
        return

    if not prompt:
        print("❌ Prompt cannot be empty")
        return

    print("\n⏳ Generating component (30-60 seconds)...")
    result = generate_react_component(prompt)

    if result:
        print("\n" + "-" * 70)
        print(f"✅ Success! Generated {result['code_length']} characters")
        print(f"\n📊 Validation Results:")
        for check, value in result["validation"].items():
            status = "✅" if value else "❌"
            print(f"   {status} {check.replace('_', ' ')}: {value}")

        print(f"\n📄 Generated Code:")
        print("-" * 70)
        print(result["code"])
        print("-" * 70)

        # Save
        output_path = save_generation(result)
        print(f"\n💾 Saved to: {output_path}")
        print(f"✓ Valid code: {result['is_valid']}")
    else:
        print("❌ Generation failed")


def batch_generation():
    """Generate multiple components in batch mode."""
    print_header("Batch Component Generation")

    prompts = [
        "Create a React button component with onClick handler",
        "Write a React counter component with increment and decrement buttons",
        "Generate a React form component with email and password inputs",
        "Create a React card component displaying user information",
        "Write a React navbar component with navigation links",
    ]

    print(f"\n📋 Will generate {len(prompts)} components:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")

    confirm = input("\n🔄 Continue with batch generation? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled")
        return

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_DIR / f"batch_{timestamp}"
    batch_dir.mkdir(exist_ok=True)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] ⏳ Generating: {prompt[:50]}...")
        result = generate_react_component(prompt)

        if result:
            results.append(result)
            status = "✅" if result["is_valid"] else "⚠️"
            print(f"{status} Done - {result['code_length']} chars")

            # Save individual
            component_file = batch_dir / f"component_{i:02d}.jsx"
            with open(component_file, "w") as f:
                f.write(result["code"])
        else:
            print("❌ Failed")

    # Save batch summary
    summary = {
        "timestamp": timestamp,
        "total": len(prompts),
        "successful": len(results),
        "success_rate": f"{len(results) / len(prompts) * 100:.1f}%",
        "components": results,
    }

    summary_file = batch_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "=" * 70)
    print(f"✅ Batch generation complete!")
    print(f"   Total: {len(prompts)}")
    print(f"   Successful: {len(results)}")
    print(f"   Success rate: {len(results) / len(prompts) * 100:.1f}%")
    print(f"   Saved to: {batch_dir}")
    print("=" * 70)


def view_logs():
    """Display generation logs."""
    print_header("Generation Logs")

    log_file = LOG_DIR / "generation_results_1_5b.jsonl"

    if not log_file.exists():
        print("📭 No logs found yet")
        return

    # Read logs
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not logs:
        print("📭 No valid logs found")
        return

    # Display summary
    total = len(logs)
    valid = sum(1 for log in logs if log.get("is_valid", False))

    print(f"\n📊 Summary:")
    print(f"   Total generations: {total}")
    print(f"   Valid components: {valid} ({valid/total*100:.1f}%)")
    print(f"   Invalid: {total - valid}")

    # Show recent logs
    print(f"\n📝 Recent generations (last 10):")
    print("-" * 70)
    for log in logs[-10:]:
        status = "✅" if log.get("is_valid") else "❌"
        timestamp = log.get("timestamp", "?")[:10]
        prompt = log.get("prompt", "?")[:40]
        print(f"{status} {timestamp} | {prompt}...")
    print("-" * 70)

    # Save stats
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_generations": total,
        "valid_components": valid,
        "success_rate": f"{valid/total*100:.1f}%",
    }

    stats_file = LOG_DIR / "generation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n💾 Stats saved to: {stats_file}")


def main():
    """Main application loop."""
    print_header("🚀 Micro AI Coder - React Component Generator")
    print("   Fine-tuned Qwen2.5-Coder-1.5B Model")
    print("   Optimized for M2 Mac mini 8GB RAM")

    # Check Ollama
    print("\n🔍 Checking Ollama connection...")
    if not check_ollama_connection():
        print("❌ Ollama is not running!")
        print("\n💡 To start Ollama, run in a new terminal:")
        print("   ollama serve")
        print("\n   Or on M2 Mac: ollama serve &")
        return 1

    print("✅ Ollama connected")

    # Check model
    models = list_available_models()
    if MODEL_NAME not in models:
        print(f"\n❌ Model '{MODEL_NAME}' not registered!")
        print("\n💡 To register the model, run:")
        print(f"   ollama create {MODEL_NAME} -f models/Modelfile_qwen_1_5b")
        return 1

    print(f"✅ Model available: {MODEL_NAME}")

    # Main loop
    while True:
        print_menu()
        choice = input("👉 Enter choice (1-4): ").strip()

        if choice == "1":
            single_generation()
        elif choice == "2":
            batch_generation()
        elif choice == "3":
            view_logs()
        elif choice == "4":
            print("\n👋 Goodbye! Happy coding! 🚀\n")
            return 0
        else:
            print("❌ Invalid choice. Please enter 1-4")

    return 0


if __name__ == "__main__":
    sys.exit(main())
