#!/usr/bin/env python3
"""
Inference module for fine-tuned Qwen2.5-Coder-1.5B model via Ollama HTTP API.
Generates React components from natural language prompts.
"""

import requests
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
MODEL_NAME = "micro-ai-coder-1-5b:latest"
BASE_URL = "http://localhost:11434"
LOG_DIR = Path("logs")
OUTPUT_DIR = Path("outputs")

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models() -> List[str]:
    """List all available models in Ollama."""
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except requests.exceptions.RequestException:
        pass
    return []


def validate_react_code(code: str) -> Dict[str, bool]:
    """
    Validate generated React code for common issues.
    Returns dict with validation results.
    """
    checks = {
        "has_import": False,
        "has_export": False,
        "balanced_braces": True,
        "balanced_brackets": True,
        "balanced_parens": True,
        "valid_jsx": False,
        "has_return": False,
    }

    # Check imports/exports
    checks["has_import"] = "import" in code.lower()
    checks["has_export"] = ("export" in code.lower()) or ("export default" in code)

    # Check balanced brackets
    checks["balanced_braces"] = code.count("{") == code.count("}")
    checks["balanced_brackets"] = code.count("[") == code.count("]")
    checks["balanced_parens"] = code.count("(") == code.count(")")

    # Check for JSX (lowercase tags or PascalCase components)
    jsx_pattern = r"<[a-z]+[^>]*>|<[A-Z][a-zA-Z0-9]*[^>]*>"
    checks["valid_jsx"] = bool(re.search(jsx_pattern, code))

    # Check for return statement
    checks["has_return"] = "return" in code.lower()

    return checks


def generate_react_component(prompt: str, temperature: float = 0.3) -> Optional[Dict]:
    """
    Generate a React component from a natural language prompt.
    Returns dict with generated code and metadata.
    """
    try:
        # Format prompt for code generation
        full_prompt = f"""You are an expert React developer. Generate only the React component code without explanations.

Prompt: {prompt}

Generate a complete, functional React component:"""

        # Call Ollama API
        response = requests.post(
            f"{BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False,
                "temperature": temperature,
                "top_k": 40,
                "top_p": 0.9,
            },
            timeout=300,  # 5 minute timeout for generation
        )

        if response.status_code != 200:
            print(f"❌ Generation failed: {response.status_code}")
            return None

        result = response.json()
        code = result.get("response", "").strip()

        # Extract actual code (remove explanations)
        if "```javascript" in code:
            code = code.split("```javascript")[1].split("```")[0].strip()
        elif "```jsx" in code:
            code = code.split("```jsx")[1].split("```")[0].strip()
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 3:
                code = parts[1].strip()

        # Validate code
        validation = validate_react_code(code)

        return {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "code": code,
            "code_length": len(code),
            "tokens_generated": result.get("eval_count", 0),
            "validation": validation,
            "is_valid": all(
                v
                for k, v in validation.items()
                if k
                not in ["valid_jsx"]  # JSX check is optional
            ),
        }

    except requests.exceptions.Timeout:
        print("❌ Request timeout (generation too long)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return None
    except json.JSONDecodeError:
        print("❌ Invalid response from Ollama")
        return None


def save_generation(result: Dict) -> str:
    """Save generation result to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / timestamp
    output_dir.mkdir(exist_ok=True)

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(result, f, indent=2)

    # Save code
    code_file = output_dir / "component.jsx"
    with open(code_file, "w") as f:
        f.write(result["code"])

    # Log to file
    log_entry = {
        "timestamp": result["timestamp"],
        "prompt": result["prompt"],
        "is_valid": result["is_valid"],
        "code_length": result["code_length"],
        "output_dir": str(output_dir),
    }
    log_file = LOG_DIR / "generation_results_1_5b.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return str(output_dir)


def interactive_mode():
    """Interactive CLI for generating React components."""
    print("\n" + "=" * 70)
    print("🚀 Micro AI Coder - React Component Generator (1.5B Model)")
    print("=" * 70)

    # Check Ollama connection
    if not check_ollama_connection():
        print("\n❌ Ollama not running!")
        print("\nTo start Ollama, run in a terminal:")
        print("  ollama serve")
        return

    # Check model availability
    models = list_available_models()
    if MODEL_NAME not in models:
        print(f"\n❌ Model '{MODEL_NAME}' not found!")
        print(f"Available models: {', '.join(models) if models else 'None'}")
        print("\nTo register the model, run:")
        print(f"  ollama create {MODEL_NAME} -f models/Modelfile_qwen_1_5b")
        return

    print(f"\n✅ Connected to Ollama")
    print(f"✅ Model available: {MODEL_NAME}")
    print(f"\n📝 Enter prompts to generate React components")
    print(f"   Type 'quit' to exit\n")

    while True:
        try:
            prompt = input("🎯 Prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not prompt:
                continue

            print("\n⏳ Generating component (this may take 30-60 seconds)...")
            result = generate_react_component(prompt)

            if result:
                print("\n" + "-" * 70)
                print(f"✅ Generated {result['code_length']} chars")
                print(f"📊 Validation: {result['validation']}")
                print(f"✓ Valid: {result['is_valid']}")
                print(f"\n📄 Code:\n{result['code'][:500]}...")
                print("-" * 70)

                # Save
                output_path = save_generation(result)
                print(f"\n💾 Saved to: {output_path}")
            else:
                print("❌ Generation failed")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    interactive_mode()
