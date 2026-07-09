#!/usr/bin/env python3
"""
VALIDATION TESTS FOR FINE-TUNED QWEN MODEL
Tests the fine-tuned model on various React component generation tasks
"""
import requests
import json
import re
from pathlib import Path
from datetime import datetime

BASE_URL = "http://localhost:11434"
GENERATION_URL = f"{BASE_URL}/api/generate"
MODEL_NAME = "micro-ai-coder-v2:latest"
RESULTS_FILE = Path(__file__).parent.parent / "test_results_qwen_finetuned.md"

def check_connection():
    """Check if model is available"""
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            tags = response.json().get("models", [])
            model_names = [m.get("name", "") for m in tags]
            return any(MODEL_NAME in m for m in model_names)
    except:
        pass
    return False

def validate_react_code(code):
    """Validate React component code"""
    issues = []
    
    # Check basic structure
    if len(code) < 50:
        issues.append("Code too short")
    
    if not re.search(r'import\s+.*from', code):
        issues.append("Missing import statements")
    
    if not re.search(r'export\s+(default\s+)?function|export\s+(default\s+)?const', code):
        issues.append("Missing export statement")
    
    if '<' not in code or '>' not in code:
        issues.append("No JSX syntax")
    
    # Check bracket balance
    if code.count('{') != code.count('}'):
        issues.append(f"Unbalanced braces: {code.count('{')} {{ vs {code.count('}')} }}")
    
    if code.count('[') != code.count(']'):
        issues.append("Unbalanced brackets")
    
    if code.count('(') != code.count(')'):
        issues.append("Unbalanced parentheses")
    
    return len(issues) == 0, issues

def generate_test_component(prompt):
    """Generate a single component"""
    payload = {
        "model": MODEL_NAME,
        "prompt": f"""Generate a complete React component for: {prompt}

Requirements:
- Use React hooks
- Include imports and export
- Valid JSX syntax
- NO explanations, just code""",
        "stream": False,
        "temperature": 0.3,
        "top_k": 40
    }
    
    try:
        response = requests.post(GENERATION_URL, json=payload, timeout=30)
        if response.status_code == 200:
            code = response.json().get("response", "").strip()
            # Clean markdown
            code = re.sub(r'^```[a-zA-Z]*\s*\n?', '', code, flags=re.MULTILINE)
            code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
            return code.strip()
    except:
        pass
    
    return None

def run_tests():
    """Run validation suite"""
    print("="*80)
    print("FINE-TUNED QWEN MODEL VALIDATION")
    print("="*80)
    
    # Check connection
    print("\n1. Checking connection...", end=" ")
    if not check_connection():
        print("❌ FAILED")
        print("\nModel not found. Please ensure:")
        print("1. Ollama is running: ollama serve")
        print("2. Model is registered: ollama create micro-ai-coder-v2:latest -f models/Modelfile_qwen_reactjs")
        return False
    print("✅")
    
    # Test prompts
    test_cases = [
        "Button component with onClick handler",
        "Counter with increment and decrement",
        "Simple form component with validation",
        "Todo list with add/remove items",
        "User profile card",
        "Input field component",
        "Navigation bar",
        "Alert/notification component",
        "Modal dialog component",
        "Loading spinner component",
        "Dropdown menu component",
        "Checkbox list",
        "Radio button group",
        "Date picker component",
        "Tab navigation",
        "Card layout component",
        "Grid layout",
        "Sidebar navigation",
        "Search filter component",
        "Pagination component"
    ]
    
    print(f"\n2. Testing {len(test_cases)} prompts...")
    print("-"*80)
    
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n[{i:2d}/{len(test_cases)}] {prompt}...", end=" ")
        
        code = generate_test_component(prompt)
        if not code:
            print("❌ (No output)")
            results["failed"] += 1
            results["details"].append({
                "prompt": prompt,
                "status": "No output",
                "issues": ["No code generated"]
            })
            continue
        
        valid, issues = validate_react_code(code)
        if valid:
            print("✅")
            results["passed"] += 1
            results["details"].append({
                "prompt": prompt,
                "status": "PASS",
                "code_length": len(code),
                "issues": []
            })
        else:
            print(f"❌ ({len(issues)} issues)")
            results["failed"] += 1
            results["details"].append({
                "prompt": prompt,
                "status": "FAIL",
                "code_length": len(code),
                "issues": issues
            })
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    success_rate = 100 * results["passed"] // results["total"]
    print(f"\nTotal:  {results['total']} tests")
    print(f"Passed: {results['passed']} ✅")
    print(f"Failed: {results['failed']} ❌")
    print(f"Rate:   {success_rate}%")
    
    # Write report
    with open(RESULTS_FILE, 'w') as f:
        f.write("# Fine-Tuned Qwen Model Validation Report\n\n")
        f.write(f"**Date**: {datetime.now().isoformat()}\n")
        f.write(f"**Model**: {MODEL_NAME}\n")
        f.write(f"**Success Rate**: {success_rate}% ({results['passed']}/{results['total']})\n\n")
        
        f.write("## Test Results\n\n")
        
        f.write("### Passed Tests\n\n")
        for detail in results["details"]:
            if detail["status"] == "PASS":
                f.write(f"- ✅ {detail['prompt']} ({detail['code_length']} chars)\n")
        
        f.write("\n### Failed Tests\n\n")
        for detail in results["details"]:
            if detail["status"] != "PASS":
                f.write(f"- ❌ {detail['prompt']}\n")
                if detail["issues"]:
                    for issue in detail["issues"]:
                        f.write(f"  - {issue}\n")
        
        f.write("\n## Recommendation\n\n")
        if success_rate >= 95:
            f.write("✅ **Model is production-ready**\n")
            f.write("The fine-tuned model is generating valid React code with high success rate.\n")
        elif success_rate >= 80:
            f.write("⚠️ **Model needs improvement**\n")
            f.write("While the model is working, some quality issues detected. Consider:\n")
            f.write("- Run more training epochs\n")
            f.write("- Adjust hyperparameters\n")
            f.write("- Add more diverse training data\n")
        else:\n            f.write("❌ **Model not ready for production**\n")
            f.write("The success rate is too low. Recommendations:\n")
            f.write("- Re-check dataset quality\n")
            f.write("- Increase training duration\n")
            f.write("- Review hyperparameters\n")
    
    print(f"\nReport saved: {RESULTS_FILE}")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
