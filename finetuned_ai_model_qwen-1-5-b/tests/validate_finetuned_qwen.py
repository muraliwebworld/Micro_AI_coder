#!/usr/bin/env python3
"""
Validation Test Suite for Fine-Tuned Qwen2.5-Coder-1.5B
20 diverse React component generation tests
Expected success rate: ≥85% valid code
"""

import requests
import json
import logging
from typing import List, Dict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:11434"
MODEL_NAME = "micro-ai-coder-1-5b:latest"
GENERATION_URL = f"{BASE_URL}/api/generate"

# Test prompts covering various React patterns
TEST_PROMPTS = [
    # Basic Components
    "Write a React button component",
    "Create a React text input component",
    "Generate a React heading component",
    
    # State & Hooks
    "Write a React counter with useState",
    "Create a React timer using useEffect",
    "Generate a React form with form validation",
    
    # Props & Children
    "Create a React card component with props",
    "Write a React list component that accepts an array",
    "Generate a React navbar with navigation links",
    
    # Events & Handlers
    "Create a React button with onClick handler",
    "Write a React input with onChange handler",
    "Generate a React form with onSubmit handler",
    
    # Conditional Rendering
    "Create a React component with if/else rendering",
    "Write a React component with ternary operator",
    "Generate a React component with logical AND operator",
    
    # Lists & Keys
    "Create a React todo list component",
    "Write a React component that maps over items",
    "Generate a React product list with keys",
    
    # Advanced
    "Create a React modal component",
    "Write a React dropdown/select component",
]


def check_model_availability():
    """Check if model is available"""
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name') for m in models]
            if any(MODEL_NAME in name for name in model_names):
                logger.info(f"✅ Model '{MODEL_NAME}' is available\n")
                return True
            else:
                logger.error(f"❌ Model '{MODEL_NAME}' not found")
                logger.info(f"Available models: {model_names}\n")
                return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to Ollama: {e}\n")
        return False


def validate_react_syntax(code: str) -> tuple[bool, str]:
    """
    Validate React code quality
    Returns: (is_valid, issues)
    """
    issues = []
    
    # Check balanced brackets
    if code.count('{') != code.count('}'):
        issues.append("Unbalanced curly braces")
    if code.count('[') != code.count(']'):
        issues.append("Unbalanced square brackets")
    if code.count('(') != code.count(')'):
        issues.append("Unbalanced parentheses")
    
    # Check for React elements
    has_jsx = ('<' in code and '>' in code and 
              ('/' in code or 'return' in code or 'React' in code))
    if not has_jsx:
        issues.append("No JSX found")
    
    # Check for common patterns (import, export, component, etc.)
    has_react_patterns = any(p in code for p in [
        'function ', 'const ', 'export', 'return', 'React', 'import'
    ])
    if not has_react_patterns:
        issues.append("Missing common React patterns")
    
    # Check code length (not too short, not too long)
    lines = len([l for l in code.split('\n') if l.strip()])
    if lines < 2:
        issues.append("Code too short")
    if lines > 50:
        issues.append("Code too long (may be truncated)")
    
    is_valid = len(issues) == 0
    return is_valid, "; ".join(issues) if issues else "OK"


def generate_and_test(prompt: str) -> Dict:
    """Generate code and test it"""
    full_prompt = f"Prompt: {prompt}\n\nCode:\n"
    
    try:
        response = requests.post(
            GENERATION_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False,
                "temperature": 0.3,
                "top_k": 40,
                "top_p": 0.9,
                "num_predict": 256,
            },
            timeout=60,
        )
        
        if response.status_code != 200:
            return {
                "prompt": prompt,
                "valid": False,
                "issue": f"API error {response.status_code}",
                "code": "",
            }
        
        result = response.json()
        generated_code = result.get('response', '').split("Code:\n")[-1].strip()
        
        is_valid, issue = validate_react_syntax(generated_code)
        
        return {
            "prompt": prompt,
            "valid": is_valid,
            "issue": issue,
            "code": generated_code[:200],  # First 200 chars
        }
    
    except requests.exceptions.Timeout:
        return {
            "prompt": prompt,
            "valid": False,
            "issue": "Request timeout",
            "code": "",
        }
    except Exception as e:
        return {
            "prompt": prompt,
            "valid": False,
            "issue": str(e),
            "code": "",
        }


def run_tests():
    """Run all validation tests"""
    logger.info("=" * 70)
    logger.info("Validation Test Suite - Qwen2.5-Coder-1.5B")
    logger.info("=" * 70)
    logger.info(f"Running {len(TEST_PROMPTS)} tests...\n")
    
    results = []
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        logger.info(f"[{i:2d}/{len(TEST_PROMPTS)}] Testing: {prompt[:50]}...")
        
        result = generate_and_test(prompt)
        results.append(result)
        
        status = "✅ PASS" if result['valid'] else "❌ FAIL"
        logger.info(f"      {status} - {result['issue']}")
    
    # Summary
    valid_count = sum(1 for r in results if r['valid'])
    total_count = len(results)
    success_rate = (valid_count / total_count) * 100
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 70)
    logger.info(f"Total tests: {total_count}")
    logger.info(f"Passed: {valid_count} ({success_rate:.1f}%)")
    logger.info(f"Failed: {total_count - valid_count} ({100-success_rate:.1f}%)")
    logger.info(f"\n✅ Target: ≥85% success rate")
    
    if success_rate >= 85:
        logger.info(f"🎉 {success_rate:.1f}% - EXCELLENT! Model is production-ready!")
    elif success_rate >= 70:
        logger.info(f"👍 {success_rate:.1f}% - GOOD. Model needs minor improvements.")
    else:
        logger.info(f"⚠️  {success_rate:.1f}% - MODEL NEEDS RETRAINING")
    
    # Save detailed results
    output_file = f"./outputs/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    save_results(results, output_file, success_rate)
    
    logger.info(f"\n📝 Detailed results saved to: {output_file}")
    
    return success_rate


def save_results(results: List[Dict], filename: str, success_rate: float):
    """Save test results to markdown file"""
    with open(filename, 'w') as f:
        f.write("# Qwen2.5-Coder-1.5B Validation Test Results\n\n")
        f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Success Rate**: {success_rate:.1f}%\n")
        f.write(f"**Total Tests**: {len(results)}\n")
        f.write(f"**Passed**: {sum(1 for r in results if r['valid'])}\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("| # | Prompt | Status | Issue |\n")
        f.write("|---|--------|--------|-------|\n")
        
        for i, result in enumerate(results, 1):
            status = "✅ PASS" if result['valid'] else "❌ FAIL"
            prompt = result['prompt'][:40]
            issue = result['issue']
            f.write(f"| {i} | {prompt}... | {status} | {issue} |\n")
        
        f.write("\n## Test Prompts\n\n")
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            f.write(f"{i}. {prompt}\n")


def main():
    """Main test runner"""
    logger.info("\n")
    
    # Check model
    if not check_model_availability():
        return
    
    # Run tests
    try:
        success_rate = run_tests()
        
        if success_rate >= 85:
            logger.info("\n✅ Model validation PASSED!")
        else:
            logger.warning("\n⚠️  Model validation FAILED - Consider retraining")
    
    except KeyboardInterrupt:
        logger.info("\n\n👋 Tests interrupted by user")


if __name__ == "__main__":
    main()
