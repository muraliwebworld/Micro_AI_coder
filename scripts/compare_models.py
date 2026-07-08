#!/usr/bin/env python3
"""
Compare outputs from original model vs. Hugging Face model
Tests both on the same prompts and shows quality differences
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80 + "\n")

def print_comparison(model_name, output):
    print(f"\n{'─'*80}")
    print(f"Model: {model_name}")
    print(f"{'─'*80}")
    print(f"Code length: {len(output['code'])} chars")
    print(f"Explanation: {len(output['Complex_CoT'])} chars")
    print(f"\nFirst 500 chars of code:")
    print(output['code'][:500])
    if len(output['code']) > 500:
        print("...")
    print(f"\nExplanation:")
    print(output['Complex_CoT'][:200])
    if len(output['Complex_CoT']) > 200:
        print("...")

def validate_code(code):
    """Quick validation check"""
    errors = []
    
    if 'import React' not in code:
        errors.append("Missing import React")
    if 'export default' not in code and 'module.exports' not in code:
        errors.append("Missing export")
    if code.count('(') != code.count(')'):
        errors.append(f"Parentheses mismatch")
    if code.count('{') != code.count('}'):
        errors.append(f"Braces mismatch")
    if code.count('[') != code.count(']'):
        errors.append(f"Brackets mismatch")
    
    return errors

def compare_models():
    """Compare original and HF models on test prompts"""
    print_header("MODEL COMPARISON: ORIGINAL vs. HUGGING FACE")
    
    # Import inference modules
    try:
        from phase3_inference.v2_inference import load_model_and_config as load_original, CodeGenerator as OriginalGenerator
        original_available = True
    except Exception as e:
        print(f"⚠️  Original model not available: {e}")
        original_available = False
    
    try:
        from phase4_agent.micro_ai_coder_agent_huggingface import load_model_and_config as load_hf, CodeGenerator as HFGenerator
        hf_available = True
    except Exception as e:
        print(f"⚠️  HF model not available: {e}")
        hf_available = False
    
    if not original_available and not hf_available:
        print("❌ Neither model is available. Please train the models first.")
        return
    
    test_prompts = [
        "React component for a login form",
        "Create a counter with increment/decrement buttons",
        "Build a todo list with add and delete functionality",
        "User profile card with name and avatar",
        "Search bar with real-time suggestions"
    ]
    
    print(f"Testing on {len(test_prompts)} prompts...\n")
    
    results = []
    
    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {idx}/{len(test_prompts)}: {prompt}")
        print(f"{'='*80}")
        
        result = {'prompt': prompt, 'original': None, 'hf': None}
        
        # Test original model
        if original_available:
            try:
                print("\n⏳ Testing original model...")
                config, enc = load_original()
                generator = OriginalGenerator(config, enc)
                output = generator.generate_react_component(prompt)
                errors = validate_code(output)
                
                result['original'] = {
                    'code': output,
                    'length': len(output),
                    'valid': len(errors) == 0,
                    'errors': errors
                }
                
                print(f"✅ Generated ({len(output)} chars)")
                if errors:
                    print(f"⚠️  Errors: {'; '.join(errors)}")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
        
        # Test HF model
        if hf_available:
            try:
                print("\n⏳ Testing Hugging Face model...")
                config, enc = load_hf()
                generator = HFGenerator(config, enc)
                output = generator.generate_react_component(prompt)
                errors = validate_code(output)
                
                result['hf'] = {
                    'code': output,
                    'length': len(output),
                    'valid': len(errors) == 0,
                    'errors': errors
                }
                
                print(f"✅ Generated ({len(output)} chars)")
                if errors:
                    print(f"⚠️  Errors: {'; '.join(errors)}")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
        
        results.append(result)
    
    # Summary Report
    print_header("📊 COMPARISON SUMMARY")
    
    if original_available:
        orig_valid = sum(1 for r in results if r['original'] and r['original']['valid'])
        orig_total = sum(1 for r in results if r['original'])
        print(f"\nOriginal Model:")
        print(f"  Valid outputs: {orig_valid}/{orig_total} ({100*orig_valid//orig_total if orig_total else 0}%)")
        
        if orig_total:
            total_chars = sum(r['original']['length'] for r in results if r['original'])
            avg_chars = total_chars // orig_total
            print(f"  Avg code length: {avg_chars} chars")
    
    if hf_available:
        hf_valid = sum(1 for r in results if r['hf'] and r['hf']['valid'])
        hf_total = sum(1 for r in results if r['hf'])
        print(f"\nHugging Face Model:")
        print(f"  Valid outputs: {hf_valid}/{hf_total} ({100*hf_valid//hf_total if hf_total else 0}%)")
        
        if hf_total:
            total_chars = sum(r['hf']['length'] for r in results if r['hf'])
            avg_chars = total_chars // hf_total
            print(f"  Avg code length: {avg_chars} chars")
    
    if original_available and hf_available:
        print(f"\n📈 Improvement:")
        if orig_total > 0 and hf_total > 0:
            orig_quality = 100 * orig_valid // orig_total
            hf_quality = 100 * hf_valid // hf_total
            improvement = hf_quality - orig_quality
            print(f"  Quality change: {improvement:+d}% (Original: {orig_quality}% → HF: {hf_quality}%)")
    
    # Save detailed results
    results_file = Path(__file__).parent / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'prompts_tested': len(test_prompts),
            'results': results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to: {results_file}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    compare_models()
