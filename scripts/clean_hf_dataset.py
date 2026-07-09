#!/usr/bin/env python3
"""
Clean HuggingFace dataset by filtering out corrupted code samples
Validates code quality before adding to training set
"""
import json
import re
from pathlib import Path
from collections import Counter

INPUT_JSONL = Path("/Users/muralidharanramasamy/Micro_AI_coder/datasets/data_cleaned_huggingface_new.jsonl")
OUTPUT_JSONL = Path("/Users/muralidharanramasamy/Micro_AI_coder/datasets/data_cleaned_huggingface.jsonl")

def is_valid_code(code):
    """Check if code sample is valid (not corrupted)"""
    
    # Balance checks
    if code.count('(') != code.count(')'):
        return False, "Unbalanced parentheses"
    if code.count('{') != code.count('}'):
        return False, "Unbalanced braces"
    if code.count('[') != code.count(']'):
        return False, "Unbalanced brackets"
    
    # Must have imports
    if 'import' not in code.lower():
        return False, "Missing imports"
    
    # Detect obvious corruption patterns
    corruption_patterns = [
        (r':\s*:', 'Double colons in ternary'),
        (r'}\s*,\s*\[', 'Malformed hook syntax'),
        (r'useState\(\s*\)', 'Empty useState'),
        (r'const\s+\[\s*,', 'Malformed destructuring'),
        (r'function\s+\(\s*\)', 'Empty function parameters'),
        (r'\?\s*:\s*:', 'Corrupted ternary operator'),
    ]
    
    for pattern, msg in corruption_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return False, msg
    
    # Check for excessive non-ASCII (garbled text)
    ascii_chars = sum(1 for c in code if ord(c) < 128)
    if len(code) > 100 and ascii_chars / len(code) < 0.7:
        return False, "Excessive non-ASCII (likely corrupted)"
    
    # Check for obvious hallucinations
    if code.count('undefined') > 5:
        return False, "Too many 'undefined'"
    if code.count('null') > 10:
        return False, "Too many 'null'"
    
    # Must be substantial code (not just imports)
    lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('//')]
    if len(lines) < 5:
        return False, "Code too short/trivial"
    
    return True, "Valid"

def extract_code_field(entry):
    """Extract code from entry - handles conversation format with messages"""
    
    # Check if entry has messages (conversation format)
    if 'messages' in entry and isinstance(entry['messages'], list):
        # Find assistant's response with code
        code_blocks = []
        
        for msg in entry['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Extract code blocks (between ``` markers)
                if '```' in content:
                    # Split by ``` and get code blocks
                    parts = content.split('```')
                    for i in range(1, len(parts), 2):
                        code = parts[i].strip()
                        # Remove language specifier
                        lines = code.split('\n')
                        if lines and lines[0].lower() in ['tsx', 'jsx', 'javascript', 'typescript', 'python', 'java', 'cpp', 'c', 'go', 'rust', 'php', 'html', 'css']:
                            code = '\n'.join(lines[1:])
                        if len(code) > 100:
                            code_blocks.append(code.strip())
        
        if code_blocks:
            return '\n'.join(code_blocks)
    
    # Fallback
    for field in ['code', 'response', 'solution', 'content', 'body', 'text', 'output']:
        if field in entry and isinstance(entry[field], str) and len(entry[field]) > 50:
            return entry[field]
    
    return None

def clean_dataset():
    print("\n" + "="*80)
    print("🧹 CLEANING HUGGING FACE DATASET")
    print("="*80 + "\n")
    
    if not INPUT_JSONL.exists():
        print("❌ Input dataset not found!")
        return
    
    stats = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'error_reasons': Counter()
    }
    
    print("Processing dataset...")
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_JSONL, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            stats['total'] += 1
            
            if line.strip():
                try:
                    entry = json.loads(line)
                    code = extract_code_field(entry)
                    
                    if not code:
                        stats['invalid'] += 1
                        stats['error_reasons']['No code found'] += 1
                        continue
                    
                    # Validate code
                    is_valid, reason = is_valid_code(code)
                    
                    if is_valid:
                        # Write valid entry with extracted code
                        valid_entry = {
                            'prompt': entry.get('messages', [{}])[1].get('content', 'React component') if 'messages' in entry else 'React component',
                            'code': code,
                            'model': entry.get('model', 'unknown'),
                            'original_timestamp': entry.get('created_at', '')
                        }
                        outfile.write(json.dumps(valid_entry, ensure_ascii=False) + '\n')
                        stats['valid'] += 1
                    else:
                        stats['invalid'] += 1
                        stats['error_reasons'][reason] += 1
                        
                except Exception as e:
                    stats['invalid'] += 1
                    stats['error_reasons']['JSON/Parse error'] += 1
            
            if line_num % 5000 == 0:
                print(f"  Processed {line_num:,} entries... ({stats['valid']} valid so far)")
    
    print(f"\n✅ Cleaning complete!\n")
    print(f"📊 Results:")
    print(f"  Total entries:     {stats['total']:,}")
    print(f"  Valid entries:     {stats['valid']:,} ({100*stats['valid']/max(1,stats['total']):.1f}%)")
    print(f"  Invalid entries:   {stats['invalid']:,} ({100*stats['invalid']/max(1,stats['total']):.1f}%)")
    
    print(f"\n📋 Rejection reasons:")
    for reason, count in stats['error_reasons'].most_common(10):
        print(f"  {reason:40s}: {count:5,}")
    
    print(f"\n✅ Cleaned dataset saved to:")
    print(f"   {OUTPUT_JSONL}")
    print(f"\n💡 Use this dataset for training:")
    print(f"   INPUT_JSONL = Path(...) / 'data_cleaned_huggingface.jsonl'")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    clean_dataset()
