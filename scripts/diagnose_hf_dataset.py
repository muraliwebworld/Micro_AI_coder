#!/usr/bin/env python3
"""
Diagnostic tool to examine Hugging Face dataset structure
"""
import json
from pathlib import Path

HF_DATASET = Path("/Users/muralidharanramasamy/Micro_AI_coder/datasets/data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl")

print("\n" + "="*80)
print("📊 HUGGING FACE DATASET STRUCTURE ANALYSIS")
print("="*80 + "\n")

if not HF_DATASET.exists():
    print("❌ Dataset not found!")
    exit(1)

print("Examining first 10 entries...\n")

with open(HF_DATASET, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        if idx > 10:
            break
        
        if line.strip():
            try:
                entry = json.loads(line)
                print(f"\n{'='*80}")
                print(f"Entry {idx}:")
                print(f"{'='*80}")
                print(f"Keys: {list(entry.keys())}")
                
                for key, value in entry.items():
                    if isinstance(value, str):
                        preview = value[:100].replace('\n', ' ')
                        print(f"\n  {key}:")
                        print(f"    Type: str")
                        print(f"    Length: {len(value)}")
                        print(f"    Preview: {preview}{'...' if len(value) > 100 else ''}")
                    elif isinstance(value, list):
                        print(f"\n  {key}:")
                        print(f"    Type: list[{len(value)}]")
                        if value:
                            preview = str(value[0])[:100]
                            print(f"    Preview: {preview}...")
                    else:
                        print(f"\n  {key}:")
                        print(f"    Type: {type(value).__name__}")
                        print(f"    Value: {value}")
                        
            except json.JSONDecodeError as e:
                print(f"Entry {idx}: JSON decode error - {e}")

print(f"\n\n{'='*80}")
print("📋 Summary")
print(f"{'='*80}\n")

# Count valid entries by checking for common code field names
field_counts = {
    'code': 0, 'response': 0, 'solution': 0, 'content': 0, 'body': 0,
    'text': 0, 'output': 0, 'completion': 0, 'generated': 0, 'result': 0,
    'instruction': 0, 'question': 0, 'prompt': 0, 'input': 0
}

with open(HF_DATASET, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        if idx > 1000:  # Check first 1000
            break
        
        if line.strip():
            try:
                entry = json.loads(line)
                for field in field_counts.keys():
                    if field in entry and isinstance(entry[field], str) and len(entry[field]) > 50:
                        field_counts[field] += 1
            except:
                pass

print("Field frequency (first 1000 entries):")
for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {field:20s}: {count:4d} entries")

print("\n" + "="*80 + "\n")
