#!/usr/bin/env python3
"""
Analyze Hugging Face dataset structure and statistics
"""
import json
from pathlib import Path
from collections import Counter

HF_DATASET = Path(__file__).parent.parent / "datasets" / "data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl"

def analyze_dataset():
    print("\n" + "="*80)
    print("📊 HUGGING FACE DATASET ANALYSIS")
    print("="*80 + "\n")
    
    entries = []
    field_names = Counter()
    code_lengths = []
    
    print("Reading dataset...")
    with open(HF_DATASET, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                    
                    # Track field names
                    for key in entry.keys():
                        field_names[key] += 1
                    
                    # Track code lengths
                    for field in ['code', 'response', 'solution', 'content', 'body']:
                        if field in entry and isinstance(entry[field], str):
                            code_lengths.append(len(entry[field]))
                            break
                    
                except json.JSONDecodeError:
                    pass
            
            if line_num % 5000 == 0:
                print(f"  Processed {line_num} lines...")
    
    print(f"\n✅ Total entries: {len(entries)}")
    print(f"\n📋 Field names and frequencies:")
    for field, count in field_names.most_common(10):
        print(f"  {field:20s}: {count:6d} ({100*count//len(entries)}%)")
    
    if code_lengths:
        print(f"\n📏 Code length statistics:")
        print(f"  Min: {min(code_lengths)} chars")
        print(f"  Max: {max(code_lengths)} chars")
        print(f"  Avg: {sum(code_lengths)//len(code_lengths)} chars")
        print(f"  Median: {sorted(code_lengths)[len(code_lengths)//2]} chars")
    
    # Show sample entries
    print(f"\n📋 Sample entries (first 3):")
    for i, entry in enumerate(entries[:3]):
        print(f"\n  Entry {i+1}:")
        for key, value in list(entry.items())[:3]:
            if isinstance(value, str):
                val_preview = value[:80].replace('\n', ' ')
            else:
                val_preview = str(value)[:80]
            print(f"    {key}: {val_preview}...")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_dataset()
