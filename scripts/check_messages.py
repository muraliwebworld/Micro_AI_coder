#!/usr/bin/env python3
"""
Deep dive into message structure
"""
import json
from pathlib import Path

HF_DATASET = Path("/Users/muralidharanramasamy/Micro_AI_coder/datasets/data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl")

print("\n" + "="*80)
print("📋 MESSAGE STRUCTURE ANALYSIS")
print("="*80 + "\n")

with open(HF_DATASET, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        if idx > 3:  # Just show first 3 entries
            break
        
        if line.strip():
            try:
                entry = json.loads(line)
                print(f"\n{'='*80}")
                print(f"Entry {idx}: {len(entry['messages'])} messages")
                print(f"{'='*80}")
                
                for msg_idx, msg in enumerate(entry['messages'], 1):
                    print(f"\nMessage {msg_idx}:")
                    print(f"  Role: {msg.get('role', 'N/A')}")
                    content = msg.get('content', '')
                    
                    # Check for code blocks
                    if '```' in content:
                        print(f"  ✅ Contains code block (```)")
                    
                    # Show preview
                    preview = content[:200].replace('\n', ' ')
                    print(f"  Content: {preview}{'...' if len(content) > 200 else ''}")
                    print(f"  Length: {len(content)} chars")
                    
            except Exception as e:
                print(f"Error: {e}")

print("\n" + "="*80 + "\n")
