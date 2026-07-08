#!/usr/bin/env python3
"""
Dataset Converter
Converts reactjs_projects_dataset.jsonl (JSON array) to generated_projects_final.jsonl (JSONL format).
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Configuration
DATASETS_DIR = Path(__file__).parent / "datasets"
INPUT_FILE = DATASETS_DIR / "reactjs_projects_dataset.jsonl"
OUTPUT_FILE = DATASETS_DIR / "generated_projects_final.jsonl"
LOG_FILE = DATASETS_DIR / "conversion_log.txt"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)

def print_step(text):
    print(f"\n➤ {text}")

def print_success(text):
    print(f"✅ {text}")

def print_warning(text):
    print(f"⚠️  {text}")

def print_error(text):
    print(f"❌ {text}")

def log_message(msg):
    """Write message to log file"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")

def validate_entry(entry, idx):
    """
    Validate that entry has required fields.
    Returns: (is_valid, error_message, cleaned_entry)
    """
    errors = []
    
    # Check required fields
    if 'prompt' not in entry:
        errors.append("Missing 'prompt' field")
    if 'code' not in entry:
        errors.append("Missing 'code' field")
    if 'Complex_CoT' not in entry:
        # Try to auto-generate from description
        if 'description' in entry:
            entry['Complex_CoT'] = f"React component: {entry['description']}"
        else:
            entry['Complex_CoT'] = f"React component for: {str(entry.get('prompt', 'Unknown'))[:80]}"
    
    # Validate field types and content
    prompt = str(entry.get('prompt', '')).strip()
    code = str(entry.get('code', '')).strip()
    cot = str(entry.get('Complex_CoT', '')).strip()
    
    if not prompt or len(prompt) < 5:
        errors.append(f"Invalid 'prompt': too short ('{prompt[:30]}')")
    
    if not code or len(code) < 50:
        errors.append(f"Invalid 'code': too short ({len(code)} chars)")
    
    if not cot or len(cot) < 10:
        errors.append(f"Invalid 'Complex_CoT': too short ({len(cot)} chars)")
    
    # Clean entry - keep only required fields
    cleaned = {
        'prompt': prompt,
        'code': code,
        'Complex_CoT': cot
    }
    
    if errors:
        return False, f"Entry {idx}: {'; '.join(errors)}", cleaned
    
    return True, "", cleaned

def convert_dataset():
    """Convert reactjs_projects_dataset.jsonl (array) to generated_projects_final.jsonl (JSONL)"""
    
    print_header("🔄 DATASET CONVERTER - JSON Array to JSONL")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Check if input file exists
    if not INPUT_FILE.exists():
        print_error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)
    
    print_step("Reading source dataset...")
    
    # Clear log file
    LOG_FILE.write_text(f"Conversion started: {datetime.now().isoformat()}\n\n", encoding="utf-8")
    
    # Load the entire JSON array
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print_error(f"Failed to parse JSON: {str(e)}")
        log_message(f"ERROR: Failed to parse JSON: {str(e)}")
        sys.exit(1)
    
    # Check if it's an array
    if not isinstance(data, list):
        print_error("Input is not a JSON array! Expected list of objects.")
        sys.exit(1)
    
    total_entries = len(data)
    print_success(f"Loaded {total_entries} entries from JSON array")
    
    # Convert and write
    print_step("Converting entries to JSONL format...")
    
    valid_entries = 0
    invalid_entries = 0
    skipped_entries = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for idx, entry in enumerate(data, 1):
            try:
                # Validate
                is_valid, error_msg, cleaned = validate_entry(entry, idx)
                
                if is_valid:
                    # Write to output
                    outfile.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    valid_entries += 1
                    
                    if valid_entries % 10 == 0:
                        print(f"  ✓ Converted {valid_entries}/{total_entries} entries")
                else:
                    # Log warning and try to salvage
                    msg = f"INVALID: {error_msg}"
                    print_warning(f"Entry {idx}: {error_msg[:80]}")
                    log_message(msg)
                    
                    # Try to save with auto-generated CoT
                    if cleaned.get('prompt') and cleaned.get('code'):
                        outfile.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                        valid_entries += 1
                        print(f"  → Salvaged with auto-generated explanation")
                    else:
                        skipped_entries += 1
                        invalid_entries += 1
                        msg = f"SKIPPED: {error_msg}"
                        log_message(msg)
            
            except Exception as e:
                invalid_entries += 1
                msg = f"ERROR at entry {idx}: {str(e)}"
                print_error(f"Entry {idx}: {str(e)}")
                log_message(msg)
    
    # Summary
    print_header("📊 CONVERSION SUMMARY")
    print(f"Total entries:    {total_entries}")
    print(f"Valid entries:    {valid_entries} ✅")
    print(f"Invalid entries:  {invalid_entries} ⚠️")
    print(f"Skipped entries:  {skipped_entries} ❌")
    
    if total_entries > 0:
        success_rate = (valid_entries / total_entries) * 100
        print(f"Success rate:     {success_rate:.1f}%")
    
    # File info
    if OUTPUT_FILE.exists():
        size_kb = OUTPUT_FILE.stat().st_size / 1024
        lines = valid_entries
        print(f"\n✅ Output file: {OUTPUT_FILE}")
        print(f"   Lines: {lines}")
        print(f"   Size:  {size_kb:.2f} KB")
        print_success(f"Conversion complete!")
        return True
    else:
        print_error("Output file was not created!")
        return False

def verify_output():
    """Verify the output file format"""
    print_step("Verifying output format...")
    
    if not OUTPUT_FILE.exists():
        print_error("Output file does not exist!")
        return False
    
    sample_entries = []
    entry_count = 0
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                entry_count += 1
                try:
                    entry = json.loads(line)
                    
                    # Check required fields
                    if not all(k in entry for k in ['prompt', 'code', 'Complex_CoT']):
                        print_error(f"Entry {i+1}: Missing required fields")
                        return False
                    
                    # Store first 3 entries as samples
                    if len(sample_entries) < 3:
                        sample_entries.append({
                            'line': entry_count,
                            'prompt': entry['prompt'][:60] + ("..." if len(entry['prompt']) > 60 else ""),
                            'code_len': len(entry['code']),
                            'cot_len': len(entry['Complex_CoT'])
                        })
                
                except json.JSONDecodeError as e:
                    print_error(f"Entry {i+1}: Invalid JSON - {str(e)}")
                    return False
    
    print_success(f"Total entries in output: {entry_count}")
    
    # Show samples
    if sample_entries:
        print("\n📋 Sample Entries:")
        for sample in sample_entries:
            print(f"\n  Entry #{sample['line']}:")
            print(f"    Prompt:     {sample['prompt']}")
            print(f"    Code size:  {sample['code_len']:,} chars")
            print(f"    CoT size:   {sample['cot_len']:,} chars")
    
    print_success("Output format verified!")
    return True

def main():
    """Main entry point"""
    try:
        # Convert dataset
        success = convert_dataset()
        
        if not success:
            print_error("Conversion failed!")
            sys.exit(1)
        
        # Verify output
        verify_success = verify_output()
        
        if verify_success:
            print_header("✅ DATASET CONVERSION SUCCESSFUL!")
            print(f"\n📝 Next steps:")
            print(f"  1. Review conversion log: {LOG_FILE}")
            print(f"  2. Train model: python phase2_training/v2_train_model.py")
            print(f"  3. Generate components: python phase4_agent/micro_ai_coder_agent.py")
        else:
            print_error("Verification failed!")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print_error("\nConversion cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        log_message(f"FATAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
