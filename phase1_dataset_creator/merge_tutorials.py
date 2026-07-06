#!/usr/bin/env python3
"""
Merge generated tutorials with the base dataset into a final training dataset.

This script takes one or more generated tutorial files (created by extract_tutorials.py)
and merges them with the base generated_projects.jsonl into a final dataset.

Usage:
    python merge_tutorials.py <file1.jsonl> [file2.jsonl] [file3.jsonl] ...
    
Example:
    python merge_tutorials.py generated_projects_reactjs.jsonl
    python merge_tutorials.py generated_projects_nodejs.jsonl generated_projects_mysql.jsonl
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

# Configuration
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
BASE_FILE = DATASETS_DIR / "generated_projects.jsonl"
OUTPUT_FILE = DATASETS_DIR / "generated_projects_final.jsonl"


def load_jsonl_file(filepath: Path) -> List[Dict]:
    """Load a JSONL file and return list of dictionaries."""
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return []
    
    items = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num} of {filepath}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return []
    
    return items


def merge_datasets(base_file: Path, tutorial_files: List[Path]) -> List[Dict]:
    """Merge base dataset with tutorial files."""
    
    print(f"\n{'='*60}")
    print("MERGING DATASETS")
    print(f"{'='*60}\n")
    
    # Load base dataset
    print(f"Loading base dataset: {base_file.name}...", end=" ", flush=True)
    merged_items = load_jsonl_file(base_file)
    print(f"✓ Loaded {len(merged_items)} items")
    
    # Track duplicates for statistics
    stats = defaultdict(int)
    stats["base_items"] = len(merged_items)
    
    # Create a set of existing file-project combinations for deduplication
    existing_entries: Set[tuple] = {
        (item.get("project"), item.get("file")) for item in merged_items
    }
    
    # Load and merge tutorial files
    total_added = 0
    for tutorial_file in tutorial_files:
        if not tutorial_file.exists():
            print(f"Warning: Tutorial file not found: {tutorial_file}")
            continue
        
        print(f"Loading tutorial file: {tutorial_file.name}...", end=" ", flush=True)
        tutorials = load_jsonl_file(tutorial_file)
        print(f"✓ Loaded {len(tutorials)} items")
        
        # Add new tutorials (avoid duplicates)
        added = 0
        duplicates = 0
        for tutorial in tutorials:
            entry_key = (tutorial.get("project"), tutorial.get("file"))
            if entry_key not in existing_entries:
                merged_items.append(tutorial)
                existing_entries.add(entry_key)
                added += 1
            else:
                duplicates += 1
        
        print(f"  → Added {added} items, skipped {duplicates} duplicates")
        total_added += added
        stats[f"tutorial_{tutorial_file.stem}"] = added
    
    print(f"\n{'='*60}")
    print(f"Total items in merged dataset: {len(merged_items)}")
    print(f"{'='*60}\n")
    
    return merged_items, stats


def save_merged_dataset(items: List[Dict], output_file: Path) -> bool:
    """Save merged dataset to output JSONL file."""
    
    # Create datasets directory if it doesn't exist
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Saving merged dataset to: {output_file.name}...", end=" ", flush=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"✓ Saved {len(items)} items")
        return True
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return False


def print_statistics(stats: Dict) -> None:
    """Print merge statistics."""
    print(f"\n{'='*60}")
    print("MERGE STATISTICS")
    print(f"{'='*60}")
    print(f"Base dataset items:   {stats.get('base_items', 0)}")
    for key, count in stats.items():
        if key != "base_items":
            print(f"Tutorial items added: {count}")
    print(f"{'='*60}\n")


def validate_output_file(output_file: Path) -> bool:
    """Validate the output file was created correctly."""
    if not output_file.exists():
        print(f"✗ Output file was not created: {output_file}")
        return False
    
    # Check file is valid JSONL
    try:
        count = 0
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)
                    count += 1
        print(f"✓ Output file validation passed ({count} valid JSON lines)")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ Output file contains invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"✗ Error validating output file: {e}")
        return False


def main():
    """Main function."""
    
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    # Get tutorial files from command line arguments
    tutorial_files = [DATASETS_DIR / filename for filename in sys.argv[1:]]
    
    # Merge datasets
    merged_items, stats = merge_datasets(BASE_FILE, tutorial_files)
    
    if not merged_items:
        print("✗ No items to save")
        sys.exit(1)
    
    # Save merged dataset
    if not save_merged_dataset(merged_items, OUTPUT_FILE):
        sys.exit(1)
    
    # Validate output
    if not validate_output_file(OUTPUT_FILE):
        sys.exit(1)
    
    # Print statistics
    print_statistics(stats)
    
    print(f"✓ Merge completed successfully!")
    print(f"✓ Output file: {OUTPUT_FILE}")
    print(f"✓ Ready to use with v2_train_model.py\n")


if __name__ == "__main__":
    main()
