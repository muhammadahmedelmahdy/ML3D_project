import sys
from pathlib import Path

# 1. Dynamically append the 'src' directory to Python's search path
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import os
import json

# 2. Clean import without the 'src.' prefix and without '.py'
from project6_qwen_pipeline.raw_layout import extract_raw_partnet_object

def batch_preprocess(raw_dataset_dir: str, output_jsonl_path: str):
    raw_root = Path(raw_dataset_dir)
    output_file = Path(output_jsonl_path)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    print(f"Starting batch preprocessing from: {raw_root}")
    
    with output_file.open("w", encoding="utf-8") as f_out:
        # Iterate through every item in the dataset directory
        for item in sorted(raw_root.iterdir()):
            # Look for subdirectories containing the vital PartNet files
            if item.is_dir() and (item / "mobility.urdf").is_file():
                try:
                    # 1. Runs the parsing, kinematic transform, Z-up flip, and [-0.5, 0.5] normalization
                    record = extract_raw_partnet_object(item)
                    
                    # 2. Convert to a compact, single-line JSON string
                    json_line = json.dumps(record, separators=(",", ":"))
                    
                    # 3. Write into the centralized layouts.jsonl file
                    f_out.write(json_line + "\n")
                    success_count += 1
                    
                    if success_count % 10 == 0:
                        print(f"Successfully processed {success_count} objects...")
                        
                except Exception as e:
                    print(f"Skipping object {item.name} due to error: {e}")
                    error_count += 1

    print("\n--- Preprocessing Complete ---")
    print(f"Successfully extracted: {success_count} objects.")
    print(f"Failed/Skipped: {error_count} objects.")
    print(f"Saved database to: {output_file.resolve()}")

if __name__ == "__main__":
    # Update these paths to match your cluster environment location
    RAW_DATASET = "/cluster/52/jonasclotten/shared/project6/data/raw/dataset"
    OUTPUT_JSONL = "/cluster/52/jonasclotten/shared/project6/data/processed/layouts.jsonl"
    
    batch_preprocess(RAW_DATASET, OUTPUT_JSONL)