"""
Script to update dataset sizes in the gallery from HuggingFace.
This script fetches the actual download_size (in bytes) from HuggingFace
and updates the size field in the dataset gallery files.
"""
import json
import os
from datasets import load_dataset_builder
from datasets.exceptions import DatasetNotFoundError


def get_dataset_download_size(dataset_id: str, config_name: str = None, dataset_config: str = None, quiet: bool = False):
    """
    Get the download size in bytes for a HuggingFace dataset.
    Returns the size in bytes, or -1 if not available.
    """
    try:
        # Handle dataset_config (used for datasets like Wikipedia)
        if dataset_config is not None:
            ds_builder = load_dataset_builder(dataset_id, dataset_config, trust_remote_code=True)
        elif config_name is not None:
            ds_builder = load_dataset_builder(path=dataset_id, name=config_name, trust_remote_code=True)
        else:
            ds_builder = load_dataset_builder(dataset_id, trust_remote_code=True)
        
        download_size = ds_builder.info.download_size
        if download_size is None:
            return -1
        return download_size
    except DatasetNotFoundError as e:
        if not quiet:
            print(f"  ⚠️  Dataset not found: {dataset_id} - {e}")
        return -1
    except Exception as e:
        if not quiet:
            print(f"  ⚠️  Error fetching size for {dataset_id}: {e}")
        return -1


def update_dataset_sizes(preset_file_path: str, quiet: bool = False):
    """
    Update the size field for all datasets in the preset.json file.
    
    Args:
        preset_file_path: Path to the preset.json file
        quiet: If True, reduce output verbosity (useful when called from other scripts)
    """
    print(f"Updating dataset sizes in: {preset_file_path}\n")
    # Read the current preset.json
    with open(preset_file_path, 'r') as f:
        datasets = json.load(f)
    
    if not quiet:
        print(f"Found {len(datasets)} datasets to update\n")
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for dataset in datasets:
        dataset_id = dataset.get("huggingfacerepo")
        current_size = dataset.get("size")
        config_name = dataset.get("config_name")
        dataset_config = dataset.get("dataset_config")
        
        if not dataset_id:
            if not quiet:
                print(f"⚠️  Skipping dataset {dataset.get('name', 'Unknown')}: No huggingfacerepo")
            skipped_count += 1
            continue
        
        if not quiet:
            print(f"Processing: {dataset.get('name', dataset_id)}")
            print(f"  Repo: {dataset_id}")
            print(f"  Current size: {current_size}")
        
        # Get the actual download size from HuggingFace
        download_size = get_dataset_download_size(dataset_id, config_name, dataset_config, quiet=quiet)
        
        if download_size == -1:
            if not quiet:
                print(f"  ❌ Could not fetch download size, keeping current value")
            error_count += 1
        else:
            # Convert to string to match the format of some entries (like "24200000")
            # Keep as integer for consistency
            dataset["size"] = download_size
            if not quiet:
                print(f"  ✅ Updated size: {download_size} bytes ({download_size / (1024*1024):.2f} MB)")
            updated_count += 1
        
        if not quiet:
            print()
    
    # Write the updated data back to the file
    with open(preset_file_path, 'w') as f:
        json.dump(datasets, f, indent=2)
    
    if not quiet:
        print("=" * 60)
        print(f"Summary:")
        print(f"  ✅ Updated: {updated_count}")
        print(f"  ⚠️  Errors: {error_count}")
        print(f"  ⏭️  Skipped: {skipped_count}")
        print(f"  📝 Total: {len(datasets)}")
        print("=" * 60)
    else:
        print(f"  ✅ Updated {updated_count} dataset sizes")


if __name__ == "__main__":
    # Get the path to preset.json relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    galleries_dir = os.path.dirname(script_dir)
    preset_file = os.path.join(galleries_dir, "datasets", "preset.json")
    
    if not os.path.exists(preset_file):
        print(f"Error: Could not find preset.json at {preset_file}")
        exit(1)
    
    print(f"Updating dataset sizes in: {preset_file}\n")
    update_dataset_sizes(preset_file)
    print("\n✅ Done! Run combineJSON.py to regenerate the gallery file.")

