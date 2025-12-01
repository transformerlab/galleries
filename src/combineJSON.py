"""
This script combines all the JSON files in the models, plugins, and datasets folders
and publishes them to the root of the repository.
"""
import os
import json
import yaml
import requests
import time
from jsonschema import validate, ValidationError
from update_dataset_sizes import update_dataset_sizes


# Global cache for URL checks to avoid duplicate requests
url_check_cache = {}


def validate_json(json, schema):
    try:
        validate(instance=json, schema=schema)
        print('JSON maps to schema')
    except ValidationError as e:
        print(f"ERROR: {e}")
        exit(1)

def check_for_duplicate_ids(combined_json):
    """Check for duplicate 'id' or 'uniqueID' fields in combined_json."""
    ids = set()
    for item in combined_json:
        id_value = item.get('id') or item.get('uniqueID')
        if id_value:
            if id_value in ids:
                print(f"ERROR: ‼️ Duplicate ID found: {id_value}")
                exit(1)
            ids.add(id_value)
    print('✅ No duplicate IDs found')


def check_url_reachable(url, timeout=10, max_retries=3):
    """Check if a URL is reachable (doesn't return 404 or other error)."""
    if not url or url == "?":
        return True  # Skip validation for placeholder URLs
    
    # Check cache first
    if url in url_check_cache:
        return url_check_cache[url]
    
    for attempt in range(max_retries):
        try:
            # Add delay between requests to avoid rate limiting
            if attempt > 0:
                wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                print(f"   ⏳ Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                # Small delay even on first request to be respectful
                time.sleep(0.5)
            
            # Try HEAD request first
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            
            # If HEAD returns 404, it's definitely unreachable
            if response.status_code == 404:
                print(f"   🔍 HEAD request returned 404 for: {url}")
                url_check_cache[url] = False
                return False
            
            # If we got rate limited (429), retry
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    print(f"   ⚠️  Still rate limited after {max_retries} attempts, assuming URL is valid: {url}")
                    url_check_cache[url] = True
                    return True
            
            # Some servers don't support HEAD, so try GET if HEAD fails
            if response.status_code >= 400:
                print(f"   🔍 HEAD failed ({response.status_code}), trying GET for: {url}")
                time.sleep(0.5)  # Extra delay before GET
                response = requests.get(url, timeout=timeout, allow_redirects=True)
                
                if response.status_code == 404:
                    print(f"   🔍 GET request returned 404 for: {url}")
                    url_check_cache[url] = False
                    return False
                
                # If we got rate limited on GET, retry
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        print(f"   ⚠️  Still rate limited after {max_retries} attempts, assuming URL is valid: {url}")
                        url_check_cache[url] = True
                        return True
                
                # For HuggingFace, check if the response contains 404 indicators in the HTML
                if 'huggingface.co' in url and response.status_code == 200:
                    content = response.text.lower()
                    if '404' in content or 'not found' in content or 'repository not found' in content:
                        print(f"   🔍 Found 404 indicators in page content for: {url}")
                        url_check_cache[url] = False
                        return False
            
            # Consider 2xx and 3xx as reachable
            result = response.status_code < 400
            url_check_cache[url] = result
            return result
            
        except requests.RequestException as e:
            # If we can't reach it due to network issues, assume it's unreachable
            print(f"   🔍 Request exception for {url}: {e}")
            url_check_cache[url] = False
            return False
    
    # If we exhausted all retries, assume it's unreachable
    url_check_cache[url] = False
    return False


def validate_model_urls(model):
    """Validate that model URLs are reachable. Returns True if valid, False otherwise."""
    resources = model.get('resources', {})
    canonical_url = resources.get('canonicalUrl')
    download_url = resources.get('downloadUrl')
    
    model_name = model.get('name', 'Unknown')
    model_id = model.get('id', model.get('uniqueID', 'Unknown'))
    
    # Check canonical URL
    if canonical_url and canonical_url != "?":
        if not check_url_reachable(canonical_url):
            print(f"⚠️  Warning: Model '{model_name}' (ID: {model_id}) has unreachable canonicalUrl: {canonical_url}")
            return False
    
    # Check download URL
    if download_url and download_url != "?":
        if not check_url_reachable(download_url):
            print(f"⚠️  Warning: Model '{model_name}' (ID: {model_id}) has unreachable downloadUrl: {download_url}")
            return False
    
    return True


def read_and_combine_json_files(directory: str):
    print(f'Combining JSON files in {directory} directory')
    
    # Check if directory exists relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    directory_path = os.path.join(parent_dir, directory)

    if not os.path.exists(directory_path):
        print(f'  ⚠️  Directory {directory_path} does not exist, skipping...')
        return

    # Get JSON/YAML files
    # Special handling for 'tasks': gather nested task definitions
    if directory == 'tasks':
        files_json: list[str] = []
        for root, _dirs, files in os.walk(directory_path):
            for f in files:
                if f == 'task.json':
                    files_json.append(os.path.join(root, f))
    else:
        files_json: list[str] = [f for f in os.listdir(
            path=directory_path) if (f.endswith('.json') or f.endswith('.yaml'))]
    
    # sort the files by name:
    files_json.sort()
    
    print(f'  📁 Found {len(files_json)} files to process')
    if len(files_json) == 0:
        print(f'  ⚠️  No JSON/YAML files found in {directory}')
        return

    # Combine the JSON files into a single dictionary
    combined_json = []

    for file in files_json:
        if directory == 'tasks':
            open_path = file
            display_name = os.path.relpath(file)
        else:
            open_path = os.path.join(directory_path, file)
            display_name = file
        with open(file=open_path, mode='r') as f:
            print(f' - Processing {display_name}')
            if file.endswith('.yaml'):
                file_contents = yaml.load(stream=f, Loader=yaml.FullLoader)
            else:
                file_contents = json.load(fp=f)

            if directory == 'tasks':
                # Transform task definitions into minimal gallery entries
                # Extract the directory name as the task ID
                task_dir = os.path.basename(os.path.dirname(open_path))
                
                if isinstance(file_contents, list):
                    for task_obj in file_contents:
                        minimal = {
                            'id': task_dir,
                            'name': task_obj.get('name'),
                            'description': task_obj.get('description'),
                            'tag': task_obj.get('tag'),
                        }
                        combined_json.append(minimal)
                else:
                    minimal = {
                        'id': task_dir,
                        'name': file_contents.get('name'),
                        'description': file_contents.get('description'),
                        'tag': file_contents.get('tag'),
                    }
                    combined_json.append(minimal)
            else:
                if isinstance(file_contents, list):
                    # Validate URLs for models directory
                    if directory == 'models':
                        valid_items = []
                        for item in file_contents:
                            if validate_model_urls(item):
                                valid_items.append(item)
                            else:
                                model_name = item.get('name', 'Unknown')
                                model_id = item.get('id', item.get('uniqueID', 'Unknown'))
                                print(f"   ❌ Skipping model '{model_name}' (ID: {model_id}) due to unreachable URLs")
                        combined_json.extend(valid_items)
                    else:
                        combined_json.extend(file_contents)
                else:
                    # Validate URLs for models directory
                    if directory == 'models':
                        if validate_model_urls(file_contents):
                            combined_json.append(file_contents)
                        else:
                            model_name = file_contents.get('name', 'Unknown')
                            model_id = file_contents.get('id', file_contents.get('uniqueID', 'Unknown'))
                            print(f"   ❌ Skipping model '{model_name}' (ID: {model_id}) due to unreachable URLs")
                    else:
                        combined_json.append(file_contents)

    # Validate the combined_json to see if it is valid JSON:
    try:
        json.dumps(obj=combined_json)
        print('JSON is valid')
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

    # Validate the combined_json against the schema (if present)
    schema_path = os.path.join(parent_dir, 'schemas', f'{directory}.json')
    if os.path.exists(schema_path):
        schema = json.load(fp=open(file=schema_path, mode='r'))
        validate_json(json=combined_json, schema=schema)
    else:
        print(f"No schema found at {schema_path}; skipping schema validation")

    check_for_duplicate_ids(combined_json)

    # generate the name of the file which matches what
    # the API expects. e.g. plugins -> plugin-gallery.json
    filename = directory[:-1]     # remove the tailing s:
    filename = f'{filename}-gallery.json'
    output_path = os.path.join(parent_dir, filename)
    
    # Write the models out
    with open(file=output_path, mode='w') as f:
        json.dump(obj=combined_json, fp=f, indent=4)

    print('---')

    if directory == 'models':
        # Load model group headers
        model_groups_dir = os.path.join(parent_dir, 'model-groups')
        model_groups = {}

        for file in os.listdir(model_groups_dir):
            if file.endswith('.json'):
                with open(os.path.join(model_groups_dir, file), 'r') as f:
                    try:
                        group_data = json.load(f)
                        group_data["models"] = []  # Reset
                        model_groups[group_data["name"]] = group_data
                    except Exception as e:
                        print(f"ERROR: Error reading model group {file}: {e}")

        # Check for blank model_group and assign models to their groups
        for model in combined_json:
            group_name = model.get("model_group")
            if not group_name or group_name.strip() == "":
                model_name = model.get('name', 'Unknown')
                model_id = model.get('id', model.get('uniqueID', 'Unknown'))
                print(f"ERROR: ‼️ Model '{model_name}' (ID: {model_id}) has a blank or missing 'model_group' field")
                exit(1)
            
            if group_name not in model_groups:
                model_name = model.get('name', 'Unknown')
                model_id = model.get('id', model.get('uniqueID', 'Unknown'))
                print(f"ERROR: ‼️ Model '{model_name}' (ID: {model_id}) has unknown 'model_group' '{group_name}'")
                exit(1)
            
            model_groups[group_name]["models"].append(model)

        # Write model-group-gallery.json
        with open(os.path.join(parent_dir, 'model-group-gallery.json'), 'w') as f:
            json.dump(list(model_groups.values()), f, indent=4)

        print('---')


read_and_combine_json_files(directory='models')

# Update dataset sizes from HuggingFace before combining datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
galleries_dir = os.path.dirname(script_dir)
preset_file = os.path.join(galleries_dir, "datasets", "preset.json")
if os.path.exists(preset_file):
    update_dataset_sizes(preset_file, quiet=True)
else:
    print(f"⚠️  Warning: Could not find preset.json at {preset_file}, skipping size update")

read_and_combine_json_files(directory='datasets')
read_and_combine_json_files(directory='plugins')
read_and_combine_json_files(directory='prompts')
read_and_combine_json_files(directory='recipes')
read_and_combine_json_files(directory='exp-recipes')
read_and_combine_json_files(directory='tasks')