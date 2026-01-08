"""
This script combines all the JSON files in the models, plugins, and datasets folders
and publishes them to the root of the repository.
"""
import os
import json
import shutil
import yaml
from jsonschema import validate, ValidationError
from update_dataset_sizes import update_dataset_sizes


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


def read_and_combine_json_files(directory: str):
    print(f'Combining JSON files in {directory} directory')

    if not os.path.exists(directory):
        return

    files_json: list[str] = [f for f in os.listdir(
        path=directory) if (f.endswith('.json') or f.endswith('.yaml'))]

    # sort the files by name:
    files_json.sort()

    # Combine the JSON files into a single dictionary
    combined_json = []

    for file in files_json:
        open_path = os.path.join(directory, file)
        display_name = file
        with open(file=open_path, mode='r') as f:
            print(f' - Processing {display_name}')
            if file.endswith('.yaml'):
                file_contents = yaml.load(stream=f, Loader=yaml.FullLoader)
            else:
                file_contents = json.load(fp=f)

            if isinstance(file_contents, list):
                combined_json.extend(file_contents)
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
    schema_path = f'schemas/{directory}.json'
    if os.path.exists(schema_path):
        schema = json.load(fp=open(file=schema_path, mode='r'))
        validate_json(json=combined_json, schema=schema)
    else:
        print(f"No schema found at {schema_path}; skipping schema validation")

    check_for_duplicate_ids(combined_json)

    # generate the name of the file which matches what
    # the API expects. e.g. plugins -> plugin-gallery.json
    filename = directory[:-1] if directory[-1] == 's' else directory    # remove the tailing s:
    filename = f'{filename}-gallery.json'
    
    # Write the models out
    with open(file=filename, mode='w') as f:
        json.dump(obj=combined_json, fp=f, indent=4)

    print('---')

    if directory == 'models':
        # Load model group headers
        model_groups_dir = 'model-groups'
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
        with open('model-group-gallery.json', 'w') as f:
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
read_and_combine_json_files(directory='interactive')

# Copy task-gallery.json to tasks-gallery.json for backward compatibility
if os.path.exists('task-gallery.json'):
    shutil.copyfile('task-gallery.json', 'tasks-gallery.json')
    print('Copied task-gallery.json to tasks-gallery.json for backward compatibility')
