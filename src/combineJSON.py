"""
This script combines all the JSON files in the models, plugins, and datasets folders
and publishes them to the root of the repository.
"""
import os
import json
import yaml
from jsonschema import validate, ValidationError


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

    # Get all the JSON files in the subdirectories
    files_json: list[str] = [f for f in os.listdir(
        path=directory) if (f.endswith('.json') or f.endswith('.yaml'))]
    
    # sort the files by name:
    files_json.sort()

    # Combine the JSON files into a single dictionary
    combined_json = []

    for file in files_json:
        with open(file=os.path.join(directory, file), mode='r') as f:
            print(f' - Processing {file}')
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

    # Validate the combined_json against the schema
    schema = json.load(fp=open(file=f'schemas/{directory}.json', mode='r'))
    validate_json(json=combined_json, schema=schema)

    check_for_duplicate_ids(combined_json)

    # generate the name of the file which matches what
    # the API expects. e.g. plugins -> plugin-gallery.json
    filename = directory[:-1]     # remove the tailing s:
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

        # Assign models to their group
        for model in combined_json:
            group_name = model.get("model_group")
            if group_name and group_name in model_groups:
                model_groups[group_name]["models"].append(model)
            else:
                print(f"Warning: Model '{model.get('name')}' missing or unknown group '{group_name}', adding to others")
                model_groups["others"]["models"].append(model)

        # Remove "others" group if it ended up empty
        if not model_groups["others"]["models"]:
            del model_groups["others"]

        # Write model-group-gallery.json
        with open('model-group-gallery.json', 'w') as f:
            json.dump(list(model_groups.values()), f, indent=4)

        print('---')


read_and_combine_json_files(directory='models')
read_and_combine_json_files(directory='datasets')
read_and_combine_json_files(directory='plugins')
read_and_combine_json_files(directory='prompts')
read_and_combine_json_files(directory='recipes')
read_and_combine_json_files(directory='exp-recipes')
