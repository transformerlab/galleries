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
        print(e)
        exit(1)


def read_and_combine_json_files(directory: str):
    print(f'Combining JSON files in {directory} directory')

    if not os.path.exists(directory):
        return

    # Get all the JSON files in the subdirectories
    files_json: list[str] = [f for f in os.listdir(
        path=directory) if (f.endswith('.json') or f.endswith('.yaml'))]

    # Combine the JSON files into a single dictionary
    combined_json = []

    for file in files_json:
        with open(file=os.path.join(directory, file), mode='r') as f:
            print(f' - Processing {file}')
            if file.endswith('.yaml'):
                file_contents = yaml.load(stream=f, Loader=yaml.FullLoader)
                combined_json.append(file_contents)
            else:
                file_contents = json.load(fp=f)
                # The files contain an array of JSON objects, so we need to iterate through them
                for sub_object in file_contents:
                    combined_json.append(sub_object)

    # Validate the combined_json to see if it is valid JSON:
    try:
        json.dumps(obj=combined_json)
        print('JSON is valid')
    except Exception as e:
        print(e)
        exit(1)

    # Validate the combined_json against the schema
    schema = json.load(fp=open(file=f'schemas/{directory}.json', mode='r'))
    validate_json(json=combined_json, schema=schema)

    # Write the models out
    with open(file=f'{directory}.json', mode='w') as f:
        json.dump(obj=combined_json, fp=f, indent=4)

    print('---')


read_and_combine_json_files(directory='models')
read_and_combine_json_files(directory='datasets')
read_and_combine_json_files(directory='plugins')
read_and_combine_json_files(directory='prompts')
