import os
from huggingface_hub import hf_hub_download
import yaml
import json
import argparse

# take in one paramter called model_id as an argument:
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str,
                    help='The model id in huggingface e.g. Qwen/Qwen2-VL-7B-Instruct you can also pass a group of models, comma separated', required=True)
# add a paramter that is the output filename you want:
parser.add_argument('--output_file_name', type=str,
                    help='The short output file name e.g. "qwen"', required=True)
parser.add_argument('--format', type=str,required=False, default='yaml',
                    help='The format of the output file e.g. "yaml" or "json"')
args = parser.parse_args()
model_id = args.model_id
output_file_name = args.output_file_name
format = args.format

# check if model_id is an array of model ids separated by commas:
if ',' in model_id:
    models = model_id.split(',')
    # 
else:
    models = [model_id]

model_objects = []  

for model_id in models:
    # verify it is a valid model id
    if '/' not in model_id:
        print("Invalid model id")
        pass

    print("-----------MODEL ID")
    print(model_id)
    # end part of model name is the part after the / in the model name
    model_author = model_id.split('/')[0]
    model_name = model_id.split('/')[1]

    card = hf_hub_download(model_id, 'README.md')
    config = hf_hub_download(model_id, 'config.json')

    # card is the README.md file, open it and print to screen:
    # with open(card, 'r') as file:
    #     print(file.read())

    print("-----------MODEL CARD")
    config_json = None
    with open(config, 'r') as file:
        config_json = json.load(file)
        print(config_json)

    readme = None
    with open(card, 'r') as file:
        readme = file.read()

    print("-----------MODEL DETAILS")
    architecture = config_json['architectures'][0]
    print(architecture)
    context_length = config_json.get('max_position_embeddings', "2048")
    context_length = str(context_length)
    print(context_length)
    transformers_version = config_json['transformers_version']
    print(transformers_version)

    # gated information is USUALLY stored in config
    # But newer models seem to not set this and then set a custom
    # gated prompt on the model card
    gated = config_json.get('gated', False)
    if not gated:
        for line in readme.split('\n'):
            if line.startswith('extra_gated_prompt:') or line.startswith('extra_gated_fields:'):
                gated = "manual"
                break
    print(gated)

    # parse the readme file and see if at the start there is metadata
    # that is set like this:
    # license: apache-2.0
    # and get the value of the second part:
    license = None
    for line in readme.split('\n'):
        if line.startswith('license:'):
            license = line.split('license:')[1].strip()
            print(license)
            break

    # Now we want to make a file that looks like:
    # uniqueID: "HuggingFaceH4/zephyr-7b-alpha"
    # name: "Zephyr 7b Alpha"
    # description: "Zephyr is a series of language models that are trained to act as helpful assistants. Zephyr-7B-α is the first model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1 that was trained on on a mix of publicly available, synthetic datasets using Direct Preference Optimization (DPO)."
    # parameters: "7B"
    # context: "4k"
    # architecture: "MistralForCausalLM"
    # formats: ["PyTorch"]
    # huggingface_repo: "HuggingFaceH4/zephyr-7b-alpha"
    # transformers_version: "4.34.0"
    # gated: "auto"
    # license: "MIT"
    # logo: "https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png"
    # size_of_model_in_mb: 27628.1
    # author:
    #   name: "HuggingFace H4"
    #   url: "https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha"
    #   blurb: ""
    # resources:
    #   canonicalUrl: "https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha"
    #   downloadUrl: "https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha"
    #   paperUrl: "?"

    # using the info we have from the config.json file, we can fill in the fields above
    # and write them to a new file

    model_object = {}
    model_object['uniqueID'] = model_id
    model_object['name'] = model_name
    model_object['description'] = ""
    model_object['parameters'] = ""
    model_object['context'] = context_length
    model_object['architecture'] = architecture
    model_object['formats'] = ["Safetensors"]
    model_object['huggingface_repo'] = model_id
    model_object['transformers_version'] = transformers_version
    model_object['gated'] = gated
    model_object['license'] = license
    model_object['logo'] = ""
    model_object['size_of_model_in_mb'] = 0
    model_object['author'] = {}
    model_object['author']['name'] = model_author
    model_object['author']['url'] = f"https://huggingface.co/{model_id}"
    model_object['author']['blurb'] = ""
    model_object['resources'] = {}
    model_object['resources']['canonicalUrl'] = f"https://huggingface.co/{model_id}"
    model_object['resources']['downloadUrl'] = f"https://huggingface.co/{model_id}"
    model_object['resources']['paperUrl'] = "?"

    print("-----------FINAL MODEL OBJECT")
    #print(yaml.dump(model_object, sort_keys=False))

    model_objects.append(model_object)


file_name = f"./models/{output_file_name}.{'yaml' if format == 'yaml' else 'json'}"
print("writing to: ", file_name)

append = False
# check if file exists already:
if os.path.exists(file_name):
    print("file exists, appending")
    append = True

if append:
    with open(file_name, 'r') as file:
        if file_name.endswith('.yaml'):
            final_model_object = yaml.load(file, Loader=yaml.FullLoader)
        else:
            final_model_object = json.load(file)
    if isinstance(final_model_object, list):
        final_model_object.append(model_objects)
    else:
        final_model_object = [final_model_object, model_objects]
else: 
    final_model_object = model_objects

# if model_objects if of size 1, then we want to write it as a single object
if len(final_model_object) == 1:
    final_model_object = final_model_object[0]

output = ""

if format == 'json':
    print("writing as json")
    output = json.dumps(final_model_object, indent=2, sort_keys=False)
else:
    print("writing as yaml")
    output = yaml.dump(final_model_object, sort_keys=False)

with open(file_name, 'w+') as file:
    file.write(output)