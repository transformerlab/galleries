import os
from huggingface_hub import hf_hub_download, list_repo_tree
from huggingface_hub.hf_api import RepoFile
import yaml
import json
import fnmatch
import argparse
from datetime import datetime

# take in one paramter called model_id as an argument:
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    type=str,
    help="The model id in huggingface e.g. Qwen/Qwen2-VL-7B-Instruct you can also pass a group of models, comma separated",
    required=True,
)
# add a paramter that is the output filename you want:
parser.add_argument(
    "--output_file_name",
    type=str,
    help='The short output file name e.g. "qwen"',
    required=True,
)
parser.add_argument(
    "--format",
    type=str,
    required=False,
    default="yaml",
    help='The format of the output file e.g. "yaml" or "json"',
)
args = parser.parse_args()
model_id = args.model_id
output_file_name = args.output_file_name
format = args.format

# check if model_id is an array of model ids separated by commas:
if "," in model_id:
    models = model_id.split(",")
    #
else:
    models = [model_id]

model_objects = []

for model_id in models:
    # verify it is a valid model id
    if "/" not in model_id:
        print("Invalid model id")
        pass

    print("-----------MODEL ID")
    print(model_id)
    # end part of model name is the part after the / in the model name
    model_author = model_id.split("/")[0]
    model_name = model_id.split("/")[1]

    card = hf_hub_download(model_id, "README.md")
    config = hf_hub_download(model_id, "config.json")

    # card is the README.md file, open it and print to screen:
    # with open(card, 'r') as file:
    #     print(file.read())

    print("-----------MODEL CARD")
    config_json = None
    with open(config, "r") as file:
        config_json = json.load(file)
        print(config_json)

    readme = None
    with open(card, "r") as file:
        readme = file.read()

    print("-----------MODEL DETAILS")
    architecture = config_json.get("architectures", [""])[0]
    print(architecture)
    context_length = config_json.get("max_position_embeddings", "2048")
    context_length = str(context_length)
    print(context_length)
    transformers_version = config_json.get("transformers_version", "")
    print(transformers_version)

    # gated information is USUALLY stored in config
    # But newer models seem to not set this and then set a custom
    # gated prompt on the model card
    gated = config_json.get("gated", False)
    if not gated:
        for line in readme.split("\n"):
            if line.startswith("extra_gated_prompt:") or line.startswith(
                "extra_gated_fields:"
            ):
                gated = "manual"
                break
    print(gated)

    # parse the readme file and see if at the start there is metadata
    # that is set like this:
    # license: apache-2.0
    # and get the value of the second part:
    license = None
    for line in readme.split("\n"):
        if line.startswith("license:"):
            license = line.split("license:")[1].strip()
            print(license)
            break

    # Date added is used to show that this is a newly added model
    date_added = datetime.now().strftime("%Y-%m-%d")

    # Try to figure out parameter count for the model by scanning through the name
    # Look for a part in the model string that is numbers followed by a B or an M
    parameter_count = ""
    model_name_parts = model_name.split("-")
    for part in model_name_parts:
        if part[:-1].replace(".", "").isdigit() and (
            part[-1].lower() == "b" or part[-1].lower() == "m"
        ):
            parameter_count = part.upper()
            break

    # Calculate download size from HuggingFace
    download_size = 0

    # This is our default allow_patterns defined in download_huggingface_model.py
    allow_patterns = [
        "*.json",
        "*.safetensors",
        "*.py",
        "tokenizer.model",
        "*.tiktoken",
        "*.npz",
        "*.bin",
    ]

    # Get a list of all files in this repo. This can throw RepositoryNotFoundError
    model_files = list_repo_tree(model_id)

    # Iterate over files in the model repo and add up size if they are included in download
    for file in model_files:
        if isinstance(file, RepoFile):
            # Only add this file if it matches one of the allow_patterns
            for pattern in allow_patterns:
                if fnmatch.fnmatch(file.path, pattern):
                    # Our gallery expects sizes in MB but the file size is in bytes
                    size_in_mb = file.size
                    download_size += size_in_mb / (1024 * 1024)
                    break

    # Try to figure out logo based on model info
    lc_model_name = model_name.lower()
    lc_architecture = architecture.lower()
    logo = ""
    if "cohere" in lc_architecture:
        logo = "https://avatars.githubusercontent.com/u/54850923?s=200&v=4"
    elif "deepseek" in lc_model_name:
        logo = "https://www.deepseek.com/favicon.ico"
    elif "exaone" in lc_architecture:
        logo = "https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-2.4B/resolve/main/assets/EXAONE_Symbol+BI_3d.png"
    elif "gemma" in lc_architecture:
        logo = "https://storage.googleapis.com/gweb-uniblog-publish-prod/images/gemma-header.width-1200.format-webp.webp"
    elif "mistral" in lc_architecture:
        logo = "https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    elif "qwen" in lc_architecture:
        logo = "https://cdn-avatars.huggingface.co/v1/production/uploads/62088594a5943c8a8fc94560/y5SEKiE8TkjBKs9xfjCx5.png"
    elif "smol" in lc_model_name:
        logo = "https://webml-community-smollm-webgpu.static.hf.space/logo.png"
    elif "vicuna" in lc_model_name:
        logo = "https://lmsys.org/images/blog/vicuna/vicuna.jpeg"

    # Do these later because I coulddn't figure out a good check without false positives
    elif "phi-" in lc_model_name:
        logo = "https://blogs.microsoft.com/wp-content/uploads/prod/2012/08/8867.Microsoft_5F00_Logo_2D00_for_2D00_screen.jpg"
    elif "llama-" in lc_model_name:
        logo = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Meta-Logo.png"

    # Do these two last as we want to capture the base architecture first if possible
    elif "mlx" in architecture:
        logo = "https://ml-explore.github.io/mlx/build/html/_static/mlx_logo.png"
    elif "gguf" in architecture:
        logo = "https://user-images.githubusercontent.com/229941/224846830-71002ab8-6c0d-49bc-bfcc-669618bbcdbe.png"

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
    model_object["uniqueID"] = model_id
    model_object["name"] = model_name
    model_object["description"] = ""
    model_object["added"] = date_added
    model_object["tags"] = []
    model_object["parameters"] = parameter_count
    model_object["context"] = context_length
    model_object["architecture"] = architecture
    model_object["formats"] = ["Safetensors"]
    model_object["huggingface_repo"] = model_id
    model_object["transformers_version"] = transformers_version
    model_object["gated"] = gated
    model_object["license"] = license
    model_object["logo"] = logo
    model_object["size_of_model_in_mb"] = download_size
    model_object["author"] = {}
    model_object["author"]["name"] = model_author
    model_object["author"]["url"] = f"https://huggingface.co/{model_id}"
    model_object["author"]["blurb"] = ""
    model_object["resources"] = {}
    model_object["resources"]["canonicalUrl"] = f"https://huggingface.co/{model_id}"
    model_object["resources"]["downloadUrl"] = f"https://huggingface.co/{model_id}"
    model_object["resources"]["paperUrl"] = "?"
    model_object["model_group"] = ""

    print("-----------FINAL MODEL OBJECT")
    # print(yaml.dump(model_object, sort_keys=False))

    model_objects.append(model_object)


file_name = f"./models/{output_file_name}.{'yaml' if format == 'yaml' else 'json'}"
print("writing to: ", file_name)

append = False
# check if file exists already:
if os.path.exists(file_name):
    print("file exists, appending")
    append = True

if append:
    with open(file_name, "r") as file:
        if file_name.endswith(".yaml"):
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

if format == "json":
    print("writing as json")
    output = json.dumps(final_model_object, indent=2, sort_keys=False)
else:
    print("writing as yaml")
    output = yaml.dump(final_model_object, sort_keys=False)

with open(file_name, "w+") as file:
    file.write(output)
