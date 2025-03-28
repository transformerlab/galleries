# Transformer Lab Galleries

Galleries for Models, Datasets, and Plugins used by Transformer Lab

## How to add new Models, Datasets, or Plugins:

Do not modify the json files in the root directory directly. They are autogenerated using a script that looks at the JSON / YAML files in the models / datasets / plugins / prompts directories.

To add to the galleries:

1. Fork this repo, create a branch and add a new JSON file in the models/datasets/plugins/prompts folder, using exiting files as a template.
2. Run:
   `python3 src/combineJSON.py`
   to validate your addition.
3. If everything works, open a pull request with your addition.

## Adding models from Hugggingface:

There is a helper function to add models from huggingface. You can pass an array of models and it will grab them all and combine them. Here is a sample command:

```bash
python src/new_model_from_hf.py --model_id=google/gemma-3-1b-pt,google/gemma-3-1b-it,google/gemma-3-4b-pt,google/gemma-3-4b-it,google/gemma-3-12b-pt,google/gemma-3-12b-it,google/gemma-3-27b-pt,google/gemma-3-27b-it,google/shieldgemma-2-4b-it --format=json --output_file_name=gemma3
```
