# from huggingface_hub import snapshot_download
# from pathlib import Path

# import os

# snapshot_download(repo_id="mistralai/Mistral-7B-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir="mistral_models/7B-v0.3")


from huggingface_hub import snapshot_download
snapshot_download(repo_id="mistralai/Mistral-Large-Instruct-2407", allow_patterns=["params.json", "consolidated-*.safetensors", "tokenizer.model.v3"], local_dir="mistral_models/Large-Instruct")



# from huggingface_hub import snapshot_download

# snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir="mistral_models/7B-Instruct-v0.3")