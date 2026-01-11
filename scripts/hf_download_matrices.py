# import torch
# from transformers import AutoModel, AutoModelForCausalLM
# import numpy as np
# from huggingface_hub import hf_hub_download
# from safetensors.torch import load_file
# import os

#     # "ResNet_FC":  ("resnet50", "fc.weight"), # Torchvision model
#     # "BERT_Query": ("bert-base-uncased", "encoder.layer.0.attention.self.query.weight"),
# # Configuration
# MODELS = {
#     "ResNet_FC": ("microsoft/resnet-50", "classifier.1.weight"),
#     "ViT_MLP": ("google/vit-base-patch16-224", "vit.encoder.layer.0.intermediate.dense.weight"),
#     "BERT_Query": ("google-bert/bert-base-uncased", "bert.encoder.layer.0.attention.self.query.weight"),
#     "Llama_MLP": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "model.layers.0.mlp.down_proj.weight"),
#     "gpt2_wte": ("openai-community/gpt2", "wte.weight"),
# }

# def export_layer_hf_efficient(model_id, target_tensor_name, output_filename, transpose=True):
#     print(f"1. Locating {target_tensor_name} in {model_id}...")
    
#     # Step A: Download the index file to find which 'shard' contains our layer
#     # (Large models are split into multiple files: model-0001.safetensors, etc.)
#     try:
#         index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
#         import json
#         with open(index_path, 'r') as f:
#             index = json.load(f)
        
#         # Find which file contains our tensor
#         if target_tensor_name in index["weight_map"]:
#             target_file = index["weight_map"][target_tensor_name]
#             print(f" -> Found tensor in file: {target_file}")
#         else:
#             print("Tensor name not found in index.")
#             return
#     except:
#         # Fallback for smaller models that aren't sharded (like TinyLlama)
#         print(" -> Model appears to be single-file (not sharded).")
#         target_file = "model.safetensors"

#     # Step B: Download ONLY that specific shard (e.g., 1GB instead of 15GB)
#     print(f"2. Downloading shard: {target_file} (this might take a moment)...")
#     file_path = hf_hub_download(repo_id=target_tensor_name, filename=target_file)

#     # Step C: Load ONLY the specific tensor from the file
#     print("3. Extracting tensor from file...")
#     # safetensors allows loading just one key!
#     try:
#         from safetensors import safe_open
#         with safe_open(file_path, framework="pt", device="cpu") as f:
#             tensor = f.get_tensor(target_tensor_name)
#     except ImportError:
#         # Fallback if it's a pytorch .bin file
#         weights = torch.load(file_path, map_location="cpu")
#         tensor = weights[target_tensor_name]
#         del weights # Free RAM immediately

#     # Step D: Save for C++
#     print(f"4. Processing {tensor.shape}...")
#     # We always convert to FP32
#     if transpose:
#         data = tensor.t().contiguous().float().numpy()
#     else:
#         data = tensor.float().numpy()
    
#     rows, cols = data.shape
    
#     # Save as: [Rows(int32)] [Cols(int32)] [Raw Data...]
#     with open(output_filename, "wb") as f:
#         np.array([rows, cols], dtype=np.int32).tofile(f)
#         data.tofile(f)
        
#     print(f"SUCCESS: Saved to {output_filename} ({os.path.getsize(output_filename)/1024/1024:.2f} MB)")

# def export_layer(model_name, layer_name, output_file):
#     print(f"Loading {model_name}...")
#     try:
#         # Load model with checking for SafeTensors/Bin
#         if "resnet" in model_name:
#             import torchvision
#             model = torchvision.models.resnet50(pretrained=True)
#         else:
#             model = AutoModel.from_pretrained(model_name, torch_dtype="auto")
        
#         # Extract weight
#         state_dict = model.state_dict()
#         if layer_name not in state_dict:
#             print(f"Layer {layer_name} not found. Available keys: {list(state_dict.keys())[:5]}...")
#             return

#         weight = state_dict[layer_name].float().numpy() # Convert to FP32 for generic usage
#         rows, cols = weight.shape
#         print(f"Extracted {layer_name}: {rows}x{cols} (Density: {np.count_nonzero(weight)/weight.size:.2f})")

#         # Save as raw binary (Row-Major)
#         # File format: [Rows (int32)][Cols (int32)][Data (float32)...]
#         with open(output_file, "wb") as f:
#             np.array([rows, cols], dtype=np.int32).tofile(f)
#             weight.tofile(f)
#         print(f"Saved to {output_file}")

#     except Exception as e:
#         print(f"Error: {e}")

import torch
from transformers import AutoModel, AutoModelForImageClassification
import numpy as np

# Configuration
# Note: I have corrected the layer names below. 
# 1. 'vit.', 'bert.', and 'model.' prefixes are removed because AutoModel loads the backbone directly.
# 2. ResNet is the exception: 'classifier' is a Head layer, so we must load the Classification model.

MODELS = {
    # ResNet: The FC layer is in the "Head", not the backbone. 
    # We must keep the name 'classifier.1.weight' and handle the loading specifically.
    "ResNet_FC":  ("microsoft/resnet-50", "classifier.1.weight"),
    
    # ViT: In AutoModel, the prefix 'vit.' is gone. 
    # Structure: encoder -> layer -> [i] -> intermediate -> dense
    "ViT_MLP":    ("google/vit-base-patch16-224", "encoder.layer.0.intermediate.dense.weight"),
    
    # BERT: In AutoModel, the prefix 'bert.' is gone.
    "BERT_Query": ("google-bert/bert-base-uncased", "encoder.layer.0.attention.self.query.weight"),
    
    # Llama: In AutoModel, the prefix 'model.' is gone. Keys start at 'layers'.
    "Llama_MLP":  ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layers.0.mlp.down_proj.weight"),
    
    # GPT2: 'wte' (Token Embeddings) is part of the base model.
    "gpt2_wte":   ("openai-community/gpt2", "wte.weight"),
}

def export_layer(model_name, layer_name, output_file):
    print(f"Loading {model_name}...")
    try:
        # LOGIC:
        # If we need the final classification layer (like ResNet FC), we must use
        # AutoModelForImageClassification because AutoModel strips the head.
        if "resnet" in model_name.lower() or "classifier" in layer_name:
            model = AutoModelForImageClassification.from_pretrained(model_name)
        else:
            # For everything else (Internal layers, Projections, Embeddings),
            # AutoModel is cleaner and standardizes the names (removes 'bert.', 'vit.' prefixes).
            model = AutoModel.from_pretrained(model_name)
        
        # Extract weight
        state_dict = model.state_dict()
        if layer_name not in state_dict:
            print(f"❌ Layer '{layer_name}' not found.")
            # Help debug by finding close matches or listing roots
            roots = set(k.split('.')[0] for k in state_dict.keys())
            print(f"   Available root keys in this model: {roots}")
            print(f"   First 3 full keys: {list(state_dict.keys())[:3]}")
            return

        weight = state_dict[layer_name].float().t().numpy() # Convert to FP32 and transpose
        rows, cols = weight.shape
        density = np.count_nonzero(weight) / weight.size
        print(f"    Extracted {layer_name}: {rows}x{cols} (Density: {density:.2f})")

        # Save as raw binary (Row-Major)
        # File format: [Rows (int32)][Cols (int32)][Data (float32)...]
        with open(output_file, "wb") as f:
            np.array([rows, cols], dtype=np.int32).tofile(f)
            weight.tofile(f)
        print(f"   Saved to {output_file}\n")

    except Exception as e:
        print(f"Error loading {model_name}: {e}\n")

# Run all models
if __name__ == "__main__":
    for key, (model_id, layer_name) in MODELS.items():
        print(f"--- Processing {key} ---")
        export_layer(model_id, layer_name, f"matrices_data/{key.lower()}.bin")