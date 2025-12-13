import torch
from transformers import AutoModel, AutoModelForCausalLM
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os

# Configuration
MODELS = {
    "BERT_Query": ("bert-base-uncased", "encoder.layer.0.attention.self.query.weight"),
    "Llama_MLP":  ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "model.layers.0.mlp.down_proj.weight"),
    "ResNet_FC":  ("resnet50", "fc.weight") # Torchvision model
}

def export_layer_hf_efficient(model_id, target_tensor_name, output_filename):
    print(f"1. Locating {target_tensor_name} in {model_id}...")
    
    # Step A: Download the index file to find which 'shard' contains our layer
    # (Large models are split into multiple files: model-0001.safetensors, etc.)
    try:
        index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
        import json
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        # Find which file contains our tensor
        if target_tensor_name in index["weight_map"]:
            target_file = index["weight_map"][target_tensor_name]
            print(f" -> Found tensor in file: {target_file}")
        else:
            print("Tensor name not found in index.")
            return
    except:
        # Fallback for smaller models that aren't sharded (like TinyLlama)
        print(" -> Model appears to be single-file (not sharded).")
        target_file = "model.safetensors"

    # Step B: Download ONLY that specific shard (e.g., 1GB instead of 15GB)
    print(f"2. Downloading shard: {target_file} (this might take a moment)...")
    file_path = hf_hub_download(repo_id=target_tensor_name, filename=target_file)

    # Step C: Load ONLY the specific tensor from the file
    print("3. Extracting tensor from file...")
    # safetensors allows loading just one key!
    try:
        from safetensors import safe_open
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(target_tensor_name)
    except ImportError:
        # Fallback if it's a pytorch .bin file
        weights = torch.load(file_path, map_location="cpu")
        tensor = weights[target_tensor_name]
        del weights # Free RAM immediately

    # Step D: Save for C++
    print(f"4. Processing {tensor.shape}...")
    # Ensure it is Float32 for your C++ baseline, or keep as is.
    # Note: Llama weights are usually BF16. Converting to Float32 for standard export.
    data = tensor.float().numpy()
    
    rows, cols = data.shape
    
    # Save as: [Rows(int32)] [Cols(int32)] [Raw Data...]
    with open(output_filename, "wb") as f:
        np.array([rows, cols], dtype=np.int32).tofile(f)
        data.tofile(f)
        
    print(f"SUCCESS: Saved to {output_filename} ({os.path.getsize(output_filename)/1024/1024:.2f} MB)")

def export_layer(model_name, layer_name, output_file):
    print(f"Loading {model_name}...")
    try:
        # Load model with checking for SafeTensors/Bin
        if "resnet" in model_name:
            import torchvision
            model = torchvision.models.resnet50(pretrained=True)
        else:
            model = AutoModel.from_pretrained(model_name, torch_dtype="auto")
        
        # Extract weight
        state_dict = model.state_dict()
        if layer_name not in state_dict:
            print(f"Layer {layer_name} not found. Available keys: {list(state_dict.keys())[:5]}...")
            return

        weight = state_dict[layer_name].float().numpy() # Convert to FP32 for generic usage
        rows, cols = weight.shape
        print(f"Extracted {layer_name}: {rows}x{cols} (Density: {np.count_nonzero(weight)/weight.size:.2f})")

        # Save as raw binary (Row-Major)
        # File format: [Rows (int32)][Cols (int32)][Data (float32)...]
        with open(output_file, "wb") as f:
            np.array([rows, cols], dtype=np.int32).tofile(f)
            weight.tofile(f)
        print(f"Saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run
    export_layer(*MODELS["BERT_Query"], "matrices_data/bert_query.bin")
    # export_layer_hf_efficient(*MODELS["Llama_MLP"], "llama_mlp.bin")
    # export_layer(*MODELS["ResNet_FC"], "resnet_fc.bin")
