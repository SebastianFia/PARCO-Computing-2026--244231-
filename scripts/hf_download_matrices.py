import torch
from transformers import AutoModel, AutoModelForImageClassification
import numpy as np

# Configuration
MODELS = {
    # ResNet: The FC layer is in the "Head", not the backbone. 
    # We must keep the name 'classifier.1.weight' and handle the loading specifically.
    "ResNet_FC":  ("microsoft/resnet-50", "classifier.1.weight"),
    
    "ViT_MLP":    ("google/vit-base-patch16-224", "encoder.layer.0.intermediate.dense.weight"),
    
    "BERT_Query": ("google-bert/bert-base-uncased", "encoder.layer.0.attention.self.query.weight"),
    
    "Llama_MLP":  ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layers.0.mlp.down_proj.weight"),
    
    "gpt2_wte":   ("openai-community/gpt2", "wte.weight"),
}

def export_layer(model_name, layer_name, output_file):
    print(f"Loading {model_name}...")
    try:
        # If we need the final classification layer (like ResNet FC), we must use
        # AutoModelForImageClassification because AutoModel strips the head.
        if "resnet" in model_name.lower() or "classifier" in layer_name:
            model = AutoModelForImageClassification.from_pretrained(model_name)
        else:
            # For everything else (Internal layers, Projections, Embeddings),
            model = AutoModel.from_pretrained(model_name)
        
        # Extract weight
        state_dict = model.state_dict()
        if layer_name not in state_dict:
            print(f"Layer '{layer_name}' not found.")
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