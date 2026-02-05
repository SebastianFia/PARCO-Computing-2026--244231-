# Reproducibility Guide: Parallel AVX-512 GEMM (Project 10)
This repository contains the source code, experimental orchestration scripts, and visualization tools for the High-Performance GEMM project.

## 1. Cluster Execution Workflow
### Step A: Start Interactive Session
Request a node with 32 physical cores to observe full scaling and hyperthreading behavior:

```
qsub -I -q short_cpuQ -l select=1:ncpus=32:mem=16gb,walltime=02:00:00
```

Hardware Note: Cluster hardware assignment is inconsistent. After starting an interactive session, verify hardware support for VNNI immediately using the following command:

```
lscpu | grep avx512_vnni
```

If this returns no output, the node lacks VNNI. You must either restart the session to request a different node or run only bf16 experiments manually 
(to run experiments manually see section 2. Manual Driver Usage).

### Step B: Load Environment & Build
The implementation uses a Header-Only approach (also known as an "inline build") to allow the compiler to perform aggressive cross-unit SIMD optimizations and register-level inlining.

Load modules:
```
module load gcc91 Intel_oneAPI_Toolkit_2021.2 python-3.8.13
```

Build:
```
make clean && make
```

The executable is generated at: ./build/main_bench_driver

### Step C: Execute Automated Experiments
Run the full benchmark suite (Strong Scaling, Workload Diversity, and Batch Size Scaling):

```
python3 scripts/run_experiments.py
```

This script automates the analysis against the oneDNN baseline.

## 2. Manual Driver Usage
You can invoke the C++ driver directly for individual tests:

```
./build/main_bench_driver (matrix_name) (dtype) (impl) (M) (K) (N)
```

dtype: bf16 or int8

impl: ours or onednn

Output Format: CSV (matrix_name,dtype,impl,M,K,N,threads,median_time,throughput)

## 3. Plot Generation (Local Machine)
As the cluster environment lacks specialized visualization packages like Seaborn and Pandas, you must generate plots locally.

Transfer Data: scp (user)@(cluster_ip):/path/to/project/data/results.csv ./data/

Generate Visuals: 

```
python3 scripts/plot_results.py
```

Generated PDFs (scaling, batch scaling) are stored in the /plots directory.

## 4. Reference Matrix Workloads
Matrix shapes (K x N) are derived from real-world Neural Network layers. To verify these, visit huggingface.co, search the model ID, and inspect the model.safetensors file metadata.

| MODEL_ID | WEIGHT NAME | ASPECT RATIO |
| ------------------- | ------------------- | ------------ |
| microsoft/resnet-50 | classifier.1.weight | 2048 x 1000 |
| google/vit-base-patch16-224 | encoder.layer.0.intermediate.dense.weight | 768 x 3072 |
| google-bert/bert-base-uncased | encoder.layer.0.attention.self.query.weight | 768 x 768 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | layers.0.mlp.down_proj.weight | 2048 x 5632 |
| openai-community/gpt2 | wte.weight | 768 x 50257 |