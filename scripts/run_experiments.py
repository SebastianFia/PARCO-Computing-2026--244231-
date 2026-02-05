import subprocess
import os
import csv

# Configuration
BINARY_PATH = "./build/main_bench_driver"
OUTPUT_FILE = "data/results.csv"
THREADS_LIST = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
T_FIXED = 32
M_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
M_FIXED = 256
DTYPES = ["bf16", "int8"]
IMPLS = ["ours", "onednn"]

# Matrix definitions from real-world models
WORKLOADS = {
    "ResNet_FC":   (2048, 1000),
    "ViT_MLP":     (768, 3072),
    "BERT_Query":  (768, 768),
    "Llama_MLP":   (2048, 5632),
    "GPT2_WTE":    (768, 50257)
}

def run_bench(matrix_name, dtype, impl, m, k, n, threads):
    """Executes the C++ driver with the specified OMP_NUM_THREADS."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    
    cmd = [BINARY_PATH, matrix_name, dtype, impl, str(m), str(k), str(n)]
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running {matrix_name}: {e}")
        return None

def main():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        # Header for the CSV
        f.write("matrix_name,dtype,impl,M,K,N,threads,median_time,throughput\n")
        f.flush()

        # Strong Scaling Experiment 
        # Fixed Llama_MLP matrix, Fixed M=256, Sweep threads
        print("Running Strong Scaling Experiment...")
        k, n = WORKLOADS["Llama_MLP"]
        for t in THREADS_LIST:
            for dtype in DTYPES:
                for impl in IMPLS:
                    out = run_bench("Llama_MLP_Scaling", dtype, impl, M_FIXED, k, n, t)
                    if out: f.write(out + "\n")
        f.flush()

        # Workload Diversity
        # Fixed M=256, Fixed Threads=32, Sweep matrices
        print("Running Workload Diversity Experiment...")
        for name, (k, n) in WORKLOADS.items():
            for dtype in DTYPES:
                for impl in IMPLS:
                    out = run_bench(name, dtype, impl, M_FIXED, k, n, T_FIXED)
                    if out: f.write(out + "\n")
        f.flush()

        # Batch Size Efficiency (Roofline Data)
        # Fixed Llama_MLP, Fixed Threads=32, Sweep M
        print("Running Batch Size Sweep (Roofline)...")
        k, n = WORKLOADS["Llama_MLP"]
        for m in M_SWEEP:
            for dtype in DTYPES:
                for impl in IMPLS:
                    out = run_bench("Llama_MLP_BatchSweep", dtype, impl, m, k, n, T_FIXED)
                    if out: f.write(out + "\n")

    print(f"Experiments complete. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()