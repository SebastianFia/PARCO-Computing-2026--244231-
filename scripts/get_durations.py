import os
import subprocess
import re
import sys
import shutil
import numpy as np

# Get experiment info from env variables
N_THREADS = os.environ.get("OMP_NUM_THREADS")
MTX_FILE_PATH = os.environ.get("MTX_FILE_PATH")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
TIMING_RUNS = int(os.environ.get("TIMING_RUNS"))
VALGRIND_RUNS = int(os.environ.get("VALGRIND_RUNS"))
MATRIX_NAME = os.path.basename(MTX_FILE_PATH)

# Parameter Sweep
algorithms = ["csr"]
if N_THREADS == "1":
    schedules = [("NA", "NA")]
else:
    schedules = [("static", "1"), ("static", "64"), ("dynamic", "10"), ("dynamic", "100"), ("guided", "NA")]

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Folder to which we redirect valgrind output
VALGRIND_OUT_FOLDER="./.valgrind_output"
os.makedirs(VALGRIND_OUT_FOLDER, exist_ok=True)

# C executable to run
C_PROGRAM = "./build/main" 

# --- Prepare CSV ---
csv_file = os.path.join(OUTPUT_DIR, f"results_{MATRIX_NAME}_{N_THREADS}t.csv")
# Write header only if file doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write(
            "matrix,alg,schedule,chunk,threads,"
            "time_ms,D1_miss_rate,LLd_miss_rate\n"
        )

def parse_valgrind_output(valgrind_out_str):
    d1_miss_rate = None
    lld_miss_rate = None

    # Regex to capture the full line for D1 miss rate
    # Pattern: D1  miss rate:         X.X% ...
    d1_pattern = re.compile(r"^==\d+==\s+D1\s+miss\s+rate:\s+([\d\.]+)\s*%", re.MULTILINE)
    
    # Regex to capture the full line for LLd miss rate
    # Pattern: LLd misses rate:        X.X% ...
    lld_pattern = re.compile(r"^==\d+==\s+LLd\s+miss\s+rate:\s+([\d\.]+)\s*%", re.MULTILINE)

    # --- 1. Extract D1 Miss Rate ---
    d1_match = d1_pattern.search(valgrind_out_str)
    if d1_match:
        try:
            # Convert the matched string to a float 
            # Divide by 100 to convert from percentage
            # Then convert back to string
            d1_miss_rate = float(d1_match.group(1)) / 100.
        except ValueError:
            pass # Keep as NA if conversion fails

    # --- 2. Extract LLd Miss Rate ---
    lld_match = lld_pattern.search(valgrind_out_str)
    if lld_match:
        try:
            lld_miss_rate = float(lld_match.group(1)) / 100.
        except ValueError:
            pass # Keep as NA if conversion fails
    
    return d1_miss_rate, lld_miss_rate

# --- Run the Sweep ---
for alg in algorithms:

    # The csb algorithm doesn't rely on OMP's schedules
    tmp_schedules = [("NA", "NA")] if alg=="csb" else schedules 

    for (sched, chunk) in tmp_schedules:
        # --- 1. Set env vars for C program ---
        env = os.environ.copy()
        env["ALG_NAME"] = alg

        if (sched == "NA"):
            pass # Leave schedule env var as it is
        elif (chunk == "NA"):
            env["OMP_SCHEDULE"] = f"{sched}"
        else:
            env["OMP_SCHEDULE"] = f"{sched},{chunk}"

        # --- 2. Run without valgrind for TIME ---
        times = []
        
        # Run a single warmup run (discard output)
        subprocess.run([C_PROGRAM], capture_output=True, env=env)

        # Run TIMING_RUNS times for timings
        for _ in range(TIMING_RUNS):
            result_time = subprocess.run(
                [C_PROGRAM], capture_output=True, text=True, env=env
            )
            
            if result_time.returncode != 0:
                print(f"ERROR running {alg},{sched},{chunk}:", file=sys.stderr)
                print(result_time.stderr, file=sys.stderr)
                times = [] # Clear times on error
                break
            
            # Append the single time float from C stdout
            try:
                times.append(float(result_time.stdout.strip()))
            except ValueError:
                print(f"ERROR: C program did not output a float.", file=sys.stderr)
                times = []
                break
        
        # Calculate 90th percentile if we have data
        if times:
            time_90p_ms = f"{np.percentile(times, 90):.4f}"
        else:
            time_90p_ms = "NA"


        # --- 3. Run with valgrind for CACHE MISS RATES ---
        d1_miss_list = []
        lld_miss_list = []
        
        # Loop VALGRIND_RUNS times for valgrind
        for _ in range(VALGRIND_RUNS):
            cmd_valgrind = [
                "valgrind", "--tool=callgrind", "--cache-sim=yes",
                f"--callgrind-out-file={VALGRIND_OUT_FOLDER}/spmv.%p",
                C_PROGRAM
            ]

            # The command will look like this:
            # valgrind --tool=callgrind --cache-sim=yes --callgrind-out-file=./.valgrind_output/spmv.%p build/main

            result_cache = subprocess.run(
                cmd_valgrind, capture_output=True, text=True, env=env
            )
        
            if result_cache.returncode != 0:
                print(f"ERROR running Valgrind {alg},{sched},{chunk}:", file=sys.stderr)
                print(result_cache.stderr, file=sys.stderr)
                d1_miss_list = [] # Wipe list on error
                lld_miss_list = []
                break
            
            # Parse the misses for this run
            d1_miss_rate, lld_miss_rate = parse_valgrind_output(result_cache.stderr)

            # Add to our lists (as numbers)
            if d1_miss_rate == None or lld_miss_rate == None:
                d1_miss_list = [] # Wipe list on failed parsing
                lld_miss_list = []
                break
            else:
                d1_miss_list.append(d1_miss_rate)
                lld_miss_list.append(lld_miss_rate)

        # Now, calculate the 90th percentile of the miss rates
        if d1_miss_list and lld_miss_list:
            d1_90p_miss_rate = f"{np.percentile(d1_miss_list, 90):.3f}"
            lld_90p_miss_rate = f"{np.percentile(lld_miss_list, 90):.3f}"
        else:
            d1_90p_miss_rate = "NA"
            lld_90p_miss_rate = "NA"
        
        # --- 4. Write final CSV row ---
        csv_row = (
            f"{MATRIX_NAME},{alg},{sched},{chunk},{N_THREADS},"
            f"{time_90p_ms},{d1_90p_miss_rate},{lld_90p_miss_rate}\n"
        )
        with open(csv_file, "a") as f:
            f.write(csv_row) 


print(f"Finished job for {MATRIX_NAME} with {N_THREADS} threads.")