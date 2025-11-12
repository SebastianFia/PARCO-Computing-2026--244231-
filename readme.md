# PARCO-Computing-2026--244231-

## Requirements
To replicate the experiments presented in this repository make sure to run on a **Linux** machine and have installed:
* `gcc-9.1.0`
* `valgrind`
* `python-3.7.2` (we run the scripts using python3)
* `make` (for building the c program with makefile)

## Run expertiments 
Here we provide 3 ways of running the experiments (more detailed explanation in the
sections below). All the methods can work both locally and on a HPC cluster 
(except **2**, which requires being on a cluster).

1.  [**The default way (Reccomended)**](#the-default-way): 
    Specify `OMP_NUM_THREADS` and `MTX_FILE_PATH` env variables, 
    then run experiments with `run_experiments.py` script

2.  [**The HPC cluster way**](#the-hpc-cluster-way): 
    Submit multiple jobs to a hpc cluster, by making use of a `job.pbs` script

3.  [**The manual way**](#the-manual-way): 
    Specify all the env variables (also `OMP_SCHEDULE_TYPE`, ...), 
    and then run directly the c program (normally or with valgrind)



### Experiments Setup
Whichever method you chose, first make sure to follow this setup instructions:

0. If you are running from the **interactive session of an hpc cluster**, first make sure to load the necessary modules:
    ```
    module load gcc91
    module load python-3.7.2
    ```

1.  **Change directory** to the **repository root**
    ```
    cd /path/to/repo/root
    ```

2.  **Download matrices** to the folder `matrices_data`. Either download manually some
    COO matrices in .mtx format or download the default matrices by running:
    ```
    python3 scripts/download_matrices.py
    ```
    Take into account that the last two matrices might take several minutes to download.

3.  **Build c program** by running in the terminal:
    ```
    make rebuild
    ```

    Optionally, we can also pass some extra compiler flags and/or ovverride the
    optimization flags (set by default to `-O3`). For example:
    ```
    make rebuild EXTRA_FLAGS="-Werror -Wpedantic" OPTIM_FLAGS="-Ofast -march=native"
    ```

### The Default Way
The simplest way to run some experiments is by following these commands
(first make sure to have followed the [**Experiments Setup**](#experiments-setup)):

1.  Set **default env variables** by running:
    ```
    source scripts/set_default_env_vars.bash
    ```

2.  Specify **threads number** (e.g. 4 threads):
    ```
    export OMP_NUM_THREADS=4
    ```

3.  Specify **matrix file path**:
    ```
    export MTX_FILE_PATH=matrices_data/your_matrix.mtx
    ```

4.  **Run experiments** by running in the terminal
    ```
    python3 scripts/run_experiments.py
    ```
    This script will write its results to `experiments_output` (see [**Results format**](#results-format))

5. Repeat **steps 2 to 4** by using different thread counts and different matrices


### The HPC Cluster Way
We can also **run all of our experiments in parallel** by submitting **jobs** to a **hpc cluster**.
All of our scripts will submit jobs to the `short_cpuQ` queue.
Each job will run `scripts/run_experiments.py`, which will write its results to `experiments_output` (see [**Results format**](#results-format)). 
To do this choose one of the options below (first make sure to have followed [**Experiments Setup**](#experiments-setup)):

* You can **submit all the jobs for a given matrix file** (one for each number of threads in `1 2 4 8 16 32 64`)
by running:
```
JOB_MTX_FILE_PATH="matrices_data/your_matrix.mtx" bash scripts/sumbit_given_matrix.bash
```

* You can **submit all the jobs for a given number of threads** (one for each matrix in `matrices_data`)
by running (e.g. with 4 threads):
```
JOB_N_THREADS=4 bash scripts/submit_given_threads.bash
```

* Optionally you can also **submit jobs individually** by specifying both the number of threads and the matrix file path:
```
JOB_N_THREADS=4 JOB_MTX_FILE_PATH="matrices_data/your_matrix.mtx" bash scripts/sumbit_job.bash
```

* Lastly you can also **sumbit all the jobs at once**, but we suggest to avoid this option, since they 
are unlikely to all fit at once inside a cluster queue:
```
bash scripts/sumbit_all.bash
```

For each one of these methods of submitting jobs, you can optionally pass to the bash script the 
**walltime and memory of the job** as **env variables** (or even set them with **export**). They default to:
```
JOB_WALLTIME="05:30:00"
JOB_MEMORY="16gb"
```


### The Manual Way
We can also run the experiments wihout using the python script, by following these instructions
(first make sure to have followed the [**Experiments Setup**](#experiments-setup)):

1.  Set **default env variables** by running:
    ```
    source scripts/set_default_env_vars.bash
    ```

2. Set all the **env variables** you want to have a value different from the default (see [**Env Variables**](#env-variables)), e.g:
    ```
    export OMP_SCHEDULE="dynamic,64"
    export OMP_NUM_THREADS=8
    export MTX_FILE_PATH="matrices_data/your_matrix.mtx"
    ```

3. **Run the c program** in the terminal:
    ```
    build/main
    ```

    Its output to stdout will be a single float: the duration of the benchmarked function.
    Optionally, you can also run it with valgrind to get cache misses instead:
    ```
    valgrind --tool=callgrind --cache-sim=yes --callgrind-out-file="./.valgrind_output/spmv.%p"  build/main
    ```
    Other tools can be used, but this one is preferred: by using CALLGRIND macros inside the c program
    we make sure to record only the cache misses of the actual operation.

4. Repeat **steps 2 and 3** for all the different combinations of env variables you want.

## Results Format
If for running the experiments you chose the [**Manual Way**](#manual-way), 
then skip this section.

The results of the experiments produced by `scripts/run_experiments.py` will be 
stored iside the `experiments_output` folder, in a `.csv` file.

## Env Variables
Here's a list of the env variables used to run the experiments and their default values.

*   `MTX_FILE_PATH = "matrices_data/1138_bus.mtx"`: The path for the matrix that we will read 
    and benchmark our operation with

*   `OMP_NUM_THREADS = 1`: The number of threads we will use

*   `OMP_SCHEDULE = "static, 64"`: The omp schedule we will use. Typically will be `static,1`, `static,64`, `dynamic,10`, `dynamic,100` or `guided`

*   `OMP_PROC_BIND = "true"`, `OMP_PLACES = "cores"`: We set these to vars to avoid moving threads between cores

*   `MATRICES_DIR = "./matrices_data"`: Folder where the matrices are found

*   `OUTPUT_DIR = "./experiments_output"`: Folder to which the `run_experiments.py` script will write its results

*   `ALG_NAME = "csr"`: Algorithm to run, can be `"csr"` or `"csb"`

*   `TIMING_RUNS = 10 `: Number of times we will run the c program to get the 90th percentilie duration

*   `VALGRIND_RUNS = 1 `: Number of times we will run with callgrind to get the 90th percentile of D1 and LLd cache miss rates. 
    It defaults to 1 because callgrind is a deterministic tool and, despite the nondeterminism introduced by the use of multiple threads, we observed that the outputs have negligible differences.


