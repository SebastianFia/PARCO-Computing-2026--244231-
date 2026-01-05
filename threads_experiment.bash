export ONEDNN_ROOT=./oneDNN
export LD_LIBRARY_PATH=$ONEDNN_ROOT/build/src:$LD_LIBRARY_PATH

for N_THREADS in 1 2 4 8 12 
do
    export OMP_NUM_THREADS=$N_THREADS
    echo "---"
    echo "Running with $N_THREADS threads"
    ./build/run_gemm
done