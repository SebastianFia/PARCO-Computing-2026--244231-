export ONEDNN_ROOT=./oneDNN
export LD_LIBRARY_PATH=$ONEDNN_ROOT/build/src:$LD_LIBRARY_PATH

g++ -Ofast -march=native -mavx2 -mfma -fopenmp -std=c++17 src/main_run_gemm.cpp -o build/run_gemm \
    -I$ONEDNN_ROOT/include \
    -I$ONEDNN_ROOT/build/include \
    -L$ONEDNN_ROOT/build/src \
    -ldnnl

./build/run_gemm
