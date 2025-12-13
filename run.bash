export ONEDNN_ROOT=./oneDNN
export LD_LIBRARY_PATH=$ONEDNN_ROOT/build/src:$LD_LIBRARY_PATH

g++ -O3 -march=native -std=c++17 src/main.cpp -o build/run_gemm \
    -I$ONEDNN_ROOT/include \
    -I$ONEDNN_ROOT/build/include \
    -L$ONEDNN_ROOT/build/src \
    -ldnnl

./build/run_gemm
