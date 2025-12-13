REPO_URL="https://github.com/oneapi-src/oneDNN.git"
REPO_NAME="oneDNN"

# Check if the repo directory already exists
if [ -d "$REPO_NAME" ]; then
    echo "Directory '$REPO_NAME' already exists. Skipping git clone."
else
    echo "Directory '$REPO_NAME' not found. Starting git clone..."
    git clone "$REPO_URL"
fi

cd oneDNN
rm -rf build
mkdir build
cd build
cmake .. -DDNNL_BUILD_EXAMPLES=ON -DDNNL_ENABLE_WORKLOAD=INFERENCE
make -j4
