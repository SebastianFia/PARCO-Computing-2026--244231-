PKGDIR=$HOME/.tmp_pkgs
pip install --no-cache-dir --target "$PKGDIR" \
    huggingface_hub \
    torch \
    transformers \
    numpy

PYTHONPATH="$PKGDIR" python3 scripts/hf_download_matrices.py

rm -rf "$PKGDIR"