PKGDIR=$HOME/.tmp_hf_pkgs
pip install --no-cache-dir --target "$PKGDIR" huggingface_hub
PYTHONPATH="$PKGDIR" python3 scripts/hf_download_matrices.py
rm -rf "$PKGDIR"
