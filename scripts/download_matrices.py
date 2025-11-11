import os
import tarfile
import requests
import shutil
from pathlib import Path
from typing import List

# --- Configuration ---
# List of matrix names (e.g., from the SuiteSparse URL path)
MATRIX_LIST: List[str] = [
    "HB/1138_bus", # rows/cols: 1138 | nnz: 4,054 | nz density: 3.13e-3
    "HB/bcsstk17", # rows/cols: 10,974 | nnz: 428,650 | nz density: 3.56e-3
    "ATandT/twotone", # rows/cols: 120,750 | nnz: 1,206,265 | nz density: 8.27e-5
    "ND/nd12k", # rows/cols: 36,000 | nnz: 14,220,946 | nz density: 1.10e-2
    "vanHeukelum/cage14", # rows/cols: 1,505,785 | nnz: 27,130,349 | nz density: 1.20e-5
]


# Base URL for the SuiteSparse Matrix Collection
SUITESPARSE_BASE_URL = "https://suitesparse-collection-website.herokuapp.com/MM/"
# OUTPUT_DIR = Path("matrices_data")
OUTPUT_DIR = Path("matrices_data")

def download_and_extract_matrix(matrix_name: str, output_dir: Path):
    """
    Downloads, extracts, and cleans up the matrix file.
    
    Args:
        matrix_name: e.g., 'ATandT/twotone'
    """
    
    # 1. Construct URL and local file paths
    # URL: https://suitesparse-collection-website.herokuapp.com/MM/ATandT/twotone.tar.gz
    # The actual file inside the archive is usually name/name.mtx
    
    # Extract the base filename (e.g., 'citeseer')
    base_filename = matrix_name.split('/')[-1]
    download_url = f"{SUITESPARSE_BASE_URL}{matrix_name}.tar.gz"
    local_archive_path = output_dir / f"{base_filename}.tar.gz"
    tmp_folder_path = output_dir / f"{base_filename}"
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Downloading {base_filename} ---")
    
    # 2. Download the archive file
    try:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            with open(local_archive_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded: {local_archive_path}")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download {download_url}. Check network/URL. {e}")
        return

    # 3. Extract the Matrix Market file (.mtx)
    try:
        with tarfile.open(local_archive_path, 'r:gz') as tar:
            # We are looking for the Matrix Market file inside the archive
            mtx_files = [member for member in tar.getnames() if member.endswith('.mtx')]
            
            if not mtx_files:
                print("ERROR: No .mtx file found in the archive.")
                return
            
            # The .mtx file is usually the first (and only) one of interest
            mtx_file_to_extract = mtx_files[0]
            
            # Extract the .mtx file directly to the output directory
            tar.extract(mtx_file_to_extract, path=output_dir)
            
            # Rename the extracted file to a clean name (e.g., 'citeseer.mtx')
            extracted_path = output_dir / Path(mtx_file_to_extract).name
            final_path = output_dir / f"{base_filename}.mtx"
            
            if extracted_path.exists():
                extracted_path.rename(final_path)
            else:
                # If the extraction preserves the folder path, find the file inside
                extracted_full_path = output_dir / mtx_file_to_extract
                if extracted_full_path.exists():
                     extracted_full_path.rename(final_path)
                
            print(f"Extracted and saved: {final_path}")
            
    except tarfile.TarError as e:
        print(f"ERROR: Failed to extract archive {local_archive_path}. {e}")
    finally:
        # 4. Cleanup: Delete the large .tar.gz archive
        if local_archive_path.exists():
            os.remove(local_archive_path)
            print(f"Cleaned up archive: {local_archive_path}")

        # 5. Cleanup: Delete the temporary folder created by extraction
        if tmp_folder_path.is_dir():
            shutil.rmtree(tmp_folder_path)
            print(f"Cleaned up temporary folder: {tmp_folder_path}")


def main():
    print("Starting matrix download process.")
    
    # Ensure requests is installed: pip install requests
    try:
        import requests
    except ImportError:
        print("\nERROR: The 'requests' library is not installed.")
        print("Please run: pip install requests")
        return

    for name in MATRIX_LIST:
        download_and_extract_matrix(name, OUTPUT_DIR)

    print("\nDownload process complete.")

if __name__ == "__main__":
    main()