"""
Download script for:
  nvidia/PhysicalAI-Spatial-Intelligence-Warehouse (Hugging Face)

Requirements:
  pip install huggingface_hub python-dotenv tqdm

Notes:
  - The dataset is GATED. You must first visit the dataset page, agree to the
    terms, and share your contact information before your token will work:
    https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse
  - Set HF_TOKEN in your .env file.
"""

import os
import tarfile
import glob
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from tqdm import tqdm 

# Config
load_dotenv()

HF_TOKEN   = os.getenv("HF_TOKEN")
DATASET_ID = "nvidia/PhysicalAI-Spatial-Intelligence-Warehouse"
LOCAL_DIR  = Path("./data/nvidia_warehouse_dataset")

# Splits whose image/depth folders are stored as chunked tar.gz archives
ARCHIVED_SPLITS = ["train", "test"]
ARCHIVED_SUBDIRS = ["images", "depths"]


def download_dataset(token: str, local_dir: Path) -> None:
    """Download the full dataset snapshot from Hugging Face Hub."""
    if not token:
        raise ValueError(
            "HF_TOKEN is not set. Add it to your .env file as HF_TOKEN=hf_..."
        )

    print(f"Downloading '{DATASET_ID}' -> {local_dir.resolve()}")
    print("(This may take a while -- the dataset is large)\n")

    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
    )
    print("\n Download complete.\n")


def extract_archives(local_dir: Path) -> None:
    """Extract chunk_*.tar.gz archives inside train/test image & depth folders."""
    print("Extracting chunked tar.gz archives ...")

    for split in ARCHIVED_SPLITS:
        for subdir in ARCHIVED_SUBDIRS:
            folder = local_dir / split / subdir
            archives = sorted(glob.glob(str(folder / "chunk_*.tar.gz")))

            if not archives:
                # Some subdirs may not exist or may not be archived
                continue

            print(f"  [{split}/{subdir}]  found {len(archives)} archive(s)")
            
            # Optimize using subprocess and Linux commands for multi-threading (much faster than tarfile)
            import subprocess
            
            # find: locate all tar.gz files
            # xargs -P 8: run up to 8 tar processes in parallel
            extract_cmd = f"find {folder} -maxdepth 1 -name 'chunk_*.tar.gz' -print0 | xargs -0 -P 8 -I {{}} tar -xzf {{}} -C {folder}"
            
            print(f"  >> Running parallel extraction (xargs -P 8 tar -xzf)...")
            subprocess.run(extract_cmd, shell=True, check=True)

            # Fast deletion of archives using find -delete
            rm_cmd = f"find {folder} -maxdepth 1 -name 'chunk_*.tar.gz' -delete"
            print(f"  >> Cleaning up .tar.gz archives...")
            subprocess.run(rm_cmd, shell=True, check=True)

    print("\n Extraction complete.\n")


def verify_structure(local_dir: Path) -> None:
    """Print a quick summary of what was downloaded."""
    print("Dataset structure:")
    for item in sorted(local_dir.iterdir()):
        if item.is_dir():
            n_files = len(list(item.rglob("*")))
            print(f"  {item.name}/  ({n_files} files)")
        else:
            size_mb = item.stat().st_size / 1e6
            print(f"  {item.name}  ({size_mb:.1f} MB)")
    print("\n")


if __name__ == "__main__":
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    download_dataset(token=HF_TOKEN, local_dir=LOCAL_DIR)
    extract_archives(local_dir=LOCAL_DIR)
    verify_structure(local_dir=LOCAL_DIR)

    print("Done! Dataset is ready at:", LOCAL_DIR.resolve())
