"""
download_model.py – Download the pre-trained DistilBERT model from GitHub Releases.

Run this once after cloning the repo instead of training from scratch.

Usage
-----
    python download_model.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "baileycg/propaganda-detector"
MODEL_ZIP = "distilbert_model_v2.zip"
MODELS_DIR = Path(__file__).parent / "models"


def get_latest_release_url(repo: str, asset_name: str) -> str:
    """Fetch the download URL for an asset in the latest GitHub release."""
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:
        release = json.load(resp)

    tag = release.get("tag_name", "unknown")
    print(f"Latest release: {tag}")

    for asset in release.get("assets", []):
        if asset["name"] == asset_name:
            return asset["browser_download_url"]

    assets = [a["name"] for a in release.get("assets", [])]
    raise FileNotFoundError(
        f"Asset '{asset_name}' not found in release {tag}.\n"
        f"Available assets: {assets}"
    )


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a simple progress bar."""
    print(f"Downloading from:\n  {url}\n")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "#" * int(pct // 2) + "-" * (50 - int(pct // 2))
            mb_done = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r  [{bar}] {pct:.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline after progress bar


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"Extracting to {dest_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("Extraction complete.")


def main() -> None:
    model_dir = MODELS_DIR / "distilbert_model"

    if model_dir.exists() and (model_dir / "model.safetensors").exists():
        print(f"Model already exists at {model_dir}")
        print("Delete it first if you want to re-download.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODELS_DIR / MODEL_ZIP

    try:
        url = get_latest_release_url(REPO, MODEL_ZIP)
    except Exception as exc:
        print(f"Could not fetch release info: {exc}", file=sys.stderr)
        sys.exit(1)

    download_with_progress(url, zip_path)
    extract_zip(zip_path, MODELS_DIR)

    # Clean up zip after extraction
    zip_path.unlink()
    print(f"\nModel ready at: {model_dir}")
    print("Run predictions with:")
    print('  python main.py --model-type transformer --model distilbert_model --interactive')


if __name__ == "__main__":
    main()
