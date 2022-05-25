import os
import urllib
import zipfile
import glob
from typing import List


def download_cached_jobs(remote: str, local: str) -> List[str]:
    """Download a zip file of cached jobs from a remote URL to a local path."""
    os.makedirs(local, exist_ok=True)
    zip_file = os.path.join(local, "cache.zip")
    urllib.request.urlretrieve(remote, zip_file)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(local)
    os.remove(zip_file)
    folder_names = [folder for folder in glob.glob(os.path.join(local, "**/*")) if "MACOS" not in folder]
    return folder_names


def download_file(remote: str, download_dir: str) -> str:
    """Download a file from a remote URL to a local directory."""
    os.makedirs(download_dir, exist_ok=True)
    filename = os.path.join(download_dir, os.path.basename(remote))
    urllib.request.urlretrieve(remote, filename)
    return filename
