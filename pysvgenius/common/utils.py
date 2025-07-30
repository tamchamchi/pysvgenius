import os
import urllib.request
from pathlib import Path

from iopath.common.file_io import g_pathmgr

from .registry import registry


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        print(f"Error creating directory: {dir_path}")
    return is_success

def download(url: str, save_dir: str = "models", filename: str = None, show_progress: bool = True) -> str:
        """
        Download a model file from a given URL and save it locally.

        Args:
            url (str): The URL of the model file (.pth, .pt, .bin, etc.)
            save_dir (str): Directory where the model will be stored
            filename (str): Name of the file to save. If None, the name is extracted from the URL
            show_progress (bool): Whether to display the download progress

        Returns:
            str: The local path of the downloaded model
        """
        # Ensure save directory exists
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = url.split("/")[-1]

        file_path = save_dir / filename

        # Skip download if the file already exists
        if file_path.exists():
            if show_progress:
                print(f"✅ Model already exists: {file_path}")
            return str(file_path)

        # Progress callback
        def progress_callback(block_num, block_size, total_size):
            if not show_progress:
                return
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                mb_downloaded = downloaded // (1024 * 1024)
                mb_total = total_size // (1024 * 1024)
                print(f"\r📥 Downloading: {percent}% ({mb_downloaded}/{mb_total} MB)", end='')

        # Start download
        try:
            urllib.request.urlretrieve(url, file_path, reporthook=progress_callback)
            if show_progress:
                print(f"\n✅ Model downloaded successfully: {file_path}")
        except Exception as e:
            print(f"\n✗ Failed to download model: {e}")
            raise RuntimeError(f"Failed to download model from {url}") from e

        return str(file_path)
