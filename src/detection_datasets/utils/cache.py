import shutil
from pathlib import Path

from detection_datasets.utils.constants import CACHE_DIR


def get_temp_dir() -> str:
    """Get the path for the temp directory, create it if needed.

    Returns:
        The path to the library's temp directory.
    """

    temp_dir = Path.home() / CACHE_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)

    return temp_dir


def clear_temp_dir() -> None:
    """Clear the temporary directory to save space.

    The temporary directory is used to extract images from parquet files donwloaded from the Hub. Each instance creates
    its own directory when using downloading images from the hub, this methods deletes all of these instance sub-
    directories.
    """

    temp_dir = get_temp_dir()
    shutil.rmtree(temp_dir)
    print(f"Temporary directory {temp_dir} has been cleared.")
