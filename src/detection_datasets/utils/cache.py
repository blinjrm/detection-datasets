import shutil
from pathlib import Path

from detection_datasets.utils.constants import CACHE_DIR


def get_temp_dir() -> str:
    """Get the path for the temp directory, create it if needed.

    Returns:
        The path to the library's temp directory.
    """

    TEMP_DIR = Path.home() / CACHE_DIR
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    return TEMP_DIR


def clear_temp_dir() -> None:
    """Clear the temporary directory to save space.

    The temporary directory is used to extract images from parquet files donwloaded from the Hub. Each instance creates
    its own directory when using downloading images from the hub, this methods deletes all of these instance sub-
    directories.
    """

    TEMP_DIR = get_temp_dir()
    shutil.rmtree(TEMP_DIR)
    print(f"Temporary directory {TEMP_DIR} has been cleared.")
