from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detection_datasets import DetectionDataset


class BaseWriter(ABC):
    def __init__(self, dataset: DetectionDataset, name: str, path: str, move_or_copy_images: str = "copy") -> None:
        """Base class for writing datasets to disk.

        Args:
            dataset: DetectionDataset instance.
            name: Name of the dataset to be created in the "path" directory.
            path: Path to the directory where the dataset will be created.
            move_or_copy_images: Wether to move or copy images from the source
                directory to the directory of the new dataset written to disk.
                Defaults to 'copy'.
        """

        self.dataset = dataset
        self.dataset_dir = os.path.join(path, name)
        self.name = name

        if move_or_copy_images.lower() in ["move", "copy"]:
            self.move_or_copy_images = move_or_copy_images
        else:
            print(f"Incorrect value ({move_or_copy_images}) for move_or_copy_images, defaulting to 'copy'.")
            self.move_or_copy_images = "copy"

    @abstractmethod
    def write(self) -> None:
        """Write the dataset to disk.

        This method is specifc to each format, and need to be implemented in the writer class.
        """

    def do_move_or_copy_image(self, in_file: str, out_file: str) -> None:
        """Move or copy an image.

        Args:
            in_file: Path to the existing image file.
            out_file: Directory where the image will be added.
        """

        if self.move_or_copy_images == "move":
            shutil.move(in_file, out_file)
        else:
            shutil.copyfile(in_file, out_file)
