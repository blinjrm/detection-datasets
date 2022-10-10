from __future__ import annotations

import os
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
        self.data = dataset.set_format(index="image").reset_index()
        self.name = name
        self.path = path
        self.dataset_dir = os.path.join(self.path, self.name)
        self.class_names = dataset.category_names
        self.n_classes = dataset.n_categories
        self.n_images = dataset.n_images
        self.split_proportions = dataset.split_proportions

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
