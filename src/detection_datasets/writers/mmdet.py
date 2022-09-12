import json
import os
import shutil
from typing import Dict, List

import pandas as pd

from detection_datasets.writers import BaseWriter


class MmdetWriter(BaseWriter):

    format = "mmdet"

    def __init__(self, **kwargs) -> None:
        """Initialize the YoloWriter."""

        super().__init__(**kwargs)
        self.data["bbox"] = [[bbox.to_voc() for bbox in bboxes] for bboxes in self.data.bbox]

    def write(self) -> None:
        """Write the dataset to disk.

        For the MMDET format, the associated steps are:
            1. Create the directories for the images and annotations.
            2. Prepare the data for any given split.
            3. Write the annotation file to disk for each split.
            4. Write the images to disk for each split.
        """

        for split in self.data.split.unique():
            os.makedirs(os.path.join(self.dataset_dir, split, "images"))

            split_data = self.data[self.data.split == split]
            dataset = self._make_mmdet_data(split_data)
            self._save_dataset(dataset, split)

    def _make_mmdet_data(self, data_split: pd.DataFrame):

        mmdet_data = []
        source_images = []

        for _, row in data_split.iterrows():
            annotations = {}
            annotations["bboxes"] = row["bbox"]
            annotations["labels"] = row["category_id"]

            data = {}
            data["filename"] = "".join((str(row["image_id"]), ".jpg"))
            data["width"] = row["width"]
            data["height"] = row["height"]
            data["ann"] = annotations

            mmdet_data.append(data)
            source_images.append(row["image_path"])

            dataset = {"mmdet_data": mmdet_data, "source_images": source_images}

        return dataset

    def _save_dataset(self, dataset: Dict[str, List[str]], split: str):
        """Create a new directory and saves the dataset and images."""

        split_path = os.path.join(self.dataset_dir, split)
        mmdet_data = dataset["mmdet_data"]
        source_images = dataset["source_images"]

        # Labels
        file = os.path.join(split_path, "annotation.json")
        with open(file, "w", encoding="utf-8") as f:
            json.dump(mmdet_data, f, ensure_ascii=False, indent=4)

        # Images
        for mmdet_data_image, original_image_path in zip(mmdet_data, source_images):
            out_file = os.path.join(self.dataset_dir, split, "images", mmdet_data_image["filename"])

            shutil.copyfile(original_image_path, out_file)
