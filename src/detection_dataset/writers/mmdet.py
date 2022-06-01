import os
import shutil
from typing import Dict, List

import pandas as pd

from detection_dataset.writers import BaseWriter


class MmdetWriter(BaseWriter):
    def __init__(self, **kwargs) -> None:
        """Initializes the YoloWriter."""

        super().__init__(**kwargs)

        self.final_data["bbox"] = [[bbox.to_voc() for bbox in bboxes] for bboxes in self.final_data.bbox]

    def write(self) -> None:
        """Writes the dataset to disk.

        For the MMDET format, the associated steps are:
            1. Create the directories for the images and annotations.
            2. Prepare the data for any given split.
            3. Write the annotation file to disk for each split.
            4. Write the images to disk for each split.
        """

        data = self.final_data.copy()

        for split in data.split.unique():
            os.makedirs(os.path.join(self.dataset_dir, split, "images"))

            split_data = data[data.split == split]
            dataset = self._make_mmdet_data(split_data)
            self._save_dataset(dataset, split)

    def _make_mmdet_data(self, data_split: pd.DataFrame):

        mmdet_data = []
        source_imges = []

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
            source_imges.append(row["image_path"])

            dataset = {"mmdet_data": mmdet_data, "source_imges": source_imges}

        return dataset

    def _save_dataset(self, dataset: Dict[str, List[str]], split: str):
        """Creates a new directory and saves the dataset and images."""

        split_path = os.path.join(self.dataset_dir, split)
        mmdet_data = dataset["mmdet_data"]
        source_images = dataset["source_imges"]

        # Labels
        with open(os.path.join(split_path, "annotation.jsonl"), "w") as f:
            f.write(str(mmdet_data))

        # Images
        for mmdet_data_image, original_image_path in zip(mmdet_data, source_images):
            out_file = os.path.join(self.dataset_dir, split, "images", mmdet_data_image["filename"])

            shutil.copyfile(original_image_path, out_file)
