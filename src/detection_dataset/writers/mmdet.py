import os
import shutil

import numpy as np
import pandas as pd

from detection_dataset.bbox import Bbox
from detection_dataset.writers import BaseWriter


class MmdetWriter(BaseWriter):
    def __init__(self, data: pd.DataFrame, path: str, name: str) -> None:
        super().__init__(data, path, name)

        self._write()

        self.data["bbox"] = [Bbox.to_mmdet(row.bbox, row.width, row.height) for _, row in self.data.iterrows()]

    def _write(self) -> None:
        data = self._data_by_image()

        os.makedirs(os.path.join(self.path, self.name))
        self._save_metadata()

        # TODO: allow selecting split
        # TODO: move to BaseWriter
        splits = ["train", "val", "test"]
        datasets = np.split(data.sample(frac=1, random_state=42), [int(0.8 * len(data)), int(0.9 * len(data))])

        for i, data_split in enumerate(datasets):
            print(f"Generate {splits[i]} dataset")
            path = os.path.join(self.path, self.name, splits[i])
            os.makedirs(path)

            data = self._make_mmdet_data(data_split=data_split)
            self._save_dataset(data=data, dataset_path=path)

    def _save_metadata(self):
        """Saves the metadata."""

        path = os.path.join(self.path, self.name, "labels_mapping.csv")
        self.labels_mapping.to_csv(path, index=False)

    def _make_mmdet_data(self, data_split: pd.DataFrame):

        mmdet_data = []

        for _, row in data_split.iterrows():
            annotations = {}
            annotations["bboxes"] = row["bbox"]

            if self.labels_mapping is not None:
                labels_mapping = self.labels_mappings
                original_ids = row["category"]
                annotations["labels"] = [
                    labels_mapping[labels_mapping["original_id"] == original_id]["id"].values[0]
                    for original_id in original_ids
                    if labels_mapping[labels_mapping["original_id"] == original_id]["id"].values[0] > 0
                ]
            else:
                annotations["labels"] = row["category"]

            data = {}
            data["filename"] = row["filename"]
            data["width"] = row["width"]
            data["height"] = row["height"]
            data["ann"] = annotations

            mmdet_data.append(data)

        return mmdet_data

    def _save_dataset(self, data: list, dataset_path: str):
        """Creates a new directory and saves the dataset and images."""

        os.makedirs(os.path.join(dataset_path, "images"))

        # Save data
        with open(os.path.join(dataset_path, "annotation.jsonl"), "w") as f:
            f.write(str(data))

        # Save corresponding images
        for image in data:
            filename = image["filename"]
            shutil.copy(os.path.join(self.dir_images, filename), os.path.join(dataset_path, "images", filename))
