import json
import os
import shutil
from typing import Any, Dict, List

from detection_datasets.writers import BaseWriter


class CocoWriter(BaseWriter):

    format = "coco"

    def __init__(self, **kwargs) -> None:
        """Initialize the CocoWriter."""

        super().__init__(**kwargs)

    def write(self) -> None:
        """Write the dataset to disk.

        For the COCO format, the associated steps are:
            1. Write the annotations json files for each split.
            2. Write the images for each split.
        """

        self._write_annotations()
        self._write_images()

    def _write_annotations(self) -> None:
        os.makedirs(os.path.join(self.dataset_dir, "annotations"))

        for split in self.data.split.unique():
            instances = {
                "info": {"description": self.name},
                "licences": [],
                "images": self._get_images(),
                "annotations": self._get_annotations(),
                "categories": self._get_categories(),
            }

            path = os.path.join(self.dataset_dir, "annotations", f"instances_{split}.json")
            with open(path, "w") as file:
                json.dump(instances, file)

    def _get_images(self) -> List[Dict[str, Any]]:
        data = self.dataset.get_data(index="image").copy().reset_index()
        result = []

        for i in range(len(data)):
            result.append(
                {
                    "file_name": str(data.loc[i, "image_path"]).split("/")[-1],
                    "height": data.loc[i, "height"],
                    "width": data.loc[i, "width"],
                    "id": int(data.loc[i, "image_id"]),  # convert from numpy int64
                }
            )

        return result

    def _get_annotations(self) -> List[Dict[str, Any]]:
        data = self.dataset.get_data(index="bbox").copy().reset_index()
        data["bbox"] = [bbox.to_coco() for bbox in data.bbox]
        result = []

        for i in range(len(data)):
            result.append(
                {
                    "area": data.loc[i, "area"],
                    "iscrowd": 0,
                    "image_id": int(data.loc[i, "image_id"]),
                    "bbox": data.loc[i, "bbox"],
                    "category_id": data.loc[i, "category_id"],
                    "id": int(data.loc[i, "bbox_id"]),
                }
            )

        return result

    def _get_categories(self) -> List[Dict[str, Any]]:
        data = self.dataset.categories.copy().reset_index()
        result = []

        for i in range(len(data)):
            result.append(
                {
                    "id": int(data.loc[i, "category_id"]),
                    "name": data.loc[i, "category"],
                }
            )

        return result

    def _write_images(self) -> None:
        for split in self.data.split.unique():
            os.makedirs(os.path.join(self.dataset_dir, split))
            split_data = self.data[self.data.split == split]

            for _, row in split_data.iterrows():
                row = row.to_frame().T

                in_file = row.image_path.values[0]
                out_file = os.path.join(self.dataset_dir, split, str(row["image_id"].values[0]) + ".jpg")

                shutil.copyfile(in_file, out_file)
