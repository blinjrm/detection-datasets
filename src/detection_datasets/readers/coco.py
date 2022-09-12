import json
import os
from typing import Dict, Tuple

import pandas as pd

from detection_datasets.bbox import Bbox
from detection_datasets.readers import BaseReader


class CocoReader(BaseReader):
    def __init__(self, path: str, splits: Dict[str, Tuple[str, str]]) -> None:
        super().__init__(path)
        self.splits = splits

    def read(self) -> pd.DataFrame():
        annotation_dataframes = []
        for split, (annotation_file, images_dir) in self.splits.items():
            images_path_prefix = os.path.join(self.path, images_dir)

            json = self._read_json(self.path, annotation_file)

            annotation_dataframe = self._read_annotations(json_data=json)
            annotation_dataframe["image_path"] = annotation_dataframe["image_name"].apply(
                lambda x: os.path.join(images_path_prefix, x)
            )
            annotation_dataframe["split"] = split
            annotation_dataframes.append(annotation_dataframe)

        annotation_by_bbox = pd.concat(annotation_dataframes, axis=0, ignore_index=True)
        annotation_by_bbox["bbox"] = [
            Bbox.from_coco(row.bbox, row.width, row.height) for _, row in annotation_by_bbox.iterrows()
        ]

        return annotation_by_bbox

    @staticmethod
    def _read_json(path: str, file: str) -> json:
        path_to_file = os.path.join(path, file)
        with open(path_to_file) as f:
            return json.load(f)

    def _read_annotations(self, json_data: json) -> pd.DataFrame:
        annotations = pd.DataFrame(json_data["annotations"])
        annotations = annotations[["image_id", "category_id", "bbox", "area", "id"]]
        annotations = annotations.rename(columns={"id": "bbox_id"})

        images = pd.DataFrame(json_data["images"])
        images = images[["id", "file_name", "height", "width"]]
        images = images.rename(columns={"id": "image_id", "file_name": "image_name"})

        categories = pd.DataFrame(json_data["categories"])
        categories = categories[["id", "name"]]
        categories = categories.rename(columns={"id": "category_id", "name": "category"})
        categories = categories.sort_values("category_id")
        self.categories = list(categories.category.unique())

        data = pd.merge(annotations, images, on="image_id", how="left")
        data = pd.merge(data, categories, on="category_id", how="left")

        return data
