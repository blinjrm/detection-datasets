import json
import os
from typing import Dict, Tuple

import pandas as pd

from detection_datasets.bbox import Bbox
from detection_datasets.readers import BaseReader
from detection_datasets.utils import Dataset


class CocoReader(BaseReader):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def load(self, splits: Dict[str, Tuple[str, str]], **kwargs) -> Dataset:
        annotation_dataframes = []
        for split, (annotation_file, images_dir) in splits.items():
            images_path_prefix = os.path.join(self.path, images_dir)

            json = self._read_json(self.path, annotation_file)

            annotation_dataframe = self._read_images_annotations(json)
            annotation_dataframe["image_path"] = annotation_dataframe["image_name"].apply(
                lambda x: os.path.join(images_path_prefix, x)
            )
            annotation_dataframe["split"] = split
            annotation_dataframes.append(annotation_dataframe)

        annotation_by_bbox = pd.concat(annotation_dataframes, axis=0, ignore_index=True)
        annotation_by_bbox["bbox"] = [
            Bbox.from_coco(row.bbox, row.width, row.height) for _, row in annotation_by_bbox.iterrows()
        ]

        return annotation_by_bbox.set_index(["image_id", "bbox_id"])

    @staticmethod
    def _read_json(path: str, file: str) -> json:
        path_to_file = os.path.join(path, file)
        with open(path_to_file) as f:
            return json.load(f)

    @staticmethod
    def _read_images_annotations(json_data: json) -> pd.DataFrame:
        annotations = pd.DataFrame(json_data["annotations"])
        annotations = annotations.drop(columns=["segmentation", "iscrowd", "attribute_ids"], errors="ignore")
        annotations = annotations.rename(columns={"id": "bbox_id"})

        images = pd.DataFrame(json_data["images"])
        images = images.drop(
            columns=["license", "time_captured", "original_url", "isstatic", "kaggle_id"], errors="ignore"
        )
        images = images.rename(columns={"id": "image_id", "file_name": "image_name"})

        categories = pd.DataFrame(json_data["categories"])
        categories = categories.drop(columns=["level", "taxonomy_id"], errors="ignore")
        categories = categories.rename(columns={"id": "category_id", "name": "category"})

        data = pd.merge(annotations, images, on="image_id", how="left")
        data = pd.merge(data, categories, on="category_id", how="left")
        return data
