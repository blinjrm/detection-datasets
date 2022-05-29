import os
import shutil

import pandas as pd
from ruamel.yaml import YAML

from detection_dataset.writers import BaseWriter

yaml = YAML()

YAML_TEMPLATE = """
path: {path}
train: images/train
val: images/val
test:  images/test

# Classes
nc: {n_classes}
names: [{class_names}]
"""


class YoloWriter(BaseWriter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.final_data["bbox"] = [[bbox.to_yolo() for bbox in bboxes] for bboxes in self.final_data.bbox]

    def write(self) -> None:

        data = self.final_data.copy()

        self._write_yaml()

        for split in data.split.unique():
            split_data = data[data.split == split]

            self._make_dirs(split)

            for _, row in split_data.iterrows():
                row = row.to_frame().T
                self._write_image(row)
                self._write_label(row)

    def _write_yaml(self) -> None:

        os.makedirs(self.dataset_dir)

        yaml_template_formated = YAML_TEMPLATE.format(
            path=self.dataset_dir,
            n_classes=self.n_classes,
            class_names=", ".join(self.class_names),
        )

        yaml_dataset = yaml.load(yaml_template_formated)

        with open(os.path.join(self.dataset_dir, "dataset.yml"), "w") as outfile:
            yaml.dump(yaml_dataset, outfile)

    def _make_dirs(self, split: str) -> None:
        os.makedirs(os.path.join(self.dataset_dir, "images", split))
        os.makedirs(os.path.join(self.dataset_dir, "labels", split))

    def _write_image(self, row: pd.DataFrame) -> None:
        out_file = self._get_filename(row, "images")
        in_file = row.image_path.values[0]

        shutil.copyfile(in_file, out_file)

    def _write_label(self, row: pd.Series) -> None:
        out_file = self._get_filename(row, "labels")
        data = row.explode(["bbox_id", "category_id", "area", "bbox"])

        with open(out_file, "w") as f:
            for _, r in data.iterrows():
                labels = " ".join((str(r.category_id), str(r.bbox[0]), str(r.bbox[1]), str(r.bbox[2]), str(r.bbox[3])))
                f.write(labels + "\n")

    def _get_filename(self, row: pd.Series, task: str) -> str:
        split = row.split.values[0]
        image_id = str(row.image_id.values[0])

        if task == "labels":
            return os.path.join(self.dataset_dir, "labels", split, image_id + ".txt")
        elif task == "images":
            return os.path.join(self.dataset_dir, "images", split, image_id + ".jpg")
        else:
            raise ValueError(f"tTask must be either 'lables' or 'images', not {task}")
