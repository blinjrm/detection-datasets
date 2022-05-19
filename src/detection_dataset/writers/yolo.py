import os

import pandas as pd
from ruamel.yaml import YAML

from detection_dataset.writers import BaseWriter

yaml = YAML()

YAML_TEMPLATE = """
path: {path}
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  images/test  # test images (relative to 'path')

# Classes
nc: {n_classes}
names: [{class_names}]
"""


class YoloWriter(BaseWriter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.final_data["bbox"] = [bbox.to_yolo() for bbox in self.data.bbox]
        self.write()

    def write(self) -> None:

        data = self._data_by_image()

        self._make_dirs()
        self._write_yaml()

        for _, row in data.iterrows():
            self._write_image(row)
            self._write_label(row)

    def _make_dirs(self) -> None:
        path = os.path.join(self.path, self.name)
        os.makedirs(path)
        os.makedir(os.path.join(path, "images", "train"))
        os.makedir(os.path.join(path, "images", "val"))
        os.makedir(os.path.join(path, "images", "test"))
        os.makedir(os.path.join(path, "labels", "train"))
        os.makedir(os.path.join(path, "labels", "val"))
        os.makedir(os.path.join(path, "labels", "test"))

    def _write_yaml(self) -> None:

        yaml_template_formated = YAML_TEMPLATE.format(
            path=self.path,
            n_classes=self.n_classes,
            class_names=", ".join(self.class_names),
        )

        yaml_dataset = yaml.load(yaml_template_formated)

        with open(os.path.join(self.path, "dataset.yml"), "w") as outfile:
            yaml.dump(yaml_dataset, outfile)

    def _write_image(self, row: pd.DataFrame) -> None:
        pass

    def _write_label(self, row: pd.DataFrame) -> None:
        pass
