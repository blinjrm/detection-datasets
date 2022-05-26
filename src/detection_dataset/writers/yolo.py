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
    def __init__(self, data: pd.DataFrame, path: str, name: str) -> None:
        super().__init__(data, path, name)

        # self.data["bbox"] = [Bbox.to_mmdet(row.bbox, row.width, row.height) for _, row in self.data.iterrows()]

        self.write()

    def write(self) -> None:
        self._make_dirs()
        self._write_yaml()
        self._write_images()
        self._write_labels()

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

    def _write_images(self) -> None:
        pass

    def _write_labels(self) -> None:
        pass
