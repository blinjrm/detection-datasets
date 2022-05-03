from __future__ import annotations

from typing import Iterable

import pandas as pd

from detection_dataset.readers import CocoReader


class Convert:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    @classmethod
    def from_coco(cls, path: str) -> Convert:
        reader = CocoReader(path)
        return Convert(reader.data)

    @classmethod
    def from_voc(self, path: str) -> None:
        raise NotImplementedError()

    @classmethod
    def from_yolo(self, path: str) -> None:
        raise NotImplementedError()

    @classmethod
    def to_coco(self, name: str, splits: Iterable[float | int] | None) -> None:
        raise NotImplementedError()
