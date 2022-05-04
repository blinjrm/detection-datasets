from __future__ import annotations

from typing import Iterable

from detection_dataset.models import Dataset
from detection_dataset.readers import CocoReader
from detection_dataset.writers import MmdetWriter


class Convert:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

        # TODO: add ability to draw image with bounding boxes

    @classmethod
    def from_coco(cls, path: str, splits: dict[str, tuple[str, str]]) -> Convert:
        reader = CocoReader(path, splits)
        return Convert(reader.dataset)

    @classmethod
    def from_voc(self, path: str) -> None:
        raise NotImplementedError()

    @classmethod
    def from_yolo(self, path: str) -> None:
        raise NotImplementedError()

    @classmethod
    def to_coco(self, name: str, splits: Iterable[float | int] | None) -> None:
        raise NotImplementedError()

    @classmethod
    def to_mmdet(self, kwargs) -> None:
        MmdetWriter(**kwargs)
