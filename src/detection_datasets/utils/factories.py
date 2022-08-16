from typing import Union

from detection_datasets.readers import BaseReader, CocoReader
from detection_datasets.writers import BaseWriter, MmdetWriter, YoloWriter


class Factory:
    def __init__(self):
        self._workers = {}

    def register(self, dataset_format: str, worker: Union[BaseReader, BaseWriter]) -> None:
        self._workers[dataset_format.lower()] = worker

    def get(self, dataset_format: str, **kwargs) -> Union[BaseReader, BaseWriter]:
        try:
            worker = self._workers[dataset_format.lower()]
        except KeyError:
            raise KeyError(f"{dataset_format} is not registered")

        return worker(**kwargs)


reader_factory = Factory()
reader_factory.register("coco", CocoReader)

writer_factory = Factory()
writer_factory.register("mmdet", MmdetWriter)
writer_factory.register("yolo", YoloWriter)
