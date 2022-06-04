import pandas as pd

from detection_dataset.utils import Dataset, reader_factory, writer_factory


class Converter:
    def __init__(self) -> None:
        self.dataset = Dataset()
        # self.category_mapping = None

    def read(self, dataset_format: str, **kwargs) -> None:
        self.reader = reader_factory.get(dataset_format, **kwargs)
        dataset = self.reader.load()
        self.dataset.concat(dataset)

    def transform(self, category_mapping: pd.DataFrame) -> None:
        self.category_mapping = category_mapping
        self.dataset.map_categories(category_mapping)

    def write(self, dataset_format: str, **kwargs) -> None:
        kwargs["dataset"] = self.dataset
        self.writer = writer_factory.get(dataset_format, **kwargs)
        self.writer.write()
