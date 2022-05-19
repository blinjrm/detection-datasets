from detection_dataset.utils import Dataset, reader_factory, writer_factory


class Converter:
    def __init__(self) -> None:
        self.dataset = Dataset()

    def read(self, dataset_format: str, **kwargs) -> None:
        reader = reader_factory.get(dataset_format, **kwargs)
        dataset = reader.load()
        self.dataset.concat(dataset)

    def write(self, dataset_format: str, **kwargs) -> None:
        kwargs["data"] = self.dataset
        writer = writer_factory.get(dataset_format, **kwargs)
        writer.write(self.dataset)
