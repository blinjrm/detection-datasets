from typing import List

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

from detection_datasets.utils import reader_factory, writer_factory

api = HfApi()

ORGANISATION = "detection-dataset"


class DetectionDataset:
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    @classmethod
    def from_hub(cls, name: str) -> "DetectionDataset":
        """Load a dataset from the Hugging Face Hub.

        Currently only datasets from the 'detection-datasets' organisation can be loaded.

        Args:
            name: name of the dataset, without the organisation's prefix.

        Returns:
            A DetectionDataset instance containing the loaded data.
        """

        if name not in cls.datasets_available_in_hub():
            raise ValueError(
                f"""{name} is not available on the Hub.
            Use `DetectionDataset.datasets_available_in_hub() to get the list of available datasets."""
            )

        path = "/".join([ORGANISATION, name])
        ds = load_dataset(path=path)

        return cls(dataset=ds)

    @staticmethod
    def datasets_available_in_hub() -> List[str]:
        """List the datasets available in the Hugging Face Hub.

        Returns:
            List of names of datasets registered in the Hugging Face Hub, under the 'detection-datasets' organisation.
        """

        datasets = api.list_datasets(author="detection-dataset")
        return [dataset.id.split("/")[-1] for dataset in datasets]

    @classmethod
    def from_disk(cls, dataset_format: str, path: str, **kwargs) -> "DetectionDataset":
        """Load a dataset from disk.

        This is a factory method that can read the dataset from different formats,
        when the dataset is already in a local directory.

        Args:
            dataset_format: Format of the dataset.
                Currently supported values and formats:
                - "coco": COCO format
            path: Path to the dataset on the local filesystem.
            **kwargs: Keyword arguments specific to the dataset_format.

        Returns:
            A DetectionDataset instance containing the loaded data.
        """

        reader = reader_factory.get(dataset_format=dataset_format, path=path, **kwargs)
        ds = reader.load()

        return cls(dataset=ds)

    def to_hub(self, dataset_name: str, repo_name: str = ORGANISATION, **kwargs) -> None:
        """Pushes the dataset to the hub as a Parquet dataset.

        This method wraps Hugging Face's DatasetDict.push_to_hub() method, check here for reference:
        https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.DatasetDict.push_to_hub

        The dataset is pushed as a DatasetDict, meaning the each split (train, val, test), if present,
        will be a separate Dataset instance inside this DatasetDict.

        Args:
            dataset_name: name of the dataset inside the user/organisation's repository.
            repo_name: user of organisation to push the dataset to.
        """

        # repo_id = "/".join([repo_name, dataset_name])

        hf_dataset_dict = DatasetDict()
        data = self._dataset

        for split in data.unique("split"):
            split_data = data.filter(lambda x: x == split, input_columns="split")
            split_data.add_column
            hf_dataset_dict[split] = split_data

        # hf_dataset_dict.push_to_hub(repo_id=repo_id, **kwargs)
        return hf_dataset_dict

    def to_disk(self, dataset_format: str, name: str, path: str) -> None:
        writer = writer_factory.get(dataset_format=dataset_format, name=name, path=path)
        writer.write()

    @property
    def dataset(self):
        return self._dataset
