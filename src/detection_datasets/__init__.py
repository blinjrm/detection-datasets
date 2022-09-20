import importlib_metadata

from detection_datasets.bbox import Bbox
from detection_datasets.detection_dataset import DetectionDataset
from detection_datasets.utils.hub import available_in_hub

__version__ = importlib_metadata.version("detection_datasets")
