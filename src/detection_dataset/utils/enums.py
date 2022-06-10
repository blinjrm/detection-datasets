from enum import Enum


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Destinations(str, Enum):
    LOCAL_DISK = "local_disk"
    WANDB = "wandb"
