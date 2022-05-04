from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Dataset:
    data: pd.DataFrame
    categories: List[str]
