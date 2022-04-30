from typing import List

from pydantic import BaseModel


class BboxData(BaseModel):
    bbox: List[float]
    width: float
    height: float
