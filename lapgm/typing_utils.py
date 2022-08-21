import numpy.typing as npt
from typing import Generic, TypeVar, Any


Shape = TypeVar('Shape')

class Array(npt.NDArray[Any], Generic[Shape]):
    pass
    