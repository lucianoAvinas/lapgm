import numpy.typing as npt
from types import ModuleType
from typing import Generic, TypeVar, Any, NewType, Callable


Shape = TypeVar('Shape')

class Array(npt.NDArray[Any], Generic[Shape]):
    """For documentation: takes in datatype and array dimension shape."""
    pass


# Function which may change at runtime
RuntimeFunc = NewType('RuntimeFunc', Callable)

# Package with numpy array capabilities
ArrayPackage = NewType('ArrayPackage', ModuleType)

# Package with sparse array capabilities
SparsePackage = NewType('SparsePackage', ModuleType)

# Package which performs linear algebgra on sparse arrays
SparseLinalgPackage = NewType('SparseLinalgPackage', ModuleType)
