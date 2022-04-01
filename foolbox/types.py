from typing import Any, Dict, NamedTuple, NewType, Optional, Tuple, Union


class Bounds(NamedTuple):
    lower: float
    upper: float


BoundsInput = Union[Bounds, Tuple[float, float]]

L0 = NewType("L0", float)
L1 = NewType("L1", float)
L2 = NewType("L2", float)
Linf = NewType("Linf", float)

Preprocessing = Optional[Dict[str, Any]]
