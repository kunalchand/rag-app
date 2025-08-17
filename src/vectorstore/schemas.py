from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class VectorRecord:
    id: str
    values: List[float]
    metadata: Dict[str, Any]
