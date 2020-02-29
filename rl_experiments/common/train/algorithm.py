from abc import ABC, abstractmethod
from typing import NoReturn, Dict, List, Any

import numpy as np


class Actor(ABC):
    @abstractmethod
    def get_actions(self, states: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, rollout: Dict[str, List[Any]]) -> NoReturn:
        pass
