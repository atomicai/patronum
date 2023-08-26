import abc
from functools import total_ordering

from typing_extensions import Dict, Iterable, Self, Union


class IState(abc.ABC):
    def next(self, x, r, **kwargs):
        if r not in self.state.keys():
            raise StateFlowException(f"Encountered input {x} that is not compatable with state {str(self)}")
        return self.state[r](**kwargs)


class StateFlowException(Exception):
    pass


@total_ordering
class IAction:
    def __init__(self, action: str):
        self.action = action

    def __str__(self):
        return self.action

    def __lt__(self, other):
        return str(self) < str(other)

    def __hash__(self):
        return hash(self.action)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, IAction):
            return self.action == other.action
        return NotImplemented


class IFormatter(abc.ABC):
    @abc.abstractmethod
    def prepare(self, debug=True, **kwargs) -> Self:
        pass

    @abc.abstractmethod
    def format(self, data: Iterable[Union[str, Dict]], **kwargs) -> Self:
        pass

    @abc.abstractmethod
    def save(self, filename_or_path=None) -> bool:
        pass


__all__ = ["IFormatter", "IState", "IAction", "StateFlowException"]
