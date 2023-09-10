import abc
from functools import total_ordering


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


__all__ = ["IState", "IAction", "StateFlowException"]
