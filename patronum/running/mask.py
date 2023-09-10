import abc

from typing_extensions import Dict, Iterable, Self, Union


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


__all__ = ["IFormatter"]
