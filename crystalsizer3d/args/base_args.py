from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Union

from crystalsizer3d.util.utils import hash_data, to_dict


class BaseArgs(ABC):
    @classmethod
    @abstractmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        pass

    @classmethod
    def from_args(cls, args: Union[Namespace, dict]) -> 'BaseArgs':
        """
        Create a BaseArgs instance from command-line arguments.
        """
        if isinstance(args, dict):
            return cls(**args)
        elif isinstance(args, Namespace):
            return cls(**vars(args))
        else:
            raise ValueError(f'Unrecognised args type: {type(args)}')

    def clone(self) -> 'BaseArgs':
        """
        Clone the arguments into a new class.
        """
        return self.__class__(**self.to_dict())

    def to_dict(self) -> dict:
        """
        Convert the class to a dictionary.
        """
        return to_dict(self)

    def hash(self) -> str:
        """
        Get a hash of the class arguments.
        """
        return hash_data(self.to_dict())
