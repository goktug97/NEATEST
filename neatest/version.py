from dataclasses import dataclass


@dataclass(frozen=True)
class Version():
    major: int = 1
    minor: int = 0
    patch: int = 7


VERSION: Version = Version()
