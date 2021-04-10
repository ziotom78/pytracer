# -*- encoding: utf-8 -*-

from dataclasses import dataclass
from colors import are_close


def _are_xyz_close(a, b):
    # This works thanks to Python's duck typing. In C++ and other languages
    # you should probably rely on function templates or something like
    return are_close(a.x, b.x) and are_close(a.y, b.y) and are_close(a.z, b.z)


@dataclass
class Vec:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other):
        return _are_xyz_close(self, other)


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other):
        return _are_xyz_close(self, other)
