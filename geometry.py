# -*- encoding: utf-8 -*-

import math
from dataclasses import dataclass
from colors import are_close


def _are_xyz_close(a, b):
    # This works thanks to Python's duck typing. In C++ and other languages
    # you should probably rely on function templates or something like
    return are_close(a.x, b.x) and are_close(a.y, b.y) and are_close(a.z, b.z)


def _add_xyz(a, b, return_type):
    # Ditto
    return return_type(a.x + b.x, a.y + b.y, a.z + b.z)


def _sub_xyz(a, b, return_type):
    # Ditto
    return return_type(a.x - b.x, a.y - b.y, a.z - b.z)


def _mul_scalar_xyz(scalar, xyz, return_type):
    return return_type(scalar * xyz.x, scalar * xyz.y, scalar * xyz.z)


@dataclass
class Vec:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other):
        assert isinstance(other, Vec)
        return _are_xyz_close(self, other)

    def __add__(self, other):
        return _add_xyz(self, other, Vec)

    def __sub__(self, other):
        return _sub_xyz(self, other, Vec)

    def __mul__(self, scalar):
        return _mul_scalar_xyz(scalar=scalar, xyz=self, return_type=Vec)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def squared_norm(self):
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def norm(self):
        return math.sqrt(self.squared_norm())

    def cross(self, other):
        return Vec(x=self.y * other.z - self.z * other.y,
                   y=self.z * other.x - self.x * other.z,
                   z=self.x * other.y - self.y * other.x)

    def normalize(self):
        norm = self.norm()
        x /= norm
        y /= norm
        z /= norm

@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other):
        assert isinstance(other, Point)
        return _are_xyz_close(self, other)

    def __add__(self, other):
        return _add_xyz(self, other, Point)

    def __sub__(self, other):
        return _sub_xyz(self, other, Vec)

    def __mul__(self, scalar):
        return _mul_scalar_xyz(scalar=scalar, xyz=self, return_type=Point)
