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


def _get_xyz_element(self, item):
    assert (item >= 0) and (item < 3), f"wrong vector index {item}"

    if item == 0:
        return self.x
    elif item == 1:
        return self.y

    return self.z


@dataclass
class Vec:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other):
        assert isinstance(other, Vec)
        return _are_xyz_close(self, other)

    def __add__(self, other):
        if isinstance(other, Vec):
            return _add_xyz(self, other, Vec)
        elif isinstance(other, Point):
            return _add_xyz(self, other, Point)
        else:
            raise TypeError(f"Unable to run Vec.__add__ on a {type(self)} and a {type(other)}.")

    def __sub__(self, other):
        if isinstance(other, Vec):
            return _sub_xyz(self, other, Vec)
        else:
            raise TypeError(f"Unable to run Vec.__sub__ on a {type(self)} and a {type(other)}.")

    def __mul__(self, scalar):
        return _mul_scalar_xyz(scalar=scalar, xyz=self, return_type=Vec)

    def __getitem__(self, item):
        return _get_xyz_element(self, item)

    def __neg__(self):
        return Vec(-self.x, -self.y, -self.z)

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
        if isinstance(other, Vec):
            return _add_xyz(self, other, Point)
        else:
            raise TypeError(f"Unable to run Point.__add__ on a {type(self)} and a {type(other)}.")

    def __sub__(self, other):
        if isinstance(other, Vec):
            return _sub_xyz(self, other, Point)
        elif isinstance(other, Point):
            return _sub_xyz(self, other, Vec)
        else:
            raise TypeError(f"Unable to run __sub__ on a {type(self)} and a {type(other)}.")

    def __mul__(self, scalar):
        return _mul_scalar_xyz(scalar=scalar, xyz=self, return_type=Point)

    def __getitem__(self, item):
        return _get_xyz_element(self, item)


@dataclass
class Normal:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other):
        assert isinstance(other, Normal)
        return _are_xyz_close(self, other)


VEC_X = Vec(1.0, 0.0, 0.0)
VEC_Y = Vec(0.0, 1.0, 0.0)
VEC_Z = Vec(0.0, 0.0, 1.0)