# -*- encoding: utf-8 -*-
#
# The MIT License (MIT)
#
# Copyright © 2021 Maurizio Tomasi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
# SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import math
from dataclasses import dataclass
from misc import are_close


def _are_xyz_close(a, b, epsilon=1e-5):
    # This works thanks to Python's duck typing. In C++ and other languages
    # you should probably rely on function templates or something like
    return (are_close(a.x, b.x, epsilon=epsilon) and
            are_close(a.y, b.y, epsilon=epsilon) and
            are_close(a.z, b.z, epsilon=epsilon))


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
    """A 3D vector.

    This class has three floating-point fields: `x`, `y`, and `z`."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other, epsilon=1e-5):
        """Return True if the object and 'other' have roughly the same direction and orientation"""
        assert isinstance(other, Vec)
        return _are_xyz_close(self, other, epsilon=epsilon)

    def __add__(self, other):
        """Sum two vectors, or one vector and one point"""
        if isinstance(other, Vec):
            return _add_xyz(self, other, Vec)
        elif isinstance(other, Point):
            return _add_xyz(self, other, Point)
        else:
            raise TypeError(f"Unable to run Vec.__add__ on a {type(self)} and a {type(other)}.")

    def __sub__(self, other):
        """Subtract one vector from another"""
        if isinstance(other, Vec):
            return _sub_xyz(self, other, Vec)
        else:
            raise TypeError(f"Unable to run Vec.__sub__ on a {type(self)} and a {type(other)}.")

    def __mul__(self, scalar):
        """Compute the product between a vector and a scalar"""
        return _mul_scalar_xyz(scalar=scalar, xyz=self, return_type=Vec)

    def __getitem__(self, item):
        """Return the i-th component of a vector, starting from 0"""
        return _get_xyz_element(self, item)

    def __neg__(self):
        """Return the reversed vector"""
        return Vec(-self.x, -self.y, -self.z)

    def dot(self, other):
        """Compute the dot product between two vectors"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def squared_norm(self):
        """Return the squared norm (Euclidean length) of a vector

        This is faster than `Vec.norm` if you just need the squared norm."""
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def norm(self):
        """Return the norm (Euclidean length) of a vector"""
        return math.sqrt(self.squared_norm())

    def cross(self, other):
        """Compute the cross (outer) product between two vectors"""
        return Vec(x=self.y * other.z - self.z * other.y,
                   y=self.z * other.x - self.x * other.z,
                   z=self.x * other.y - self.y * other.x)

    def normalize(self):
        """Modify the vector's norm so that it becomes equal to 1"""
        norm = self.norm()
        self.x /= norm
        self.y /= norm
        self.z /= norm


@dataclass
class Point:
    """A point in 3D space

    This class has three floating-point fields: `x`, `y`, and `z`."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def is_close(self, other, epsilon=1e-5):
        """Return True if the object and 'other' have roughly the same position"""
        assert isinstance(other, Point)
        return _are_xyz_close(self, other, epsilon=epsilon)

    def __add__(self, other):
        """Sum a point and a vector"""
        if isinstance(other, Vec):
            return _add_xyz(self, other, Point)
        else:
            raise TypeError(f"Unable to run Point.__add__ on a {type(self)} and a {type(other)}.")

    def __sub__(self, other):
        """Subtract a vector from a point"""
        if isinstance(other, Vec):
            return _sub_xyz(self, other, Point)
        elif isinstance(other, Point):
            return _sub_xyz(self, other, Vec)
        else:
            raise TypeError(f"Unable to run __sub__ on a {type(self)} and a {type(other)}.")

    def __mul__(self, scalar):
        """Multiply the point by a scalar value"""
        return _mul_scalar_xyz(scalar=scalar, xyz=self, return_type=Point)

    def __getitem__(self, item):
        """Return the i-th component of a point, starting from 0"""
        return _get_xyz_element(self, item)

    def to_vec(self):
        """Convert a `Point` into a `Vec`"""
        return Vec(self.x, self.y, self.z)


@dataclass
class Normal:
    """A normal vector in 3D space

    This class has three floating-point fields: `x`, `y`, and `z`."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __neg__(self):
        return Normal(-self.x, -self.y, -self.z)

    def is_close(self, other, epsilon=1e-5):
        """Return True if the object and 'other' have roughly the same direction and orientation"""
        assert isinstance(other, Normal)
        return _are_xyz_close(self, other, epsilon=epsilon)

    def to_vec(self) -> Vec:
        """Convert a normal into a :class:`Vec` type"""
        return Vec(self.x, self.y, self.z)

    def squared_norm(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self):
        return math.sqrt(self.squared_norm())

    def normalize(self):
        norm = self.norm()
        return Normal(self.x / norm, self.y / norm, self.z / norm)


VEC_X = Vec(1.0, 0.0, 0.0)
VEC_Y = Vec(0.0, 1.0, 0.0)
VEC_Z = Vec(0.0, 0.0, 1.0)


@dataclass
class Vec2d:
    """A 2D vector used to represent a point on a surface

    The fields are named `u` and `v` to distinguish them from the usual 3D coordinates `x`, `y`, `z`."""
    u: float = 0.0
    v: float = 0.0

    def is_close(self, other: "Vec2d", epsilon=1e-5):
        """Check whether two `Vec2d` points are roughly the same or not"""
        return (abs(self.u - other.u) < epsilon) and (abs(self.v - other.v) < epsilon)
