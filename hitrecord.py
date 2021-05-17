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

from dataclasses import dataclass
from typing import Union

from geometry import Point, Normal, Vec2d
from ray import Ray


@dataclass
class HitRecord:
    """
    A class holding information about a ray-shape intersection

    The parameters defined in this dataclass are the following:

    -   `world_point`: a :class:`.Point` object holding the world coordinates of the hit point
    -   `normal`: a :class:`.Normal` object holding the orientation of the normal to the surface where the hit happened
    -   `surface_point`: a :class:`.Vec2d` object holding the position of the hit point on the surface of the object
    -   `t`: a floating-point value specifying the distance from the origin of the ray where the hit happened
    -   `ray`: the ray that hit the surface
    """
    world_point: Point
    normal: Normal
    surface_point: Vec2d
    t: float
    ray: Ray
    shape: "Shape"

    def is_close(self, other: Union["HitRecord", None], epsilon=1e-5) -> bool:
        """Check whether two `HitRecord` represent the same hit event or not"""
        if not other:
            return False

        return (
                self.world_point.is_close(other.world_point) and
                self.normal.is_close(other.normal) and
                self.surface_point.is_close(other.surface_point) and
                (abs(self.t - other.t) < epsilon) and
                self.ray.is_close(other.ray)
        )
