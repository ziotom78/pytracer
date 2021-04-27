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

from __future__ import annotations

from dataclasses import dataclass

from geometry import Point, Vec
from transformations import Transformation
from misc import are_close


@dataclass
class Ray:
    """A ray of light propagating in space

    The class contains the following members:

    -   `origin` (``Point``): the 3D point where the ray originated
    -   `dir` (``Vec``): the 3D direction along which this ray propagates
    -   `tmin` (float): the minimum distance travelled by the ray is this number times `dir`
    -   `tmax` (float): the maximum distance travelled by the ray is this number times `dir`
    -   `depth` (int): number of times this ray was reflected/refracted"""

    origin: Point = Point()
    dir: Vec = Vec()
    tmin: float = 1e-5
    tmax: float = inf
    depth: int = 0

    def is_close(self, other: Ray, epsilon=1e-5):
        """Check if two rays are similar enough to be considered equal"""
        return (self.origin.is_close(other.origin, epsilon=epsilon) and
                self.dir.is_close(other.dir, epsilon=epsilon))

    def at(self, t):
        """Compute the point along the ray's path at some distance from the origin

        Return a ``Point`` object representing the point in 3D space whose distance from the
        ray's origin is equal to `t`, measured in units of the length of `Vec.dir`."""
        return self.origin + self.dir * t

    def transform(self, transformation: Transformation):
        """Transform a ray

        This method returns a new ray whose origin and direction are the transformation of the original ray"""
        return Ray(origin=transformation * self.origin,
                   dir=transformation * self.dir,
                   tmin=self.tmin,
                   tmax=self.tmax,
                   depth=self.depth)
