# -*- encoding: utf-8 -*-

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
