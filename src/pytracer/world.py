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

from typing import Union, List

from pytracer.geometry import Point
from pytracer.hitrecord import HitRecord
from pytracer.lights import PointLight
from pytracer.ray import Ray
from pytracer.shapes import Shape


class World:
    """A class holding a list of shapes, which make a «world»

    You can add shapes to a world using :meth:`.World.add`. Typically, you call
    :meth:`.World.ray_intersection` to check whether a light ray intersects any
    of the shapes in the world.
    """

    shapes: List[Shape]
    point_lights: List[PointLight]

    def __init__(self):
        self.shapes = []
        self.point_lights = []

    def add_shape(self, shape: Shape):
        """Append a new shape to this world"""
        self.shapes.append(shape)

    def add_light(self, light: PointLight):
        """Append a new point light to this world"""
        self.point_lights.append(light)

    def ray_intersection(self, ray: Ray) -> Union[HitRecord, None]:
        """Determine whether a ray intersects any of the objects in this world"""
        closest: Union[HitRecord, None] = None

        for shape in self.shapes:
            intersection = shape.ray_intersection(ray)

            if not intersection:
                # The ray missed this shape, skip to the next one
                continue

            if (not closest) or (intersection.t < closest.t):
                # There was a hit, and it was closer than any other hit found before
                closest = intersection

        if closest:
            closest.normal = closest.normal.normalize()

        return closest

    def is_point_visible(self, point: Point, observer_pos: Point):
        direction = point - observer_pos
        dir_norm = direction.norm()

        ray = Ray(origin=observer_pos, dir=direction, tmin=1e-2 / dir_norm, tmax=1.0)
        for shape in self.shapes:
            if shape.quick_ray_intersection(ray):
                return False

        return True
