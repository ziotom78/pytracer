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

from hitrecord import HitRecord
from ray import Ray
from shapes import Shape


class World:
    """A class holding a list of shapes, which make a «world»

    You can add shapes to a world using :meth:`.World.add`. Typically, you call
    :meth:`.World.ray_intersection` to check whether a light ray intersects any
    of the shapes in the world.
    """

    def __init__(self):
        self.shapes = []

    def add(self, shape: Shape):
        """Append a new shape to this world"""
        self.shapes.append(shape)

    def ray_intersection(self, ray: Ray) -> HitRecord:
        """Determine whether a ray intersects any of the objects in this world"""
        closest = None

        for shape in self.shapes:
            intersection = shape.ray_intersection(ray)

            if not intersection:
                # The ray missed this shape, skip to the next one
                continue

            if (not closest) or (intersection.t < closest.t):
                # There was a hit, and it was closer than any other hit found before
                closest = intersection

        return closest
