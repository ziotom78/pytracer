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
from pytracer.geometry import Point, Vec, VEC_X
from pytracer.ray import Ray
from pytracer.transformations import Transformation


class Camera:
    """An abstract class representing an observer

    Concrete subclasses are `OrthogographicCamera` and `PerspectiveCamera`.
    """

    def fire_ray(self, u, v):
        """Fire a ray through the camera.

        This is an abstract method. You should redefine it in derived classes.

        Fire a ray that goes through the screen at the position (u, v). The exact meaning
        of these coordinates depend on the projection used by the camera.
        """
        raise NotImplementedError(f"Camera.fire_ray(u={u}, v={v}) is not implemented")


class OrthogonalCamera(Camera):
    """A camera implementing an orthogonal 3D → 2D projection

    This class implements an observer seeing the world through an orthogonal projection.
    """

    def __init__(self, aspect_ratio=1.0, transformation=Transformation()):
        """Create a new orthographic camera

        The parameter `aspect_ratio` defines how larger than the height is the image. For fullscreen
        images, you should probably set `aspect_ratio` to 16/9, as this is the most used aspect ratio
        used in modern monitors.

        The `transformation` parameter is an instance of the :class:`.Transformation` class."""
        self.aspect_ratio = aspect_ratio
        self.transformation = transformation

    def fire_ray(self, u, v):
        """Shoot a ray through the camera's screen

        The coordinates (u, v) specify the point on the screen where the ray crosses it. Coordinates (0, 0) represent
        the bottom-left corner, (0, 1) the top-left corner, (1, 0) the bottom-right corner, and (1, 1) the top-right
        corner, as in the following diagram::

            (0, 1)                          (1, 1)
               +------------------------------+
               |                              |
               |                              |
               |                              |
               +------------------------------+
            (0, 0)                          (1, 0)
        """
        origin = Point(-1.0, (1.0 - 2 * u) * self.aspect_ratio, 2 * v - 1)
        direction = VEC_X
        return Ray(origin=origin, dir=direction, tmin=1.0e-5).transform(
            self.transformation
        )


class PerspectiveCamera(Camera):
    """A camera implementing a perspective 3D → 2D projection

    This class implements an observer seeing the world through a perspective projection.
    """

    def __init__(
        self, screen_distance=1.0, aspect_ratio=1.0, transformation=Transformation()
    ):
        """Create a new perspective camera

        The parameter `screen_distance` tells how much far from the eye of the observer is the screen,
        and it influences the so-called «aperture» (the field-of-view angle along the horizontal direction).
        The parameter `aspect_ratio` defines how larger than the height is the image. For fullscreen
        images, you should probably set `aspect_ratio` to 16/9, as this is the most used aspect ratio
        used in modern monitors.

        The `transformation` parameter is an instance of the :class:`.Transformation` class."""
        self.screen_distance = screen_distance
        self.aspect_ratio = aspect_ratio
        self.transformation = transformation

    def fire_ray(self, u, v):
        """Shoot a ray through the camera's screen

        The coordinates (u, v) specify the point on the screen where the ray crosses it. Coordinates (0, 0) represent
        the bottom-left corner, (0, 1) the top-left corner, (1, 0) the bottom-right corner, and (1, 1) the top-right
        corner, as in the following diagram::

            (0, 1)                          (1, 1)
               +------------------------------+
               |                              |
               |                              |
               |                              |
               +------------------------------+
            (0, 0)                          (1, 0)
        """
        origin = Point(-self.screen_distance, 0.0, 0.0)
        direction = Vec(
            self.screen_distance, (1.0 - 2 * u) * self.aspect_ratio, 2 * v - 1
        )
        return Ray(origin=origin, dir=direction, tmin=1.0e-5).transform(
            self.transformation
        )

    def aperture_deg(self):
        """Compute the aperture of the camera in degrees

        The aperture is the angle of the field-of-view along the horizontal direction (Y axis)"""
        return (
            2.0
            * math.atan(self.screen_distance / self.aspect_ratio)
            * 180.0
            / 3.14159265359
        )
