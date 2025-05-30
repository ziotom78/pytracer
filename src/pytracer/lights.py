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

from pytracer.colors import Color
from pytracer.geometry import Point


@dataclass
class PointLight:
    """A point light (used by the point-light renderer)

    This class holds information about a point light (a Dirac's delta in the rendering equation). The class has
    the following fields:

    -   `position`: a :class:`Point` object holding the position of the point light in 3D space
    -   `color`: the color of the point light (an instance of :class:`.Color`)
    -   `linear_radius`: a floating-point number. If non-zero, this «linear radius» `r` is used to compute the solid
        angle subtended by the light at a given distance `d` through the formula `(r / d)²`."""

    position: Point
    color: Color
    linear_radius: float = 0.0
