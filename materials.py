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

from math import floor, pi

from colors import Color, BLACK, WHITE
from geometry import Normal, Vec, Vec2d
from hdrimages import HdrImage


class Pigment:
    """A «pigment»

    This abstract class represents a pigment, i.e., a function that associates a color with
    each point on a parametric surface (u,v). Call the method :meth:`.Pigment.get_color` to
    retrieve the color of the surface given a :class:`.Vec2d` object."""

    def get_color(self, uv: Vec2d) -> Color:
        """Return the color of the pigment at the specified coordinates"""
        raise NotImplementedError("Method Pigment.get_color is abstract and cannot be called")


class UniformPigment(Pigment):
    """A uniform pigment

    This is the most boring pigment: a uniform hue over the whole surface."""

    def __init__(self, color=Color()):
        self.color = color

    def get_color(self, uv: Vec2d) -> Color:
        return self.color


class ImagePigment(Pigment):
    """A textured pigment

    The texture is given through a PFM image."""

    def __init__(self, image: HdrImage):
        self.image = image

    def get_color(self, uv: Vec2d) -> Color:
        col = int(uv.u * self.image.width)
        row = int(uv.v * self.image.height)

        if col >= self.image.width:
            col = self.image.width - 1

        if row >= self.image.height:
            row = self.image.height - 1

        # A nicer solution would implement bilinear interpolation to reduce pixelization artifacts
        # See https://en.wikipedia.org/wiki/Bilinear_interpolation
        return self.image.get_pixel(col, row)


class CheckeredPigment(Pigment):
    """A checkered pigment

    The number of rows/columns in the checkered pattern is tunable, but you cannot have a different number of
    repetitions along the u/v directions."""

    def __init__(self, color1: Color, color2: Color, num_of_steps=10):
        self.color1 = color1
        self.color2 = color2
        self.num_of_steps = num_of_steps

    def get_color(self, uv: Vec2d) -> Color:
        int_u = int(floor(uv.u * self.num_of_steps))
        int_v = int(floor(uv.v * self.num_of_steps))

        return self.color1 if ((int_u % 2) == (int_v % 2)) else self.color2


class BRDF:
    """An abstract class representing a Bidirectional Reflectance Distribution Function"""
    def __init__(self, pigment: Pigment = UniformPigment(WHITE)):
        self.pigment = pigment

    def eval(self, normal: Normal, in_dir: Vec, out_dir: Vec, uv: Vec2d) -> Color:
        return BLACK


class DiffuseBRDF(BRDF):
    """A class representing an ideal diffuse BRDF (also called «Lambertian»)"""
    def __init__(self, pigment: Pigment = UniformPigment(WHITE), reflectance: float = 1.0):
        super().__init__(pigment)
        self.reflectance = reflectance

    def eval(self, normal: Normal, in_dir: Vec, out_dir: Vec, uv: Vec2d) -> Color:
        return self.pigment.get_color(uv) * (self.reflectance / pi)


@dataclass
class Material:
    """A material"""
    brdf: BRDF = DiffuseBRDF()
    emitted_radiance: Pigment = UniformPigment(BLACK)
