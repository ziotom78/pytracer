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

from colors import Color, WHITE, BLACK
from ray import Ray
from world import World


class Renderer:
    """A class implementing a solver of the rendering equation.

    This is an abstract class; you should use a derived concrete class."""

    def __init__(self, world: World, background_color: Color = BLACK):
        self.world = world
        self.background_color = background_color

    def __call__(self, ray: Ray) -> Color:
        """Estimate the radiance along a ray"""
        raise NotImplementedError("Unable to call Renderer.radiance, it is an abstract method")


class OnOffRenderer(Renderer):
    """A on/off renderer

    This renderer is mostly useful for debugging purposes, as it is really fast, but it produces boring images."""

    def __init__(self, world: World, background_color: Color = BLACK, color=WHITE):
        super().__init__(world, background_color)
        self.world = world
        self.color = color

    def __call__(self, ray: Ray) -> Color:
        return self.color if self.world.ray_intersection(ray) else self.background_color


class FlatRenderer(Renderer):
    """A «flat» renderer

    This renderer estimates the solution of the rendering equation by neglecting any contribution of the light.
    It just uses the pigment of each surface to determine how to compute the final radiance."""

    def __init__(self, world: World, background_color: Color = BLACK):
        super().__init__(world, background_color)

    def __call__(self, ray: Ray) -> Color:
        hit = self.world.ray_intersection(ray)
        if not hit:
            return self.background_color

        material = hit.shape.material

        return (material.brdf.pigment.get_color(hit.surface_point) +
                material.emitted_radiance.get_color(hit.surface_point))
