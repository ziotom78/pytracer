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

from pytracer.colors import Color, WHITE, BLACK
from pytracer.geometry import normalized_dot
from pytracer.pcg import PCG
from pytracer.ray import Ray
from pytracer.world import World


class Renderer:
    """A class implementing a solver of the rendering equation.

    This is an abstract class; you should use a derived concrete class."""

    def __init__(self, world: World, background_color: Color = BLACK):
        self.world = world
        self.background_color = background_color

    def __call__(self, ray: Ray) -> Color:
        """Estimate the radiance along a ray"""
        raise NotImplementedError(
            "Unable to call Renderer.radiance, it is an abstract method"
        )


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

        material = hit.material

        return material.brdf.pigment.get_color(
            hit.surface_point
        ) + material.emitted_radiance.get_color(hit.surface_point)


class PathTracer(Renderer):
    """A simple path-tracing renderer

    The algorithm implemented here allows the caller to tune number of rays thrown at each iteration, as well as the
    maximum depth. It implements Russian roulette to reduce the number of recursive calls.
    """

    def __init__(
        self,
        world: World,
        background_color: Color = BLACK,
        pcg: PCG = PCG(),
        num_of_rays: int = 10,
        max_depth: int = 10,
        russian_roulette_limit=3,
    ):
        super().__init__(world, background_color)
        self.pcg = pcg
        self.num_of_rays = num_of_rays
        self.max_depth = max_depth
        self.russian_roulette_limit = russian_roulette_limit

    def __call__(self, ray: Ray) -> Color:
        if ray.depth > self.max_depth:
            return Color(0.0, 0.0, 0.0)

        hit_record = self.world.ray_intersection(ray)
        if not hit_record:
            return self.background_color

        hit_material = hit_record.material
        hit_color = hit_material.brdf.pigment.get_color(hit_record.surface_point)
        emitted_radiance = hit_material.emitted_radiance.get_color(
            hit_record.surface_point
        )

        hit_color_lum = max(hit_color.r, hit_color.g, hit_color.b)

        # Russian roulette
        if ray.depth >= self.russian_roulette_limit:
            q = max(0.05, 1 - hit_color_lum)
            if self.pcg.random_float() > q:
                # Keep the recursion going, but compensate for other potentially discarded rays
                hit_color *= 1.0 / (1.0 - q)
            else:
                # Terminate prematurely
                return emitted_radiance

        cum_radiance = Color(0.0, 0.0, 0.0)
        if hit_color_lum > 0.0:  # Only do costly recursions if it's worth it
            for ray_index in range(self.num_of_rays):
                new_ray = hit_material.brdf.scatter_ray(
                    pcg=self.pcg,
                    incoming_dir=hit_record.ray.dir,
                    interaction_point=hit_record.world_point,
                    normal=hit_record.normal,
                    depth=ray.depth + 1,
                )
                # Recursive call
                new_radiance = self(new_ray)
                cum_radiance += hit_color * new_radiance

        return emitted_radiance + cum_radiance * (1.0 / self.num_of_rays)


class PointLightRenderer(Renderer):
    """A simple point-light renderer

    This renderer is similar to what POV-Ray provides by default.
    """

    def __init__(
        self,
        world: World,
        background_color: Color = BLACK,
        ambient_color: Color = Color(0.1, 0.1, 0.1),
    ):
        super().__init__(world, background_color)
        self.ambient_color = ambient_color

    def __call__(self, ray: Ray) -> Color:
        hit_record = self.world.ray_intersection(ray)
        if not hit_record:
            return self.background_color

        hit_material = hit_record.material

        emitted_color = hit_material.emitted_radiance.get_color(
            hit_record.surface_point
        )
        result_color = self.ambient_color + emitted_color
        for cur_light in self.world.point_lights:
            if self.world.is_point_visible(
                point=cur_light.position, observer_pos=hit_record.world_point
            ):
                distance_vec = hit_record.world_point - cur_light.position
                distance = distance_vec.norm()
                in_dir = distance_vec * (1.0 / distance)
                cos_theta = max(0.0, normalized_dot(-in_dir, hit_record.normal))

                distance_factor = (
                    (cur_light.linear_radius / distance) ** 2
                    if (cur_light.linear_radius > 0)
                    else 1.0
                )

                brdf_color = hit_material.brdf.eval(
                    normal=hit_record.normal,
                    in_dir=in_dir,
                    out_dir=-ray.dir,
                    uv=hit_record.surface_point,
                )
                result_color += (
                    brdf_color * cur_light.color * cos_theta * distance_factor
                )

        return result_color
