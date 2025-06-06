# -*- encoding: utf-8 -*-
#
# The MIT License (MIT)
#
# Copyright © 2021 Maurizio Tomasi
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the “Software”), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software. THE
# SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import sqrt, atan2, acos, pi, floor
from typing import Union

from pytracer.geometry import Point, Vec, Normal
from pytracer.hitrecord import Vec2d, HitRecord
from pytracer.ray import Ray
from pytracer.transformations import Transformation
from pytracer.materials import Material


def _sphere_point_to_uv(point: Point) -> Vec2d:
    """Convert a 3D point on the surface of the unit sphere into a (u, v) 2D point"""
    u = atan2(point.y, point.x) / (2.0 * pi)
    return Vec2d(
        u=u if u >= 0.0 else u + 1.0,
        v=acos(point.z) / pi,
    )


def _sphere_normal(point: Point, ray_dir: Vec) -> Normal:
    """Compute the normal of a unit sphere

    The normal is computed for `point` (a point on the surface of the
    sphere), and it is chosen so that it is always in the opposite
    direction with respect to `ray_dir`.

    """
    result = Normal(point.x, point.y, point.z)
    return result if (point.to_vec().dot(ray_dir) < 0.0) else -result


class Shape:
    """A generic 3D shape

    This is an abstract class, and you should only use it to derive
    concrete classes. Be sure to redefine the method
    :meth:`.Shape.ray_intersection`.

    """

    def __init__(
        self,
        transformation: Transformation = Transformation(),
        material: Material = Material(),
    ):
        """Create a shape, potentially associating a transformation to it"""
        self.transformation = transformation
        self.material = material

    def ray_intersection(self, ray: Ray) -> Union[HitRecord, None]:
        """Compute the intersection between a ray and this shape"""
        raise NotImplementedError(
            "Shape.ray_intersection is an abstract method and cannot be called directly"
        )

    def quick_ray_intersection(self, ray: Ray) -> bool:
        """Determine whether a ray hits the shape or not"""
        raise NotImplementedError(
            "Shape.quick_ray_intersection is an abstract method and cannot be called directly"
        )


class Sphere(Shape):
    """A 3D unit sphere centered on the origin of the axes"""

    def __init__(
        self, transformation=Transformation(), material: Material = Material()
    ):
        """Create a unit sphere, potentially associating a transformation to it"""
        super().__init__(transformation, material)

    def ray_intersection(self, ray: Ray) -> Union[HitRecord, None]:
        """Checks if a ray intersects the sphere

        Return a `HitRecord`, or `None` if no intersection was found.
        """
        inv_ray = ray.transform(self.transformation.inverse())
        origin_vec = inv_ray.origin.to_vec()
        a = inv_ray.dir.squared_norm()
        b = 2.0 * origin_vec.dot(inv_ray.dir)
        c = origin_vec.squared_norm() - 1.0

        delta = b * b - 4.0 * a * c
        if delta <= 0.0:
            return None

        sqrt_delta = sqrt(delta)
        tmin = (-b - sqrt_delta) / (2.0 * a)
        tmax = (-b + sqrt_delta) / (2.0 * a)

        if (tmin > inv_ray.tmin) and (tmin < inv_ray.tmax):
            first_hit_t = tmin
        elif (tmax > inv_ray.tmin) and (tmax < inv_ray.tmax):
            first_hit_t = tmax
        else:
            return None

        hit_point = inv_ray.at(first_hit_t)
        return HitRecord(
            world_point=self.transformation * hit_point,
            normal=self.transformation * _sphere_normal(hit_point, inv_ray.dir),
            surface_point=_sphere_point_to_uv(hit_point),
            t=first_hit_t,
            ray=ray,
            material=self.material,
        )

    def quick_ray_intersection(self, ray: Ray) -> bool:
        """Quickly checks if a ray intersects the sphere"""
        inv_ray = ray.transform(self.transformation.inverse())
        origin_vec = inv_ray.origin.to_vec()
        a = inv_ray.dir.squared_norm()
        b = 2.0 * origin_vec.dot(inv_ray.dir)
        c = origin_vec.squared_norm() - 1.0

        delta = b * b - 4.0 * a * c
        if delta <= 0.0:
            return False

        sqrt_delta = sqrt(delta)
        tmin = (-b - sqrt_delta) / (2.0 * a)
        tmax = (-b + sqrt_delta) / (2.0 * a)

        return (inv_ray.tmin < tmin < inv_ray.tmax) or (
            inv_ray.tmin < tmax < inv_ray.tmax
        )


class Plane(Shape):
    """A 3D infinite plane parallel to the x and y axis and passing through the origin."""

    def __init__(
        self, transformation=Transformation(), material: Material = Material()
    ):
        """Create a xy plane, potentially associating a transformation to it"""
        super().__init__(transformation, material)

    def ray_intersection(self, ray: Ray) -> Union[HitRecord, None]:
        """Checks if a ray intersects the plane

        Return a `HitRecord`, or `None` if no intersection was found.
        """
        inv_ray = ray.transform(self.transformation.inverse())
        if abs(inv_ray.dir.z) < 1e-5:
            return None

        t = -inv_ray.origin.z / inv_ray.dir.z

        if (t <= inv_ray.tmin) or (t >= inv_ray.tmax):
            return None

        hit_point = inv_ray.at(t)

        return HitRecord(
            world_point=self.transformation * hit_point,
            normal=self.transformation
            * Normal(0.0, 0.0, 1.0 if inv_ray.dir.z < 0.0 else -1.0),
            surface_point=Vec2d(
                hit_point.x - floor(hit_point.x), hit_point.y - floor(hit_point.y)
            ),
            t=t,
            ray=ray,
            material=self.material,
        )

    def quick_ray_intersection(self, ray: Ray) -> bool:
        """Quickly checks if a ray intersects the plane"""
        inv_ray = ray.transform(self.transformation.inverse())
        if abs(inv_ray.dir.z) < 1e-5:
            return False

        t = -inv_ray.origin.z / inv_ray.dir.z
        return inv_ray.tmin < t < inv_ray.tmax
