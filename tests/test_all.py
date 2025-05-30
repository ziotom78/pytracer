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

from copy import deepcopy
from math import pi, sqrt

import unittest
from io import BytesIO, StringIO
from colors import Color, BLACK, WHITE
from hdrimages import (
    HdrImage,
    InvalidPfmFileFormat,
    Endianness,
    read_pfm_image,
    _read_line,
    _parse_img_size,
    _parse_endianness,
)
from geometry import Vec, Point, Normal, VEC_X, VEC_Y, VEC_Z, create_onb_from_z
from scene_file import InputStream, KeywordEnum, Token, KeywordToken, IdentifierToken, SymbolToken, LiteralNumberToken, \
    StringToken, parse_scene, GrammarError
from transformations import (
    Transformation,
    translation,
    scaling,
    rotation_x,
    rotation_y,
    rotation_z,
)
from camera import OrthogonalCamera, PerspectiveCamera
from ray import Ray
from imagetracer import ImageTracer
from hitrecord import HitRecord, Vec2d
from shapes import Sphere, Plane
from misc import are_close
from world import World
from pcg import PCG
from materials import UniformPigment, ImagePigment, CheckeredPigment, DiffuseBRDF, Material, SpecularBRDF
from render import OnOffRenderer, FlatRenderer, PathTracer

import pytest


class TestColor(unittest.TestCase):
    def test_create(self):
        col = Color(1.0, 2.0, 3.0)
        assert col.is_close(Color(1.0, 2.0, 3.0))

    def test_close(self):
        col = Color(1.0, 2.0, 3.0)
        assert not col.is_close(Color(3.0, 4.0, 5.0))

    def test_operations(self):
        col1 = Color(1.0, 2.0, 3.0)
        col2 = Color(5.0, 7.0, 9.0)

        assert (col1 + col2).is_close(Color(6.0, 9.0, 12.0))
        assert (col1 * col2).is_close(Color(5.0, 14.0, 27.0))

        prod_col = Color(1.0, 2.0, 3.0) * 2.0
        assert prod_col.is_close(Color(2.0, 4.0, 6.0))

    def test_luminosity(self):
        col1 = Color(1.0, 2.0, 3.0)
        col2 = Color(9.0, 5.0, 7.0)

        assert pytest.approx(2.0) == col1.luminosity()
        assert pytest.approx(7.0) == col2.luminosity()


# fmt: off

# This is the content of "reference_le.pfm" (little-endian file)
LE_REFERENCE_BYTES = bytes(
    [
        0x50, 0x46, 0x0A, 0x33, 0x20, 0x32, 0x0A, 0x2D,
        0x31, 0x2E, 0x30, 0x0A, 0x00, 0x00, 0xC8, 0x42,
        0x00, 0x00, 0x48, 0x43, 0x00, 0x00, 0x96, 0x43,
        0x00, 0x00, 0xC8, 0x43, 0x00, 0x00, 0xFA, 0x43,
        0x00, 0x00, 0x16, 0x44, 0x00, 0x00, 0x2F, 0x44,
        0x00, 0x00, 0x48, 0x44, 0x00, 0x00, 0x61, 0x44,
        0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0xA0, 0x41,
        0x00, 0x00, 0xF0, 0x41, 0x00, 0x00, 0x20, 0x42,
        0x00, 0x00, 0x48, 0x42, 0x00, 0x00, 0x70, 0x42,
        0x00, 0x00, 0x8C, 0x42, 0x00, 0x00, 0xA0, 0x42,
        0x00, 0x00, 0xB4, 0x42,
    ]
)

# This is the content of "reference_be.pfm" (big-endian file)
BE_REFERENCE_BYTES = bytes(
    [
        0x50, 0x46, 0x0A, 0x33, 0x20, 0x32, 0x0A, 0x31,
        0x2E, 0x30, 0x0A, 0x42, 0xC8, 0x00, 0x00, 0x43,
        0x48, 0x00, 0x00, 0x43, 0x96, 0x00, 0x00, 0x43,
        0xC8, 0x00, 0x00, 0x43, 0xFA, 0x00, 0x00, 0x44,
        0x16, 0x00, 0x00, 0x44, 0x2F, 0x00, 0x00, 0x44,
        0x48, 0x00, 0x00, 0x44, 0x61, 0x00, 0x00, 0x41,
        0x20, 0x00, 0x00, 0x41, 0xA0, 0x00, 0x00, 0x41,
        0xF0, 0x00, 0x00, 0x42, 0x20, 0x00, 0x00, 0x42,
        0x48, 0x00, 0x00, 0x42, 0x70, 0x00, 0x00, 0x42,
        0x8C, 0x00, 0x00, 0x42, 0xA0, 0x00, 0x00, 0x42,
        0xB4, 0x00, 0x00,
    ]
)


# fmt: on


class TestHdrImage(unittest.TestCase):
    def test_image_creation(self):
        img = HdrImage(7, 4)
        assert img.width == 7
        assert img.height == 4

    def test_coordinates(self):
        img = HdrImage(7, 4)

        assert img.valid_coordinates(0, 0)
        assert img.valid_coordinates(6, 3)
        assert not img.valid_coordinates(-1, 0)
        assert not img.valid_coordinates(0, -1)

    def test_pixel_offset(self):
        img = HdrImage(7, 4)

        assert img.pixel_offset(0, 0) == 0
        assert img.pixel_offset(3, 2) == 17
        assert img.pixel_offset(6, 3) == 7 * 4 - 1

    def test_get_set_pixel(self):
        img = HdrImage(7, 4)

        reference_color = Color(1.0, 2.0, 3.0)
        img.set_pixel(3, 2, reference_color)
        assert reference_color.is_close(img.get_pixel(3, 2))

    def test_pfm_save(self):
        img = HdrImage(3, 2)

        img.set_pixel(0, 0, Color(1.0e1, 2.0e1, 3.0e1))
        img.set_pixel(1, 0, Color(4.0e1, 5.0e1, 6.0e1))
        img.set_pixel(2, 0, Color(7.0e1, 8.0e1, 9.0e1))
        img.set_pixel(0, 1, Color(1.0e2, 2.0e2, 3.0e2))
        img.set_pixel(1, 1, Color(4.0e2, 5.0e2, 6.0e2))
        img.set_pixel(2, 1, Color(7.0e2, 8.0e2, 9.0e2))

        le_buf = BytesIO()
        img.write_pfm(le_buf, endianness=Endianness.LITTLE_ENDIAN)
        assert le_buf.getvalue() == LE_REFERENCE_BYTES

        be_buf = BytesIO()
        img.write_pfm(be_buf, endianness=Endianness.BIG_ENDIAN)
        assert be_buf.getvalue() == BE_REFERENCE_BYTES

    def test_pfm_read_line(self):
        line = BytesIO(b"hello\nworld")
        assert _read_line(line) == "hello"
        assert _read_line(line) == "world"
        assert _read_line(line) == ""

    def test_pfm_parse_img_size(self):
        assert _parse_img_size("3 2") == (3, 2)

        with pytest.raises(InvalidPfmFileFormat):
            _ = _parse_img_size("-1 3")

        with pytest.raises(InvalidPfmFileFormat):
            _ = _parse_img_size("3 2 1")

    def test_pfm_parse_endianness(self):
        assert _parse_endianness("1.0") == Endianness.BIG_ENDIAN
        assert _parse_endianness("-1.0") == Endianness.LITTLE_ENDIAN

        with pytest.raises(InvalidPfmFileFormat):
            _ = _parse_endianness("2.0")

        with pytest.raises(InvalidPfmFileFormat):
            _ = _parse_endianness("abc")

    def test_pfm_read(self):
        for reference_bytes in [LE_REFERENCE_BYTES, BE_REFERENCE_BYTES]:
            img = read_pfm_image(BytesIO(reference_bytes))
            assert img.width == 3
            assert img.height == 2

            assert img.get_pixel(0, 0).is_close(Color(1.0e1, 2.0e1, 3.0e1))
            assert img.get_pixel(1, 0).is_close(Color(4.0e1, 5.0e1, 6.0e1))
            assert img.get_pixel(2, 0).is_close(Color(7.0e1, 8.0e1, 9.0e1))
            assert img.get_pixel(0, 1).is_close(Color(1.0e2, 2.0e2, 3.0e2))
            assert img.get_pixel(0, 0).is_close(Color(1.0e1, 2.0e1, 3.0e1))
            assert img.get_pixel(1, 1).is_close(Color(4.0e2, 5.0e2, 6.0e2))
            assert img.get_pixel(2, 1).is_close(Color(7.0e2, 8.0e2, 9.0e2))

    def test_pfm_read_wrong(self):
        buf = BytesIO(b"PF\n3 2\n-1.0\nstop")
        with pytest.raises(InvalidPfmFileFormat):
            _ = read_pfm_image(buf)

    def test_average_luminosity(self):
        img = HdrImage(2, 1)

        img.set_pixel(0, 0, Color(0.5e1, 1.0e1, 1.5e1))
        img.set_pixel(1, 0, Color(0.5e3, 1.0e3, 1.5e3))

        assert pytest.approx(100.0) == img.average_luminosity(delta=0.0)

    def test_normalize_image(self):
        img = HdrImage(2, 1)

        img.set_pixel(0, 0, Color(0.5e1, 1.0e1, 1.5e1))
        img.set_pixel(1, 0, Color(0.5e3, 1.0e3, 1.5e3))

        img.normalize_image(factor=1000.0, luminosity=100.0)
        assert img.get_pixel(0, 0).is_close(Color(0.5e2, 1.0e2, 1.5e2))
        assert img.get_pixel(1, 0).is_close(Color(0.5e4, 1.0e4, 1.5e4))

    def test_clamp_image(self):
        img = HdrImage(2, 1)

        img.set_pixel(0, 0, Color(0.5e1, 1.0e1, 1.5e1))
        img.set_pixel(1, 0, Color(0.5e3, 1.0e3, 1.5e3))

        img.clamp_image()

        for cur_pixel in img.pixels:
            assert (cur_pixel.r >= 0) and (cur_pixel.r <= 1)
            assert (cur_pixel.g >= 0) and (cur_pixel.g <= 1)
            assert (cur_pixel.b >= 0) and (cur_pixel.b <= 1)


class TestGeometry(unittest.TestCase):
    def test_vectors(self):
        a = Vec(1.0, 2.0, 3.0)
        b = Vec(4.0, 6.0, 8.0)
        assert a.is_close(a)
        assert not a.is_close(b)

    def test_vector_operations(self):
        a = Vec(1.0, 2.0, 3.0)
        b = Vec(4.0, 6.0, 8.0)
        assert (-a).is_close(Vec(-1.0, -2.0, -3.0))
        assert (a + b).is_close(Vec(5.0, 8.0, 11.0))
        assert (b - a).is_close(Vec(3.0, 4.0, 5.0))
        assert (a * 2).is_close(Vec(2.0, 4.0, 6.0))
        assert pytest.approx(40.0) == a.dot(b)
        assert a.cross(b).is_close(Vec(-2.0, 4.0, -2.0))
        assert b.cross(a).is_close(Vec(2.0, -4.0, 2.0))
        assert pytest.approx(14.0) == a.squared_norm()
        assert pytest.approx(14.0) == a.norm() ** 2

    def test_points(self):
        a = Point(1.0, 2.0, 3.0)
        b = Point(4.0, 6.0, 8.0)
        assert a.is_close(a)
        assert not a.is_close(b)

    def test_point_operations(self):
        p1 = Point(1.0, 2.0, 3.0)
        v = Vec(4.0, 6.0, 8.0)
        p2 = Point(4.0, 6.0, 8.0)
        assert (p1 * 2).is_close(Point(2.0, 4.0, 6.0))
        assert (p1 + v).is_close(Point(5.0, 8.0, 11.0))
        assert (p2 - p1).is_close(Vec(3.0, 4.0, 5.0))
        assert (p1 - v).is_close(Point(-3.0, -4.0, -5.0))


class TestTransformations(unittest.TestCase):
    def test_is_close(self):
        m1 = Transformation(
            m=[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 9.0, 8.0, 7.0],
                [6.0, 5.0, 4.0, 1.0],
            ],
            invm=[
                [-3.75, 2.75, -1, 0],
                [4.375, -3.875, 2.0, -0.5],
                [0.5, 0.5, -1.0, 1.0],
                [-1.375, 0.875, 0.0, -0.5],
            ],
        )

        assert m1.is_consistent()

        # Not using "deepcopy" here would make Python pass a pointer to the *same* matrices and vectors
        m2 = Transformation(m=deepcopy(m1.m), invm=deepcopy(m1.invm))
        assert m1.is_close(m2)

        m3 = Transformation(m=deepcopy(m1.m), invm=deepcopy(m1.invm))
        m3.m[2][
            2
        ] += 1.0  # Note: this makes "m3" not consistent (m3.is_consistent() == False)
        assert not m1.is_close(m3)

        m4 = Transformation(m=deepcopy(m1.m), invm=deepcopy(m1.invm))
        m4.invm[2][
            2
        ] += 1.0  # Note: this makes "m4" not consistent (m4.is_consistent() == False)
        assert not m1.is_close(m4)

    def test_multiplication(self):
        m1 = Transformation(
            m=[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 9.0, 8.0, 7.0],
                [6.0, 5.0, 4.0, 1.0],
            ],
            invm=[
                [-3.75, 2.75, -1, 0],
                [4.375, -3.875, 2.0, -0.5],
                [0.5, 0.5, -1.0, 1.0],
                [-1.375, 0.875, 0.0, -0.5],
            ],
        )
        assert m1.is_consistent()

        m2 = Transformation(
            m=[
                [3.0, 5.0, 2.0, 4.0],
                [4.0, 1.0, 0.0, 5.0],
                [6.0, 3.0, 2.0, 0.0],
                [1.0, 4.0, 2.0, 1.0],
            ],
            invm=[
                [0.4, -0.2, 0.2, -0.6],
                [2.9, -1.7, 0.2, -3.1],
                [-5.55, 3.15, -0.4, 6.45],
                [-0.9, 0.7, -0.2, 1.1],
            ],
        )
        assert m2.is_consistent()

        expected = Transformation(
            m=[
                [33.0, 32.0, 16.0, 18.0],
                [89.0, 84.0, 40.0, 58.0],
                [118.0, 106.0, 48.0, 88.0],
                [63.0, 51.0, 22.0, 50.0],
            ],
            invm=[
                [-1.45, 1.45, -1.0, 0.6],
                [-13.95, 11.95, -6.5, 2.6],
                [25.525, -22.025, 12.25, -5.2],
                [4.825, -4.325, 2.5, -1.1],
            ],
        )
        assert expected.is_consistent()

        assert expected.is_close(m1 * m2)

    def test_vec_point_multiplication(self):
        m = Transformation(
            m=[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 9.0, 8.0, 7.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            invm=[
                [-3.75, 2.75, -1, 0],
                [5.75, -4.75, 2.0, 1.0],
                [-2.25, 2.25, -1.0, -2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        assert m.is_consistent()

        expected_v = Vec(14.0, 38.0, 51.0)
        assert expected_v.is_close(m * Vec(1.0, 2.0, 3.0))

        expected_p = Point(18.0, 46.0, 58.0)
        assert expected_p.is_close(m * Point(1.0, 2.0, 3.0))

        expected_n = Normal(-8.75, 7.75, -3.0)
        assert expected_n.is_close(m * Normal(3.0, 2.0, 4.0))

    def test_inverse(self):
        m1 = Transformation(
            m=[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 9.0, 8.0, 7.0],
                [6.0, 5.0, 4.0, 1.0],
            ],
            invm=[
                [-3.75, 2.75, -1, 0],
                [4.375, -3.875, 2.0, -0.5],
                [0.5, 0.5, -1.0, 1.0],
                [-1.375, 0.875, 0.0, -0.5],
            ],
        )

        m2 = m1.inverse()
        assert m2.is_consistent()

        prod = m1 * m2
        assert prod.is_consistent()
        assert prod.is_close(Transformation())

    def test_translations(self):
        tr1 = translation(Vec(1.0, 2.0, 3.0))
        assert tr1.is_consistent()

        tr2 = translation(Vec(4.0, 6.0, 8.0))
        assert tr1.is_consistent()

        prod = tr1 * tr2
        assert prod.is_consistent()

        expected = translation(Vec(5.0, 8.0, 11.0))
        assert prod.is_close(expected)

    def test_rotations(self):
        assert rotation_x(0.1).is_consistent()
        assert rotation_y(0.1).is_consistent()
        assert rotation_z(0.1).is_consistent()

        assert (rotation_x(angle_deg=90) * VEC_Y).is_close(VEC_Z)
        assert (rotation_y(angle_deg=90) * VEC_Z).is_close(VEC_X)
        assert (rotation_z(angle_deg=90) * VEC_X).is_close(VEC_Y)

    def test_scalings(self):
        tr1 = scaling(Vec(2.0, 5.0, 10.0))
        assert tr1.is_consistent()

        tr2 = scaling(Vec(3.0, 2.0, 4.0))
        assert tr2.is_consistent()

        expected = scaling(Vec(6.0, 10.0, 40.0))
        assert expected.is_close(tr1 * tr2)


class TestRays(unittest.TestCase):
    def test_is_close(self):
        ray1 = Ray(origin=Point(1.0, 2.0, 3.0), dir=Vec(5.0, 4.0, -1.0))
        ray2 = Ray(origin=Point(1.0, 2.0, 3.0), dir=Vec(5.0, 4.0, -1.0))
        ray3 = Ray(origin=Point(5.0, 1.0, 4.0), dir=Vec(3.0, 9.0, 4.0))

        assert ray1.is_close(ray2)
        assert not ray1.is_close(ray3)

    def test_at(self):
        ray = Ray(origin=Point(1.0, 2.0, 4.0), dir=Vec(4.0, 2.0, 1.0))
        assert ray.at(0.0).is_close(ray.origin)
        assert ray.at(1.0).is_close(Point(5.0, 4.0, 5.0))
        assert ray.at(2.0).is_close(Point(9.0, 6.0, 6.0))

    def test_transform(self):
        ray = Ray(origin=Point(1.0, 2.0, 3.0), dir=Vec(6.0, 5.0, 4.0))
        transformation = translation(Vec(10.0, 11.0, 12.0)) * rotation_x(90.0)
        transformed = ray.transform(transformation)
        assert transformed.origin.is_close(Point(11.0, 8.0, 14.0))
        assert transformed.dir.is_close(Vec(6.0, -4.0, 5.0))


class TestCameras(unittest.TestCase):
    def test_orthogonal_camera(self):
        cam = OrthogonalCamera(aspect_ratio=2.0)

        # Fire one ray for each corner of the image plane
        ray1 = cam.fire_ray(0.0, 0.0)
        ray2 = cam.fire_ray(1.0, 0.0)
        ray3 = cam.fire_ray(0.0, 1.0)
        ray4 = cam.fire_ray(1.0, 1.0)

        # Verify that the rays are parallel by verifying that cross-products vanish
        assert are_close(0.0, ray1.dir.cross(ray2.dir).squared_norm())
        assert are_close(0.0, ray1.dir.cross(ray3.dir).squared_norm())
        assert are_close(0.0, ray1.dir.cross(ray4.dir).squared_norm())

        # Verify that the ray hitting the corners have the right coordinates
        assert ray1.at(1.0).is_close(Point(0.0, 2.0, -1.0))
        assert ray2.at(1.0).is_close(Point(0.0, -2.0, -1.0))
        assert ray3.at(1.0).is_close(Point(0.0, 2.0, 1.0))
        assert ray4.at(1.0).is_close(Point(0.0, -2.0, 1.0))

    def test_orthogonal_camera_transform(self):
        cam = OrthogonalCamera(
            transformation=translation(-VEC_Y * 2.0) * rotation_z(angle_deg=90)
        )

        ray = cam.fire_ray(0.5, 0.5)
        assert ray.at(1.0).is_close(Point(0.0, -2.0, 0.0))

    def test_perspective_camera(self):
        cam = PerspectiveCamera(screen_distance=1.0, aspect_ratio=2.0)

        # Fire one ray for each corner of the image plane
        ray1 = cam.fire_ray(0.0, 0.0)
        ray2 = cam.fire_ray(1.0, 0.0)
        ray3 = cam.fire_ray(0.0, 1.0)
        ray4 = cam.fire_ray(1.0, 1.0)

        # Verify that all the rays depart from the same point
        assert ray1.origin.is_close(ray2.origin)
        assert ray1.origin.is_close(ray3.origin)
        assert ray1.origin.is_close(ray4.origin)

        # Verify that the ray hitting the corners have the right coordinates
        assert ray1.at(1.0).is_close(Point(0.0, 2.0, -1.0))
        assert ray2.at(1.0).is_close(Point(0.0, -2.0, -1.0))
        assert ray3.at(1.0).is_close(Point(0.0, 2.0, 1.0))
        assert ray4.at(1.0).is_close(Point(0.0, -2.0, 1.0))

    def test_perspective_camera_transform(self):
        cam = PerspectiveCamera(
            transformation=translation(-VEC_Y * 2.0) * rotation_z(pi / 2.0)
        )

        ray = cam.fire_ray(0.5, 0.5)
        assert ray.at(1.0).is_close(Point(0.0, -2.0, 0.0))


class TestImageTracer(unittest.TestCase):
    def setUp(self) -> None:
        self.image = HdrImage(width=4, height=2)
        self.camera = PerspectiveCamera(aspect_ratio=2)
        self.tracer = ImageTracer(image=self.image, camera=self.camera)

    def test_orientation(self):
        top_left_ray = self.tracer.fire_ray(0, 0, u_pixel=0.0, v_pixel=0.0)
        assert Point(0.0, 2.0, 1.0).is_close(top_left_ray.at(1.0))

        bottom_right_ray = self.tracer.fire_ray(3, 1, u_pixel=1.0, v_pixel=1.0)
        assert Point(0.0, -2.0, -1.0).is_close(bottom_right_ray.at(1.0))

    def test_uv_sub_mapping(self):
        # Here we're cheating: we are asking `ImageTracer.fire_ray` to fire one ray *outside*
        # the pixel we're specifying
        ray1 = self.tracer.fire_ray(0, 0, u_pixel=2.5, v_pixel=1.5)
        ray2 = self.tracer.fire_ray(2, 1, u_pixel=0.5, v_pixel=0.5)
        assert ray1.is_close(ray2)

    def test_image_coverage(self):
        self.tracer.fire_all_rays(lambda ray: Color(1.0, 2.0, 3.0))
        for row in range(self.image.height):
            for col in range(self.image.width):
                assert self.image.get_pixel(col, row) == Color(1.0, 2.0, 3.0)

    def test_antialiasing(self):
        num_of_rays = 0
        small_image = HdrImage(width=1, height=1)
        camera = OrthogonalCamera(aspect_ratio=1)
        tracer = ImageTracer(small_image, camera, samples_per_side=10, pcg=PCG())

        def trace_ray(ray: Ray) -> Color:
            nonlocal num_of_rays
            point = ray.at(1)

            # Check that all the rays intersect the screen within the region [−1, 1] × [−1, 1]
            assert pytest.approx(0.0) == point.x
            assert -1.0 <= point.y <= 1.0
            assert -1.0 <= point.z <= 1.0

            num_of_rays += 1

            return Color(0.0, 0.0, 0.0)

        tracer.fire_all_rays(trace_ray)

        # Check that the number of rays that were fired is what we expect (10²)
        assert num_of_rays == 100


class TestSphere(unittest.TestCase):
    def testHit(self):
        sphere = Sphere()

        ray1 = Ray(origin=Point(0, 0, 2), dir=-VEC_Z)
        intersection1 = sphere.ray_intersection(ray1)
        assert intersection1
        assert HitRecord(
            world_point=Point(0.0, 0.0, 1.0),
            normal=Normal(0.0, 0.0, 1.0),
            surface_point=Vec2d(0.0, 0.0),
            t=1.0,
            ray=ray1,
            material=sphere.material,
        ).is_close(intersection1)

        ray2 = Ray(origin=Point(3, 0, 0), dir=-VEC_X)
        intersection2 = sphere.ray_intersection(ray2)
        assert intersection2
        assert HitRecord(
            world_point=Point(1.0, 0.0, 0.0),
            normal=Normal(1.0, 0.0, 0.0),
            surface_point=Vec2d(0.0, 0.5),
            t=2.0,
            ray=ray2,
            material=sphere.material,
        ).is_close(intersection2)

        assert not sphere.ray_intersection(Ray(origin=Point(0, 10, 2), dir=-VEC_Z))

    def testInnerHit(self):
        sphere = Sphere()

        ray = Ray(origin=Point(0, 0, 0), dir=VEC_X)
        intersection = sphere.ray_intersection(ray)
        assert intersection
        assert HitRecord(
            world_point=Point(1.0, 0.0, 0.0),
            normal=Normal(-1.0, 0.0, 0.0),
            surface_point=Vec2d(0.0, 0.5),
            t=1.0,
            ray=ray,
            material=sphere.material,
        ).is_close(intersection)

    def testTransformation(self):
        sphere = Sphere(transformation=translation(Vec(10.0, 0.0, 0.0)))

        ray1 = Ray(origin=Point(10, 0, 2), dir=-VEC_Z)
        intersection1 = sphere.ray_intersection(ray1)
        assert intersection1
        assert HitRecord(
            world_point=Point(10.0, 0.0, 1.0),
            normal=Normal(0.0, 0.0, 1.0),
            surface_point=Vec2d(0.0, 0.0),
            t=1.0,
            ray=ray1,
            material=sphere.material,
        ).is_close(intersection1)

        ray2 = Ray(origin=Point(13, 0, 0), dir=-VEC_X)
        intersection2 = sphere.ray_intersection(ray2)
        assert intersection2
        assert HitRecord(
            world_point=Point(11.0, 0.0, 0.0),
            normal=Normal(1.0, 0.0, 0.0),
            surface_point=Vec2d(0.0, 0.5),
            t=2.0,
            ray=ray2,
            material=sphere.material,
        ).is_close(intersection2)

        # Check if the sphere failed to move by trying to hit the untransformed shape
        assert not sphere.ray_intersection(Ray(origin=Point(0, 0, 2), dir=-VEC_Z))

        # Check if the *inverse* transformation was wrongly applied
        assert not sphere.ray_intersection(Ray(origin=Point(-10, 0, 0), dir=-VEC_Z))

    def testNormals(self):
        sphere = Sphere(transformation=scaling(Vec(2.0, 1.0, 1.0)))

        ray = Ray(origin=Point(1.0, 1.0, 0.0), dir=Vec(-1.0, -1.0))
        intersection = sphere.ray_intersection(ray)
        # We normalize "intersection.normal", as we are not interested in its length
        assert intersection.normal.normalize().is_close(Normal(1.0, 4.0, 0.0).normalize())

    def testNormalDirection(self):
        # Scaling a sphere by -1 keeps the sphere the same but reverses its
        # reference frame
        sphere = Sphere(transformation=scaling(Vec(-1.0, -1.0, -1.0)))

        ray = Ray(origin=Point(0.0, 2.0, 0.0), dir=-VEC_Y)
        intersection = sphere.ray_intersection(ray)
        # We normalize "intersection.normal", as we are not interested in its length
        assert intersection.normal.normalize().is_close(Normal(0.0, 1.0, 0.0).normalize())

    def testUVCoordinates(self):
        sphere = Sphere()

        # The first four rays hit the unit sphere at the
        # points P1, P2, P3, and P4.
        #
        #                    ^ y
        #                    | P2
        #              , - ~ * ~ - ,
        #          , '       |       ' ,
        #        ,           |           ,
        #       ,            |            ,
        #      ,             |             , P1
        # -----*-------------+-------------*---------> x
        #   P3 ,             |             ,
        #       ,            |            ,
        #        ,           |           ,
        #          ,         |        , '
        #            ' - , _ * _ ,  '
        #                    | P4
        #
        # P5 and P6 are aligned along the x axis and are displaced
        # along z (ray5 in the positive direction, ray6 in the negative
        # direction).

        ray1 = Ray(origin=Point(2.0, 0.0, 0.0), dir=-VEC_X)
        assert sphere.ray_intersection(ray1).surface_point.is_close(Vec2d(0.0, 0.5))

        ray2 = Ray(origin=Point(0.0, 2.0, 0.0), dir=-VEC_Y)
        assert sphere.ray_intersection(ray2).surface_point.is_close(Vec2d(0.25, 0.5))

        ray3 = Ray(origin=Point(-2.0, 0.0, 0.0), dir=VEC_X)
        assert sphere.ray_intersection(ray3).surface_point.is_close(Vec2d(0.5, 0.5))

        ray4 = Ray(origin=Point(0.0, -2.0, 0.0), dir=VEC_Y)
        assert sphere.ray_intersection(ray4).surface_point.is_close(Vec2d(0.75, 0.5))

        ray5 = Ray(origin=Point(2.0, 0.0, 0.5), dir=-VEC_X)
        assert sphere.ray_intersection(ray5).surface_point.is_close(Vec2d(0.0, 1 / 3))

        ray6 = Ray(origin=Point(2.0, 0.0, -0.5), dir=-VEC_X)
        assert sphere.ray_intersection(ray6).surface_point.is_close(Vec2d(0.0, 2 / 3))


class TestPlane(unittest.TestCase):
    def testHit(self):
        plane = Plane()

        ray1 = Ray(origin=Point(0, 0, 1), dir=-VEC_Z)
        intersection1 = plane.ray_intersection(ray1)
        assert intersection1
        assert HitRecord(
            world_point=Point(0.0, 0.0, 0.0),
            normal=Normal(0.0, 0.0, 1.0),
            surface_point=Vec2d(0.0, 0.0),
            t=1.0,
            ray=ray1,
            material=plane.material,
        ).is_close(intersection1)

        ray2 = Ray(origin=Point(0, 0, 1), dir=VEC_Z)
        intersection2 = plane.ray_intersection(ray2)
        assert not intersection2

        ray3 = Ray(origin=Point(0, 0, 1), dir=VEC_X)
        intersection3 = plane.ray_intersection(ray3)
        assert not intersection3

        ray4 = Ray(origin=Point(0, 0, 1), dir=VEC_Y)
        intersection4 = plane.ray_intersection(ray4)
        assert not intersection4

    def testTransformation(self):
        plane = Plane(transformation=rotation_y(angle_deg=90.0))

        ray1 = Ray(origin=Point(1, 0, 0), dir=-VEC_X)
        intersection1 = plane.ray_intersection(ray1)
        assert intersection1
        assert HitRecord(
            world_point=Point(0.0, 0.0, 0.0),
            normal=Normal(1.0, 0.0, 0.0),
            surface_point=Vec2d(0.0, 0.0),
            t=1.0,
            ray=ray1,
            material=plane.material,
        ).is_close(intersection1)

        ray2 = Ray(origin=Point(0, 0, 1), dir=VEC_Z)
        intersection2 = plane.ray_intersection(ray2)
        assert not intersection2

        ray3 = Ray(origin=Point(0, 0, 1), dir=VEC_X)
        intersection3 = plane.ray_intersection(ray3)
        assert not intersection3

        ray4 = Ray(origin=Point(0, 0, 1), dir=VEC_Y)
        intersection4 = plane.ray_intersection(ray4)
        assert not intersection4

    def testUVCoordinates(self):
        plane = Plane()

        ray1 = Ray(origin=Point(0, 0, 1), dir=-VEC_Z)
        intersection1 = plane.ray_intersection(ray1)
        assert intersection1.surface_point.is_close(Vec2d(0.0, 0.0))

        ray2 = Ray(origin=Point(0.25, 0.75, 1), dir=-VEC_Z)
        intersection2 = plane.ray_intersection(ray2)
        assert intersection2.surface_point.is_close(Vec2d(0.25, 0.75))

        ray3 = Ray(origin=Point(4.25, 7.75, 1), dir=-VEC_Z)
        intersection3 = plane.ray_intersection(ray3)
        assert intersection3.surface_point.is_close(Vec2d(0.25, 0.75))


class TestWorld(unittest.TestCase):
    def testRayIntersections(self):
        world = World()

        sphere1 = Sphere(transformation=translation(VEC_X * 2))
        sphere2 = Sphere(transformation=translation(VEC_X * 8))
        world.add_shape(sphere1)
        world.add_shape(sphere2)

        intersection1 = world.ray_intersection(Ray(
            origin=Point(0.0, 0.0, 0.0), dir=VEC_X
        ))
        assert intersection1
        assert intersection1.world_point.is_close(Point(1.0, 0.0, 0.0))

        intersection2 = world.ray_intersection(Ray(
            origin=Point(10.0, 0.0, 0.0), dir=-VEC_X
        ))

        assert intersection2
        assert intersection2.world_point.is_close(Point(9.0, 0.0, 0.0))

    def test_quick_ray_intersection(self):
        world = World()

        sphere1 = Sphere(transformation=translation(VEC_X * 2))
        sphere2 = Sphere(transformation=translation(VEC_X * 8))
        world.add_shape(sphere1)
        world.add_shape(sphere2)

        assert not world.is_point_visible(point=Point(10.0, 0.0, 0.0),
                                          observer_pos=Point(0.0, 0.0, 0.0))
        assert not world.is_point_visible(point=Point(5.0, 0.0, 0.0),
                                          observer_pos=Point(0.0, 0.0, 0.0))
        assert world.is_point_visible(point=Point(5.0, 0.0, 0.0),
                                      observer_pos=Point(4.0, 0.0, 0.0))
        assert world.is_point_visible(point=Point(0.5, 0.0, 0.0),
                                      observer_pos=Point(0.0, 0.0, 0.0))
        assert world.is_point_visible(point=Point(0.0, 10.0, 0.0),
                                      observer_pos=Point(0.0, 0.0, 0.0))
        assert world.is_point_visible(point=Point(0.0, 0.0, 10.0),
                                      observer_pos=Point(0.0, 0.0, 0.0))


class TestPCG(unittest.TestCase):
    def test_random(self):
        pcg = PCG()
        assert pcg.state == 1753877967969059832
        assert pcg.inc == 109

        for expected in [
            2707161783,
            2068313097,
            3122475824,
            2211639955,
            3215226955,
            3421331566,
        ]:
            result = pcg.random()
            assert expected == result


class TestPigments(unittest.TestCase):
    def testUniformPigment(self):
        color = Color(1.0, 2.0, 3.0)
        pigment = UniformPigment(color=color)

        assert pigment.get_color(Vec2d(0.0, 0.0)).is_close(color)
        assert pigment.get_color(Vec2d(1.0, 0.0)).is_close(color)
        assert pigment.get_color(Vec2d(0.0, 1.0)).is_close(color)
        assert pigment.get_color(Vec2d(1.0, 1.0)).is_close(color)

    def testImagePigment(self):
        image = HdrImage(width=2, height=2)
        image.set_pixel(0, 0, Color(1.0, 2.0, 3.0))
        image.set_pixel(1, 0, Color(2.0, 3.0, 1.0))
        image.set_pixel(0, 1, Color(2.0, 1.0, 3.0))
        image.set_pixel(1, 1, Color(3.0, 2.0, 1.0))

        pigment = ImagePigment(image)
        assert pigment.get_color(Vec2d(0.0, 0.0)).is_close(Color(1.0, 2.0, 3.0))
        assert pigment.get_color(Vec2d(1.0, 0.0)).is_close(Color(2.0, 3.0, 1.0))
        assert pigment.get_color(Vec2d(0.0, 1.0)).is_close(Color(2.0, 1.0, 3.0))
        assert pigment.get_color(Vec2d(1.0, 1.0)).is_close(Color(3.0, 2.0, 1.0))

    def testCheckeredPigment(self):
        color1 = Color(1.0, 2.0, 3.0)
        color2 = Color(10.0, 20.0, 30.0)

        pigment = CheckeredPigment(color1=color1, color2=color2, num_of_steps=2)

        # With num_of_steps == 2, the pattern should be the following:
        #
        #              (0.5, 0)
        #   (0, 0) +------+------+ (1, 0)
        #          |      |      |
        #          | col1 | col2 |
        #          |      |      |
        # (0, 0.5) +------+------+ (1, 0.5)
        #          |      |      |
        #          | col2 | col1 |
        #          |      |      |
        #   (0, 1) +------+------+ (1, 1)
        #              (0.5, 1)
        assert pigment.get_color(Vec2d(0.25, 0.25)).is_close(color1)
        assert pigment.get_color(Vec2d(0.75, 0.25)).is_close(color2)
        assert pigment.get_color(Vec2d(0.25, 0.75)).is_close(color2)
        assert pigment.get_color(Vec2d(0.75, 0.75)).is_close(color1)


class TestRenderers(unittest.TestCase):
    def testOnOffRenderer(self):
        sphere = Sphere(transformation=translation(Vec(2, 0, 0)) * scaling(Vec(0.2, 0.2, 0.2)),
                        material=Material(brdf=DiffuseBRDF(pigment=UniformPigment(WHITE))))
        image = HdrImage(width=3, height=3)
        camera = OrthogonalCamera()
        tracer = ImageTracer(image=image, camera=camera)
        world = World()
        world.add_shape(sphere)
        renderer = OnOffRenderer(world=world)
        tracer.fire_all_rays(renderer)

        assert image.get_pixel(0, 0).is_close(BLACK)
        assert image.get_pixel(1, 0).is_close(BLACK)
        assert image.get_pixel(2, 0).is_close(BLACK)

        assert image.get_pixel(0, 1).is_close(BLACK)
        assert image.get_pixel(1, 1).is_close(WHITE)
        assert image.get_pixel(2, 1).is_close(BLACK)

        assert image.get_pixel(0, 2).is_close(BLACK)
        assert image.get_pixel(1, 2).is_close(BLACK)
        assert image.get_pixel(2, 2).is_close(BLACK)

    def testFlatRenderer(self):
        sphere_color = Color(1.0, 2.0, 3.0)
        sphere = Sphere(transformation=translation(Vec(2, 0, 0)) * scaling(Vec(0.2, 0.2, 0.2)),
                        material=Material(brdf=DiffuseBRDF(pigment=UniformPigment(sphere_color))))
        image = HdrImage(width=3, height=3)
        camera = OrthogonalCamera()
        tracer = ImageTracer(image=image, camera=camera)
        world = World()
        world.add_shape(sphere)
        renderer = FlatRenderer(world=world)
        tracer.fire_all_rays(renderer)

        assert image.get_pixel(0, 0).is_close(BLACK)
        assert image.get_pixel(1, 0).is_close(BLACK)
        assert image.get_pixel(2, 0).is_close(BLACK)

        assert image.get_pixel(0, 1).is_close(BLACK)
        assert image.get_pixel(1, 1).is_close(sphere_color)
        assert image.get_pixel(2, 1).is_close(BLACK)

        assert image.get_pixel(0, 2).is_close(BLACK)
        assert image.get_pixel(1, 2).is_close(BLACK)
        assert image.get_pixel(2, 2).is_close(BLACK)


class TestOnbCreation(unittest.TestCase):
    def testOnbFromNormal(self):
        pcg = PCG()

        expected_zero = pytest.approx(0.0)
        expected_one = pytest.approx(1.0)

        for i in range(100):
            normal = Vec(pcg.random_float(), pcg.random_float(), pcg.random_float())
            normal.normalize()
            e1, e2, e3 = create_onb_from_z(normal)

            assert e3.is_close(normal)

            assert expected_one == e1.squared_norm()
            assert expected_one == e2.squared_norm()
            assert expected_one == e3.squared_norm()

            assert expected_zero == e1.dot(e2)
            assert expected_zero == e2.dot(e3)
            assert expected_zero == e3.dot(e1)


class TestPathTracer(unittest.TestCase):
    def testFurnace(self):
        pcg = PCG()

        # Run the furnace test several times using random values for the emitted radiance and reflectance
        for i in range(5):
            world = World()

            emitted_radiance = pcg.random_float()
            reflectance = pcg.random_float() * 0.9  # Be sure to pick a reflectance that's not too close to 1
            enclosure_material = Material(
                brdf=DiffuseBRDF(pigment=UniformPigment(Color(1.0, 1.0, 1.0) * reflectance)),
                emitted_radiance=UniformPigment(Color(1.0, 1.0, 1.0) * emitted_radiance),
            )

            world.add_shape(Sphere(material=enclosure_material))

            path_tracer = PathTracer(pcg=pcg, num_of_rays=1, world=world, max_depth=100, russian_roulette_limit=101)

            ray = Ray(origin=Point(0, 0, 0), dir=Vec(1, 0, 0))
            color = path_tracer(ray)

            expected = emitted_radiance / (1.0 - reflectance)
            assert pytest.approx(expected, 1e-3) == color.r
            assert pytest.approx(expected, 1e-3) == color.g
            assert pytest.approx(expected, 1e-3) == color.b


def _assert_is_keyword(token: Token, keyword: KeywordEnum):
    assert isinstance(token, KeywordToken)
    assert token.keyword == keyword, f"Token '{token}' is not equal to keyword '{keyword}'"


def _assert_is_identifier(token: Token, identifier: str):
    assert isinstance(token, IdentifierToken)
    assert token.identifier == identifier, f"expecting identifier '{identifier}' instead of '{token}'"


def _assert_is_symbol(token: Token, symbol: str):
    assert isinstance(token, SymbolToken)
    assert token.symbol == symbol, f"expecting symbol '{symbol}' instead of '{token}'"


def _assert_is_number(token: Token, number: float):
    assert isinstance(token, LiteralNumberToken)
    assert token.value == number, f"Token '{token}' is not equal to number '{number}'"


def _assert_is_string(token: Token, s: str):
    assert isinstance(token, StringToken)
    assert token.string == s, f"Token '{token}' is not equal to string '{s}'"


class TestSceneFile(unittest.TestCase):
    def test_input_file(self):
        stream = InputStream(StringIO("abc   \nd\nef"))

        assert stream.location.line_num == 1
        assert stream.location.col_num == 1

        assert stream.read_char() == "a"
        assert stream.location.line_num == 1
        assert stream.location.col_num == 2

        stream.unread_char("A")
        assert stream.location.line_num == 1
        assert stream.location.col_num == 1

        assert stream.read_char() == "A"
        assert stream.location.line_num == 1
        assert stream.location.col_num == 2

        assert stream.read_char() == "b"
        assert stream.location.line_num == 1
        assert stream.location.col_num == 3

        assert stream.read_char() == "c"
        assert stream.location.line_num == 1
        assert stream.location.col_num == 4

        stream.skip_whitespaces_and_comments()

        assert stream.read_char() == "d"
        assert stream.location.line_num == 2
        assert stream.location.col_num == 2

        assert stream.read_char() == "\n"
        assert stream.location.line_num == 3
        assert stream.location.col_num == 1

        assert stream.read_char() == "e"
        assert stream.location.line_num == 3
        assert stream.location.col_num == 2

        assert stream.read_char() == "f"
        assert stream.location.line_num == 3
        assert stream.location.col_num == 3

        assert stream.read_char() == ""

    def test_lexer(self):
        stream = StringIO("""
        # This is a comment
        # This is another comment
        new material sky_material(
            diffuse(image("my file.pfm")),
            <5.0, 500.0, 300.0>
        ) # Comment at the end of the line
""")

        input_file = InputStream(stream)

        _assert_is_keyword(input_file.read_token(), KeywordEnum.NEW)
        _assert_is_keyword(input_file.read_token(), KeywordEnum.MATERIAL)
        _assert_is_identifier(input_file.read_token(), "sky_material")
        _assert_is_symbol(input_file.read_token(), "(")
        _assert_is_keyword(input_file.read_token(), KeywordEnum.DIFFUSE)
        _assert_is_symbol(input_file.read_token(), "(")
        _assert_is_keyword(input_file.read_token(), KeywordEnum.IMAGE)
        _assert_is_symbol(input_file.read_token(), "(")
        _assert_is_string(input_file.read_token(), "my file.pfm")
        _assert_is_symbol(input_file.read_token(), ")")

    def test_parser(self):
        stream = StringIO("""
        float clock(150)
    
        material sky_material(
            diffuse(uniform(<0, 0, 0>)),
            uniform(<0.7, 0.5, 1>)
        )
    
        # Here is a comment
    
        material ground_material(
            diffuse(checkered(<0.3, 0.5, 0.1>,
                              <0.1, 0.2, 0.5>, 4)),
            uniform(<0, 0, 0>)
        )
    
        material sphere_material(
            specular(uniform(<0.5, 0.5, 0.5>)),
            uniform(<0, 0, 0>)
        )
    
        plane (sky_material, translation([0, 0, 100]) * rotation_y(clock))
        plane (ground_material, identity)
    
        sphere(sphere_material, translation([0, 0, 1]))
    
        camera(perspective, rotation_z(30) * translation([-4, 0, 1]), 1.0, 2.0)
        """)

        scene = parse_scene(input_file=InputStream(stream))

        # Check that the float variables are ok

        assert len(scene.float_variables) == 1
        assert "clock" in scene.float_variables.keys()
        assert scene.float_variables["clock"] == 150.0

        # Check that the materials are ok

        assert len(scene.materials) == 3
        assert "sphere_material" in scene.materials
        assert "sky_material" in scene.materials
        assert "ground_material" in scene.materials

        sphere_material = scene.materials["sphere_material"]
        sky_material = scene.materials["sky_material"]
        ground_material = scene.materials["ground_material"]

        assert isinstance(sky_material.brdf, DiffuseBRDF)
        assert isinstance(sky_material.brdf.pigment, UniformPigment)
        assert sky_material.brdf.pigment.color.is_close(Color(0, 0, 0))

        assert isinstance(ground_material.brdf, DiffuseBRDF)
        assert isinstance(ground_material.brdf.pigment, CheckeredPigment)
        assert ground_material.brdf.pigment.color1.is_close(Color(0.3, 0.5, 0.1))
        assert ground_material.brdf.pigment.color2.is_close(Color(0.1, 0.2, 0.5))
        assert ground_material.brdf.pigment.num_of_steps == 4

        assert isinstance(sphere_material.brdf, SpecularBRDF)
        assert isinstance(sphere_material.brdf.pigment, UniformPigment)
        assert sphere_material.brdf.pigment.color.is_close(Color(0.5, 0.5, 0.5))

        assert isinstance(sky_material.emitted_radiance, UniformPigment)
        assert sky_material.emitted_radiance.color.is_close(Color(0.7, 0.5, 1.0))
        assert isinstance(ground_material.emitted_radiance, UniformPigment)
        assert ground_material.emitted_radiance.color.is_close(Color(0, 0, 0))
        assert isinstance(sphere_material.emitted_radiance, UniformPigment)
        assert sphere_material.emitted_radiance.color.is_close(Color(0, 0, 0))

        # Check that the shapes are ok

        assert len(scene.world.shapes) == 3
        assert isinstance(scene.world.shapes[0], Plane)
        assert scene.world.shapes[0].transformation.is_close(translation(Vec(0, 0, 100)) * rotation_y(150.0))
        assert isinstance(scene.world.shapes[1], Plane)
        assert scene.world.shapes[1].transformation.is_close(Transformation())
        assert isinstance(scene.world.shapes[2], Sphere)
        assert scene.world.shapes[2].transformation.is_close(translation(Vec(0, 0, 1)))

        # Check that the camera is ok

        assert isinstance(scene.camera, PerspectiveCamera)
        assert scene.camera.transformation.is_close(rotation_z(30) * translation(Vec(-4, 0, 1)))
        assert pytest.approx(1.0) == scene.camera.aspect_ratio
        assert pytest.approx(2.0) == scene.camera.screen_distance

    def test_parser_undefined_material(self):
        # Check that unknown materials raises a GrammarError
        stream = StringIO("""
        plane(this_material_does_not_exist, identity)
        """)

        try:
            _ = parse_scene(input_file=InputStream(stream))
            assert False, "the code did not throw an exception"
        except GrammarError:
            pass

    def test_parser_double_camera(self):
        # Check that defining two cameras in the same file raises a GrammarError
        stream = StringIO("""
        camera(perspective, rotation_z(30) * translation([-4, 0, 1]), 1.0, 1.0)
        camera(orthogonal, identity, 1.0, 1.0)
        """)

        try:
            _ = parse_scene(input_file=InputStream(stream))
            assert False, "the code did not throw an exception"
        except GrammarError:
            pass


if __name__ == "__main__":
    unittest.main()
