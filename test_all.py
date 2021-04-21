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

from copy import deepcopy
from math import pi, sin, cos

import unittest
from io import BytesIO
from colors import Color
from hdrimages import HdrImage, InvalidPfmFileFormat, Endianness, read_pfm_image, _read_line, _parse_img_size, \
    _parse_endianness
from geometry import Vec, Point, Normal, VEC_X, VEC_Y, VEC_Z
from transformations import Transformation, _matr_prod, translation, scaling, rotation_x, rotation_y, rotation_z
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
        assert (col1 - col2).is_close(Color(-4.0, -5.0, -6.0))
        assert (col1 * col2).is_close(Color(5.0, 14.0, 27.0))

        prod_col = Color(1.0, 2.0, 3.0) * 2.0
        assert prod_col.is_close(Color(2.0, 4.0, 6.0))

    def test_luminosity(self):
        col1 = Color(1.0, 2.0, 3.0)
        col2 = Color(9.0, 5.0, 7.0)

        assert pytest.approx(2.0) == col1.luminosity()
        assert pytest.approx(7.0) == col2.luminosity()


# This is the content of "reference_le.pfm" (little-endian file)
LE_REFERENCE_BYTES = bytes([
    0x50, 0x46, 0x0a, 0x33, 0x20, 0x32, 0x0a, 0x2d, 0x31, 0x2e, 0x30, 0x0a,
    0x00, 0x00, 0xc8, 0x42, 0x00, 0x00, 0x48, 0x43, 0x00, 0x00, 0x96, 0x43,
    0x00, 0x00, 0xc8, 0x43, 0x00, 0x00, 0xfa, 0x43, 0x00, 0x00, 0x16, 0x44,
    0x00, 0x00, 0x2f, 0x44, 0x00, 0x00, 0x48, 0x44, 0x00, 0x00, 0x61, 0x44,
    0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0xa0, 0x41, 0x00, 0x00, 0xf0, 0x41,
    0x00, 0x00, 0x20, 0x42, 0x00, 0x00, 0x48, 0x42, 0x00, 0x00, 0x70, 0x42,
    0x00, 0x00, 0x8c, 0x42, 0x00, 0x00, 0xa0, 0x42, 0x00, 0x00, 0xb4, 0x42
])

# This is the content of "reference_be.pfm" (big-endian file)
BE_REFERENCE_BYTES = bytes([
    0x50, 0x46, 0x0a, 0x33, 0x20, 0x32, 0x0a, 0x31, 0x2e, 0x30, 0x0a, 0x42,
    0xc8, 0x00, 0x00, 0x43, 0x48, 0x00, 0x00, 0x43, 0x96, 0x00, 0x00, 0x43,
    0xc8, 0x00, 0x00, 0x43, 0xfa, 0x00, 0x00, 0x44, 0x16, 0x00, 0x00, 0x44,
    0x2f, 0x00, 0x00, 0x44, 0x48, 0x00, 0x00, 0x44, 0x61, 0x00, 0x00, 0x41,
    0x20, 0x00, 0x00, 0x41, 0xa0, 0x00, 0x00, 0x41, 0xf0, 0x00, 0x00, 0x42,
    0x20, 0x00, 0x00, 0x42, 0x48, 0x00, 0x00, 0x42, 0x70, 0x00, 0x00, 0x42,
    0x8c, 0x00, 0x00, 0x42, 0xa0, 0x00, 0x00, 0x42, 0xb4, 0x00, 0x00
])


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
        print(be_buf.getvalue())
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

        print(img.average_luminosity(delta=0.0))
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
        m1 = Transformation(m=[[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [9.0, 9.0, 8.0, 7.0],
                               [6.0, 5.0, 4.0, 1.0]],
                            invm=[[-3.75, 2.75, -1, 0],
                                  [4.375, -3.875, 2.0, -0.5],
                                  [0.5, 0.5, -1.0, 1.0],
                                  [-1.375, 0.875, 0.0, -0.5]])

        assert m1.is_consistent()

        # Not using "deepcopy" here would make Python pass a pointer to the *same* matrices and vectors
        m2 = Transformation(m=deepcopy(m1.m), invm=deepcopy(m1.invm))
        assert m1.is_close(m2)

        m3 = Transformation(m=deepcopy(m1.m), invm=deepcopy(m1.invm))
        m3.m[2][2] += 1.0   # Note: this makes "m3" not consistent (m3.is_consistent() == False)
        assert not m1.is_close(m3)

        m4 = Transformation(m=deepcopy(m1.m), invm=deepcopy(m1.invm))
        m4.invm[2][2] += 1.0   # Note: this makes "m4" not consistent (m4.is_consistent() == False)
        assert not m1.is_close(m4)

    def test_multiplication(self):
        m1 = Transformation(m=[[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [9.0, 9.0, 8.0, 7.0],
                               [6.0, 5.0, 4.0, 1.0]],
                            invm=[[-3.75, 2.75, -1, 0],
                                  [4.375, -3.875, 2.0, -0.5],
                                  [0.5, 0.5, -1.0, 1.0],
                                  [-1.375, 0.875, 0.0, -0.5]])
        assert m1.is_consistent()

        m2 = Transformation(m=[[3.0, 5.0, 2.0, 4.0],
                               [4.0, 1.0, 0.0, 5.0],
                               [6.0, 3.0, 2.0, 0.0],
                               [1.0, 4.0, 2.0, 1.0]],
                            invm=[[0.4, -0.2, 0.2, -0.6],
                                  [2.9, -1.7, 0.2, -3.1],
                                  [-5.55, 3.15, -0.4, 6.45],
                                  [-0.9, 0.7, -0.2, 1.1]])
        assert m2.is_consistent()

        expected = Transformation(
            m=[[33.0, 32.0, 16.0, 18.0],
               [89.0, 84.0, 40.0, 58.0],
               [118.0, 106.0, 48.0, 88.0],
               [63.0, 51.0, 22.0, 50.0]],
            invm=[[-1.45, 1.45, -1.0, 0.6],
                  [-13.95, 11.95, -6.5, 2.6],
                  [25.525, -22.025, 12.25, -5.2],
                  [4.825, -4.325, 2.5, -1.1]],
        )
        assert expected.is_consistent()

        assert expected.is_close(m1 * m2)

    def test_vec_point_multiplication(self):
        m = Transformation(m=[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 9.0, 8.0, 7.0],
                              [0.0, 0.0, 0.0, 1.0]],
                           invm=[[-3.75, 2.75, -1, 0],
                                 [5.75, -4.75, 2.0, 1.0],
                                 [-2.25, 2.25, -1.0, -2.0],
                                 [0.0, 0.0, 0.0, 1.0]])
        assert m.is_consistent()

        expected_v = Vec(14.0, 38.0, 51.0)
        assert expected_v.is_close(m * Vec(1.0, 2.0, 3.0))

        expected_p = Point(18.0, 46.0, 58.0)
        assert expected_p.is_close(m * Point(1.0, 2.0, 3.0))

        expected_n = Normal(-8.75, 7.75, -3.0)
        assert expected_n.is_close(m * Normal(3.0, 2.0, 4.0))

    def test_inverse(self):
        m1 = Transformation(m=[[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [9.0, 9.0, 8.0, 7.0],
                               [6.0, 5.0, 4.0, 1.0]],
                            invm=[[-3.75, 2.75, -1, 0],
                                  [4.375, -3.875, 2.0, -0.5],
                                  [0.5, 0.5, -1.0, 1.0],
                                  [-1.375, 0.875, 0.0, -0.5]])

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

        assert (rotation_x(angle_rad=pi / 2) * VEC_Y).is_close(VEC_Z)
        assert (rotation_y(angle_rad=pi / 2) * VEC_Z).is_close(VEC_X)
        assert (rotation_z(angle_rad=pi / 2) * VEC_X).is_close(VEC_Y)

    def test_scalings(self):
        tr1 = scaling(Vec(2.0, 5.0, 10.0))
        assert tr1.is_consistent()

        tr2 = scaling(Vec(3.0, 2.0, 4.0))
        assert tr2.is_consistent()

        expected = scaling(Vec(6.0, 10.0, 40.0))
        assert expected.is_close(tr1 * tr2)


if __name__ == '__main__':
    unittest.main()
