import unittest
from io import BytesIO
from colors import Color
from hdrimages import HdrImage, InvalidPfmFileFormat, Endianness, read_pfm_image, _read_line, _parse_img_size, \
    _parse_endianness
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

if __name__ == '__main__':
    unittest.main()
