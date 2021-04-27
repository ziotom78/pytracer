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


from colors import Color
import math
import struct
from enum import Enum


class Endianness(Enum):
    """Kinds of byte/bit endianness"""
    LITTLE_ENDIAN = 1
    BIG_ENDIAN = 2


# "<": little endian
# ">": big endian
# "f": single-precision floating point value (32 bit)
_FLOAT_STRUCT_FORMAT = {
    Endianness.LITTLE_ENDIAN: "<f",
    Endianness.BIG_ENDIAN: ">f",
}


def _write_float(stream, value, endianness=Endianness.LITTLE_ENDIAN):
    format_str = _FLOAT_STRUCT_FORMAT[endianness]
    stream.write(struct.pack(format_str, value))


def _read_float(stream, endianness=Endianness.LITTLE_ENDIAN):
    format_str = _FLOAT_STRUCT_FORMAT[endianness]

    try:
        return struct.unpack(format_str, stream.read(4))[0]
    except struct.error:
        raise InvalidPfmFileFormat("impossible to read binary data from the file")


def _clamp(x: float) -> float:
    return x / (1 + x)


class HdrImage:
    """A High-Dynamic-Range 2D image

    This class has the following members:

    -   `width` (int): number of columns in the 2D matrix of colors
    -   `height` (int): number of rows in the 2D matrix of colors
    -   `pixel` (array of `Color`): the 2D matrix, represented as a 1D array
    """

    def __init__(self, width=0, height=0):
        """Create a black image with the specified resolution"""
        (self.width, self.height) = (width, height)
        self.pixels = [Color() for i in range(self.width * self.height)]

    def valid_coordinates(self, x, y):
        """Return True if ``(x, y)`` are coordinates within the 2D matrix"""
        return ((x >= 0) and (x < self.width) and
                (y >= 0) and (y < self.height))

    def pixel_offset(self, x, y):
        """Return the position in the 1D array of the specified pixel"""
        return y * self.width + x

    def get_pixel(self, x, y):
        """Return the `Color` value for a pixel in the image

        The pixel at the top-left corner has coordinates (0, 0)."""
        assert self.valid_coordinates(x, y)
        return self.pixels[self.pixel_offset(x, y)]

    def set_pixel(self, x, y, new_color):
        """Set the new color for a pixel in the image

        The pixel at the top-left corner has coordinates (0, 0)."""
        assert self.valid_coordinates(x, y)
        self.pixels[self.pixel_offset(x, y)] = new_color

    def write_pfm(self, stream, endianness=Endianness.LITTLE_ENDIAN):
        """Write the image in a PFM file

        The `stream` parameter must be a I/O stream. The parameter `endianness` specifies the byte endianness
        to be used in the file."""
        if endianness == Endianness.LITTLE_ENDIAN:
            endianness_str = "-1.0"
        else:
            endianness_str = "1.0"

        # The PFM header, as a Python string (UTF-8)
        header = f"PF\n{self.width} {self.height}\n{endianness_str}\n"

        # Convert the header into a sequence of bytes
        stream.write(header.encode("ascii"))

        # Write the image (bottom-to-up, left-to-right)
        for y in reversed(range(self.height)):
            for x in range(self.width):
                color = self.get_pixel(x, y)
                _write_float(stream, color.r, endianness=endianness)
                _write_float(stream, color.g, endianness=endianness)
                _write_float(stream, color.b, endianness=endianness)

    def average_luminosity(self, delta=1e-10):
        """Return the average luminosity of the image

        The `delta` parameter is used to prevent  numerical problems for underilluminated pixels"""
        cumsum = 0.0
        for pix in self.pixels:
            cumsum += math.log10(delta + pix.luminosity())

        return math.pow(10, cumsum / len(self.pixels))

    def normalize_image(self, factor, luminosity=None):
        """Normalize the image for a given luminosity

        If the `luminosity` parameter is ``None`` (the default), the image will be normalized according to
        the result of ``HdrImage.average_luminosity()``."""
        if not luminosity:
            luminosity = self.average_luminosity()

        for i in range(len(self.pixels)):
            self.pixels[i] = self.pixels[i] * (factor / luminosity)

    def clamp_image(self):
        """Adjust the color levels of the brightest pixels in the image"""
        for i in range(len(self.pixels)):
            self.pixels[i].r = _clamp(self.pixels[i].r)
            self.pixels[i].g = _clamp(self.pixels[i].g)
            self.pixels[i].b = _clamp(self.pixels[i].b)

    def write_ldr_image(self, stream, format, gamma=1.0):
        """Save the image in a LDR format

        Before calling this function, you should apply a tone-mapping algorithm to the image and be sure that
        the R, G, and B values of the colors in the image are all in the range [0, 1]. Use ``HdrImage.normalize_image``
        and ``HdrImage.clamp_image`` to do this.
        """
        from PIL import Image
        img = Image.new("RGB", (self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                cur_color = self.get_pixel(x, y)
                img.putpixel(xy=(x, y), value=(
                    int(255 * math.pow(cur_color.r, 1 / gamma)),
                    int(255 * math.pow(cur_color.g, 1 / gamma)),
                    int(255 * math.pow(cur_color.b, 1 / gamma)),
                ))

        img.save(stream, format=format)


class InvalidPfmFileFormat(Exception):
    """Invalid PFM file format exception"""

    def __init__(self, error_message):
        super().__init__(error_message)


def _read_line(stream):
    result = b""
    while True:
        cur_byte = stream.read(1)
        if cur_byte in [b"", b"\n"]:
            return result.decode("ascii")

        result += cur_byte


def _parse_img_size(line: str):
    elements = line.split(" ")
    if len(elements) != 2:
        raise InvalidPfmFileFormat("invalid image size specification")

    try:
        width, height = (int(elements[0]), int(elements[1]))
        if (width < 0) or (height < 0):
            raise ValueError()
    except ValueError:
        raise InvalidPfmFileFormat("invalid width/height")

    return width, height


def _parse_endianness(line: str):
    try:
        value = float(line)
    except ValueError:
        raise InvalidPfmFileFormat("missing endianness specification")

    if value == 1.0:
        return Endianness.BIG_ENDIAN
    elif value == -1.0:
        return Endianness.LITTLE_ENDIAN
    else:
        raise InvalidPfmFileFormat("invalid endianness specification")


def read_pfm_image(stream):
    """Read a PFM image from a stream

    Return a ``HdrImage`` object containing the image. If an error occurs, raise a
    ``InvalidPfmFileFormat`` exception."""
    magic = _read_line(stream)
    if magic != "PF":
        raise InvalidPfmFileFormat("invalid magic in PFM file")

    img_size = _read_line(stream)
    (width, height) = _parse_img_size(img_size)

    endianness_line = _read_line(stream)
    endianness = _parse_endianness(endianness_line)

    result = HdrImage(width=width, height=height)
    for y in range(height - 1, -1, -1):
        for x in range(width):
            (r, g, b) = [_read_float(stream, endianness) for i in range(3)]
            result.set_pixel(x, y, Color(r, g, b))

    return result
