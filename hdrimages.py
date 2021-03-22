from colors import Color
import struct
from enum import Enum


class Endianness(Enum):
    LITTLE_ENDIAN = 1
    BIG_ENDIAN = 2


def _write_float(stream, value, endianness=Endianness.LITTLE_ENDIAN):
    # Meaning of "<f":
    # "<": little endian
    # "f": single-precision floating point value (32 bit)
    if endianness == Endianness.LITTLE_ENDIAN:
        format_str = "<f"
    else:
        format_str = ">f"

    stream.write(struct.pack(format_str, value))


class HdrImage:
    def __init__(self, width=0, height=0):
        (self.width, self.height) = (width, height)
        self.pixels = [Color() for i in range(self.width * self.height)]

    def valid_coordinates(self, x, y):
        return ((x >= 0) and (x < self.width) and
                (y >= 0) and (y < self.height))

    def pixel_offset(self, x, y):
        return y * self.width + x

    def get_pixel(self, x, y):
        assert self.valid_coordinates(x, y)
        return self.pixels[self.pixel_offset(x, y)]

    def set_pixel(self, x, y, new_color):
        assert self.valid_coordinates(x, y)
        self.pixels[self.pixel_offset(x, y)] = new_color

    def write_pfm(self, stream):
        # The PFM header, as a Python string (UTF-8)
        header = f"PF\n{self.width} {self.height}\n-1.0\n"

        # Convert the header into a sequence of bytes
        stream.write(header.encode("utf-8"))

        # Write the image (bottom-to-up, left-to-right)
        for y in reversed(range(self.height)):
            for x in range(self.width):
                color = self.get_pixel(x, y)
                _write_float(stream, color.r, endianness=Endianness.LITTLE_ENDIAN)
                _write_float(stream, color.g, endianness=Endianness.LITTLE_ENDIAN)
                _write_float(stream, color.b, endianness=Endianness.LITTLE_ENDIAN)
