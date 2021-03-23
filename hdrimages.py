from colors import Color
import struct
from enum import Enum


class Endianness(Enum):
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
    return struct.unpack(format_str, stream.read(4))[0]


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

    def write_pfm(self, stream, endianness=Endianness.LITTLE_ENDIAN):
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


class InvalidPfmFileFormat(Exception):
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
        raise InvalidPfmFileFormat("invalid specification of the endianness")

    if value > 0:
        return Endianness.BIG_ENDIAN
    else:
        return Endianness.LITTLE_ENDIAN


def read_pfm_image(stream):
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

