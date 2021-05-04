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


from dataclasses import dataclass
from misc import are_close


@dataclass
class Color:
    """
    A RGB color

    The class has three floating-point members: `r` (red), `g` (green), and `b` (blue).
    """
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0

    def __add__(self, other):
        """Sum two colors"""
        return Color(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b,
        )

    def __sub__(self, other):
        """Subtract two colors"""
        return Color(
            self.r - other.r,
            self.g - other.g,
            self.b - other.b,
        )

    def __mul__(self, other):
        """Multiply two colors, or one color with one number"""
        try:
            # Try a color-times-color operation
            return Color(
                self.r * other.r,
                self.g * other.g,
                self.b * other.b,
            )
        except AttributeError:
            # Fall back to a color-times-scalar operation
            return Color(
                self.r * other,
                self.g * other,
                self.b * other,
            )

    def luminosity(self):
        """Return a rough measure of the luminosity associated with the color"""
        return (max(self.r, self.g, self.b) + min(self.r, self.g, self.b)) / 2

    def is_close(self, other, epsilon=1e-6):
        """Return True if the three RGB components of two colors are close by less than `epsilon`"""
        return (are_close(self.r, other.r, epsilon=epsilon) and
                are_close(self.g, other.g, epsilon=epsilon) and
                are_close(self.b, other.b, epsilon=epsilon))


BLACK = Color(0.0, 0.0, 0.0)
WHITE = Color(1.0, 1.0, 1.0)
