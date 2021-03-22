from dataclasses import dataclass

def are_close(num1, num2, epsilon=1e-6):
    return abs(num1 - num2) < epsilon

@dataclass
class Color:
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0

    def __add__(self, other):
        return Color(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b,
        )

    def __sub__(self, other):
        return Color(
            self.r - other.r,
            self.g - other.g,
            self.b - other.b,
        )

    def __mul__(self, other):
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

    def is_close(self, other, epsilon=1e-6):
        return (are_close(self.r, other.r, epsilon=epsilon) and
                are_close(self.g, other.g, epsilon=epsilon) and
                are_close(self.b, other.b, epsilon=epsilon))
