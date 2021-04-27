# -*- encoding: utf-8 -*-

from math import sin, cos, radians
from geometry import Vec, Point, Normal
from colors import are_close


def _matr_prod(a, b):
    result = [[0.0 for i in range(4)] for j in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += a[i][k] * b[k][j]

    return result


def _are_matr_close(m1, m2):
    for i in range(4):
        for j in range(4):
            if not are_close(m1[i][j], m2[i][j]):
                return False

    return True


def _diff_of_products(a: float, b: float, c: float, d: float):
    # On systems supporting the FMA instruction (e.g., C++, Julia), you
    # might want to implement this function using the trick explained here:
    #
    # https://pharr.org/matt/blog/2019/11/03/difference-of-floats.html
    #
    # as it prevents roundoff errors.

    return a * b - c * d


IDENTITY_MATR4x4 = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]


class Transformation:
    def __init__(self, m=IDENTITY_MATR4x4, invm=IDENTITY_MATR4x4):
        self.m = m
        self.invm = invm

    def __mul__(self, other):
        if isinstance(other, Vec):
            row0, row1, row2, row3 = self.m
            return Vec(x=other.x * row0[0] + other.y * row0[1] + other.z * row0[2],
                       y=other.x * row1[0] + other.y * row1[1] + other.z * row1[2],
                       z=other.x * row2[0] + other.y * row2[1] + other.z * row2[2])
        elif isinstance(other, Point):
            row0, row1, row2, row3 = self.m
            p = Point(x=other.x * row0[0] + other.y * row0[1] + other.z * row0[2] + row0[3],
                      y=other.x * row1[0] + other.y * row1[1] + other.z * row1[2] + row1[3],
                      z=other.x * row2[0] + other.y * row2[1] + other.z * row2[2] + row2[3])
            w = other.x * row3[0] + other.y * row3[1] + other.z * row3[2] + row3[3]

            if w == 1.0:
                return p
            else:
                return Point(p.x / w, p.y / w, p.z / w)
        elif isinstance(other, Normal):
            row0, row1, row2, _ = self.invm
            return Normal(x=other.x * row0[0] + other.y * row1[0] + other.z * row2[0],
                          y=other.x * row0[1] + other.y * row1[1] + other.z * row2[1],
                          z=other.x * row0[2] + other.y * row1[2] + other.z * row2[2])
        elif isinstance(other, Transformation):
            result_m = _matr_prod(self.m, other.m)
            result_invm = _matr_prod(other.invm, self.invm)  # Reverse order! (A B)^-1 = B^-1 A^-1
            return Transformation(m=result_m, invm=result_invm)
        else:
            raise TypeError(f"Invalid type {type(other)} multiplied to a Transformation object")

    def is_consistent(self):
        prod = _matr_prod(self.m, self.invm)
        return _are_matr_close(prod, IDENTITY_MATR4x4)

    def __repr__(self):
        row0, row1, row2, row3 = self.m
        fmtstring = "   [{0:6.3e} {1:6.3e} {2:6.3e} {3:6.3e}],\n"
        result = "[\n"
        result += fmtstring.format(*row0)
        result += fmtstring.format(*row1)
        result += fmtstring.format(*row2)
        result += fmtstring.format(*row3)
        result += "]"
        return result

    def is_close(self, other):
        return _are_matr_close(self.m, other.m) and _are_matr_close(self.invm, other.invm)

    def inverse(self):
        return Transformation(m=self.invm, invm=self.m)


def translation(vec):
    return Transformation(
        m=[[1.0, 0.0, 0.0, vec.x],
           [0.0, 1.0, 0.0, vec.y],
           [0.0, 0.0, 1.0, vec.z],
           [0.0, 0.0, 0.0, 1.0]],
        invm=[[1.0, 0.0, 0.0, -vec.x],
              [0.0, 1.0, 0.0, -vec.y],
              [0.0, 0.0, 1.0, -vec.z],
              [0.0, 0.0, 0.0, 1.0]],
    )


def scaling(vec):
    return Transformation(
        m=[[vec.x, 0.0, 0.0, 0.0],
           [0.0, vec.y, 0.0, 0.0],
           [0.0, 0.0, vec.z, 0.0],
           [0.0, 0.0, 0.0, 1.0]],
        invm=[[1 / vec.x, 0.0, 0.0, 0.0],
              [0.0, 1 / vec.y, 0.0, 0.0],
              [0.0, 0.0, 1 / vec.z, 0.0],
              [0.0, 0.0, 0.0, 1.0]],
    )


def rotation_x(angle_deg: float):
    sinang, cosang = sin(radians(angle_deg)), cos(radians(angle_deg))
    return Transformation(
        m=[[1.0, 0.0, 0.0, 0.0],
           [0.0, cosang, -sinang, 0.0],
           [0.0, sinang, cosang, 0.0],
           [0.0, 0.0, 0.0, 1.0]],
        invm=[[1.0, 0.0, 0.0, 0.0],
              [0.0, cosang, sinang, 0.0],
              [0.0, -sinang, cosang, 0.0],
              [0.0, 0.0, 0.0, 1.0]],
    )


def rotation_y(angle_deg: float):
    sinang, cosang = sin(radians(angle_deg)), cos(radians(angle_deg))
    return Transformation(
        m=[[cosang, 0.0, sinang, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [-sinang, 0.0, cosang, 0.0],
           [0.0, 0.0, 0.0, 1.0]],
        invm=[[cosang, 0.0, -sinang, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [sinang, 0.0, cosang, 0.0],
              [0.0, 0.0, 0.0, 1.0]],
    )


def rotation_z(angle_deg: float):
    sinang, cosang = sin(radians(angle_deg)), cos(radians(angle_deg))
    return Transformation(
        m=[[cosang, -sinang, 0.0, 0.0],
           [sinang, cosang, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]],
        invm=[[cosang, sinang, 0.0, 0.0],
              [-sinang, cosang, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]],
    )
