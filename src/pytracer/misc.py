# -*- encoding: utf-8 -*-


def are_close(num1, num2, epsilon=1e-6):
    """Return True if the two numbers differ by less than `epsilon`"""
    return abs(num1 - num2) < epsilon
