# -*- encoding: utf-8 -*-

from dataclasses import dataclass


# The routines to_uint64 and to_uint32 are needed in Python, as it does
# not have the concept of "typed integers". In other languages like C++
# it is enough to declare a variable as `uint32_t` or `uint64_t`, and
# clipping will be done automatically by the CPU/virtual machine.


def to_uint64(x: int) -> int:
    """Clip an integer so that it occupies 64 bits"""
    return x & 0xFFFFFFFFFFFFFFFF


def to_uint32(x: int) -> int:
    """Clip an integer so that it occupies 32 bits"""
    return x & 0xFFFFFFFF


@dataclass
class PCG:
    """PCG Uniform Pseudo-random Number Generator"""

    state: int = 0
    inc: int = 0

    def __init__(self, init_state=42, init_seq=54):
        # 64-bit
        self.state = 0

        # 64-bit
        self.inc = (init_seq << 1) | 1

        self.random()

        # 64-bit
        self.state += init_state

        self.random()

    def random(self):
        """Return a new random number and advance PCG's internal state"""
        # 64-bit
        oldstate = self.state

        # 64-bit
        self.state = to_uint64((oldstate * 6364136223846793005 + self.inc))

        # 32-bit
        xorshifted = to_uint32((((oldstate >> 18) ^ oldstate) >> 27))

        # 32-bit
        rot = oldstate >> 59

        # 32-bit
        return to_uint32((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))

    def random_float(self):
        """Return a new random number uniformly distributed over [0, 1]"""
        return self.random() / 0xFFFFFFFF
