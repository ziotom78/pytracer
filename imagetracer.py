# -*- encoding: utf-8 -*-
#
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
from time import process_time

from hdrimages import HdrImage
from camera import Camera


class ImageTracer:
    """Trace an image by shooting light rays through each of its pixels
    """

    def __init__(self, image: HdrImage, camera: Camera):
        """Initialize an ImageTracer object

        The parameter `image` must be a :class:`.HdrImage` object that has already been initialized.
        The parameter `camera` must be a descendeant of the :class:`.Camera` object."""
        self.image = image
        self.camera = camera

    def fire_ray(self, col: int, row: int, u_pixel=0.5, v_pixel=0.5):
        """Shoot one light ray through image pixel (col, row)

        The parameters (col, row) are measured in the same way as they are in :class:`.HdrImage`: the bottom left
        corner is placed at (0, 0).

        The values of `u_pixel` and `v_pixel` are floating-point numbers in the range [0, 1]. They specify where
        the ray should cross the pixel; passing 0.5 to both means that the ray will pass through the pixel's center."""
        u = (col + u_pixel) / self.image.width
        v = 1.0 - (row + v_pixel) / self.image.height
        return self.camera.fire_ray(u, v)

    def fire_all_rays(self, func, callback=None, callback_time_s: float = 2.0, **callback_kwargs):
        """Shoot several light rays crossing each of the pixels in the image

        For each pixel in the :class:`.HdrImage` object fire one ray, and pass it to the function `func`, which
        must accept a :class:`.Ray` as its only parameter and must return a :class:`.Color` instance telling the
        color to assign to that pixel in the image."""
        last_call_time = process_time()
        callback(0, 0, **callback_kwargs)
        for row in range(self.image.height):
            for col in range(self.image.width):
                ray = self.fire_ray(col, row)
                color = func(ray)
                self.image.set_pixel(col, row, color)

                current_time = process_time()
                if callback and (current_time - last_call_time > callback_time_s):
                    callback(row, col, **callback_kwargs)
                    last_call_time = current_time
