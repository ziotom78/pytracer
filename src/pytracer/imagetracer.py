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

from pytracer.colors import Color
from pytracer.hdrimages import HdrImage
from pytracer.camera import Camera
from pytracer.pcg import PCG


class ImageTracer:
    """Trace an image by shooting light rays through each of its pixels"""

    def __init__(
        self,
        image: HdrImage,
        camera: Camera,
        samples_per_side: int = 0,
        pcg: PCG = PCG(),
    ):
        """Initialize an ImageTracer object

        The parameter `image` must be a :class:`.HdrImage` object that has already been initialized.
        The parameter `camera` must be a descendeant of the :class:`.Camera` object.

        If `samples_per_side` is larger than zero, stratified sampling will be applied to each pixel in the
        image, using the random number generator `pcg`."""
        self.image = image
        self.camera = camera
        self.samples_per_side = samples_per_side
        self.pcg = pcg

    def fire_ray(self, col: int, row: int, u_pixel=0.5, v_pixel=0.5):
        """Shoot one light ray through image pixel (col, row)

        The parameters (col, row) are measured in the same way as they are in :class:`.HdrImage`: the bottom left
        corner is placed at (0, 0).

        The values of `u_pixel` and `v_pixel` are floating-point numbers in the range [0, 1]. They specify where
        the ray should cross the pixel; passing 0.5 to both means that the ray will pass through the pixel's center."""
        u = (col + u_pixel) / self.image.width
        v = 1.0 - (row + v_pixel) / self.image.height
        return self.camera.fire_ray(u, v)

    def fire_all_rays(
        self, func, callback=None, callback_time_s: float = 2.0, **callback_kwargs
    ):
        """Shoot several light rays crossing each of the pixels in the image

        For each pixel in the :class:`.HdrImage` object fire one ray, and pass it to the function `func`, which
        must accept a :class:`.Ray` as its only parameter and must return a :class:`.Color` instance telling the
        color to assign to that pixel in the image.

        If `callback` is not none, it must be a function accepting at least two parameters named `col` and `row`.
        This function is called periodically during the rendering, and the two mandatory arguments are the row and
        column number of the last pixel that has been traced. (Both the row and column are increased by one starting
        from zero: first the row and then the column.) The time between two consecutive calls to the callback can be
        tuned using the parameter `callback_time_s`. Any keyword argument passed to `fire_all_rays` is passed to the
        callback.
        """
        last_call_time = process_time()
        if callback:
            callback(col=0, row=0, **callback_kwargs)

        for row in range(self.image.height):
            for col in range(self.image.width):
                cum_color = Color(0.0, 0.0, 0.0)

                if self.samples_per_side > 0:
                    # Run stratified sampling over the pixel's surface
                    for inter_pixel_row in range(self.samples_per_side):
                        for inter_pixel_col in range(self.samples_per_side):
                            u_pixel = (
                                inter_pixel_col + self.pcg.random_float()
                            ) / self.samples_per_side
                            v_pixel = (
                                inter_pixel_row + self.pcg.random_float()
                            ) / self.samples_per_side
                            ray = self.fire_ray(
                                col=col, row=row, u_pixel=u_pixel, v_pixel=v_pixel
                            )
                            cum_color += func(ray)

                    self.image.set_pixel(
                        col, row, cum_color * (1 / self.samples_per_side**2)
                    )
                else:
                    ray = self.fire_ray(col=col, row=row)
                    self.image.set_pixel(col, row, func(ray))

                # Call the callback, if necessary
                current_time = process_time()
                if callback and (current_time - last_call_time > callback_time_s):
                    callback(col=col, row=row, **callback_kwargs)
                    last_call_time = current_time
