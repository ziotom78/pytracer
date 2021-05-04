#!/usr/bin/env python3

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

from hdrimages import HdrImage, Endianness, read_pfm_image
from camera import OrthogonalCamera, PerspectiveCamera
from colors import Color, BLACK, WHITE
from geometry import Point, Vec
from imagetracer import ImageTracer
from ray import Ray
from shapes import Sphere
from transformations import translation, scaling, rotation_z
from world import World

import click


@dataclass
class Parameters:
    input_pfm_file_name: str = ""
    factor: float = 0.2
    gamma: float = 1.0
    output_png_file_name: str = ""

    def parse_command_line(self, argv):
        if len(sys.argv) != 5:
            raise RuntimeError(
                "Usage: main.py INPUT_PFM_FILE FACTOR GAMMA OUTPUT_PNG_FILE"
            )

        self.input_pfm_file_name = sys.argv[1]

        try:
            self.factor = float(sys.argv[2])
        except ValueError:
            raise RuntimeError(
                f"Invalid factor ('{sys.argv[2]}'), it must be a floating-point number."
            )

        try:
            self.gamma = float(sys.argv[3])
        except ValueError:
            raise RuntimeError(
                f"Invalid gamma ('{sys.argv[3]}'), it must be a floating-point number."
            )

        self.output_png_file_name = sys.argv[4]


@click.group()
def cli():
    pass


@click.command("demo")
@click.option("--width", type=int, default=640, help="Width of the image to render")
@click.option("--height", type=int, default=480, help="Height of the image to render")
@click.option("--angle-deg", type=float, default=0.0, help="Angle of view")
@click.option(
    "--pfm-output",
    type=str,
    default="output.pfm",
    help="Name of the PFM file to create",
)
@click.option(
    "--png-output",
    type=str,
    default="output.png",
    help="Name of the PNG file to create",
)
@click.option(
    "--orthogonal",
    is_flag=True,
    help="Use an orthogonal camera instead of a perspective camera",
)
def demo(width, height, angle_deg, orthogonal, pfm_output, png_output):
    image = HdrImage(width, height)

    # Create a world and populate it with a few shapes
    world = World()

    for x in [-0.5, 0.5]:
        for y in [-0.5, 0.5]:
            for z in [-0.5, 0.5]:
                world.add(
                    Sphere(
                        transformation=translation(Vec(x, y, z))
                        * scaling(Vec(0.1, 0.1, 0.1))
                    )
                )

    # Place two other balls in the bottom/left part of the cube, so
    # that we can check if there are issues with the orientation of
    # the image
    world.add(
        Sphere(
            transformation=translation(Vec(0.0, 0.0, -0.5))
            * scaling(Vec(0.1, 0.1, 0.1))
        )
    )
    world.add(
        Sphere(
            transformation=translation(Vec(0.0, 0.5, 0.0)) * scaling(Vec(0.1, 0.1, 0.1))
        )
    )

    # Initialize a camera
    camera_tr = rotation_z(angle_deg=angle_deg) * translation(Vec(-1.0, 0.0, 0.0))
    if orthogonal:
        camera = OrthogonalCamera(aspect_ratio=width / height, transformation=camera_tr)
    else:
        camera = PerspectiveCamera(
            aspect_ratio=width / height, transformation=camera_tr
        )

    # Run the ray-tracer

    tracer = ImageTracer(image=image, camera=camera)

    def compute_color(ray: Ray) -> Color:
        if world.ray_intersection(ray):
            return WHITE
        else:
            return BLACK

    tracer.fire_all_rays(compute_color)

    # Save the HDR image
    with open(pfm_output, "wb") as outf:
        image.write_pfm(outf)
    print(f"HDR demo image written to {pfm_output}")

    # Apply tone-mapping to the image
    image.normalize_image(factor=1.0)
    image.clamp_image()

    # Save the LDR image
    with open(png_output, "wb") as outf:
        image.write_ldr_image(outf, "PNG")
    print(f"PNG demo image written to {png_output}")


@click.command("pfm2png")
@click.option("--factor", type=float, default=0.2, help="Multiplicative factor")
@click.option(
    "--gamma", type=float, default=1.0, help="Value to be used for gamma correction"
)
@click.argument("input_pfm_file_name")
@click.argument("output_png_file_name")
def pfm2png(factor, gamma, input_pfm_file_name, output_png_file_name):
    with open(input_pfm_file_name, "rb") as inpf:
        img = read_pfm_image(inpf)

    print(f"File {input_pfm_file_name} has been read from disk.")

    img.normalize_image(factor=factor)
    img.clamp_image()

    with open(output_png_file_name, "wb") as outf:
        img.write_ldr_image(stream=outf, format="PNG", gamma=gamma)

    print(f"File {output_png_file_name} has been written to disk.")


cli.add_command(demo)
cli.add_command(pfm2png)

if __name__ == "__main__":
    cli()
