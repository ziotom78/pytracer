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
import sys

from hdrimages import HdrImage, Endianness, read_pfm_image
from camera import OrthogonalCamera, PerspectiveCamera
from colors import Color, BLACK, WHITE
from geometry import Point, Vec
from imagetracer import ImageTracer
from ray import Ray
from shapes import Sphere
from transformations import translation, scaling, rotation_z
from world import World
from materials import UniformPigment, CheckeredPigment, ImagePigment, DiffuseBRDF, Material
from render import OnOffRenderer, FlatRenderer

import click


@dataclass
class Parameters:
    input_pfm_file_name: str = ""
    factor: float = 0.2
    gamma: float = 1.0
    output_png_file_name: str = ""


@click.group()
def cli():
    pass


RENDERERS = ["onoff", "flat"]


@click.command("demo")
@click.option("--width", type=int, default=640, help="Width of the image to render")
@click.option("--height", type=int, default=480, help="Height of the image to render")
@click.option("--angle-deg", type=float, default=0.0, help="Angle of view")
@click.option('--algorithm', type=click.Choice(RENDERERS), default="flat")
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
def demo(width, height, angle_deg, algorithm, orthogonal, pfm_output, png_output):
    # Create a world and populate it with a few shapes
    world = World()

    material1 = Material(
        brdf=DiffuseBRDF(UniformPigment(Color(0.7, 0.3, 0.2)))
    )

    material2 = Material(
        brdf=DiffuseBRDF(CheckeredPigment(Color(0.2, 0.7, 0.3), Color(0.3, 0.2, 0.7), num_of_steps=4)),
    )

    sphere_texture = HdrImage(2, 2)
    sphere_texture.set_pixel(0, 0, Color(0.1, 0.2, 0.3))
    sphere_texture.set_pixel(0, 1, Color(0.2, 0.1, 0.3))
    sphere_texture.set_pixel(1, 0, Color(0.3, 0.2, 0.1))
    sphere_texture.set_pixel(1, 1, Color(0.3, 0.1, 0.2))

    material3 = Material(
        brdf=DiffuseBRDF(ImagePigment(sphere_texture))
    )

    for x in [-0.5, 0.5]:
        for y in [-0.5, 0.5]:
            for z in [-0.5, 0.5]:
                world.add(
                    Sphere(
                        transformation=translation(Vec(x, y, z))
                                       * scaling(Vec(0.1, 0.1, 0.1)),
                        material=material1,
                    )
                )

    # Place two other balls in the bottom/left part of the cube, so
    # that we can check if there are issues with the orientation of
    # the image
    world.add(
        Sphere(
            transformation=translation(Vec(0.0, 0.0, -0.5))
                           * scaling(Vec(0.1, 0.1, 0.1)),
            material=material2,
        )
    )

    world.add(
        Sphere(
            transformation=translation(Vec(0.0, 0.5, 0.0)) * scaling(Vec(0.1, 0.1, 0.1)),
            material=material3,
        )
    )

    image = HdrImage(width, height)
    print(f"Generating a {width}×{height} image, with the camera tilted by {angle_deg}°")

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

    if algorithm == "onoff":
        print("Using on/off renderer")
        renderer = OnOffRenderer(world=world, background_color=BLACK)
    elif algorithm == "flat":
        print("Using flat renderer")
        renderer = FlatRenderer(world=world, background_color=BLACK)
    else:
        print(f"Unknown renderer: {algorithm}")
        sys.exit(1)

    tracer.fire_all_rays(renderer)

    # Save the HDR image
    with open(pfm_output, "wb") as outf:
        image.write_pfm(outf)
    print(f"HDR demo image written to {pfm_output}")

    # Apply tone-mapping to the image
    image.normalize_image(factor=0.3)
    image.clamp_image()

    # Save the LDR image
    with open(png_output, "wb") as outf:
        image.write_ldr_image(outf, "PNG")
    print(f"PNG demo image written to {png_output}")


@click.command("pfm2png")
@click.option("--factor", type=float, default=0.7, help="Multiplicative factor")
@click.option("--gamma", type=float, default=1.0, help="Exponent for gamma-correction")
@click.option("--luminosity", type=float, default=None, help="Average luminosity")
@click.argument("input_pfm_file_name", type=str)
@click.argument("output_png_file_name", type=str)
def pfm2png(factor, gamma, luminosity, input_pfm_file_name, output_png_file_name):
    with open(input_pfm_file_name, "rb") as inpf:
        img = read_pfm_image(inpf)

    print(f"File {input_pfm_file_name} has been read from disk.")

    img.normalize_image(factor=factor, luminosity=luminosity)
    img.clamp_image()

    with open(output_png_file_name, "wb") as outf:
        img.write_ldr_image(stream=outf, format="PNG", gamma=gamma)

    print(f"File {output_png_file_name} has been written to disk.")


cli.add_command(demo)
cli.add_command(pfm2png)

if __name__ == "__main__":
    cli()
