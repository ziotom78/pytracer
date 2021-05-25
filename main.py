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
from math import sqrt
from time import process_time
import sys

from hdrimages import HdrImage, Endianness, read_pfm_image
from camera import OrthogonalCamera, PerspectiveCamera
from colors import Color, BLACK, WHITE
from geometry import Point, Vec
from imagetracer import ImageTracer
from pcg import PCG
from shapes import Sphere, Plane
from transformations import translation, scaling, rotation_z
from world import World
from materials import UniformPigment, CheckeredPigment, ImagePigment, DiffuseBRDF, Material, SpecularBRDF
from render import OnOffRenderer, FlatRenderer, PathTracer

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


RENDERERS = ["onoff", "flat", "pathtracing"]


@click.command("demo")
@click.option("--width", type=int, default=640, help="Width of the image to render")
@click.option("--height", type=int, default=480, help="Height of the image to render")
@click.option("--angle-deg", type=float, default=0.0, help="Angle of view")
@click.option('--algorithm', type=click.Choice(RENDERERS), default="pathtracing")
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
@click.option(
    "--num-of-rays",
    type=int,
    default=10,
    help="Number of rays departing from each surface point (only applicable with --algorithm=pathtracing)."
)
@click.option(
    "--max-depth",
    type=int,
    default=3,
    help="Maximum allowed ray depth (only applicable with --algorithm=pathtracing)."
)
@click.option(
    "--init-state",
    type=int,
    help="Initial seed for the random number generator (positive number).",
    default=45,
)
@click.option(
    "--init-seq",
    type=int,
    help="Identifier of the sequence produced by the random number generator (positive number).",
    default=54
)
@click.option(
    "--samples-per-pixel",
    type=int,
    help="Number of samples per pixel (must be a perfect square, e.g., 16).",
    default=1,
)
def demo(width, height, angle_deg, algorithm, orthogonal, pfm_output, png_output, num_of_rays, max_depth, init_state,
         init_seq, samples_per_pixel):

    samples_per_side = int(sqrt(samples_per_pixel))
    if samples_per_side**2 != samples_per_pixel:
        print(f"Error, the number of samples per pixel ({samples_per_pixel}) must be a perfect square")
        return

    # Create a world and populate it with a few shapes
    world = World()

    sky_material = Material(
        brdf=DiffuseBRDF(pigment=UniformPigment(Color(0, 0, 0))),
        emitted_radiance=UniformPigment(Color(1.0, 0.9, 0.5)),
    )
    ground_material = Material(
        brdf=DiffuseBRDF(
            pigment=CheckeredPigment(
                color1=Color(0.3, 0.5, 0.1),
                color2=Color(0.1, 0.2, 0.5),
            )
        )
    )
    sphere_material = Material(brdf=DiffuseBRDF(pigment=UniformPigment(Color(0.3, 0.4, 0.8))))
    mirror_material = Material(brdf=SpecularBRDF(pigment=UniformPigment(color=Color(0.6, 0.2, 0.3))))
    world.shapes.append(
        Sphere(
            material=sky_material,
            transformation=scaling(Vec(200, 200, 200)) * translation(Vec(0, 0, 0.4))
        )
    )
    world.shapes.append(
        Plane(
            material=ground_material,
        )
    )
    world.shapes.append(Sphere(
        material=sphere_material,
        transformation=translation(Vec(0, 0, 1)),
    ))
    world.shapes.append(Sphere(
        material=mirror_material,
        transformation=translation(Vec(1, 2.5, 0)),
    ))

    image = HdrImage(width, height)
    print(f"Generating a {width}×{height} image, with the camera tilted by {angle_deg}°")

    # Initialize a camera
    camera_tr = rotation_z(angle_deg=angle_deg) * translation(Vec(-1.0, 0.0, 1.0))
    if orthogonal:
        camera = OrthogonalCamera(aspect_ratio=width / height, transformation=camera_tr)
    else:
        camera = PerspectiveCamera(
            aspect_ratio=width / height, transformation=camera_tr
        )

    # Run the ray-tracer

    tracer = ImageTracer(image=image, camera=camera, samples_per_side=samples_per_side)

    if algorithm == "onoff":
        print("Using on/off renderer")
        renderer = OnOffRenderer(world=world, background_color=BLACK)
    elif algorithm == "flat":
        print("Using flat renderer")
        renderer = FlatRenderer(world=world, background_color=BLACK)
    elif algorithm == "pathtracing":
        print("Using a path tracer")
        renderer = PathTracer(
            world=world,
            pcg=PCG(init_state=init_state, init_seq=init_seq),
            num_of_rays=num_of_rays,
            max_depth=max_depth,
        )
    else:
        print(f"Unknown renderer: {algorithm}")
        sys.exit(1)

    def print_progress(row, col):
        print(f"Rendering row {row + 1}/{image.height}\r", end="")

    start_time = process_time()
    tracer.fire_all_rays(renderer, callback=print_progress)
    elapsed_time = process_time() - start_time

    print(f"Rendering completed in {elapsed_time:.1f} s")

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
