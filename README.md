# Pytracer

This is a simple raytracer written in Python. It is meant to be used as a reference for the course *Numerical techniques for photorealistic image generation* (AY2020-2021).

## Installation

You need Python 3.8 or higher. To install the dependencies, run the following commands (possibly within a virtual environment):

    pip install -e .

To check that the code works as expected, you can run the suite of tests using the following command:

    pytest


## Usage

You can run the program in two ways:

1.  Run `src/pytracer/main.py`:

        python3 src/pytracer/main.py [ARGS]

2.  Use `python3 -m pytracer`:

        python3 -m pytracer [ARGS]

You can create a demo image with the following command:

    python3 -m pytracer render examples/demo.txt

Beware that it will take a very long time to produce the image!

To get command-line help, run

    python3 -m pytracer --help


## Scene files

Describe here the syntax used in the files. See [`examples/demo.txt`](./examples/demo.txt) for an example.


## History

See the file [CHANGELOG.md](https://github.com/ziotom78/pytracer/blob/master/CHANGELOG.md).


## License

The code is released under a MIT license. See the file [LICENSE.md](./LICENSE.md)
