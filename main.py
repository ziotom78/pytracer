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
import hdrimages

@dataclass
class Parameters:
    input_pfm_file_name: str = ""
    factor:float = 0.2
    gamma:float = 1.0
    output_png_file_name: str = ""

    def parse_command_line(self, argv):
        if len(sys.argv) != 5:
            raise RuntimeError("Usage: main.py INPUT_PFM_FILE FACTOR GAMMA OUTPUT_PNG_FILE")

        self.input_pfm_file_name = sys.argv[1]

        try:
            self.factor = float(sys.argv[2])
        except ValueError:
            raise RuntimeError(f"Invalid factor ('{sys.argv[2]}'), it must be a floating-point number.")

        try:
            self.gamma = float(sys.argv[3])
        except ValueError:
            raise RuntimeError(f"Invalid gamma ('{sys.argv[3]}'), it must be a floating-point number.")

        self.output_png_file_name = sys.argv[4]


def main(argv):
    parameters = Parameters()
    try:
        parameters.parse_command_line(argv)
    except RuntimeError as err:
        print("Error: ", err)
        return

    with open(parameters.input_pfm_file_name, "rb") as inpf:
        img = hdrimages.read_pfm_image(inpf)

    print(f"File {parameters.input_pfm_file_name} has been read from disk.")

    img.normalize_image(factor=parameters.factor)
    img.clamp_image()

    with open(parameters.output_png_file_name, "wb") as outf:
        img.write_ldr_image(stream=outf, format="PNG", gamma=parameters.gamma)

    print(f"File {parameters.output_png_file_name} has been written to disk.")


if __name__ == '__main__':
    import sys
    main(sys.argv)
