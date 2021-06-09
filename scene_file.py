# -*- encoding: utf-8 -*-
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Dict, Union, List, Tuple

from camera import Camera, PerspectiveCamera, OrthogonalCamera
from colors import Color
from geometry import Vec
from hdrimages import read_pfm_image
from materials import Material, BRDF, Pigment, UniformPigment, CheckeredPigment, ImagePigment, DiffuseBRDF, SpecularBRDF
from shapes import Shape, Sphere, Plane
from transformations import translation, rotation_x, rotation_y, rotation_z, scaling, Transformation

WHITESPACE = " \t\n\r"
SYMBOLS = "()<>[],*"


@dataclass
class SourceLocation:
    """A specific position in a source file

    This class has the following fields:
    - file_name: the name of the file, or the empty string if there is no file associated with this location
      (e.g., because the source code was provided as a memory stream, or through a network connection)
    - line_num: number of the line (starting from 1)
    - col_num: number of the column (starting from 1)
    """
    file_name: str = ""
    line_num: int = 0
    col_num: int = 0


@dataclass
class Token:
    """A lexical token, used when parsing a scene file"""
    location: SourceLocation


class StopToken(Token):
    """A token signalling the end of a file"""

    def __init__(self, location: SourceLocation):
        super().__init__(location=location)


class KeywordEnum(Enum):
    """Enumeration for all the possible keywords recognized by the lexer"""
    NEW = 1
    MATERIAL = 2
    PLANE = 3
    SPHERE = 4
    DIFFUSE = 5
    SPECULAR = 6
    UNIFORM = 7
    CHECKERED = 8
    IMAGE = 9
    IDENTITY = 10
    TRANSLATION = 11
    ROTATION_X = 12
    ROTATION_Y = 13
    ROTATION_Z = 14
    SCALING = 15
    CAMERA = 16
    ORTHOGONAL = 17
    PERSPECTIVE = 18
    FLOAT = 19


KEYWORDS: Dict[str, KeywordEnum] = {
    "new": KeywordEnum.NEW,
    "material": KeywordEnum.MATERIAL,
    "plane": KeywordEnum.PLANE,
    "sphere": KeywordEnum.SPHERE,
    "diffuse": KeywordEnum.DIFFUSE,
    "specular": KeywordEnum.SPECULAR,
    "uniform": KeywordEnum.UNIFORM,
    "checkered": KeywordEnum.CHECKERED,
    "image": KeywordEnum.IMAGE,
    "identity": KeywordEnum.IDENTITY,
    "translation": KeywordEnum.TRANSLATION,
    "rotation_x": KeywordEnum.ROTATION_X,
    "rotation_y": KeywordEnum.ROTATION_Y,
    "rotation_z": KeywordEnum.ROTATION_Z,
    "scaling": KeywordEnum.SCALING,
    "camera": KeywordEnum.CAMERA,
    "orthogonal": KeywordEnum.ORTHOGONAL,
    "perspective": KeywordEnum.PERSPECTIVE,
    "float": KeywordEnum.FLOAT,
}


class KeywordToken(Token):
    """A token containing a keyword"""

    def __init__(self, location: SourceLocation, keyword: KeywordEnum):
        super().__init__(location=location)
        self.keyword = keyword

    def __str__(self) -> str:
        return str(self.keyword)


class IdentifierToken(Token):
    """A token containing an identifier"""

    def __init__(self, location: SourceLocation, s: str):
        super().__init__(location=location)
        self.identifier = s

    def __str__(self) -> str:
        return self.identifier


class StringToken(Token):
    """A token containing a literal string"""

    def __init__(self, location: SourceLocation, s: str):
        super().__init__(location=location)
        self.string = s

    def __str__(self) -> str:
        return self.string


class LiteralNumberToken(Token):
    """A token containing a literal number"""

    def __init__(self, location: SourceLocation, value: float):
        super().__init__(location=location)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)


class SymbolToken(Token):
    """A token containing a symbol (i.e., a variable name)"""

    def __init__(self, location: SourceLocation, symbol: str):
        super().__init__(location=location)
        self.symbol = symbol

    def __str__(self) -> str:
        return self.symbol


@dataclass
class GrammarError(BaseException):
    """An error found by the lexer/parser while reading a scene file

    The fields of this type are the following:

    - `file_name`: the name of the file, or the empty string if there is no real file
    - `line_num`: the line number where the error was discovered (starting from 1)
    - `col_num`: the column number where the error was discovered (starting from 1)
    - `message`: a user-frendly error message
    """
    location: SourceLocation
    message: str


class InputStream:
    """A high-level wrapper around a stream, used to parse scene files

    This class implements a wrapper around a stream, with the following additional capabilities:
    - It tracks the line number and column number;
    - It permits to "un-read" characters and tokens.
    """

    def __init__(self, stream, file_name="", tabulations=8):
        self.stream = stream

        self.location = SourceLocation(file_name=file_name, line_num=1, col_num=1)

        self.saved_char = ""
        self.saved_location = self.location
        self.tabulations = tabulations

        self.saved_token: Union[Token, None] = None

    def _update_pos(self, ch):
        """Update `location` after having read `ch` from the stream"""
        if ch == "":
            # Nothing to do!
            return
        elif ch == "\n":
            self.location.line_num += 1
            self.location.col_num = 1
        elif ch == "\t":
            self.location.col_num += self.tabulations
        else:
            self.location.col_num += 1

    def read_char(self) -> str:
        """Read a new character from the stream"""
        if self.saved_char != "":
            ch = self.saved_char
            self.saved_char = ""
        else:
            ch = self.stream.read(1)

        self.saved_location = copy(self.location)
        self._update_pos(ch)

        return ch

    def unread_char(self, ch):
        """Push a character back to the stream"""
        assert self.saved_char == ""
        self.saved_char = ch
        self.location = copy(self.saved_location)

    def skip_whitespaces(self):
        """Keep reading characters until a non-whitespace character is found"""
        ch = self.read_char()
        while ch in WHITESPACE:
            ch = self.read_char()
            if ch == "":
                return

        # Put the non-whitespace character back
        self.unread_char(ch)

    def _parse_string_token(self, token_location: SourceLocation) -> StringToken:
        token = ""
        while True:
            ch = self.read_char()

            if ch == '"':
                break

            if ch == "":
                raise GrammarError(token_location, "unterminated string")

            token += ch

        return StringToken(token_location, token)

    def _parse_float_token(self, first_char: str, token_location: SourceLocation) -> LiteralNumberToken:
        token = first_char
        while True:
            ch = self.read_char()

            if not (ch.isdigit() or ch == "." or ch in ["e", "E"]):
                self.unread_char(ch)
                break

            token += ch

        try:
            value = float(token)
        except ValueError:
            raise GrammarError(token_location, f"'{token}' is an invalid floating-point number")

        return LiteralNumberToken(token_location, value)

    def _parse_keyword_or_identifier_token(
            self,
            first_char: str,
            token_location: SourceLocation,
    ) -> Union[KeywordToken, IdentifierToken]:
        token = first_char
        while True:
            ch = self.read_char()
            # Note that here we do not call "isalpha" but "isalnum": digits are ok after the first character
            if not (ch.isalnum() or ch == "_"):
                self.unread_char(ch)
                break

            token += ch

        try:
            # If it is a keyword, it must be listed in the KEYWORDS dictionary
            return KeywordToken(token_location, KEYWORDS[token])
        except KeyError:
            # If we got KeyError, it is not a keyword and thus it must be an identifier
            return IdentifierToken(token_location, token)

    def read_token(self) -> Token:
        """Read a token from the stream

        Raise :class:`.ParserError` if a lexical error is found."""
        if self.saved_token:
            result = self.saved_token
            self.saved_token = None
            return result

        self.skip_whitespaces()

        # At this point we're sure that ch does *not* contain a whitespace character
        ch = self.read_char()
        if ch == "":
            # No more characters in the file, so return a StopToken
            return StopToken(location=self.location)

        # However, if it signals the start of a comment, we need to treat this condition as if it were a whitespace
        if ch == "#":
            # Keep reading until the end of the line (include the case "", which means end-of-file)
            while self.read_char() not in ["\r", "\n", ""]:
                pass

            # Now skip all the whitespaces (including the characters '\r' and '\n')
            self.skip_whitespaces()

            # Note that self.skip_whitespaces() reads the first non-whitespace and then puts it back
            # using self.unread_char()
        else:
            self.unread_char(ch)

        # At this point we must check what kind of token begins with the "ch" character (which has been
        # put back in the stream with self.unread_char). First, we save the position in the stream
        token_location = copy(self.location)

        # Now we read again ch (which was put back using self.unread_char)
        ch = self.read_char()

        if ch in SYMBOLS:
            # One-character symbol, like '(' or ','
            return SymbolToken(token_location, ch)
        elif ch == '"':
            # A literal string (used for file names)
            return self._parse_string_token(token_location=token_location)
        elif ch.isdecimal() or ch in ["+", "-", "."]:
            # A floating-point number
            return self._parse_float_token(first_char=ch, token_location=token_location)
        elif ch.isalpha() or ch == "_":
            # Since it begins with an alphabetic character, it must either be a keyword or a identifier
            return self._parse_keyword_or_identifier_token(first_char=ch, token_location=token_location)
        else:
            # We got some weird character, like '@` or `&`
            raise GrammarError(self.location, f"Invalid character {ch}")

    def unread_token(self, token: Token):
        """Make as if `token` were never read from `input_file`"""
        assert not self.saved_token
        self.saved_token = token


@dataclass
class Scene:
    """A scene read from a scene file"""
    materials: Dict[str, Material] = field(default_factory=dict)
    objects: List[Shape] = field(default_factory=list)
    camera: Union[Camera, None] = None
    float_variables: Dict[str, float] = field(default_factory=dict)


def expect_symbol(input_file: InputStream, symbol: str) -> SymbolToken:
    """Read a token from `input_file` and check that it matches `symbol`."""
    token = input_file.read_token()
    if not isinstance(token, SymbolToken) or token.symbol != symbol:
        raise GrammarError(token.location, f"got '{token}' instead of '{symbol}'")


def expect_keywords(input_file: InputStream, keywords: List[KeywordEnum]) -> KeywordEnum:
    """Read a token from `input_file` and check that it is one of the keywords in `keywords`.

    Return the keyword as a :class:`.KeywordEnum` object."""
    token = input_file.read_token()
    if not isinstance(token, KeywordToken):
        raise GrammarError(token.location, f"expected a keyword instead of '{token}'")

    if not token.keyword in keywords:
        raise GrammarError(token.location,
                           f"expected one of the keywords {','.join([str(x) for x in keywords])} instead of '{token}'")

    return token.keyword


def expect_number(input_file: InputStream, scene: Scene) -> float:
    """Read a token from `input_file` and check that it is either a literal number or a variable in `scene`.

    Return the number as a ``float``."""
    token = input_file.read_token()
    if isinstance(token, LiteralNumberToken):
        return token.value
    elif isinstance(token, IdentifierToken):
        variable_name = token.identifier
        if variable_name not in scene.float_variables:
            raise GrammarError(token.location, f"unknown variable '{token}'")
        return scene.float_variables[variable_name]

    raise GrammarError(token.location, f"got '{token}' instead of a number")


def expect_string(input_file: InputStream) -> str:
    """Read a token from `input_file` and check that it is a literal string.

    Return the value of the string (a ``str``)."""
    token = input_file.read_token()
    if not isinstance(token, StringToken):
        raise GrammarError(token.location, f"got '{token}' instead of a string")

    return token.string


def expect_identifier(input_file: InputStream) -> str:
    """Read a token from `input_file` and check that it is an identifier.

    Return the name of the identifier."""
    token = input_file.read_token()
    if not isinstance(token, IdentifierToken):
        raise GrammarError(token.location, f"got '{token}' instead of an identifier")

    return token.identifier


def parse_vector(input_file: InputStream, scene: Scene) -> Vec:
    expect_symbol(input_file, "[")
    x = expect_number(input_file, scene)
    expect_symbol(input_file, ",")
    y = expect_number(input_file, scene)
    expect_symbol(input_file, ",")
    z = expect_number(input_file, scene)
    expect_symbol(input_file, "]")

    return Vec(x, y, z)


def parse_color(input_file: InputStream, scene: Scene) -> Color:
    expect_symbol(input_file, "<")
    red = expect_number(input_file, scene)
    expect_symbol(input_file, ",")
    green = expect_number(input_file, scene)
    expect_symbol(input_file, ",")
    blue = expect_number(input_file, scene)
    expect_symbol(input_file, ">")

    return Color(red, green, blue)


def parse_pigment(input_file: InputStream, scene: Scene) -> Pigment:
    keyword = expect_keywords(input_file, [KeywordEnum.UNIFORM, KeywordEnum.CHECKERED, KeywordEnum.IMAGE])

    expect_symbol(input_file, "(")
    if keyword == KeywordEnum.UNIFORM:
        color = parse_color(input_file, scene)
        result = UniformPigment(color=color)
    elif keyword == KeywordEnum.CHECKERED:
        color1 = parse_color(input_file, scene)
        expect_symbol(input_file, ",")
        color2 = parse_color(input_file, scene)
        expect_symbol(input_file, ",")
        num_of_steps = int(expect_number(input_file, scene))
        result = CheckeredPigment(color1=color1, color2=color2, num_of_steps=num_of_steps)
    elif keyword == KeywordEnum.IMAGE:
        file_name = expect_string(input_file)
        with open(file_name, "rb") as image_file:
            image = read_pfm_image(image_file)
        result = ImagePigment(image=image)
    else:
        assert False, "This line should be unreachable"

    expect_symbol(input_file, ")")
    return result


def parse_brdf(input_file: InputStream, scene: Scene) -> BRDF:
    brdf_keyword = expect_keywords(input_file, [KeywordEnum.DIFFUSE, KeywordEnum.SPECULAR])
    expect_symbol(input_file, "(")
    pigment = parse_pigment(input_file, scene)
    expect_symbol(input_file, ")")

    if brdf_keyword == KeywordEnum.DIFFUSE:
        return DiffuseBRDF(pigment=pigment)
    elif brdf_keyword == KeywordEnum.SPECULAR:
        return SpecularBRDF(pigment=pigment)

    assert False, "This line should be unreachable"


def parse_material(input_file: InputStream, scene: Scene) -> Tuple[str, Material]:
    name = expect_identifier(input_file)

    expect_symbol(input_file, "(")
    brdf = parse_brdf(input_file, scene)
    expect_symbol(input_file, ",")
    emitted_radiance = parse_pigment(input_file, scene)
    expect_symbol(input_file, ")")

    return name, Material(brdf=brdf, emitted_radiance=emitted_radiance)


def parse_transformation(input_file, scene: Scene):
    result = Transformation()

    while True:
        transformation_kw = expect_keywords(input_file, [
            KeywordEnum.IDENTITY,
            KeywordEnum.TRANSLATION,
            KeywordEnum.ROTATION_X,
            KeywordEnum.ROTATION_Y,
            KeywordEnum.ROTATION_Z,
            KeywordEnum.SCALING,
        ])

        if transformation_kw == KeywordEnum.IDENTITY:
            pass  # Do nothing (this is a primitive form of optimization!)
        elif transformation_kw == KeywordEnum.TRANSLATION:
            expect_symbol(input_file, "(")
            result *= translation(parse_vector(input_file, scene))
            expect_symbol(input_file, ")")
        elif transformation_kw == KeywordEnum.ROTATION_X:
            expect_symbol(input_file, "(")
            result *= rotation_x(expect_number(input_file, scene))
            expect_symbol(input_file, ")")
        elif transformation_kw == KeywordEnum.ROTATION_Y:
            expect_symbol(input_file, "(")
            result *= rotation_y(expect_number(input_file, scene))
            expect_symbol(input_file, ")")
        elif transformation_kw == KeywordEnum.ROTATION_Z:
            expect_symbol(input_file, "(")
            result *= rotation_z(expect_number(input_file, scene))
            expect_symbol(input_file, ")")
        elif transformation_kw == KeywordEnum.SCALING:
            expect_symbol(input_file, "(")
            result *= scaling(parse_vector(input_file, scene))
            expect_symbol(input_file, ")")

        # We must peek the next token to check if there is another transformation that is being
        # chained or if the sequence ends. Thus, this is a LL(1) parser.
        next_kw = input_file.read_token()
        if (not isinstance(next_kw, SymbolToken)) or (next_kw.symbol != "*"):
            # Pretend you never read this token and put it back!
            input_file.unread_token(next_kw)
            break

    return result


def parse_sphere(input_file: InputStream, scene: Scene) -> Sphere:
    expect_symbol(input_file, "(")

    material_name = expect_identifier(input_file)
    if material_name not in scene.materials.keys():
        # We raise the exception here because input_file is pointing to the end of the wrong identifier
        raise GrammarError(input_file.location, f"unknown material {material_name}")

    expect_symbol(input_file, ",")
    transformation = parse_transformation(input_file, scene)
    expect_symbol(input_file, ")")

    return Sphere(transformation=transformation, material=scene.materials[material_name])


def parse_plane(input_file: InputStream, scene: Scene) -> Plane:
    expect_symbol(input_file, "(")

    material_name = expect_identifier(input_file)
    if material_name not in scene.materials.keys():
        # We raise the exception here because input_file is pointing to the end of the wrong identifier
        raise GrammarError(input_file.location, f"unknown material {material_name}")

    expect_symbol(input_file, ",")
    transformation = parse_transformation(input_file, scene)
    expect_symbol(input_file, ")")

    return Plane(transformation=transformation, material=scene.materials[material_name])


def parse_camera(input_file: InputStream, scene) -> Camera:
    expect_symbol(input_file, "(")
    type_kw = expect_keywords(input_file, [KeywordEnum.PERSPECTIVE, KeywordEnum.ORTHOGONAL])
    expect_symbol(input_file, ",")
    transformation = parse_transformation(input_file, scene)
    expect_symbol(input_file, ",")
    aspect_ratio = expect_number(input_file, scene)
    expect_symbol(input_file, ",")
    distance = expect_number(input_file, scene)
    expect_symbol(input_file, ")")

    if type_kw == KeywordEnum.PERSPECTIVE:
        result = PerspectiveCamera(screen_distance=distance, aspect_ratio=aspect_ratio, transformation=transformation)
    elif type_kw == KeywordEnum.ORTHOGONAL:
        result = OrthogonalCamera(aspect_ratio=aspect_ratio, transformation=transformation)

    return result


def parse_scene(input_file: InputStream):
    scene = Scene()

    while True:
        what = input_file.read_token()
        if isinstance(what, StopToken):
            break

        if not isinstance(what, KeywordToken):
            raise GrammarError(what.location, f"expected a keyword instead of '{what}'")

        if what.keyword == KeywordEnum.FLOAT:
            variable_name = expect_identifier(input_file)
            expect_symbol(input_file, "(")
            variable_value = expect_number(input_file, scene)
            expect_symbol(input_file, ")")

            scene.float_variables[variable_name] = variable_value
        elif what.keyword == KeywordEnum.SPHERE:
            scene.objects.append(parse_sphere(input_file, scene))
        elif what.keyword == KeywordEnum.PLANE:
            scene.objects.append(parse_plane(input_file, scene))
        elif what.keyword == KeywordEnum.CAMERA:
            if scene.camera:
                raise GrammarError(what.location, "You cannot define more than one camera")

            scene.camera = parse_camera(input_file, scene)
        elif what.keyword == KeywordEnum.MATERIAL:
            name, material = parse_material(input_file, scene)
            scene.materials[name] = material

    return scene


def test_input_file():
    stream = InputStream(StringIO("abc   \nd\nef"))

    assert stream.location.line_num == 1
    assert stream.location.col_num == 1

    assert stream.read_char() == "a"
    assert stream.location.line_num == 1
    assert stream.location.col_num == 2

    stream.unread_char("A")
    assert stream.location.line_num == 1
    assert stream.location.col_num == 1

    assert stream.read_char() == "A"
    assert stream.location.line_num == 1
    assert stream.location.col_num == 2

    assert stream.read_char() == "b"
    assert stream.location.line_num == 1
    assert stream.location.col_num == 3

    assert stream.read_char() == "c"
    assert stream.location.line_num == 1
    assert stream.location.col_num == 4

    stream.skip_whitespaces()

    assert stream.read_char() == "d"
    assert stream.location.line_num == 2
    assert stream.location.col_num == 2

    assert stream.read_char() == "\n"
    assert stream.location.line_num == 3
    assert stream.location.col_num == 1

    assert stream.read_char() == "e"
    assert stream.location.line_num == 3
    assert stream.location.col_num == 2

    assert stream.read_char() == "f"
    assert stream.location.line_num == 3
    assert stream.location.col_num == 3

    assert stream.read_char() == ""


def _assert_is_keyword(token: Token, keyword: KeywordEnum):
    assert isinstance(token, KeywordToken)
    assert token.keyword == keyword, f"Token '{token}' is not equal to keyword '{keyword}'"


def _assert_is_identifier(token: Token, identifier: str):
    assert isinstance(token, IdentifierToken)
    assert token.identifier == identifier, f"expecting identifier '{identifier}' instead of '{token}'"


def _assert_is_symbol(token: Token, symbol: str):
    assert isinstance(token, SymbolToken)
    assert token.symbol == symbol, f"expecting symbol '{symbol}' instead of '{token}'"


def _assert_is_number(token: Token, number: float):
    assert isinstance(token, LiteralNumberToken)
    assert token.value == number, f"Token '{token}' is not equal to number '{number}'"


def _assert_is_string(token: Token, s: str):
    assert isinstance(token, StringToken)
    assert token.string == s, f"Token '{token}' is not equal to string '{s}'"


def test_lexer():
    stream = StringIO("""
    # This is a comment
    new material sky_material(
        diffuse(image("my file.pfm")),
        <5.0, 500.0, 300.0>
    )
""")

    input_file = InputStream(stream)

    _assert_is_keyword(input_file.read_token(), KeywordEnum.NEW)
    _assert_is_keyword(input_file.read_token(), KeywordEnum.MATERIAL)
    _assert_is_identifier(input_file.read_token(), "sky_material")
    _assert_is_symbol(input_file.read_token(), "(")
    _assert_is_keyword(input_file.read_token(), KeywordEnum.DIFFUSE)
    _assert_is_symbol(input_file.read_token(), "(")
    _assert_is_keyword(input_file.read_token(), KeywordEnum.IMAGE)
    _assert_is_symbol(input_file.read_token(), "(")
    _assert_is_string(input_file.read_token(), "my file.pfm")
    _assert_is_symbol(input_file.read_token(), ")")


def test_parser():
    stream = StringIO("""
    float clock(150)

    material sky_material(
        diffuse(uniform(<0, 0, 0>)),
        uniform(<0.7, 0.5, 1>)
    )

    # Here is a comment

    material ground_material(
        diffuse(checkered(<0.3, 0.5, 0.1>,
                          <0.1, 0.2, 0.5>, 4)),
        uniform(<0, 0, 0>)
    )

    material sphere_material(
        specular(uniform(<0.5, 0.5, 0.5>)),
        uniform(<0, 0, 0>)
    )

    plane (sky_material, translation([0, 0, 100]) * rotation_y(clock))
    plane (ground_material, identity)

    sphere(sphere_material, translation([0, 0, 1]))

    camera(perspective, rotation_z(30) * translation([-4, 0, 1]), 1.0, 1.0)
    """)

    scene = parse_scene(input_file=InputStream(stream))

    # Check that the float variables are ok
    assert len(scene.float_variables) == 1
    assert "clock" in scene.float_variables.keys()
    assert scene.float_variables["clock"] == 150.0

    # Check that the materials are ok

    assert len(scene.materials) == 3
    assert "sphere_material" in scene.materials
    assert "sky_material" in scene.materials
    assert "ground_material" in scene.materials

    sphere_material = scene.materials["sphere_material"]
    sky_material = scene.materials["sky_material"]
    ground_material = scene.materials["ground_material"]

    assert isinstance(sky_material.brdf, DiffuseBRDF)
    assert isinstance(sky_material.brdf.pigment, UniformPigment)
    assert sky_material.brdf.pigment.color.is_close(Color(0, 0, 0))

    assert isinstance(ground_material.brdf, DiffuseBRDF)
    assert isinstance(ground_material.brdf.pigment, CheckeredPigment)
    assert ground_material.brdf.pigment.color1.is_close(Color(0.3, 0.5, 0.1))
    assert ground_material.brdf.pigment.color2.is_close(Color(0.1, 0.2, 0.5))
    assert ground_material.brdf.pigment.num_of_steps == 4

    assert isinstance(sphere_material.brdf, SpecularBRDF)
    assert isinstance(sphere_material.brdf.pigment, UniformPigment)
    assert sphere_material.brdf.pigment.color.is_close(Color(0.5, 0.5, 0.5))

    assert isinstance(sky_material.emitted_radiance, UniformPigment)
    assert sky_material.emitted_radiance.color.is_close(Color(0.7, 0.5, 1.0))
    assert isinstance(ground_material.emitted_radiance, UniformPigment)
    assert ground_material.emitted_radiance.color.is_close(Color(0, 0, 0))
    assert isinstance(sphere_material.emitted_radiance, UniformPigment)
    assert sphere_material.emitted_radiance.color.is_close(Color(0, 0, 0))

    # Check that the shapes are ok

    assert len(scene.objects) == 3
    assert isinstance(scene.objects[0], Plane)
    assert isinstance(scene.objects[1], Plane)
    assert isinstance(scene.objects[2], Sphere)

    # Check that the camera is ok

    assert isinstance(scene.camera, PerspectiveCamera)


def test_parser_undefined_material():
    stream = StringIO("""
    plane(this_material_does_not_exist, identity)
    """)

    try:
        _ = parse_scene(input_file=InputStream(stream))
        assert False, "the code did not throw an exception"
    except GrammarError:
        pass


def test_parser_double_camera():
    stream = StringIO("""
    camera(perspective, rotation_z(30) * translation([-4, 0, 1]), 1.0, 1.0)
    camera(orthogonal, identity, 1.0, 1.0)
    """)

    try:
        _ = parse_scene(input_file=InputStream(stream))
        assert False, "the code did not throw an exception"
    except GrammarError:
        pass


def main():
    test_input_file()
    test_lexer()
    try:
        test_parser()
    except GrammarError as e:
        print(f"{e.file_name}:{e.line_num}:{e.col_num}: {e.message}")

    test_parser_double_camera()
    test_parser_undefined_material()


if __name__ == "__main__":
    main()
