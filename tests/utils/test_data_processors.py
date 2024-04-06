import sys
from pyprojroot import here

# set to top level directory of project
sys.path.insert(0, str(here()))

from elements.utils.data_processors import (
    snake_case,
)


def test_snake_case_space_and_capitals():
    assert snake_case("Hello World") == "hello_world"


def test_snake_case_from_camel_case():
    assert snake_case("HelloWorld") == "hello_world"


def test_snake_case_special_characters():
    assert snake_case("(Hello/World')") == "hello_world"
