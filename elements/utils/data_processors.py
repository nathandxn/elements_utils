"""Helper functions for data manipulation and processing"""

__version__ = "0.0.1"
__author__ = "Nathan Dixon"

from re import sub
from typing import Union


# some formatting helper functions
def snake_case(text: str) -> Union[str, None]:
    """This function is used for converting strings to snake_case."""

    if text is None:
        return None

    else:
        return "_".join(
            sub(
                "([A-Z][a=z]+)",
                r" \1",
                sub(
                    "([A-Z]+)",
                    r" \1",
                    text.replace("-", " ")
                    .replace("’", "")
                    .replace("–", "")
                    .replace("&", "and")
                    .replace(",", "")
                    .replace("/", " ")
                    .replace(".", "")
                    .replace("'", "")
                    .replace(")", "")
                    .replace("(", "")
                    .replace(":", ""),
                ),
            ).split()
        ).lower()
