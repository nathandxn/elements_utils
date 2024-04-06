from re import sub


# some formatting helper functions
def snake_case(text):
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
