from setuptools import find_packages, setup

with open("elements/README.md", "r") as f:
    long_description = f.read()

setup(
    name="elements_utils",
    version="0.1.0",
    description="Utility functions and classes for data applications",
    package_dir={"": "elements"},
    packages=find_packages(where="elements"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathandxn/elements_utils",
    author="Nathan Dixon",
    author_email="nathandxn@gmail.com",
    license="GNU v3.0",
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)