from skbuild import setup
from setuptools import find_packages

setup(
    name="streamtomocupy",
    version=open("VERSION").read().strip(),
    author="Viktor Nikitin",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    extras_require={
        "dev": [
            "ruff",
        ],
    },
)
