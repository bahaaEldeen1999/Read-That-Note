from setuptools import setup
from Cython.Build import cythonize

setup(
    name='staff line detection',
    ext_modules=cythonize("staff_detection.pyx"),
    zip_safe=False,
)
