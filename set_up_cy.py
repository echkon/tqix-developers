from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("./tqix/pis/accumulated_state.pyx"),include_dirs=[numpy.get_include()]
)