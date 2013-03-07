from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = "seeded_region_growing",
    include_dirs=[np.get_include(), '../../include'],
    ext_modules = cythonize('seeded_region_growing.pyx')
)
