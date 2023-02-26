from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    name='LAF',
    version='1.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            "LAF",
            sources=["laf.pyx"],
            language="c++",
            include_dirs=[numpy.get_include()], 
            build_dir="build"
        ), 
    ],
    author='Yifan Wang',
    author_email='wangyifan@ufl.edu'
)
