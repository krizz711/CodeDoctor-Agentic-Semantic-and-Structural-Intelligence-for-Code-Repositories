from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'codedoctor_cpp',
        ['src/hybrid.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='codedoctor_cpp',
    version='1.0',
    ext_modules=ext_modules,
)
