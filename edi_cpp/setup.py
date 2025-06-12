#!/usr/bin/env python3
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='edi_cpp',
      ext_modules=[cpp_extension.CppExtension('edi_cpp', ['edi.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
