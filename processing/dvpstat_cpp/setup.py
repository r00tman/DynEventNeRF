#!/usr/bin/env python3
from setuptools import setup, Extension
# from torch.utils import cpp_extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess

def pkgconfig(package, kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    output = subprocess.getoutput(
        'pkg-config --cflags --libs {}'.format(package))
    for token in output.strip().split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw

kw = {
   'extra_compile_args': ['-std=c++23']
}

kw = pkgconfig('dv-processing', kw)
kw = pkgconfig('eigen3', kw)

ext = Pybind11Extension(
   name='dvpstat_cpp',
   sources=['dvpstat.cpp'],
   **kw)

setup(name='dvpstat_cpp',
      ext_modules=[ext])
      # cmdclass={"build_ext": build_ext})
