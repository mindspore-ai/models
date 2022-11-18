"""Build fast_coco_eval library"""
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

args = ["-O2"] if sys.platform == "win32" else ["-O3", "-std=c++14", "-g", "-Wno-reorder"]

ext_modules = [
    Pybind11Extension(
        name="fast_coco_eval",
        sources=['cocoeval/cocoeval.cpp'],
        extra_compile_args=args
    ),
]

setup(ext_modules=ext_modules)
