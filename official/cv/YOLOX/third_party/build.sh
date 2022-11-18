#!/bin/bash
# Build dynamic library
python setup.py build_ext --inplace && echo "Build fast_coco_eval successfully."
