# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# This file was copied from project [sshaoshuai][https://github.com/sshaoshuai/PointRCNN]
"""setup for python"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='iou3d',
      ext_modules=[
          CUDAExtension('iou3d_cuda', [
              'src/iou3d.cpp',
              'src/ms_ext.cpp',
              'src/iou3d_kernel.cu',
          ],
                        extra_compile_args={
                            'cxx': ['-g'],
                            'nvcc': ['-O2']
                        })
      ],
      cmdclass={'build_ext': BuildExtension})
