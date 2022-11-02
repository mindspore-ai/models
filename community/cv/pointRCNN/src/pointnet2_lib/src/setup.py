# Copyright 2021 Huawei Technologies Co., Ltd
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
"""setup fot python"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='pointnet2',
      ext_modules=[
          CUDAExtension('pointnet2_cuda', [
              'pointnet2_cuda/pointnet2_api.cpp',
              'pointnet2_cuda/ms_ext.cpp',
              'pointnet2_cuda/ball_query.cpp',
              'pointnet2_cuda/ball_query_gpu.cu',
              'pointnet2_cuda/group_points.cpp',
              'pointnet2_cuda/group_points_gpu.cu',
              'pointnet2_cuda/interpolate.cpp',
              'pointnet2_cuda/interpolate_gpu.cu',
              'pointnet2_cuda/sampling.cpp',
              'pointnet2_cuda/sampling_gpu.cu',
          ],
                        extra_compile_args={
                            'cxx': ['-g'],
                            'nvcc': ['-O2']
                        })
      ],
      cmdclass={'build_ext': BuildExtension})
