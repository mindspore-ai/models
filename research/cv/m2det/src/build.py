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

from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext

_EXTENSION_NAME = 'nms.cpu_nms'
_SOURCES = ["nms/cpu_nms.pyx"]
_EXTA_ARGS = ["-Wno-cpp", "-Wno-unused-function"]

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

modules = [Extension(_EXTENSION_NAME, _SOURCES, extra_compile_args=_EXTA_ARGS, include_dirs=[numpy_include])]

setup(name='mot_utils', ext_modules=modules, cmdclass={'build_ext': build_ext})
