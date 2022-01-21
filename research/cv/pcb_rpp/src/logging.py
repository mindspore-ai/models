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
""" Logging helper """

import os
import sys


class Logger:
    """ Logger to file """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            if os.path.exists(fpath):
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def __del__(self):
        """ Close file in the end """
        self.close()

    def __enter__(self):
        """ On context init """

    def __exit__(self, *args):
        """ Close file in the end """
        self.close()

    def write(self, msg):
        """ Write msg to file """
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        """ Flush msg """
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        """ Close file """
        self.console.close()
        if self.file is not None:
            self.file.close()
