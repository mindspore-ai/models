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

"""split_by_user"""

import random

with open("local_test", "r") as fi:
    with open("local_train_splitByUser", "w") as ftrain:
        with open("local_test_splitByUser", "w") as ftest:
            while True:
                rand_int = random.randint(1, 10)
                noclk_line = fi.readline().strip()
                clk_line = fi.readline().strip()
                if noclk_line == "" or clk_line == "":
                    break
                if rand_int == 2:
                    print(noclk_line, file=ftest)
                    print(clk_line, file=ftest)
                else:
                    print(noclk_line, file=ftrain)
                    print(clk_line, file=ftrain)
