# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from data_provider.mnist_to_mindrecord import generate_mindrecord

file_path_train = "/moving-mnist-train.npz"
mindrecord_name_train = "mnist_train.mindrecord"
generate_mindrecord(file_path_train, mindrecord_name_train)

file_path_valid = "/moving-mnist-valid.npz"
mindrecord_name_valid = "mnist_valid.mindrecord"
generate_mindrecord(file_path_valid, mindrecord_name_valid)

file_path_test = "/moving-mnist-test.npz"
mindrecord_name_test = "mnist_test.mindrecord"
generate_mindrecord(file_path_test, mindrecord_name_test)
