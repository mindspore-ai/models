/*
 * Copyright(C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>

#include "CommandFlagParser.h"


DEFINE_string(feat_ids, "../../data/feat_ids.bin", "feat_ids.");
DEFINE_string(feat_vals, "../../data/feat_vals.bin", "feat_vals.");
DEFINE_int32(sample_num, 2, "num of samples");
DEFINE_string(pipeline, "../../data/config/wide_and_deep_ms.pipeline",
              "config file for this model.");

int main(int argc, char *argv[]) {
    OptionManager::getInstance()->parseCommandLineFlags(argc, argv);

    std::cout << "FEAT ID: " << FLAGS_feat_ids << std::endl;
    std::cout << "sample number: " << FLAGS_sample_num << std::endl;
    return 0;
}
