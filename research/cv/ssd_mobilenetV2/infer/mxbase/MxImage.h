/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace sdk_infer {
namespace mxbase_infer {

class MxImage {
 public:
    void ConvertToDeviceFormat(const cv::Mat &input_mat,
                               aclDataType acl_data_type, aclFormat acl_format,
                               cv::Mat *output_mat);
};

}  // namespace mxbase_infer
}  // namespace sdk_infer
