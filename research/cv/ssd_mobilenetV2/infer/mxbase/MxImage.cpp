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

#include "infer/mxbase/MxImage.h"

#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>

namespace sdk_infer {
namespace mxbase_infer {

void MxImage::ConvertToDeviceFormat(const cv::Mat &input_mat,
                                    aclDataType acl_data_type,
                                    aclFormat acl_format, cv::Mat *output_mat) {
    if (acl_data_type == ACL_UINT8) {
        if (acl_format == ACL_FORMAT_NCHW) {
            *output_mat = cv::dnn::blobFromImage(
                input_mat, 1.0, cv::Size(), cv::Scalar(), false, false, CV_8U);
        }
    } else if (acl_data_type == ACL_FLOAT16) {
        if (acl_format == ACL_FORMAT_NCHW) {
            *output_mat = cv::dnn::blobFromImage(
                input_mat, 1.0, cv::Size(), cv::Scalar(), false, false, CV_8U);
            output_mat->convertTo(*output_mat, CV_16U);
        } else if (acl_format == ACL_FORMAT_NHWC) {
            output_mat->convertTo(*output_mat, CV_16U);
        } else {
            assert(0);
        }
    } else if (acl_data_type == ACL_FLOAT) {
        if (acl_format == ACL_FORMAT_NCHW) {
            *output_mat = cv::dnn::blobFromImage(input_mat, 1.0 / 255.0);
        } else if (acl_format == ACL_FORMAT_NHWC) {
            output_mat->convertTo(*output_mat, CV_32F, 1.0 / 255.0);
        } else {
            assert(0);
        }
    } else {
        assert(0);
    }

    if (!output_mat->isContinuous()) {
        *output_mat = output_mat->clone();
    }
}

}  // namespace mxbase_infer
}  // namespace sdk_infer
