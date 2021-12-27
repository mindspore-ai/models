/*
 * Copyright (c) 2021.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OFFICIAL_CV_CENTERFACE_INFER_SDK_MXPI_MXPIMSCENTERFACEPOSTPROCESSOR_H_
#define OFFICIAL_CV_CENTERFACE_INFER_SDK_MXPI_MXPIMSCENTERFACEPOSTPROCESSOR_H_
#include <memory>

#include "../../mxbase/MxCenterfacePostProcessor.h"


extern "C" {
std::shared_ptr<MxCenterfacePostProcessor> GetObjectInstance();
}

#endif  // OFFICIAL_CV_CENTERFACE_INFER_SDK_MXPI_MXPIMSCENTERFACEPOSTPROCESSOR_H_
