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
#ifndef SDKMEMORY_MxpiTransposePlugin_H
#define SDKMEMORY_MxpiTransposePlugin_H
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"

using namespace MxTools;

/**
* @api
* @brief Definition of MxpiTransposePlugin class.
*/
class MxpiTransposePlugin : public MxTools::MxPluginBase {
 public:
    /**
     * @api
     * @brief Initialize configure parameter.
     * @param configParamMap
     * @return APP_ERROR
     */
    APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) override;
    /**
     * @api
     * @brief DeInitialize configure parameter.
     * @return APP_ERROR
     */
    APP_ERROR DeInit() override;
    /**
     * @api
     * @brief Process the data of MxpiBuffer.
     * @param mxpiBuffer
     * @return APP_ERROR
     */
    APP_ERROR Process(std::vector<MxTools::MxpiBuffer *> &mxpiBuffer) override;
    /**
     * @api
     * @brief Definition the parameter of configure properties.
     * @return std::vector<std::shared_ptr<void>>
     */
    static std::vector<std::shared_ptr<void>> DefineProperties();
    /**
     * @api
     * @brief convert from HWC to CHW.
     * @param key
     * @param buffer
     * @return APP_ERROR
     */
    APP_ERROR Transpose(MxTools::MxpiVisionList srcMxpiVisionList, MxTools::MxpiVisionList &dstMxpiVisionList);

 private:
    APP_ERROR SetMxpiErrorInfo(
        MxTools::MxpiBuffer &buffer, std::string pluginName, const MxTools::MxpiErrorInfo mxpiErrorInfo);
    std::string parentName_;
    std::ostringstream ErrorInfo_;
};
#endif  // SDKMEMORY_MxpiTransposePlugin_H
