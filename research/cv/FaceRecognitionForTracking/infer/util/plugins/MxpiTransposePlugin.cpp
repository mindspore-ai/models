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
#include "MxpiTransposePlugin.h"
#include <math.h>
#include <iostream>
#include "MxBase/Log/Log.h"

namespace {
    const std::string SAMPLE_KEY = "MxpiVisionList";
}

APP_ERROR MxpiTransposePlugin::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) {
    LogInfo << "MxpiTransposePlugin::Init start.";
    // Get the property values by key
    std::shared_ptr<std::string> parentNamePropSptr = std::static_pointer_cast<std::string>(
        configParamMap["dataSource"]);
    // parentName_: mxpi_imageresize0
    parentName_ = *parentNamePropSptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiTransposePlugin::DeInit() {
    LogInfo << "MxpiTransposePlugin::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiTransposePlugin::SetMxpiErrorInfo(
    MxTools::MxpiBuffer& buffer, const std::string pluginName, const MxTools::MxpiErrorInfo mxpiErrorInfo) {
    APP_ERROR ret = APP_ERR_OK;
    // Define an object of MxTools::MxpiMetadataManager
    MxTools::MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(pluginName, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

APP_ERROR MxpiTransposePlugin::Transpose(
    MxTools::MxpiVisionList srcMxpiVisionList, MxTools::MxpiVisionList& dstMxpiVisionList) {
    dstMxpiVisionList = srcMxpiVisionList;
    for (int i = 0; i < dstMxpiVisionList.visionvec_size(); i++) {
        MxTools::MxpiVision* dstMxpiVision = dstMxpiVisionList.mutable_visionvec(i);
        // set MxTools::MxpiVisionData
        MxTools::MxpiVisionData* dstMxpiVisionData = dstMxpiVision->mutable_visiondata();
        // height：96，width：64，channel：3
        float HWCData[96*64*3], CHWData[3*96*64];
        std::memcpy(HWCData, reinterpret_cast<void*>(dstMxpiVisionData->dataptr()), sizeof(HWCData));
        for (int c = 0; c < 3; c++) {
            for (int k = 0; k < 96*64; k++) {
                CHWData[k+c*96*64] = HWCData[3*k+c];
            }
        }
        std::memcpy(reinterpret_cast<void*>(dstMxpiVisionData->dataptr()), CHWData, sizeof(CHWData));
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiTransposePlugin::Process(std::vector<MxTools::MxpiBuffer*>& mxpiBuffer) {
    LogInfo << "MxpiTransposePlugin::Process start";
    // Get the data from buffer
    MxTools::MxpiBuffer* buffer = mxpiBuffer[0];
    // Get metadata by key.
    MxTools::MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxTools::MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiTransposePlugin process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiTransposePlugin process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the data from buffer(mxpi_imageresize0)
    std::shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL;
    }
    // check whether the proto struct name is MxTools::MxpiVisionList(plugin mxpi_imageresize's output format)
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != SAMPLE_KEY) {
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_);
        ErrorInfo_ << "Proto struct name is not MxTools::MxpiVisionList, failed";
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_PROTOBUF_NAME_MISMATCH;
    }
    // Generate sample output
    std::shared_ptr<MxTools::MxpiVisionList> src = std::static_pointer_cast<MxTools::MxpiVisionList>(metadata);
    std::shared_ptr<MxTools::MxpiVisionList> dst = std::make_shared<MxTools::MxpiVisionList>();
    APP_ERROR ret = Transpose(*src, *dst);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiTransposePlugin gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Add Generated data to metedata
    // call function AddProtoMetadata(). Mount the result to the corresponding Buffer when getting the input
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, std::static_pointer_cast<void>(dst));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiTransposePlugin add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiTransposePlugin::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiTransposePlugin::DefineProperties() {
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<MxTools::ElementProperty<std::string>>(
        MxTools::ElementProperty<std::string>{
        MxTools::STRING, "dataSource", "name", "the name of previous plugin", "mxpi_imageresize0", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiTransposePlugin)
