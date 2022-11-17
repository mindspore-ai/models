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

#ifndef OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXBASEINFER_H_
#define OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXBASEINFER_H_

#include <memory>
#include <string>
#include <vector>

#include "FunctionTimer.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxImage.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "acl/acl.h"

extern FunctionStats g_infer_stats;

static inline const char *aclFormatToStr(int fmt) {
    switch (fmt) {
        case ACL_FORMAT_NCHW:
            return "NCHW";
        case ACL_FORMAT_NHWC:
            return "NHWC";
        case ACL_FORMAT_ND:
            return "ND";
        case ACL_FORMAT_NC1HWC0:
            return "NC1HWC0";
        case ACL_FORMAT_FRACTAL_Z:
            return "FRACTAL_Z";
        case ACL_FORMAT_FRACTAL_NZ:
            return "FRACTAL_NZ";
        default:
            return "UNDEFINED";
    }
}
static inline const char *aclDatatypeToStr(MxBase::TensorDataType ty) {
    switch (ty) {
        case ACL_FLOAT:
            return "float";
        case ACL_FLOAT16:
            return "float16";
        case ACL_INT8:
            return "int8";
        case ACL_INT32:
            return "int32";
        case ACL_UINT8:
            return "uint8";
        case ACL_INT16:
            return "int16";
        case ACL_UINT16:
            return "uint16";
        case ACL_UINT32:
            return "uint32";
        case ACL_INT64:
            return "int64";
        case ACL_UINT64:
            return "uint64";
        case ACL_DOUBLE:
            return "double";
        case ACL_BOOL:
            return "bool";
        default:
            return "undefined";
    }
}

class CBaseInfer {
 public:
    explicit CBaseInfer(uint32_t deviceId = 0) : DeviceId(deviceId) {}
    virtual ~CBaseInfer() {
        DisposeDvpp();
    }

    bool IsLoad() const { return IsLoaded; }

    virtual bool Init(const std::string &omPath);

    virtual void UnInit();

    void Dump();

    MxBase::DvppWrapper *GetDvpp();

    virtual void DoPostProcess(
        std::vector<std::shared_ptr<void>> &featLayerData) {}

    virtual void DoPostProcess(
        std::vector<MxBase::TensorBase> &outputTensors) = 0;

    virtual APP_ERROR DoInference();

 public:
    // helper function to load data into model input memory
    // should be override by specific model subclass
    virtual void GetImagePreprocessConfig(std::string &color,
                                          cv::Scalar *&means,
                                          cv::Scalar *&stds) {}

    // default preprocess : resize to dest size & padding width or height with
    // black
    virtual bool PreprocessImage(CVImage &input, uint32_t w, uint32_t h,
                                 const std::string &color, bool isCenter,
                                 CVImage &output, MxBase::PostImageInfo &info);

    virtual bool LoadImageAsInput(const std::string &file, size_t index = 0);

 protected:
    void GetImageInputHW(uint32_t &w, uint32_t &h, size_t index = 0);

    // dispose dvpp processor
    void DisposeDvpp();
    void DisposeIOBuffers();

    // prepare batch memory for inputs
    void AllocateInputs(std::vector<MxBase::MemoryData> &data);

    // prepare batch memory for outputs
    void AllocateOutputs(std::vector<MxBase::MemoryData> &data);
    void PrepareTensors(
        std::vector<MxBase::MemoryData> &data,
        const std::vector<std::vector<int64_t>> &tensorShapes,
        const std::vector<MxBase::TensorDataType> &tensorDataTypes,
        std::vector<MxBase::TensorBase> &tensors);

    void PrepareTensors(std::vector<MxBase::MemoryData> &data,
                        std::vector<MxBase::TensorDesc> &tensorDesc,
                        std::vector<MxBase::BaseTensor> &tensor);

    APP_ERROR FetchDeviceBuffers(std::vector<MxBase::MemoryData> &data,
                                 std::vector<std::shared_ptr<void>> &out);

    // helper function to join array data into string
    template <class T>
    static std::string Array2Str(const std::vector<T> &vec) {
        std::ostringstream ostr;
        for (auto &pos : vec) ostr << pos << " ";
        return ostr.str();
    }

 protected:
    // om file path
    std::string strModelPath;
    // model description
    MxBase::ModelDesc ModelDesc;
    // MxBase inference module
    MxBase::ModelInferenceProcessor m_oInference;
    // MxBase singleton dvpp processor
    MxBase::DvppWrapper *DvppProcessor = nullptr;
    // @todo:  private use of acl API
    aclmdlDesc *m_pMdlDesc = nullptr;

    bool IsLoaded = false;

    // device id used
    uint32_t DeviceId = 0;
    // statically input & output buffers
    std::vector<MxBase::MemoryData> MemoryInputs;
    std::vector<MxBase::MemoryData> MemoryOutputs;
};

#endif  // OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXBASEINFER_H_
