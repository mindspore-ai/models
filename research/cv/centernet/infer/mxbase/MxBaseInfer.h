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

#pragma once
#include <algorithm>
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
    explicit CBaseInfer(uint32_t deviceId = 0) : m_nDeviceId(deviceId) {}
    virtual ~CBaseInfer() {
        DisposeDvpp();
        // UnInit();
    }

    bool IsLoad() const { return m_bIsLoaded; }

    virtual bool Init(const std::string &omPath) {
        APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
        if (ret != APP_ERR_OK) {
            LogError << "Init devices failed, ret=" << ret << ".";
            return ret;
        }
        ret = MxBase::TensorContext::GetInstance()->SetContext(m_nDeviceId);
        if (ret != APP_ERR_OK) {
            LogError << "Set context failed, ret=" << ret << ".";
            return ret;
        }
        assert(!IsLoad());
        if (m_oInference.Init(omPath, m_oModelDesc) == APP_ERR_OK) {
            m_strModelPath = omPath;
            AllocateInputs(m_oMemoryInputs);
            AllocateOutputs(m_oMemoryOutpus);
            m_bIsLoaded = true;
            return true;
        }
        return false;
    }

    virtual void UnInit() {
        if (IsLoad()) {
            DisposeIOBuffers();
            m_oInference.DeInit();
            m_strModelPath.clear();
            m_bIsLoaded = false;
            MxBase::DeviceManager::GetInstance()->DestroyDevices();
        }
    }

    void Dump() {
        LogInfo << "om file path:" << m_strModelPath;
        LogInfo << "model has " << m_oModelDesc.inputTensors.size()
                << " inputs, " << m_oModelDesc.outputTensors.size()
                << " outputs ...";

        size_t idx = 0;
        auto formats = m_oInference.GetInputFormat();
        auto dataTypes = m_oInference.GetInputDataType();
        for (auto &tensor : m_oModelDesc.inputTensors) {
            LogInfo << "input[" << idx
                    << "] dims: " << Array2Str(tensor.tensorDims)
                    << " size:" << tensor.tensorSize;

            LogInfo << "name: " << tensor.tensorName
                    << " format:" << aclFormatToStr(formats[idx])
                    << " dataType:" << aclDatatypeToStr(dataTypes[idx]);

            idx++;
        }

        idx = 0;
        formats = m_oInference.GetOutputFormat();
        dataTypes = m_oInference.GetOutputDataType();
        for (auto &tensor : m_oModelDesc.outputTensors) {
            LogInfo << "output[" << idx
                    << "] dims: " << Array2Str(tensor.tensorDims)
                    << " size:" << tensor.tensorSize;
            LogInfo << " name:" << tensor.tensorName
                    << " format:" << aclFormatToStr(formats[idx])
                    << " dataType:" << aclDatatypeToStr(dataTypes[idx]);
            idx++;
        }

        if (m_oModelDesc.dynamicBatch) {
            LogInfo << "dynamic batchSize: "
                    << Array2Str(m_oModelDesc.batchSizes);
        }
    }

    MxBase::DvppWrapper *GetDvpp() {
        if (!m_pDvppProcessor) {
            m_pDvppProcessor = new MxBase::DvppWrapper;
            m_pDvppProcessor->deviceId_ = m_nDeviceId;
            if (m_pDvppProcessor->Init() != APP_ERR_OK) {
                delete m_pDvppProcessor;
                m_pDvppProcessor = nullptr;
            }
        }
        return m_pDvppProcessor;
    }

    virtual void DoPostProcess(
        std::vector<std::shared_ptr<void>> &featLayerData) {}

    virtual void DoPostProcess(
        std::vector<MxBase::TensorBase> &outputTensors) = 0;

    virtual APP_ERROR DoInference() {
        FunctionTimer timer;
        assert(IsLoad());

        std::vector<MxBase::TensorBase> inputTensors;
        std::vector<MxBase::TensorBase> outputTensors;
        PrepareTensors(m_oMemoryInputs, m_oInference.GetInputShape(),
                       m_oInference.GetInputDataType(), inputTensors);
        PrepareTensors(m_oMemoryOutpus, m_oInference.GetOutputShape(),
                       m_oInference.GetOutputDataType(), outputTensors);

        timer.start_timer();
        MxBase::DynamicInfo dynamicInfo = {};
        dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
        APP_ERROR ret = m_oInference.ModelInference(inputTensors, outputTensors,
                                                    dynamicInfo);
        timer.calculate_time();

        if (ret == APP_ERR_OK) {
            DoPostProcess(outputTensors);
        }

        g_infer_stats.update_time(timer.get_elapsed_time_in_milliseconds());
        return ret;
    }

 public:  // help function to load data into model input memory
    // should be override by specific model subclass
    virtual void GetImagePreprocessConfig(std::string &color,
                                          cv::Scalar **means,
                                          cv::Scalar **stds) {}

    // default preprocess : resize to dest size & padding width or height with
    // black
    virtual bool PreprocessImage(CVImage &input, uint32_t w, uint32_t h,
                                 const std::string &color, bool isCenter,
                                 CVImage &output, MxBase::PostImageInfo &info) {
        double scale;

        output = input.Preprocess(w, h, color, scale, isCenter);

        info.widthOriginal = input.Width();
        info.heightOriginal = input.Height();
        info.widthResize = w;
        info.heightResize = h;

        if (isCenter) {
            info.x0 = (w - info.widthOriginal * scale) / 2.0;
            info.y0 = (h - info.heightOriginal * scale) / 2.0;
        } else {
            info.x0 = 0;
            info.y0 = 0;
        }
        info.x1 = info.x0 + info.widthOriginal * scale;
        info.y1 = info.y0 + info.heightOriginal * scale;

        return !!output;
    }

    virtual bool LoadImageAsInput(const std::string &file, size_t index = 0) {
        CVImage img;
        if (img.Load(file) != APP_ERR_OK) {
            LogError << "load image :" << file << " as CVImage error !!!";
            return false;
        }
        uint32_t w, h;
        GetImageInputHW(w, h, index);
        CVImage outImg;
        MxBase::PostImageInfo postImageInfo;
        std::string color;
        cv::Scalar *means = nullptr;
        cv::Scalar *stds = nullptr;
        // fetch image config from subclass
        GetImagePreprocessConfig(color, &means, &stds);
        if (!PreprocessImage(img, w, h, color, true, outImg, postImageInfo)) {
            LogError << "Preprocess image error !!!";
            return false;
        }

        return outImg.FetchToDevice(
            m_oMemoryInputs[index],
            (aclDataType)m_oInference.GetInputDataType()[index],
            (aclFormat)m_oInference.GetInputFormat()[index], means, stds);
    }

 protected:
    void GetImageInputHW(uint32_t &w, uint32_t &h, size_t index = 0) {
        assert(m_oModelDesc.inputTensors[index].tensorDims.size() == 4);

        // int input_format = aclmdlGetInputFormat(m_pMdlDesc, index);
        aclFormat input_format =
            (aclFormat)m_oInference.GetInputFormat()[index];

        if (input_format == ACL_FORMAT_NCHW) {
            w = m_oModelDesc.inputTensors[index].tensorDims[3];
            h = m_oModelDesc.inputTensors[index].tensorDims[2];
        } else if (input_format == ACL_FORMAT_NHWC) {
            w = m_oModelDesc.inputTensors[index].tensorDims[2];
            h = m_oModelDesc.inputTensors[index].tensorDims[1];
        } else {
            assert(0);
        }
    }

    // dispose dvpp processor
    void DisposeDvpp() {
        if (m_pDvppProcessor) {
            m_pDvppProcessor->DeInit();
            delete m_pDvppProcessor;
            m_pDvppProcessor = nullptr;
        }
    }
    void DisposeIOBuffers() {
        for (auto &pos : m_oMemoryInputs) {
            MxBase::MemoryHelper::MxbsFree(pos);
        }
        m_oMemoryInputs.clear();
        for (auto &pos : m_oMemoryOutpus) {
            MxBase::MemoryHelper::MxbsFree(pos);
        }
        m_oMemoryOutpus.clear();
    }
    // prepare batch memory for inputs
    void AllocateInputs(std::vector<MxBase::MemoryData> &data) {
        data.resize(m_oModelDesc.inputTensors.size());
        size_t idx = 0;
        for (auto &mem : data) {
            mem.deviceId = m_nDeviceId;
            mem.type = MxBase::MemoryData::MEMORY_DEVICE;
            mem.size = m_oModelDesc.inputTensors[idx].tensorSize;
            MxBase::MemoryHelper::MxbsMalloc(mem);
            idx++;
        }
    }
    // prepare batch memory for outputs
    void AllocateOutputs(std::vector<MxBase::MemoryData> &data) {
        data.resize(m_oModelDesc.outputTensors.size());
        size_t idx = 0;
        for (auto &mem : data) {
            mem.deviceId = m_nDeviceId;
            mem.type = MxBase::MemoryData::MEMORY_DEVICE;
            mem.size = m_oModelDesc.outputTensors[idx].tensorSize;
            MxBase::MemoryHelper::MxbsMalloc(mem);
            idx++;
        }
    }
    void PrepareTensors(
        std::vector<MxBase::MemoryData> &data,
        const std::vector<std::vector<int64_t>> &tensorShapes,
        const std::vector<MxBase::TensorDataType> &tensorDataTypes,
        std::vector<MxBase::TensorBase> &tensors) {
        size_t idx = 0;

        for (auto &pos : data) {
            std::vector<uint32_t> shapes(tensorShapes[idx].begin(),
                                         tensorShapes[idx].end());
            tensors.emplace_back(
                MxBase::TensorBase(pos, true, shapes, tensorDataTypes[idx]));
            idx++;
        }
    }

    static MxBase::BaseTensor tensor_transform(const MxBase::MemoryData &pos,
                                               const MxBase::TensorDesc &desc) {
        return MxBase::BaseTensor{
            pos.ptrData,
            std::vector<int>(desc.tensorDims.begin(), desc.tensorDims.end()),
            pos.size};
    }

    void PrepareTensors(std::vector<MxBase::MemoryData> &data,
                        std::vector<MxBase::TensorDesc> &tensorDesc,
                        std::vector<MxBase::BaseTensor> &tensor) {
        tensor.reserve(data.size());
        std::transform(data.begin(), data.end(), tensorDesc.begin(),
                       tensor.begin(), tensor_transform);
    }

    APP_ERROR FetchDeviceBuffers(std::vector<MxBase::MemoryData> &data,
                                 std::vector<std::shared_ptr<void>> &out) {
        for (auto &mem : data) {
            MxBase::MemoryData memoryDst(mem.size);

            APP_ERROR ret =
                MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, mem);
            if (ret != APP_ERR_OK) {
                LogError << GetError(ret)
                         << " Fail to copy device memory to host for "
                            "ModelPostProcessor.";
                return ret;
            }
            std::shared_ptr<void> buffer = nullptr;
            buffer.reset(memoryDst.ptrData, memoryDst.free);
            out.emplace_back(buffer);
        }
        return APP_ERR_OK;
    }
    // helper function to join array data into string
    template <class T>
    static std::string Array2Str(const std::vector<T> &vec) {
        std::ostringstream ostr;
        for (auto &pos : vec) ostr << pos << " ";
        return ostr.str();
    }

 protected:
    // om file path
    std::string m_strModelPath;
    // model description
    MxBase::ModelDesc m_oModelDesc;
    // MxBase inference module
    MxBase::ModelInferenceProcessor m_oInference;
    // MxBase singleton dvpp processor
    MxBase::DvppWrapper *m_pDvppProcessor = nullptr;
    // @todo:  private use of acl API
    aclmdlDesc *m_pMdlDesc = nullptr;

    bool m_bIsLoaded = false;

    // device id used
    uint32_t m_nDeviceId = 0;
    // statically input & output buffers
    std::vector<MxBase::MemoryData> m_oMemoryInputs;
    std::vector<MxBase::MemoryData> m_oMemoryOutpus;
};
