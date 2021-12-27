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

#include "MxBaseInfer.h"
#include <algorithm>
#include <string>
#include <memory>
#include <vector>

bool CBaseInfer::Init(const std::string &omPath) {
    if (MxBase::DeviceManager::GetInstance()->InitDevices() != APP_ERR_OK) {
        return false;
    }
    if (MxBase::TensorContext::GetInstance()->SetContext(DeviceId) !=
        APP_ERR_OK) {
        return false;
    }
    if (m_oInference.Init(omPath, ModelDesc) != APP_ERR_OK) {
        return false;
    }

    strModelPath = omPath;
    AllocateInputs(MemoryInputs);
    AllocateOutputs(MemoryOutputs);
    IsLoaded = true;
    return true;
}

void CBaseInfer::UnInit() {
    if (IsLoad()) {
        DisposeIOBuffers();
        m_oInference.DeInit();
        strModelPath.clear();
        IsLoaded = false;
        MxBase::DeviceManager::GetInstance()->DestroyDevices();
    }
}

void CBaseInfer::Dump() {
    LogInfo << "om file path:" << strModelPath;
    LogInfo << "model has " << ModelDesc.inputTensors.size() << " inputs, "
            << ModelDesc.outputTensors.size() << " outputs ...";

    size_t idx = 0;
    auto formats = m_oInference.GetInputFormat();
    auto dataTypes = m_oInference.GetInputDataType();
    for (auto &tensor : ModelDesc.inputTensors) {
        LogInfo << "input[" << idx << "] dims: " << Array2Str(tensor.tensorDims)
                << " size:" << tensor.tensorSize;

        LogInfo << "name: " << tensor.tensorName
                << " format:" << aclFormatToStr(formats[idx])
                << " dataType:" << aclDatatypeToStr(dataTypes[idx]);

        idx++;
    }

    idx = 0;
    formats = m_oInference.GetOutputFormat();
    dataTypes = m_oInference.GetOutputDataType();
    for (auto &tensor : ModelDesc.outputTensors) {
        LogInfo << "output[" << idx
                << "] dims: " << Array2Str(tensor.tensorDims)
                << " size:" << tensor.tensorSize;
        LogInfo << " name:" << tensor.tensorName
                << " format:" << aclFormatToStr(formats[idx])
                << " dataType:" << aclDatatypeToStr(dataTypes[idx]);
        idx++;
    }

    if (ModelDesc.dynamicBatch) {
        LogInfo << "dynamic batchSize: " << Array2Str(ModelDesc.batchSizes);
    }
}

MxBase::DvppWrapper *CBaseInfer::GetDvpp() {
    if (!DvppProcessor) {
        DvppProcessor = new MxBase::DvppWrapper;
        DvppProcessor->deviceId_ = DeviceId;
        if (DvppProcessor->Init() != APP_ERR_OK) {
            delete DvppProcessor;
            DvppProcessor = nullptr;
        }
    }
    return DvppProcessor;
}

APP_ERROR CBaseInfer::DoInference() {
    FunctionTimer timer;

    std::vector<MxBase::TensorBase> inputTensors;
    std::vector<MxBase::TensorBase> outputTensors;
    PrepareTensors(MemoryInputs, m_oInference.GetInputShape(),
                   m_oInference.GetInputDataType(), inputTensors);
    PrepareTensors(MemoryOutputs, m_oInference.GetOutputShape(),
                   m_oInference.GetOutputDataType(), outputTensors);

    timer.start_timer();
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret =
        m_oInference.ModelInference(inputTensors, outputTensors, dynamicInfo);
    timer.calculate_time();

    if (ret == APP_ERR_OK) {
        DoPostProcess(outputTensors);
    }

    g_infer_stats.update_time(timer.get_elapsed_time_in_milliseconds());
    return ret;
}

bool CBaseInfer::PreprocessImage(CVImage &input, uint32_t w, uint32_t h,
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

    return static_cast<bool>(output);
}

bool CBaseInfer::LoadImageAsInput(const std::string &file, size_t index) {
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
    GetImagePreprocessConfig(color, means, stds);
    if (!PreprocessImage(img, w, h, color, true, outImg, postImageInfo)) {
        LogError << "Preprocess image error !!!";
        return false;
    }

    return outImg.FetchToDevice(
        MemoryInputs[index],
        (aclDataType)m_oInference.GetInputDataType()[index],
        (aclFormat)m_oInference.GetInputFormat()[index], means, stds);
}

void CBaseInfer::GetImageInputHW(uint32_t &w, uint32_t &h, size_t index) {
    aclFormat input_format = (aclFormat)m_oInference.GetInputFormat()[index];

    if (input_format == ACL_FORMAT_NCHW) {
        w = ModelDesc.inputTensors[index].tensorDims[3];
        h = ModelDesc.inputTensors[index].tensorDims[2];
    } else if (input_format == ACL_FORMAT_NHWC) {
        w = ModelDesc.inputTensors[index].tensorDims[2];
        h = ModelDesc.inputTensors[index].tensorDims[1];
    } else {
        LogError << "incorrect input format";
    }
}

// dispose dvpp processor
void CBaseInfer::DisposeDvpp() {
    if (DvppProcessor) {
        DvppProcessor->DeInit();
        delete DvppProcessor;
        DvppProcessor = nullptr;
    }
}
void CBaseInfer::DisposeIOBuffers() {
    for (auto &pos : MemoryInputs) {
        MxBase::MemoryHelper::MxbsFree(pos);
    }
    MemoryInputs.clear();
    for (auto &pos : MemoryOutputs) {
        MxBase::MemoryHelper::MxbsFree(pos);
    }
    MemoryOutputs.clear();
}

// prepare batch memory for inputs
void CBaseInfer::AllocateInputs(std::vector<MxBase::MemoryData> &data) {
    data.resize(ModelDesc.inputTensors.size());
    size_t idx = 0;
    for (auto &mem : data) {
        mem.deviceId = DeviceId;
        mem.type = MxBase::MemoryData::MEMORY_DEVICE;
        mem.size = ModelDesc.inputTensors[idx].tensorSize;
        MxBase::MemoryHelper::MxbsMalloc(mem);
        idx++;
    }
}

// prepare batch memory for outputs
void CBaseInfer::AllocateOutputs(std::vector<MxBase::MemoryData> &data) {
    data.resize(ModelDesc.outputTensors.size());
    size_t idx = 0;
    for (auto &mem : data) {
        mem.deviceId = DeviceId;
        mem.type = MxBase::MemoryData::MEMORY_DEVICE;
        mem.size = ModelDesc.outputTensors[idx].tensorSize;
        MxBase::MemoryHelper::MxbsMalloc(mem);
        idx++;
    }
}
void CBaseInfer::PrepareTensors(
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

MxBase::BaseTensor transform_tensor(const MxBase::MemoryData &pos,
                                    const MxBase::TensorDesc &desc) {
    return MxBase::BaseTensor{
        pos.ptrData,
        std::vector<int>(desc.tensorDims.begin(), desc.tensorDims.end()),
        pos.size};
}

void CBaseInfer::PrepareTensors(std::vector<MxBase::MemoryData> &data,
                                std::vector<MxBase::TensorDesc> &tensorDesc,
                                std::vector<MxBase::BaseTensor> &tensor) {
    tensor.reserve(data.size());
    std::transform(data.begin(), data.end(), tensorDesc.begin(), tensor.begin(),
                   transform_tensor);
}

APP_ERROR CBaseInfer::FetchDeviceBuffers(
    std::vector<MxBase::MemoryData> &data,
    std::vector<std::shared_ptr<void>> &out) {
    for (auto &mem : data) {
        MxBase::MemoryData memoryDst(mem.size);

        APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, mem);
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
