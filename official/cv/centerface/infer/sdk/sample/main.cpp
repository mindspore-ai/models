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

#include <gflags/gflags.h>
#include <stdlib.h>

#include <cstring>
#include <iostream>
#include <boost/property_tree/json_parser.hpp>


#include "../../mxbase/FunctionTimer.h"
#include "../../mxbase/MxCenterfacePostProcessor.h"
#include "../../mxbase/MxImage.h"
#include "../../mxbase/MxUtil.h"
#include "MxBase/Log/Log.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/Proto/MxpiDataTypeDeleter.h"

FunctionStats g_infer_stats("inference",
                            "ms");  // use microseconds as time unit
FunctionStats g_total_stats("total Process",
                            "ms");  // use milliseconds as time unit

uint32_t g_original_width;
uint32_t g_original_height;

DEFINE_string(images, "../../data/images/", "test image file or path.");
DEFINE_string(pipeline, "../../data/config/centerface_no_aipp.pipeline",
              "config file for this model.");
DEFINE_uint32(width, 832, "om file accept width");
DEFINE_uint32(height, 832, "om file accept height");
DEFINE_bool(show, false,
            "whether show result as bbox on top of original picture");
DEFINE_string(color, "bgr", "input color space need for model.");
DEFINE_string(config, "../../data/models/centerface.cfg",
              "config file for this model.");

APP_ERROR WriteResult(std::string &imgPath,
                      const std::vector<MxBase::ObjectInfo> &objInfos) {
    std::string m_strOutputJpg = "mark_images/";
    std::string m_strOutputTxt = "infer_results/";
    std::string m_strInputFileName = basename(imgPath.c_str());
    m_strOutputJpg.append(imgPath.substr(FLAGS_images.size() + 1));
    m_strOutputTxt.append(imgPath.substr(FLAGS_images.size() + 1));
    m_strOutputJpg.replace(m_strOutputJpg.rfind('.'), -1, "_out.jpg");
    m_strOutputTxt.replace(m_strOutputTxt.rfind('.'), -1, ".txt");

    if (!MkdirRecursive(ResolvePathName(m_strOutputJpg))) return false;
    if (!MkdirRecursive(ResolvePathName(m_strOutputTxt))) return false;
    FILE *fp = fopen(m_strOutputTxt.c_str(), "w");
    fprintf(fp, "%s\r\n1\r\n", m_strInputFileName.c_str());

    CVImage img;
    if (FLAGS_show) img.Load(imgPath);
    for (auto &pos : objInfos) {
        if (FLAGS_show)
            img.DrawBox(pos.x0, pos.y0, pos.x1, pos.y1, pos.confidence);
        fprintf(fp, "%f %f %f %f %f\r\n", pos.x0, pos.y0, pos.x1 - pos.x0 + 1,
                pos.y1 - pos.y0 + 1, pos.confidence);
    }
    if (FLAGS_show) img.Save(m_strOutputJpg.c_str());
    fclose(fp);

    return APP_ERR_OK;
}

void GetTensors(
    const std::shared_ptr<MxTools::MxpiTensorPackageList> &tensorPackageList,
    std::vector<MxBase::TensorBase> &tensors) {
    for (int i = 0; i < tensorPackageList->tensorpackagevec_size(); ++i) {
        for (int j = 0;
             j < tensorPackageList->tensorpackagevec(i).tensorvec_size(); j++) {
            MxBase::MemoryData memoryData = {};
            memoryData.deviceId =
                tensorPackageList->tensorpackagevec(i).tensorvec(j).deviceid();
            memoryData.type = (MxBase::MemoryData::MemoryType)tensorPackageList
                                  ->tensorpackagevec(i)
                                  .tensorvec(j)
                                  .memtype();
            memoryData.size = (uint32_t)tensorPackageList->tensorpackagevec(i)
                                  .tensorvec(j)
                                  .tensordatasize();
            memoryData.ptrData =
                reinterpret_cast<void *>(tensorPackageList->tensorpackagevec(i)
                                             .tensorvec(j)
                                             .tensordataptr());
            if (memoryData.type == MxBase::MemoryData::MEMORY_HOST ||
                memoryData.type == MxBase::MemoryData::MEMORY_HOST_MALLOC ||
                memoryData.type == MxBase::MemoryData::MEMORY_HOST_NEW) {
                memoryData.deviceId = -1;
            }
            std::vector<uint32_t> outputShape = {};
            for (int k = 0; k < tensorPackageList->tensorpackagevec(i)
                                    .tensorvec(j)
                                    .tensorshape_size();
                 ++k) {
                outputShape.push_back(
                    (uint32_t)tensorPackageList->tensorpackagevec(i)
                        .tensorvec(j)
                        .tensorshape(k));
            }
            MxBase::TensorBase tmpTensor(
                memoryData, true, outputShape,
                (MxBase::TensorDataType)tensorPackageList->tensorpackagevec(0)
                    .tensorvec(j)
                    .tensordatatype());
            tensors.push_back(tmpTensor);
        }
    }
}

bool PreprocessedImage(const std::string &file, uint32_t wdith, uint32_t height,
                       CVImage &dstImg) {
    CVImage img;
    if (img.Load(file) != APP_ERR_OK) {
        LogError << "load image :" << file << " as CVImage error !!!";
        return false;
    }

    g_original_width = img.Width();
    g_original_height = img.Height();

    static cv::Scalar __means = {0.408, 0.447, 0.470};
    static cv::Scalar __stds = {0.289, 0.274, 0.278};

    dstImg = img.WarpAffinePreprocess(wdith, height, FLAGS_color);
    dstImg = dstImg.ConvertToDeviceFormat(ACL_FLOAT, ACL_FORMAT_NCHW, &__means,
                                          &__stds);
    return true;
}

// This function is only for reading sample txt.
APP_ERROR SendEachProtobuf(MxStream::MxStreamManager &mxStreamManager,
                           int inPluginId, const std::string &streamName,
                           MxBase::MemoryData &inputMem,
                           MxBase::TensorDataType dType) {
    MxBase::MemoryData memoryDst(inputMem.size,
                                 MxBase::MemoryData::MEMORY_HOST_NEW);
    APP_ERROR ret =
        MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, inputMem);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }
    auto tensorPackageList = std::shared_ptr<MxTools::MxpiTensorPackageList>(
        new MxTools::MxpiTensorPackageList,
        MxTools::g_deleteFuncMxpiTensorPackageList);
    auto tensorPackage = tensorPackageList->add_tensorpackagevec();
    auto tensorVec = tensorPackage->add_tensorvec();
    tensorVec->set_tensordataptr((uint64_t)memoryDst.ptrData);
    tensorVec->set_tensordatasize(memoryDst.size);
    tensorVec->set_tensordatatype(dType);
    tensorVec->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);
    tensorVec->set_deviceid(0);
    // nchw
    tensorVec->add_tensorshape(1);
    tensorVec->add_tensorshape(3);
    tensorVec->add_tensorshape(FLAGS_height);
    tensorVec->add_tensorshape(FLAGS_width);

    MxStream::MxstProtobufIn dataBuffer;
    std::ostringstream dataSource;
    dataSource << "appsrc" << inPluginId;
    dataBuffer.key = dataSource.str();
    dataBuffer.messagePtr =
        std::static_pointer_cast<google::protobuf::Message>(tensorPackageList);
    std::vector<MxStream::MxstProtobufIn> dataBufferVec;
    dataBufferVec.push_back(dataBuffer);

    // @add: MxpiVisionList
    /*
    auto mxVisionPackageList = std::make_shared<MxTools::MxpiVisionList>();
    auto mxVisionInfo =   mxVisionPackageList->add_visionvec();
    auto visioninfo = mxVisionInfo->mutable_visioninfo();

    visioninfo->set_format(13);
    visioninfo->set_width(832);
    visioninfo->set_height(832);
    visioninfo->set_widthaligned(832);
    visioninfo->set_heightaligned(832);

    auto visiondata =  mxVisionInfo->mutable_visiondata();
    visiondata->set_dataptr((uint64_t)memoryDst.ptrData);
    visiondata->set_datasize(memoryDst.size);
    visiondata->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);

    dataBuffer.messagePtr =
    std::static_pointer_cast<google::protobuf::Message>(mxVisionPackageList);
    dataBufferVec.push_back(dataBuffer);*/
    ret = mxStreamManager.SendProtobuf(streamName, inPluginId, dataBufferVec);
    return ret;
}

static bool g_loop_stop = false;

void softINT(int signo) {
    printf("user ctr-c quit loop!\n");
    g_loop_stop = true;
}

int main(int argc, char *argv[]) {
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = google::GLOG_ERROR;
    std::cout << "Usage: --images ../data/images/ --pipeline "
                 "../../data/config/centerface_ms.pipeline --show"
              << std::endl;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::vector<std::string> files;
    if (!FetchTestFiles(FLAGS_images, files)) return -1;

    for (auto &file : files) {
        LogInfo << "scan file:" << file;
    }

    // Init stream manager
    MxStream::MxStreamManager mxStreamManager;
    LogError << "begin to Init stream manager...";
    APP_ERROR ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to Init Stream manager, ret = " << ret << ".";
        return ret;
    }
    // FLAGS_minloglevel=google::GLOG_ERROR;
    // create stream by pipeline config file
    ret = mxStreamManager.CreateMultipleStreamsFromFile(FLAGS_pipeline);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create Stream, ret = " << ret << ".";
        return ret;
    }
    std::vector<std::string> keyVec = {"mxpi_tensorinfer0"};
    CVImage destImage;
    FunctionTimer timer;

    signal(SIGINT, softINT);
    FLAGS_minloglevel = google::GLOG_ERROR;

    for (auto &file : files) {
        if (g_loop_stop) break;

        timer.start_timer();

        if (!PreprocessedImage(file, FLAGS_width, FLAGS_height, destImage)) {
            LogError << "Preprocess image file error:" << file;
        } else {
            // read image file and build stream input
            MxBase::MemoryData dataBuffer(destImage.FetchImageBuf(),
                                          destImage.FetchImageBytes());
            // send protobuf data into stream
            ret = SendEachProtobuf(mxStreamManager, 0, "im_centerface",
                                   dataBuffer, MxBase::TENSOR_DTYPE_FLOAT32);

            if (APP_ERR_OK != ret) {
                LogError << "Failed to send data to stream, ret = " << ret
                         << ".";
                return ret;
            }
            std::vector<MxStream::MxstProtobufOut> output =
                mxStreamManager.GetProtobuf("im_centerface", 0, keyVec);
            if (output.size() < 1) {
                LogError << "output size less then 1 !!!";
                return APP_ERR_ACL_FAILURE;
            }
            if (output[0].errorCode != APP_ERR_OK) {
                LogError << "GetProtobuf error. errorCode="
                         << output[0].errorCode;
                return output[0].errorCode;
            }
            std::shared_ptr<MxTools::MxpiTensorPackageList> objectList =
                std::dynamic_pointer_cast<MxTools::MxpiTensorPackageList>(
                    output[0].messagePtr);

            std::vector<MxBase::TensorBase> tensors = {};
            GetTensors(objectList, tensors);

            // post process
            MxCenterfacePostProcessor postProcessor;
            std::vector<std::vector<MxBase::ObjectInfo>> objectInfos;

            if (APP_ERR_OK != postProcessor.Init(FLAGS_config, "")) {
                LogError << "Failed to init post processor!";
            }
            std::vector<MxBase::ResizedImageInfo> imgInfos;
            imgInfos.emplace_back(MxBase::ResizedImageInfo{
                FLAGS_width, FLAGS_height, g_original_width, g_original_height,
                MxBase::RESIZER_RESCALE, 0.0});
            if (APP_ERR_OK !=
                postProcessor.Process(tensors, objectInfos, imgInfos)) {
                LogError << "Failed in post process!";
            }

            WriteResult(file, objectInfos[0]);

            timer.calculate_time();
            g_total_stats.update_time(timer.get_elapsed_time_in_milliseconds());
        }
    }
    // destroy streams
    mxStreamManager.DestroyAllStreams();

    g_total_stats.print_stats();
}
