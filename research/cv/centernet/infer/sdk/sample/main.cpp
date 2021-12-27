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

#include <stdlib.h>


#include <cstring>
#include <iostream>
#include <boost/property_tree/json_parser.hpp>

#include "../../mxbase/FunctionTimer.h"
#include "../../mxbase/MxCenterNetPostProcessor.h"
#include "../../mxbase/MxImage.h"
#include "../../mxbase/MxUtil.h"
#include "CommandFlagParser.h"
#include "MxBase/Log/Log.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/Proto/MxpiDataTypeDeleter.h"
#include "cpprest/json.h"


namespace json = web::json;

FunctionStats g_infer_stats("inference",
                            "ms");  // use microseconds as time unit
FunctionStats g_total_stats("total Process",
                            "ms");  // use milliseconds as time unit
uint32_t g_original_width;
uint32_t g_original_height;

namespace {
const char STREAM_NAME[] = "im_centernet";
}  // namespace

DEFINE_string(images, "../../data/images/val2017/", "test image file or path.");
DEFINE_string(image_name_path, "../../data/config/test.txt",
              "json file for getting image info");
DEFINE_string(pipeline, "../../data/config/centernet_ms.pipeline",
              "config file for this model.");
DEFINE_uint32(width, 512, "om file accept width");
DEFINE_uint32(height, 512, "om file accept height");
DEFINE_bool(show, false,
            "whether show result as bbox on top of original picture");
DEFINE_string(color, "bgr", "input color space need for model.");
DEFINE_string(config, "../../data/models/centernet.cfg",
              "config file for this model.");

struct ImgInfo {
    std::string imgPath;
    uint64_t imageId;
};

APP_ERROR DrawRectangle(const std::string& imagePath,
                        const std::vector<MxBase::ObjectInfo>& objInfos) {
    std::string m_strOutputJpg = "mark_images/";
    std::string m_strInputFileName = basename(imagePath.c_str());
    m_strOutputJpg.append(m_strInputFileName);
    m_strOutputJpg.replace(m_strOutputJpg.rfind('.'), -1, "_out.jpg");

    if (!MkdirRecursive(ResolvePathName(m_strOutputJpg))) return false;

    CVImage img;
    if (FLAGS_show) img.Load(imagePath);
    for (auto& pos : objInfos) {
        if (FLAGS_show)
            img.DrawBox(pos.x0, pos.y0, pos.x1, pos.y1, pos.confidence);
    }
    if (FLAGS_show) img.Save(m_strOutputJpg.c_str());

    return APP_ERR_OK;
}

void GetTensors(
    const std::shared_ptr<MxTools::MxpiTensorPackageList>& tensorPackageList,
    std::vector<MxBase::TensorBase>& tensors) {
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
                reinterpret_cast<void*>(tensorPackageList->tensorpackagevec(i)
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

bool PreprocessedImage(const std::string& file, uint32_t wdith, uint32_t height,
                       CVImage& dstImg) {
    CVImage img;
    if (img.Load(file) != APP_ERR_OK) {
        LogError << "load image :" << file << " as CVImage error !!!";
        return false;
    }

    // char buf[100];
    // snprintf(buf,sizeof(buf),"%u",img.Width());
    // setenv("ORG_WIDTH",buf,1);
    // snprintf(buf,sizeof(buf),"%u",img.Height());
    // setenv("ORG_HEIGHT",buf,1);
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
APP_ERROR SendEachProtobuf(MxStream::MxStreamManager& mxStreamManager,
                           int inPluginId, const std::string& streamName,
                           MxBase::MemoryData& inputMem,
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

    ret = mxStreamManager.SendProtobuf(streamName, inPluginId, dataBufferVec);
    return ret;
}

static bool g_loop_stop = false;

void softINT(int signo) {
    printf("user ctr-c quit loop!\n");
    g_loop_stop = true;
}

bool FetchImageNameAndId(const std::string& imageFolderPath,
                         std::string& fileNamePath,
                         std::vector<ImgInfo>& files) {
    std::ifstream inFile(fileNamePath);
    std::string imageName;
    while (std::getline(inFile, imageName)) {
        std::string imagePath = imageFolderPath + imageName;
        char absPath[PATH_MAX];
        if (realpath(imagePath.c_str(), absPath) == nullptr) {
            LOG_SYS_ERROR("get the absolute file path: " << absPath);
            return false;
        }

        struct stat buffer;
        if (stat(absPath, &buffer) != 0) {
            LOG_SYS_ERROR("stat file path:" << absPath);
            return false;
        }

        std::string imageId = imageName.substr(0, imageName.size() - 4);
        int idx = 0;
        while (imageId[idx] == '0') ++idx;
        imageId = imageId.substr(idx);

        if (S_ISREG(buffer.st_mode)) {
            files.push_back({absPath, std::stoul(imageId)});
        } else {
            LogFatal << "not a regular file";
            return false;
        }
    }

    return true;
}

// write object information to JSON
bool writeImageObjectList(std::vector<json::value>& jsonObjs,
                          const std::vector<MxBase::ObjectInfo>& objs,
                          const uint64_t imgId) {
    for (auto& obj : objs) {
        json::value jsonObj;
        jsonObj["image_id"] = json::value::number(imgId);
        jsonObj["category_id"] = json::value::number(1);
        jsonObj["score"] = json::value::number(obj.confidence);
        jsonObj["bbox"][0] = json::value::number(obj.x0);
        jsonObj["bbox"][1] = json::value::number(obj.y0);
        jsonObj["bbox"][2] = json::value::number(obj.x1 - obj.x0);
        jsonObj["bbox"][3] = json::value::number(obj.y1 - obj.y0);

        std::vector<json::value> keyPoints;
        for (auto& point : obj.mask) {
            keyPoints.push_back(
                json::value::number(static_cast<float>(point[0]) / 1000));
            keyPoints.push_back(
                json::value::number(static_cast<float>(point[1]) / 1000));
            keyPoints.push_back(json::value::number(static_cast<float>(1.0)));
        }
        jsonObj["keypoints"] = json::value::array(keyPoints);

        jsonObjs.push_back(jsonObj);
    }

    return true;
}

int main(int argc, char* argv[]) {
    FLAGS_logtostderr = 1;
    std::cout << "Usage: --images " << FLAGS_images << " --pipeline "
              << FLAGS_pipeline << " --show" << std::endl;
    OptionManager::getInstance()->parseCommandLineFlags(argc, argv);

    std::vector<ImgInfo> files;
    if (!FetchImageNameAndId(FLAGS_images, FLAGS_image_name_path, files))
        return -1;

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
    std::vector<json::value> jsonResults;

    signal(SIGINT, softINT);
    FLAGS_minloglevel = google::GLOG_ERROR;

    for (auto& file : files) {
        if (g_loop_stop) break;

        timer.start_timer();

        if (!PreprocessedImage(file.imgPath, FLAGS_width, FLAGS_height,
                               destImage)) {
            LogError << "Preprocess image file error:" << file.imgPath;
        } else {
            // read image file and build stream input
            MxBase::MemoryData dataBuffer(destImage.FetchImageBuf(),
                                          destImage.FetchImageBytes());
            // send protobuf data into stream
            ret = SendEachProtobuf(mxStreamManager, 0, STREAM_NAME, dataBuffer,
                                   MxBase::TENSOR_DTYPE_FLOAT32);

            if (APP_ERR_OK != ret) {
                LogError << "Failed to send data to stream, ret = " << ret
                         << ".";
                return ret;
            }
            std::vector<MxStream::MxstProtobufOut> output =
                mxStreamManager.GetProtobuf(STREAM_NAME, 0, keyVec);
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
            MxCenterNetPostProcessor postProcessor;
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

            writeImageObjectList(jsonResults, objectInfos[0], file.imageId);
            DrawRectangle(file.imgPath, objectInfos[0]);

            timer.calculate_time();
            g_total_stats.update_time(timer.get_elapsed_time_in_milliseconds());
        }
    }
    // destroy streams
    mxStreamManager.DestroyAllStreams();

    // write inference result to the json file
    std::string m_strOutputTxt = "infer_results/sdk_infer_result.json";
    if (!MkdirRecursive(ResolvePathName(m_strOutputTxt))) return false;

    std::ofstream result_file;
    result_file.open(m_strOutputTxt);
    result_file << json::value::array(jsonResults);

    g_total_stats.print_stats();
}
