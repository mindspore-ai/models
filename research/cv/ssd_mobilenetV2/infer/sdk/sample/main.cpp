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

#include <cstring>
#include <functional>
#include <memory>
#include <queue>
#include <type_traits>

#include "MxBase/Log/Log.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "boost/filesystem.hpp"
#include "infer/mxbase/CommandFlagParser.h"
#include "infer/mxbase/FunctionTimer.h"
#include "infer/mxbase/MxImage.h"
#include "infer/sdk/sample/ResultProcess.h"
#include "opencv2/opencv.hpp"

namespace {

DEFINE_string(coco_path, "../../data/coco2017/", "coco data dir");
DEFINE_string(classes_id_path, "../../data/classes_id.json", "classes id path");
DEFINE_int32(width, 320, "width");
DEFINE_int32(height, 320, "height");

APP_ERROR ReadFile(const std::string &filePath,
                   std::vector<MxStream::MxstProtobufIn> *dataBufferVec,
                   int *origin_width, int *origin_height) {
    cv::Mat image = cv::imread(filePath);
    if (image.empty()) {
        LogError << "read image from " << filePath << "fail" << std::endl;
        return false;
    }

    *origin_width = image.cols;
    *origin_height = image.rows;

    uint32_t model_width = FLAGS_width;
    uint32_t model_height = FLAGS_height;

    LogInfo << "LoadImageToModel : " << model_height << " " << model_width;

    cv::Size dsize = cv::Size(model_width, model_height);
    cv::Mat out_image = cv::Mat(dsize, image.type());
    cv::resize(image, out_image, dsize);
    cv::cvtColor(out_image, out_image, static_cast<int>(cv::COLOR_BGR2RGB));
    static cv::Mat device_mat;
    device_mat = cv::dnn::blobFromImage(out_image, 1.0 / 255.0);
    cv::Scalar means = {0.485, 0.456, 0.406};
    cv::Scalar stds = {0.229, 0.224, 0.225};
    size_t plane = device_mat.step1(1);
    // first channel
    float *ptr = reinterpret_cast<float *>(device_mat.ptr(0));
    for (size_t i = 0; i < plane; i++) {
        *ptr = (*ptr - means[0]) / stds[0];
        ptr++;
    }
    // second channel
    for (size_t i = 0; i < plane; i++) {
        *ptr = (*ptr - means[1]) / stds[1];
        ptr++;
    }
    // third channel
    for (size_t i = 0; i < plane; i++) {
        *ptr = (*ptr - means[2]) / stds[2];
        ptr++;
    }

    MxBase::MemoryData memoryDst(device_mat.data,
                                 device_mat.total() * device_mat.elemSize());

    auto tensorPackageList = std::make_shared<MxTools::MxpiTensorPackageList>();
    auto tensorPackage = tensorPackageList->add_tensorpackagevec();
    auto tensorVec = tensorPackage->add_tensorvec();
    tensorVec->set_tensordataptr((uint64_t)memoryDst.ptrData);
    tensorVec->set_tensordatasize(memoryDst.size);
    tensorVec->set_tensordatatype(MxBase::TENSOR_DTYPE_FLOAT32);
    tensorVec->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);
    tensorVec->set_deviceid(0);
    // nchw
    tensorVec->add_tensorshape(1);
    tensorVec->add_tensorshape(3);
    tensorVec->add_tensorshape(FLAGS_height);
    tensorVec->add_tensorshape(FLAGS_width);

    MxStream::MxstProtobufIn dataBuffer;
    dataBuffer.key = "appsrc0";
    dataBuffer.messagePtr =
        std::static_pointer_cast<google::protobuf::Message>(tensorPackageList);

    dataBufferVec->push_back(dataBuffer);

    return APP_ERR_OK;
}

std::string ReadPipelineConfig(const std::string &pipelineConfigPath) {
    std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
    if (!file) {
        LogError << pipelineConfigPath << " file dose not exist.";
        return "";
    }
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);
    std::unique_ptr<char[]> data(new char[fileSize]);
    file.read(data.get(), fileSize);
    file.close();
    std::string pipelineConfig(data.get(), fileSize);
    return pipelineConfig;
}
}  // namespace

static bool g_loop_stop = false;
MxStream::MxStreamManager mxStreamManager;
const char *streamName = "ssd_mobile_net_v2_coco";
int inPluginId = 0;
std::vector<int> classes_ids;
std::vector<web::json::value> preditions;
std::vector<web::json::value> image_ids;

void softINT(int signo) {
    LogInfo << "user ctr-c quit loop!\n";
    g_loop_stop = true;
}

APP_ERROR init(int argc, char *argv[]) {
    // read image file and build stream input
    // read pipeline config file
    OptionManager::getInstance()->parseCommandLineFlags(argc, argv);

    APP_ERROR ret;
    std::string pipelineConfigPath =
        "../../data/ssd_mobile_net_v2.pipeline";
    std::string pipelineConfig = ReadPipelineConfig(pipelineConfigPath);
    if (pipelineConfig == "") {
        return APP_ERR_COMM_INIT_FAIL;
    }

    signal(SIGINT, softINT);

    // init stream manager
    ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // create stream by pipeline config file
    ret = mxStreamManager.CreateMultipleStreams(pipelineConfig);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    if (!boost::filesystem::is_regular_file(FLAGS_classes_id_path)) {
        return -1;
    }

    if (!boost::filesystem::is_regular_file(FLAGS_classes_id_path)) {
        return -1;
    }
    std::fstream classes_idx_stream(FLAGS_classes_id_path);
    web::json::value tmp = web::json::value::parse(classes_idx_stream);

    for (size_t i = 0; i < tmp.size(); i++) {
        classes_ids.push_back(tmp[i].as_integer());
    }

    return 0;
}

void finalize() {
    web::json::value output = web::json::value::array(preditions);
    std::fstream out(std::string("../../data/") + "predictions.json",
                     std::ios::out | std::ios::trunc);
    out << output.serialize();

    output = web::json::value::array(image_ids);
    std::fstream out1(std::string("../../data/") + "imageid.json",
                      std::ios::out | std::ios::trunc);
    out1 << output.serialize();

    // destroy streams
    mxStreamManager.DestroyAllStreams();
}

int main(int argc, char *argv[]) {
    APP_ERROR ret = init(argc, argv);
    if (ret != 0) {
        return ret;
    }

    // load coco data
    std::fstream coco_stream(FLAGS_coco_path +
                             "annotations/instances_val2017.json");
    if (!coco_stream.is_open()) {
        return -1;
    }
    web::json::value coco_json = web::json::value::parse(coco_stream);
    web::json::array images_info = coco_json["images"].as_array();
    web::json::array annotations = coco_json["annotations"].as_array();

    std::map<int, int> image_category_id;
    std::map<int, int> image_iscrowd;

    for (auto &ann_info : annotations) {
        int iscrowd = ann_info["iscrowd"].as_integer();
        int category_id = ann_info["category_id"].as_integer();
        int id = ann_info["image_id"].as_integer();
        image_category_id[id] = category_id;
        image_iscrowd[id] = iscrowd;
    }

    sdk_infer::mxbase_infer::FunctionStats read_jpg("read_jpg", "us");
    sdk_infer::mxbase_infer::FunctionStats send_data("send_data", "us");

    for (auto &image_info : images_info) {
        if (g_loop_stop) {
            break;
        }
        std::string image_name =
            FLAGS_coco_path + "val2017/" + image_info["file_name"].as_string();

        int id = image_info["id"].as_integer();
        if (image_iscrowd[id] == 1 || image_category_id[id] >= 91) {
            continue;
        }

        std::vector<MxStream::MxstProtobufIn> dataBuffer;
        int origin_width;
        int origin_height;
        {
            sdk_infer::mxbase_infer::FunctionTimer timer;
            timer.start_timer();

            ret = ReadFile(image_name, &dataBuffer, &origin_width,
                           &origin_height);

            timer.calculate_time();
            read_jpg.update_time(timer.get_elapsed_time_in_microseconds());
        }
        if (ret != APP_ERR_OK) {
            return ret;
        }
        {
            sdk_infer::mxbase_infer::FunctionTimer timer;
            timer.start_timer();

            // send data into stream
            ret = mxStreamManager.SendProtobuf(streamName, inPluginId,
                                               dataBuffer);
            if (ret != APP_ERR_OK) {
                return ret;
            }
            // get stream output
            std::vector<MxStream::MxstProtobufOut> output =
                mxStreamManager.GetProtobuf(streamName, inPluginId,
                                            {"mxpi_objectpostprocessor0"});

            timer.calculate_time();
            send_data.update_time(timer.get_elapsed_time_in_microseconds());

            if (output.size() == 0) {
                return APP_ERR_ACL_FAILURE;
            }
            if (output[0].errorCode != APP_ERR_OK) {
                return output[0].errorCode;
            }

            std::shared_ptr<MxTools::MxpiObjectList> objectList =
                std::dynamic_pointer_cast<MxTools::MxpiObjectList>(
                    output[0].messagePtr);
            for (int i = 0; i < objectList->objectvec_size(); i++) {
                auto &object = objectList->objectvec(i);
                object.x0();

                web::json::value json_data;
                json_data["bbox"] = web::json::value::array(
                    {object.x0() * origin_width, object.y0() * origin_height,
                     (object.x1() - object.x0()) * origin_width,
                     (object.y1() - object.y0()) * origin_height});
                json_data["image_id"] = id;
                json_data["score"] = object.classvec(0).confidence();
                json_data["category_id"] =
                    classes_ids[object.classvec(0).classid()];

                preditions.push_back(json_data);
            }
            image_ids.push_back(id);
        }
    }


    finalize();
    read_jpg.print_stats();
    send_data.print_stats();

    return 0;
}
