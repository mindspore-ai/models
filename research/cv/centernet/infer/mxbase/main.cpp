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
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include "CommandFlagParser.h"
#include "FunctionTimer.h"
#include "MxBase/Log/Log.h"
#include "MxBaseInfer.h"
#include "MxCenterNetPostProcessor.h"
#include "MxImage.h"
#include "cpprest/json.h"

namespace pt = boost::property_tree;
namespace json = web::json;

FunctionStats g_infer_stats("inference",
                            "ms");  // use microseconds as time unit
FunctionStats g_total_stats("total Process",
                            "ms");  // use milliseconds as time unit

DEFINE_string(model, "../data/models/centerface_export.om", "om model path.");
DEFINE_string(images, "../data/images/val2017/", "test image file or path.");
DEFINE_string(image_name_path, "../data/config/test.txt",
              "json file for getting image info");
DEFINE_string(config, "../data/models/centernet.cfg",
              "config file for this model.");
DEFINE_int32(device, 0, "device id used");
DEFINE_string(color, "bgr", "input color space need for model.");
DEFINE_bool(show, false,
            "whether show result as bbox on top of original picture");

class CCenterNetInfer : public CBaseInfer {
    MxCenterNetPostProcessor m_oPostProcessor;
    MxBase::PostImageInfo m_LastPostImageInfo;
    CVImage m_LastImage;
    std::string m_strOutputJpg;
    std::string m_strInputFileName;
    std::vector<std::vector<MxBase::ObjectInfo>> objectInfos;

    // should be override by specific model subclass
    void GetImagePreprocessConfig(std::string &color, cv::Scalar **means,
                                  cv::Scalar **stds) override {
        color = FLAGS_color;

        static cv::Scalar __means = {0.40789654, 0.44719302, 0.47026115};
        static cv::Scalar __stds = {0.28863828, 0.27408164, 0.27809835};

        *means = &__means;
        *stds = &__stds;
    }

 public:
    explicit CCenterNetInfer(uint32_t deviceId) : CBaseInfer(deviceId) {}

    const std::vector<MxBase::ObjectInfo> &getObjInfo() {
        return objectInfos[0];
    }

    void DoPostProcess(std::vector<MxBase::TensorBase> &tensors) override {
        objectInfos.clear();

        std::vector<MxBase::ResizedImageInfo> imgInfos;
        imgInfos.emplace_back(MxBase::ResizedImageInfo{
            m_LastPostImageInfo.widthResize, m_LastPostImageInfo.heightResize,
            m_LastPostImageInfo.widthOriginal,
            m_LastPostImageInfo.heightOriginal, MxBase::RESIZER_RESCALE, 0.0});

        if (APP_ERR_OK ==
            m_oPostProcessor.Process(tensors, objectInfos, imgInfos)) {
            for (auto &pos : objectInfos[0]) {
                if (FLAGS_show)
                    m_LastImage.DrawBox(pos.x0, pos.y0, pos.x1, pos.y1,
                                        pos.confidence);
            }
            if (FLAGS_show) m_LastImage.Save(m_strOutputJpg.c_str());
        }
    }
    bool PreprocessImage(CVImage &input, uint32_t w, uint32_t h,
                         const std::string &color, bool isCenter,
                         CVImage &output,
                         MxBase::PostImageInfo &info) override {
        m_LastImage = input;
        if (m_oPostProcessor.IsUseAffineTransform()) {
            output = input.WarpAffinePreprocess(w, h, color);
            info.widthOriginal = input.Width();
            info.heightOriginal = input.Height();
            info.widthResize = w;
            info.heightResize = h;
            info.x0 = info.y0 = 0;
            info.x1 = w;
            info.y1 = h;

            m_LastPostImageInfo = info;
            return !!output;
        } else {
            return CBaseInfer::PreprocessImage(
                input, w, h, color, !m_oPostProcessor.IsResizeNoMove(), output,
                m_LastPostImageInfo);
        }
    }

    bool LoadImageAsInput(const std::string &file, size_t index = 0) override {
        m_strOutputJpg = "mark_images/";
        m_strInputFileName = basename(file.c_str());
        m_strOutputJpg.append(file.substr(FLAGS_images.size() + 1));
        m_strOutputJpg.replace(m_strOutputJpg.rfind('.'), -1, "_out.jpg");

        if (!MkdirRecursive(ResolvePathName(m_strOutputJpg))) return false;
        return CBaseInfer::LoadImageAsInput(file, index);
    }

 public:
    bool InitPostProcessor(const std::string &configPath = "",
                           const std::string &labelPath = "") {
        return m_oPostProcessor.Init(configPath, labelPath) == APP_ERR_OK;
    }
    void UnInitPostProcessor() { m_oPostProcessor.DeInit(); }
};

static bool g_loop_stop = false;

void softINT(int signo) {
    printf("user ctr-c quit loop!\n");
    g_loop_stop = true;
}

struct ImgInfo {
    std::string imgPath;
    uint64_t imageId;
};

bool FetchImageNameAndId(const std::string &imageFolderPath,
                         std::string &fileNamePath,
                         std::vector<ImgInfo> &files) {
    std::cout << "FOLDER NAME: " << imageFolderPath << std::endl;
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

// read file names from a JSON file and return a vector of all the file names
bool FetchTestFilesFromJson(const std::string &imageFolderPath,
                            std::string &jsonPath,
                            std::vector<ImgInfo> &files) {
    namespace pt = boost::property_tree;
    pt::ptree root;
    pt::read_json(jsonPath, root);
    pt::ptree img_infos = root.get_child("images");

    BOOST_FOREACH(boost::property_tree::ptree::value_type &v, img_infos) {
        std::string imagePath =
            imageFolderPath +
            v.second.get_child("file_name").get_value<std::string>();
        uint64_t imageId = v.second.get_child("id").get_value<u_int64_t>();
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

        if (S_ISREG(buffer.st_mode)) {
            files.push_back({absPath, imageId});
        } else {
            LogFatal << "not a regular file";
            return false;
        }
    }

    return true;
}

// write object information to JSON
bool writeImageObjectList(std::vector<json::value> &jsonObjs,
                          const std::vector<MxBase::ObjectInfo> &objs,
                          const uint64_t imgId) {
    for (auto &obj : objs) {
        json::value jsonObj;
        jsonObj["image_id"] = json::value::number(imgId);
        jsonObj["category_id"] = json::value::number(1);
        jsonObj["score"] = json::value::number(obj.confidence);
        jsonObj["bbox"][0] = json::value::number(obj.x0);
        jsonObj["bbox"][1] = json::value::number(obj.y0);
        jsonObj["bbox"][2] = json::value::number(obj.x1 - obj.x0);
        jsonObj["bbox"][3] = json::value::number(obj.y1 - obj.y0);

        std::vector<json::value> keyPoints;

        float x, y;
        auto temp_x = reinterpret_cast<unsigned char*> (&x);
        auto temp_y = reinterpret_cast<unsigned char*> (&y);
        for (auto &point : obj.mask) {
            *temp_x = point[0];
            *(temp_x+1) = point[1];
            *(temp_x+2) = point[2];
            *(temp_x+3) = point[3];

            *temp_y = point[4];
            *(temp_y+1) = point[5];
            *(temp_y+2) = point[6];
            *(temp_y+3) = point[7];

            keyPoints.push_back(
                json::value::number(x));
            keyPoints.push_back(
                json::value::number(y));
            keyPoints.push_back(json::value::number(static_cast<float>(1.0)));
        }
        jsonObj["keypoints"] = json::value::array(keyPoints);
        jsonObjs.push_back(jsonObj);
    }

    return true;
}

int main(int argc, char *argv[]) {
    FLAGS_logtostderr = 1;
    std::cout << "Usage: --model ../data/models/centerface.om --images "
                 "../data/images/ --image_name_path ../data/config/test.txt "
                 " --device 0"
              << std::endl;
    OptionManager::getInstance()->parseCommandLineFlags(argc, argv);

    LogInfo << "OM File Path :" << FLAGS_model;
    LogInfo << "config File Path :" << FLAGS_config;
    LogInfo << "input image Path :" << FLAGS_images;
    LogInfo << "image info JSON file: " << FLAGS_image_name_path;
    LogInfo << "deviceId :" << FLAGS_device;
    LogInfo << "showResult :" << FLAGS_show;

    std::vector<ImgInfo> files;
    if (!FetchImageNameAndId(FLAGS_images, FLAGS_image_name_path, files))
        return -1;

    CCenterNetInfer infer(FLAGS_device);
    if (!infer.Init(FLAGS_model)) {
        LogError << "Init inference module failed !!!";
        return -1;
    }
    if (!infer.InitPostProcessor(FLAGS_config)) {
        LogError << "Init post processor failed !!!";
        return -1;
    }
    infer.Dump();

    FunctionTimer timer;
    signal(SIGINT, softINT);

    std::vector<json::value> resultJsonVec;

    for (size_t i = 0; i < files.size(); ++i) {
        if (g_loop_stop) break;

        timer.start_timer();
        if (!infer.LoadImageAsInput(files[i].imgPath)) {
            LogError << "load image:" << files[i].imgPath
                     << " to device input[0] error!!!";
        }
        infer.DoInference();
        timer.calculate_time();
        g_total_stats.update_time(timer.get_elapsed_time_in_milliseconds());
        writeImageObjectList(resultJsonVec, infer.getObjInfo(),
                             files[i].imageId);
        if (i % 100 == 0 && i != 0) {
            LogInfo << "finish data index " << i;
        }
    }

    g_infer_stats.print_stats();
    g_total_stats.print_stats();

    std::string m_strOutputTxt = "infer_results/mxbase_infer_result.json";
    if (!MkdirRecursive(ResolvePathName(m_strOutputTxt))) return false;
    std::ofstream result_file;
    result_file.open(m_strOutputTxt);
    result_file << json::value::array(resultJsonVec);

    infer.UnInitPostProcessor();
    infer.UnInit();
    return 0;
}
