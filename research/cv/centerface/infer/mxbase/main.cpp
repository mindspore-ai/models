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
#include <gflags/gflags.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>

#include "FunctionTimer.h"
#include "MxBase/Log/Log.h"
#include "MxBaseInfer.h"
#include "MxCenterfacePostProcessor.h"
#include "MxImage.h"

FunctionStats g_infer_stats("inference",
                            "ms");  // use microseconds as time unit
FunctionStats g_total_stats("total Process",
                            "ms");  // use milliseconds as time unit

DEFINE_string(model, "../data/models/centerface_131_NCHW.om", "om model path.");
DEFINE_string(images, "../data/images/", "test image file or path.");
DEFINE_string(config, "../data/models/centerface.cfg",
              "config file for this model.");
DEFINE_int32(device, 0, "device id used");
DEFINE_string(color, "bgr", "input color space need for model.");
DEFINE_bool(show, false,
            "whether show result as bbox on top of original picture");

class CCenterfaceInfer : public CBaseInfer {
    MxCenterfacePostProcessor m_oPostProcessor;
    MxBase::PostImageInfo m_LastPostImageInfo;
    CVImage m_LastImage;
    std::string m_strOutputJpg;
    std::string m_strOutputTxt;
    std::string m_strInputFileName;

    // should be override by specific model subclass
    void GetImagePreprocessConfig(std::string& color, cv::Scalar*& means,
                                  cv::Scalar*& stds) override {
        color = FLAGS_color;
        static cv::Scalar __means = {0.408, 0.447, 0.470};
        static cv::Scalar __stds = {0.289, 0.274, 0.278};

        means = &__means;
        stds = &__stds;
    }

 public:
    explicit CCenterfaceInfer(uint32_t deviceId) : CBaseInfer(deviceId) {}
    void DoPostProcess(std::vector<MxBase::TensorBase>& tensors) override {
        std::vector<std::vector<MxBase::ObjectInfo>> objectInfos;
        FILE* fp = fopen(m_strOutputTxt.c_str(), "w");
        fprintf(fp, "%s\r\n1\r\n", m_strInputFileName.c_str());

        std::vector<MxBase::ResizedImageInfo> imgInfos;
        imgInfos.emplace_back(MxBase::ResizedImageInfo{
            m_LastPostImageInfo.widthResize, m_LastPostImageInfo.heightResize,
            m_LastPostImageInfo.widthOriginal,
            m_LastPostImageInfo.heightOriginal, MxBase::RESIZER_RESCALE, 0.0});

        if (APP_ERR_OK ==
            m_oPostProcessor.Process(tensors, objectInfos, imgInfos)) {
            int predictNum = 0;
            for (auto& pos : objectInfos[0]) {
                if (FLAGS_show)
                    m_LastImage.DrawBox(pos.x0, pos.y0, pos.x1, pos.y1,
                                        pos.confidence);
                fprintf(fp, "%.1f %.1f %.1f %.1f %.3f\r\n", pos.x0, pos.y0,
                        pos.x1 - pos.x0 + 1, pos.y1 - pos.y0 + 1,
                        pos.confidence);

                LogInfo << "[" << ++predictNum << "] " << pos.x0 << " "
                        << pos.y0 << " " << pos.x1 - pos.x0 + 1 << " "
                        << pos.y1 - pos.y0 + 1 << " score: " << pos.confidence;
            }
            if (FLAGS_show) m_LastImage.Save(m_strOutputJpg.c_str());
        }
        fclose(fp);
    }

    bool PreprocessImage(CVImage& input, uint32_t w, uint32_t h,
                         const std::string& color, bool isCenter,
                         CVImage& output,
                         MxBase::PostImageInfo& info) override {
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

    bool LoadImageAsInput(const std::string& file, size_t index = 0) override {
        m_strOutputJpg = "mark_images/";
        m_strOutputTxt = "infer_results/";
        m_strInputFileName = basename(file.c_str());
        m_strOutputJpg.append(file.substr(FLAGS_images.size() + 1));
        m_strOutputTxt.append(file.substr(FLAGS_images.size() + 1));
        m_strOutputJpg.replace(m_strOutputJpg.rfind('.'), -1, "_out.jpg");
        m_strOutputTxt.replace(m_strOutputTxt.rfind('.'), -1, ".txt");

        if (!MkdirRecursive(ResolvePathName(m_strOutputJpg))) return false;
        if (!MkdirRecursive(ResolvePathName(m_strOutputTxt))) return false;
        return CBaseInfer::LoadImageAsInput(file, index);
    }

 public:
    bool InitPostProcessor(const std::string& configPath = "",
                           const std::string& labelPath = "") {
        return m_oPostProcessor.Init(configPath, labelPath) == APP_ERR_OK;
    }
    void UnInitPostProcessor() { m_oPostProcessor.DeInit(); }
};

static bool g_loop_stop = false;

void softINT(int signo) {
    printf("user ctr-c quit loop!\n");
    g_loop_stop = true;
}

int main(int argc, char* argv[]) {
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = google::GLOG_ERROR;
    std::cout << "Usage: --model ../data/models/centerface.om --images "
                 "../data/images/ --show --color bgr --width 832 --height 832 "
                 "--device 0"
              << std::endl;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    LogInfo << "OM File Path :" << FLAGS_model;
    LogInfo << "config File Path :" << FLAGS_config;
    LogInfo << "input image Path :" << FLAGS_images;
    LogInfo << "deviceId :" << FLAGS_device;
    LogInfo << "showResult :" << FLAGS_show;

    std::vector<std::string> files;
    if (!FetchTestFiles(FLAGS_images, files)) return -1;

    CCenterfaceInfer infer(FLAGS_device);
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
    int i = 0;
    for (auto& file : files) {
        if (g_loop_stop) break;

        timer.start_timer();
        if (!infer.LoadImageAsInput(file)) {
            LogError << "load image:" << file << " to device input[0] error!!!";
        }
        infer.DoInference();
        timer.calculate_time();
        g_total_stats.update_time(timer.get_elapsed_time_in_milliseconds());
        if (i % 100 == 0) {
            LogInfo << "loading data index " << i;
        }
        i++;
    }


    infer.UnInitPostProcessor();
    infer.UnInit();

    return 0;
}
