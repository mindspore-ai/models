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
#include <ctime>
#include <fstream>
#include <iostream>

#include "MxBase/Log/Log.h"
#include "MxBaseInfer.h"
#include "MxDeepFmPostProcessor.h"

#define LOG_SYS_ERROR(msg) LogError << msg << " error:" << strerror(errno)
#define D_ISREG(ty) ((ty) == DT_REG)

DEFINE_string(model, "../data/deepfm.om", "om model path.");
DEFINE_string(feat_ids, "../data/feat_ids.bin", "feat_ids.");
DEFINE_string(feat_vals, "../data/feat_vals.bin", "feat_vals.");
DEFINE_int32(sample_num, 1, "num of samples");
DEFINE_int32(device, 0, "device id used");

class DeepFmInfer : public CBaseInfer {
 private:
    MSDeepfmPostProcessor m_oPostProcessor;

 public:
    // the shape of feat_ids & feat_vals. It's defined by the inference model
    const int col_per_sample = 39;

    explicit DeepFmInfer(uint32_t deviceId) : CBaseInfer(deviceId) {}

    MSDeepfmPostProcessor &get_post_processor() override {
        return m_oPostProcessor;
    }

    void OutputResult(const std::vector<ObjDetectInfo> &objs) override {}

 public:
    bool InitPostProcessor(const std::string &configPath,
                           const std::string &labelPath) {
        return m_oPostProcessor.Init(configPath, labelPath, m_oModelDesc) ==
               APP_ERR_OK;
    }
    void UninitPostProcessor() { m_oPostProcessor.DeInit(); }
};

/*
 * @description Initialize and run AclProcess module
 * @param resourceInfo resource info of deviceIds, model info, single Operator
 * Path, etc
 * @param file the absolute path of input file
 * @return int int code
 */
int main(int argc, char *argv[]) {
    FLAGS_logtostderr = 1;
    LogInfo << "Usage: --model ../data/models/deepfm.om --device 0";
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    LogInfo << "OM File Path :" << FLAGS_model;
    LogInfo << "deviceId :" << FLAGS_device;
    LogInfo << "feat_ids :" << FLAGS_feat_ids;
    LogInfo << "feat_vals :" << FLAGS_feat_vals;
    LogInfo << "sample_num :" << FLAGS_sample_num;

    DeepFmInfer infer(FLAGS_device);
    if (!infer.Init(FLAGS_model)) {
        LogError << "init inference module failed !!!";
        return -1;
    }

    if (!infer.InitPostProcessor("", "")) {
        LogError << "init post processor failed !!!";
        return -1;
    }
    infer.Dump();

    timespec start_time, stop_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int i = 0; i < FLAGS_sample_num; ++i) {
        if (!infer.loadBinaryAsInput<int>(FLAGS_feat_ids, infer.col_per_sample,
                                          i * infer.col_per_sample, 0)) {
            LogError << "load text:" << FLAGS_feat_ids
                     << " to device input[0] error!!!";
            break;
        }
        if (!infer.loadBinaryAsInput<float>(FLAGS_feat_vals,
                                            infer.col_per_sample,
                                            i * infer.col_per_sample, 1)) {
            LogError << "load text:" << FLAGS_feat_vals
                     << " to device input[0] error!!!";
            break;
        }

        if (FLAGS_sample_num < 1000 || i % 1000 == 0) {
            LogInfo << "loading data index " << i;
        }

        int ret = infer.DoInference();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to do inference, ret = " << ret;
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &stop_time);
    double duration = (stop_time.tv_sec - start_time.tv_sec) +
                      (stop_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;

    LogInfo << "sec: " << duration;
    LogInfo << "fps: " << FLAGS_sample_num / duration;

    infer.UninitPostProcessor();
    // infer.UnInit();

    return 0;
}
