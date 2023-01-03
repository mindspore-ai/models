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
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>

#include "FunctionTimer.h"
#include "MxBase/Log/Log.h"
#include "MxBaseInfer.h"
#include "MxDeepFmPostProcessor.h"

#define LOG_SYS_ERROR(msg) LogError << msg << " error:" << strerror(errno)
#define D_ISREG(ty) ((ty) == DT_REG)

FunctionStats g_infer_stats("WideAndDeep Inference", 1e6);

DEFINE_string(model, "../data/models/wide_and_deep.om", "om model path.");
DEFINE_string(feat_ids, "../data/feat_ids.bin", "feat_ids.");
DEFINE_string(feat_vals, "../data/feat_vals.bin", "feat_vals.");
DEFINE_int32(sample_num, 1, "num of samples");
DEFINE_int32(device, 0, "device id used");

class WideAndDeepInfer : public CBaseInfer {
 private:
    MSDeepfmPostProcessor m_oPostProcessor;

 public:
    // the shape of feat_ids & feat_vals. It's defined by the inference model
    const int col_per_sample = 39;

    // total number of samples
    uint64_t sample_num = 1;

    explicit WideAndDeepInfer(uint32_t deviceId) : CBaseInfer(deviceId) {}

    MSDeepfmPostProcessor &GetPostProcessor() override {
        return m_oPostProcessor;
    }

    void OutputResult(const std::vector<ObjDetectInfo> &objs) override {
    }

 public:
    bool InitPostProcessor(const std::string &configPath,
                           const std::string &labelPath) {
        return m_oPostProcessor.Init(configPath, labelPath, m_oModelDesc) ==
               APP_ERR_OK;
    }
    void UnInitPostProcessor() { m_oPostProcessor.DeInit(); }

    bool CheckInputSize(const std::string &featIdPath,
                        const std::string &featValPath) {
        struct stat featIdStat;
        struct stat featValStat;
        stat(featIdPath.c_str(), &featIdStat);
        stat(featValPath.c_str(), &featValStat);

        uint64_t featIdSize = featValStat.st_size;
        uint64_t idSampleSize = sizeof(int) * col_per_sample;
        uint64_t valSampelSize = sizeof(float) * col_per_sample;
        sample_num = featIdSize / idSampleSize;
        // if the size of feat_ids.bin or feat_vals.bin is not divisible by the
        // size of a single sample or the size of two files is not equal to each
        // other return false;
        if (featIdSize % idSampleSize != 0 ||
            featIdSize % valSampelSize != 0 ||
            sample_num != (featIdSize / valSampelSize))
            return false;

        return true;
    }
};

template <class Type>
int load_bin_file(char *buffer, const std::string &filename, int struct_num) {
    std::ifstream rf(filename, std::ios::out | std::ios::binary);

    if (!rf) {
        LogError << "Cannot open file!";
        return -1;
    }

    rf.read(buffer, sizeof(Type) * struct_num);
    rf.close();

    if (!rf.good()) {
        LogError << "Error occurred at reading time!";
        return -2;
    }

    return 0;
}

/*
 * @description Initialize and run AclProcess module
 * @param resourceInfo resource info of deviceIds, model info, single Operator
 * Path, etc
 * @param file the absolute path of input file
 * @return int int code
 */
int main(int argc, char *argv[]) {
    FLAGS_logtostderr = 1;
    LogInfo << "Usage: --model ../data/models/wide_and_deep.om --device 0";
    if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }

    LogInfo << "OM File Path :" << FLAGS_model;
    LogInfo << "deviceId :" << FLAGS_device;
    LogInfo << "feat_ids :" << FLAGS_feat_ids;
    LogInfo << "feat_vals :" << FLAGS_feat_vals;
    LogInfo << "sample_num :" << FLAGS_sample_num;

    WideAndDeepInfer infer(FLAGS_device);
    if (!infer.CheckInputSize(FLAGS_feat_ids, FLAGS_feat_vals)) {
        LogError << "invalid files: " << FLAGS_feat_ids << ", "
                 << FLAGS_feat_vals;
        LogError
            << "File sizes are not equal or the file size is not divisible by "
               "the size of a single sample";
    }

    if (!infer.Init(FLAGS_model)) {
        LogError << "init inference module failed !!!";
        return -1;
    }

    if (!infer.InitPostProcessor("", "")) {
        LogError << "init post processor failed !!!";
        return -1;
    }
    infer.Dump();

    FunctionTimer timer;

    if (FLAGS_sample_num == -1) {
        FLAGS_sample_num = infer.sample_num;
    }

    for (int i = 0; i < FLAGS_sample_num; ++i) {
        if (!infer.LoadBinaryAsInput<int>(FLAGS_feat_ids, infer.col_per_sample,
                                          i * infer.col_per_sample, 0)) {
            LogError << "load text:" << FLAGS_feat_ids
                     << " to device input[0] error!!!";
            break;
        }
        if (!infer.LoadBinaryAsInput<float>(FLAGS_feat_vals,
                                            infer.col_per_sample,
                                            i * infer.col_per_sample, 1)) {
            LogError << "load text:" << FLAGS_feat_vals
                     << " to device input[0] error!!!";
            break;
        }

        if (FLAGS_sample_num < 1000 || i % 1000 == 0) {
            LogInfo << "loading data index " << i;
        }

        timer.start_timer();
        int ret = infer.DoInference();
        timer.calculate_time();
        g_infer_stats.update_time(timer.get_elapsed_time_in_microseconds());

        if (ret != APP_ERR_OK) {
            LogError << "Failed to do inference, ret = " << ret;
            break;
        }
    }

    infer.UnInitPostProcessor();
    infer.UnInit();

    return 0;
}
