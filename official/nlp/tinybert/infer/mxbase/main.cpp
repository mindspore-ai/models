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
#include <bits/stdc++.h>
#include <dirent.h>
#include <gflags/gflags.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>

#include "../opensource/tokenizer.h"
#include "MxBase/Log/Log.h"
#include "MxBaseInfer.h"
#include "MxTinyBERTPostProcessor.h"

#define LOG_SYS_ERROR(msg) LogError << msg << " error:" << strerror(errno)
#define D_ISREG(ty) ((ty) == DT_REG)

DEFINE_string(model, "../data/model/tinybert.om", "om model path.");
DEFINE_string(input_file, "../data/dataset/input_file.txt", "input_file.");
DEFINE_string(vocab_txt, "../data/dataset/vocab.txt", "input vocab_txt.");
DEFINE_int32(max_seq_length, 128, "max_seq_length");
DEFINE_int32(device, 0, "device id used");
DEFINE_string(eval_labels_file, "../data/dataset/eval_labels.txt",
              "eval labels file path.");

class TinyBERTInfer : public CBaseInfer {
 private:
    MSTinyBERTPostProcessor m_oPostProcessor;

 public:
    // the shape of seq_length. It's defined by the     inference model
    const int col_per_sample = 128;

    explicit TinyBERTInfer(uint32_t deviceId) : CBaseInfer(deviceId) {}

    MSTinyBERTPostProcessor &GetPostProcessor() override {
        return m_oPostProcessor;
    }

    void OutputResult(const std::vector<ObjDetectInfo> &objs) override {}

 public:
    bool InitPostProcessor(const std::string &configPath,
                           const std::string &labelPath) {
        return m_oPostProcessor.Init(configPath, labelPath, m_oModelDesc) ==
               APP_ERR_OK;
    }
    void UnInitPostProcessor() { m_oPostProcessor.DeInit(); }
};

vector<int> ReadEvalLabelTxt() {
    ifstream in(FLAGS_eval_labels_file.c_str());
    string line;
    vector<int> labels_vector;
    while (getline(in, line) && line.length() != 0) {
        if (line.empty()) {
            continue;
        }
        int label = atoi(line.c_str());
        labels_vector.push_back(label);
    }
    LogInfo << "eval labels size : " << labels_vector.size();
    return labels_vector;
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
    LogInfo << "Usage: --model ../data/models/tinybert.om --device 0";
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    LogInfo << "OM File Path :" << FLAGS_model;
    LogInfo << "deviceId :" << FLAGS_device;
    LogInfo << "vocab_txt :" << FLAGS_vocab_txt;
    LogInfo << "input_file :" << FLAGS_input_file;
    LogInfo << "max_seq_length :" << FLAGS_max_seq_length;

    vector<int> eval_labels_vector = ReadEvalLabelTxt();

    if (eval_labels_vector.size() == 0) {
        LogError << "eval labels vector is empty !!!";
    }

    TinyBERTInfer infer(FLAGS_device);
    if (!infer.Init(FLAGS_model)) {
        LogError << "init inference module failed !!!";
        return -1;
    }

    if (!infer.InitPostProcessor("", "")) {
        LogError << "init post processor failed !!!";
        return -1;
    }
    infer.Dump();

    const clock_t begin_time = clock();

    ifstream in(FLAGS_input_file.c_str());
    string line;
    BertTokenizer tokenizer;
    tokenizer.add_vocab(FLAGS_vocab_txt.c_str());
    int sample_index = 0;
    while (getline(in, line) && line.length() != 0) {
        vector<int> input_ids;
        vector<int> input_segments;
        vector<int> input_masks;
        //      only cut first sentence .
        tokenizer.encode(line, "", input_ids, input_masks, input_segments,
                         FLAGS_max_seq_length, "only_first");

        if (!infer.LoadVectorAsInput(input_ids, 0)) {
            LogError << "load vector:" << input_ids[0]
                     << " to device input[0] error!!!";
        }
        if (!infer.LoadVectorAsInput(input_segments, 1)) {
            LogError << "load vector:" << input_segments[0]
                     << " to device input[1] error!!!";
        }
        if (!infer.LoadVectorAsInput(input_masks, 2)) {
            LogError << "load vector:" << input_masks[0]
                     << " to device input[2] error!!!";
        }

        if (sample_index % 100 == 0) {
            LogInfo << "loading data index " << sample_index;
        }
        int ret = infer.DoInference();
        if (ret != APP_ERR_OK) {
            LogError << "Failed to do inference, ret = " << ret;
        }
        sample_index++;
    }
    int correct_count = 0;
    for (int i = 0; i < infer.predict_vector.size(); ++i) {
        if (infer.predict_vector[i] == eval_labels_vector[i]) {
            correct_count++;
        }
    }
    LogInfo << "performance summary";
    LogInfo << "#####################";
    LogInfo << "total samples: " << sample_index;
    LogInfo << "accuracy: "
            << (static_cast<float>(correct_count) /
                static_cast<float>(sample_index));
    LogInfo << "cost time: "
            << (static_cast<float>(clock() - begin_time) / CLOCKS_PER_SEC);
    LogInfo << "sentences/s: "
            << sample_index /
                   (static_cast<float>(clock() - begin_time) / CLOCKS_PER_SEC);

    infer.UnInitPostProcessor();
    infer.UnInit();

    return 0;
}
