/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>

#include "inc/utils.h"

using mindspore::MSTensor;
using mindspore::DataType;

std::string RealPath(std::string_view path) {
  char realPathMem[PATH_MAX] = {0};
  char *realPathRet = nullptr;
  realPathRet = realpath(path.data(), realPathMem);
  if (realPathRet == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  std::string realPath(realPathMem);
  std::cout << path << " realpath is: " << realPath << std::endl;

  return realPath;
}

std::unordered_map< int, std::unordered_set<int> > GetGraph(std::string gfp, bool is_bidir) {
  std::unordered_map< int, std::unordered_set<int> > graph;
  if (RealPath(gfp).empty()) {
    std::cout << "Invalid graph file path." << std::endl;
    return graph;
  }
  std::ifstream ifs(gfp);
  if (!ifs.is_open()) {
    std::cout << "File: " << gfp << " open failed" << std::endl;
    return graph;
  }

  std::string line;
  int cnt = 0;
  while (getline(ifs, line)) {
    std::stringstream nodes(line);
    int node1, node2;
    nodes >> node1 >> node2;
    if (graph.find(node1) != graph.end()) {
      graph.insert({node1, std::unordered_set<int>()});
    }
    if (graph.find(node2) != graph.end()) {
      graph.insert({node2, std::unordered_set<int>()});
    }
    graph[node1].insert(node2);
    if (is_bidir) {
      graph[node2].insert(node1);
    }
    cnt++;
  }
  std::cout << "The number of nodes: " << graph.size() << std::endl;
  std::cout << "The number of edges: " << cnt << std::endl;

  return graph;
}

std::vector<float> GetDataFromGraph(const std::unordered_set<int>& subgraph, int data_size) {
  std::vector<float> ret(data_size);
  for (int i = 0; i < data_size; i++) {
    if (subgraph.find(i) != subgraph.end())
      ret[i] = 1;
    else
      ret[i] = 0;
  }

  return ret;
}

std::vector<float> Tensor2Vector(const float* pdata, int data_size) {
  std::vector<float> ret(data_size, 0);
  for (int i = 0; i < data_size; i++) {
    ret[i] = pdata[i];
  }

  return ret;
}

int WriteResult(const std::vector< std::vector<float> >& embeddings,
                const std::vector< std::vector<float> >& reconstructions) {
  std::string home_path = "./result_Files";
  std::string emb_file_name = home_path + std::string("/embeddings.txt");
  std::string rec_file_name = home_path + std::string("/reconstructions.txt");
  std::ofstream emb_file_stream(emb_file_name.c_str(), std::ios::trunc);
  std::ofstream rec_file_stream(rec_file_name.c_str(), std::ios::trunc);
  int row = 0;
  for (auto embedding : embeddings) {
    for (auto val : embedding) {
      emb_file_stream << val << " ";
    }
    emb_file_stream << std::endl;
  }
  emb_file_stream.close();
  for (auto reconstruction : reconstructions) {
    int col = 0;
    for (auto val : reconstruction) {
      rec_file_stream << row << " " << col << " " << val << " ";
      rec_file_stream << std::endl;
      col++;
    }
    row++;
  }
  rec_file_stream.close();

  return 0;
}
