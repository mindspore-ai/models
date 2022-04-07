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

std::vector<float> GetSimilarity1(const std::vector< std::vector<float> >& embeddings) {
  std::vector<float> similarity(embeddings.size() * embeddings.size());
  for (unsigned int i = 0; i < embeddings.size(); i++) {
    for (unsigned int j = i; j < embeddings.size(); j++) {
      float prod = 0;
      for (unsigned int k = 0; k < embeddings[i].size(); k++) {
        prod += embeddings[i][k] * embeddings[j][k];
      }
      similarity[i * embeddings.size() + j] = prod;
      similarity[j * embeddings.size() + i] = prod;
    }
  }

  return similarity;
}

std::vector<int> SortIndexes(const std::vector<float>& vec) {
  std::vector<int> indexes(vec.size());
  for (unsigned int i = 0; i < indexes.size(); ++i) {
    indexes[i] = i;
  }
  sort(indexes.begin(), indexes.end(), [&vec](int i, int j) -> bool {return vec[i] > vec[j]; });

  return indexes;
}

std::vector<float> CheckReconstruction1(const std::unordered_map< int, std::unordered_set<int> >& graph,
                                       const std::vector< std::vector<float> >& embeddings,
                                       const std::vector<int>& k_query) {
  auto similarity = GetSimilarity1(embeddings);
  auto sorted_idx = SortIndexes(similarity);
  int cur = 0, count = 0;
  std::vector<float> precisions;
  std::vector<float> ret(k_query.size(), 0);
  int max_k_query = k_query.back();
  for (auto idx : sorted_idx) {
    int x = idx / graph.size();
    int y = idx % graph.size();
    count++;
    if (graph.at(x).find(y) != graph.at(x).end() || graph.at(y).find(x) != graph.at(y).end() || x == y) {
      cur++;
    }
    precisions.push_back(1.0 * cur / count);
    if (count > max_k_query) {
      break;
    }
  }
  std::transform(k_query.begin(), k_query.end(), ret.begin(), [&precisions](int k) {return precisions[k]; });

  return ret;
}

std::vector<float> GetSimilarity2(const std::vector< std::vector<float> >& reconstructions) {
  std::vector<float> similarity(reconstructions.size() * reconstructions.size());
  for (unsigned int i = 0; i < reconstructions.size(); i++) {
    for (unsigned int j = 0; j < reconstructions.size(); j++) {
      similarity[i * reconstructions.size() + j] = reconstructions[i][j];
    }
  }

  return similarity;
}

std::vector<float> CheckReconstruction2(const std::unordered_map< int, std::unordered_set<int> >& graph,
                                        const std::vector< std::vector<float> >& reconstructions,
                                        const std::vector<int>& k_query) {
  auto similarity = GetSimilarity2(reconstructions);
  auto sorted_idx = SortIndexes(similarity);
  int cur = 0, count = 0;
  int true_edge_num = std::accumulate(graph.begin(), graph.end(), 0,
                                      [](int num, const std::pair< int, std::unordered_set<int> >& item)
                                      {return num + item.second.size(); });
  std::vector<float> precision_list;
  std::vector<float> precision_k_list;
  std::vector<float> ret(k_query.size(), 0);
  int max_k_query = k_query.back();

  for (auto idx : sorted_idx) {
    int x = idx / graph.size();
    int y = idx % graph.size();
    if (x == y) {
      continue;
    }
    count++;
    if (graph.at(x).find(y) != graph.at(x).end()) {
      cur++;
      if (cur <= true_edge_num) {
        precision_list.push_back(1.0 * cur / count);
      }
    }
    precision_k_list.push_back(1.0 * cur / count);
    if (cur > true_edge_num && count > max_k_query) {
      break;
    }
  }
  float map = std::accumulate(precision_list.begin(), precision_list.end(), 0.0);
  map = map / true_edge_num;
  std::transform(k_query.begin(), k_query.end(), ret.begin(),
                 [&precision_k_list](int k) {return precision_k_list[k]; });
  ret.push_back(map);

  return ret;
}

int WriteResult(const std::vector< std::vector<float> >& embeddings) {
  std::string home_path = "./result_Files";
  std::string file_name = home_path + std::string("/embeddings.txt");
  std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
  for (auto embedding : embeddings) {
    for (auto val : embedding) {
      file_stream << val << " ";
    }
    file_stream << std::endl;
  }
  file_stream.close();

  return 0;
}
