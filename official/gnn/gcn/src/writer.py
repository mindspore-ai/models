# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
######################## write mindrecord example ########################
Write mindrecord by data dictionary:
python writer.py mindrecord_script /YourScriptPath ...
"""
import os
import time
from importlib import import_module
from multiprocessing import Pool

from mindspore.mindrecord import FileWriter
from src.graph_map_schema import GraphMapSchema


def exec_task(task_id, mindrecord_dict_data, graph_map_schema, writer, parallel_writer=True):
    """
    Execute task with specified task id
    """
    print("exec task {}, parallel: {} ...".format(task_id, parallel_writer))
    imagenet_iter = mindrecord_dict_data(task_id)
    batch_size = 512
    transform_count = 0
    while True:
        data_list = []
        try:
            for _ in range(batch_size):
                data = imagenet_iter.__next__()
                if 'dst_id' in data:
                    data = graph_map_schema.transform_edge(data)
                else:
                    data = graph_map_schema.transform_node(data)
                data_list.append(data)
                transform_count += 1
            writer.write_raw_data(data_list, parallel_writer=parallel_writer)
            print("transformed {} record...".format(transform_count))
        except StopIteration:
            if data_list:
                writer.write_raw_data(data_list, parallel_writer=parallel_writer)
                print("transformed {} record...".format(transform_count))
            break


def init_writer(mr_schema, mindrecord_file, mindrecord_partitions, mindrecord_header_size_by_bit,
                mindrecord_page_size_by_bit):
    """
    init writer
    """
    print("Init writer  ...")
    mr_writer = FileWriter(mindrecord_file, mindrecord_partitions)

    # set the header size
    if mindrecord_header_size_by_bit != 24:
        header_size = 1 << mindrecord_header_size_by_bit
        mr_writer.set_header_size(header_size)

    # set the page size
    if mindrecord_page_size_by_bit != 25:
        page_size = 1 << mindrecord_page_size_by_bit
        mr_writer.set_page_size(page_size)

    # create the schema
    mr_writer.add_schema(mr_schema, "mindrecord_graph_schema")

    # open file and set header
    mr_writer.open_and_set_header()

    return mr_writer


def run_parallel_workers(num_tasks, mindrecord_workers, mindrecord_dict_data, graph_map_schema, writer):
    """
    run parallel workers
    """
    # set number of workers
    num_workers = mindrecord_workers

    task_list = list(range(num_tasks))

    if num_workers > num_tasks:
        num_workers = num_tasks

    if os.name == 'nt':
        for window_task_id in task_list:
            exec_task(window_task_id, mindrecord_dict_data, graph_map_schema, writer, False)
    elif num_tasks > 1:
        with Pool(num_workers) as p:
            p.map(exec_task, task_list)
    else:
        exec_task(0, mindrecord_dict_data, graph_map_schema, writer, False)


def writer_data(mindrecord_script="template",
                mindrecord_file="/tmp/mindrecord/xyz",
                mindrecord_partitions=1,
                mindrecord_header_size_by_bit=24,
                mindrecord_page_size_by_bit=25,
                mindrecord_workers=8,
                num_node_tasks=1,
                num_edge_tasks=1,
                graph_api_args="/tmp/nodes.csv:/tmp/edges.csv"):
    """Write the data in mindrecord format"""
    print(mindrecord_file)
    print(graph_api_args)
    if os.path.exists(os.path.join(mindrecord_file, mindrecord_script)):
        os.remove(os.path.join(mindrecord_file, mindrecord_script))
        os.remove(os.path.join(mindrecord_file, mindrecord_script + ".db"))

    mindrecord_file = os.path.join(mindrecord_file, mindrecord_script)
    start_time = time.time()

    # pass mr_api arguments
    os.environ['graph_api_args'] = graph_api_args

    try:
        mr_api = import_module('src.' + mindrecord_script + '.mr_api')
    except ModuleNotFoundError:
        raise RuntimeError("Unknown module path: {}".format(mindrecord_script + '.mr_api'))

    # init graph schema
    graph_map_schema = GraphMapSchema()

    num_features, feature_data_types, feature_shapes = mr_api.node_profile
    graph_map_schema.set_node_feature_profile(num_features, feature_data_types, feature_shapes)

    num_features, feature_data_types, feature_shapes = mr_api.edge_profile
    graph_map_schema.set_edge_feature_profile(num_features, feature_data_types, feature_shapes)

    graph_schema = graph_map_schema.get_schema

    # init writer
    writer = init_writer(graph_schema, mindrecord_file, mindrecord_partitions, mindrecord_header_size_by_bit,
                         mindrecord_page_size_by_bit)

    # write nodes data
    mindrecord_dict_data = mr_api.yield_nodes
    run_parallel_workers(num_node_tasks, mindrecord_workers, mindrecord_dict_data, graph_map_schema, writer)

    # write edges data
    mindrecord_dict_data = mr_api.yield_edges
    run_parallel_workers(num_edge_tasks, mindrecord_workers, mindrecord_dict_data, graph_map_schema, writer)

    # writer wrap up
    writer.commit()

    end_time = time.time()
    print("")
    print("END. Total time: {}".format(end_time - start_time))
    print("")
