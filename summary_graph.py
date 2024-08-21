from models.deebert.src.berthighway import BertHighway
import collections
import onnx
import networkx as nx
import torch
import torch.nn as nn
import torchvision
import argparse
from torch.onnx import TrainingMode
import pprint
import models
from decimal import Decimal
import os
import sys
import pickle
import utils
sys.path.insert(1, os.path.join(os.getcwd(), 'profiling'))
from profiler import TIDSProfiler

def map_torch_to_onnx(op_type):
    """ Map PyTorch operator to ONNX operator

    Args:
        op_type (str): PyTorch nn module type

    Returns:
        str: ONNX operator type
    """
    if "Linear" in op_type:
        return "Gemm"
    elif "Conv" in op_type:
        return "Conv"
    elif "BatchNorm" in op_type:
        return "BatchNormalization"
    elif "AdaptiveAvgPool2d" in op_type:
        return "GlobalAveragePool"
    elif "MaxPool" in op_type:
        return "MaxPool"
    elif "AvgPool" in op_type:
        return "AveragePool"
    elif "ReLU" in op_type:
        return "Relu"
    elif "Sigmoid" in op_type:
        return "Sigmoid"
    elif "Tanh" in op_type:
        return "Tanh"
    elif "Dropout" in op_type:
        return "Dropout"
    elif "Embedding" in op_type:
        return "Gather"
    else:
        return None
        # raise NotImplementedError("Operator {} not implemented".format(op_type))


def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output


def get_op_type(op_type, node_name):
    temp_op = op_type
    if node_name:
        for annotation in set(['key', 'value', 'query']):
            if annotation in node_name:
                temp_op = temp_op+'_'+annotation

    return temp_op


def get_tensor_shapes(model_graph):

    node_shapes = dict()
    num_of_trainable_tensors = 0

    if model_graph.initializer:
        for init in model_graph.initializer:
            if '.weight' in init.name:
                num_of_trainable_tensors += 1
            # print(init.name, init.dims)
            node_shapes[init.name] = tuple(init.dims)
    else:
        for node in model_graph.input:
            node_shapes[node.name] = tuple(
                [p.dim_value for p in node.type.tensor_type.shape.dim])

    return node_shapes, num_of_trainable_tensors


def split_inputs(in_list):
    # input list may contain trainable weights
    input_nodes = []
    layer_name = None
    for _input in in_list:
        # tensor nodes are numeric by default
        # if _input.isnumeric():
        input_nodes.append(_input)
        # in onnx model, weight comes ahead of other trainable weights
        # in some cases, bias itself may be a tensor
        if '.weight' in _input:
            layer_name = _input
            # break
        elif layer_name is None and '.bias' in _input:
            layer_name = _input
            # break
    return input_nodes, layer_name


def onnx_layer_to_torch_layer(outputs):
    """
    output_nodes: a list of output nodes
    """

    assert len(outputs) == 1, "Node has more than one output"

    l = outputs[0].split('/')[1:-1]

    if len(l) == 0:
        return None

    res = [l[0]]

    if len(l) == 1:
        return res[0]
    
    for i in range(1, len(l)):
        if res[-1] in l[i]:
            res[-1] = l[i]
        else:
            res.append(l[i])

    layer_name = '.'.join(res)
    return layer_name


def load_model_meta(meta_file='sample__accuracy.onnx'):
    """
    meta_file: input files are onnx. return the weight meta graph of this model
    """

    # meta file is rather small
    onnx_model = onnx.load(meta_file)
    model_graph = onnx_model.graph

    # record the shape of each weighted nodes
    node_shapes, num_of_trainable_tensors = get_tensor_shapes(model_graph)
    # construct the computation graph and align their attribution
    nodes = [
        n for n in onnx_model.graph.node
         if n.op_type != 'Constant' and n.op_type != 'Identity'] 
    graph = nx.DiGraph(
        name=meta_file,
        num_tensors=num_of_trainable_tensors,
        num_nodes=len(nodes))

    edge_source = collections.defaultdict(list)

    opt_dir = collections.defaultdict(int)
    input_nodes_list = []
    for idx, node in enumerate(nodes):
        input_nodes, trainable_weights = split_inputs(node.input)

        opt_dir[node.op_type] += 1
        layer_name = onnx_layer_to_torch_layer(node.output)
        attr = {
            'dims': [] if not trainable_weights else node_shapes[trainable_weights],
            'op_type': get_op_type(node.op_type, layer_name),
            'name': node.name,
            'layer_name': layer_name,
            'path_weight': Decimal(0.0)
        }
        graph.add_node(idx, attr=attr)
        # register node
        for out_node in node.output:
            edge_source[out_node].append(idx)
        input_nodes_list.append(input_nodes)

    for idx, node in enumerate(nodes):
        input_nodes = input_nodes_list[idx]
        # add edges
        for input_node in input_nodes:

            for s in edge_source[input_node]:
                graph.add_edge(s, idx)
    return graph, onnx_model


def dfs_iterative(start_vertex, graph, ret=[], in_degrees=None):
    stack = [start_vertex]

    while stack:
        vertex = stack.pop()
        ret.append(vertex)

        temp_out = []
        for edge in graph.out_edges(vertex):
            if in_degrees[edge[1]] == 1:
                temp_out.append(edge[1])
                del in_degrees[edge[1]]
            else:
                in_degrees[edge[1]] -= 1

        stack += temp_out


def topological_sorting(graph):
    """DFS based topological sort to maximize length of each chain"""
    ret = []
    in_degrees = {n: graph.in_degree(n)
                  for n in graph.nodes if graph.in_degree(n) > 0}

    [dfs_iterative(node, graph, ret, in_degrees)
     for node in graph.nodes() if graph.in_degree(node) == 0]
    assert len(ret) == graph.number_of_nodes()

    return ret


def get_bottleneck_nodes(graph):

    topo_order = topological_sorting(graph)

    root = graph.nodes[0]
    root['attr']['path_weight'] = Decimal(1)
    queue = topo_order

    while queue:
        node = queue.pop(0)
        for out_edge in graph.out_edges(node):
            child = out_edge[1]
            graph.nodes[child]['attr']['path_weight'] += \
                graph.nodes[node]['attr']['path_weight'] / \
                Decimal(len(graph.out_edges(node)))

    res = []

    for idx in graph.nodes():
        if float(graph.nodes[idx]['attr']['path_weight']) == 1.0 \
                and graph.in_degree(idx) == 1:
            if idx < 5 or abs(idx - len(graph.nodes())) < 7:
                print("Too early or too late to add a ramp at node",
                      idx, graph.nodes[idx]['attr'])
            else:
                res.append(idx)
    return res


def get_profile_node_list(profile):
    """
    get the list of profile node
    Args
        profile (class Profiler): the profile of the model
    Returns
        node_list (list): the list of profile node
    """
    nodes = []

    def _get_leaf_nodes(node, res):
        if node is not None:
            if len(node.children) == 0:
                res.append(node)
            for n in node.children:
                _get_leaf_nodes(n, res)
    _get_leaf_nodes(profile, nodes)
    # for n in nodes:
    #     print(n.full_name, n.type, n.output_shape)
    return nodes


def find_node_by_child(profile, child_info, ordered_node_list):
    """
    find the node name by its child node info

    Args
        profile (class Profiler): the profile of the model
        child_info (list): [child_name, child_op_type] both are from ONNX model
        ordered_node_list (list): the list of profile node ordered by the forward order
    Returns
        node: the profile node that contains the child node
    """
    if child_info[0] != None:
        for i, node in enumerate(ordered_node_list):
            if node.full_name == child_info[0]:
                return ordered_node_list[i-1]
    else:
        for i, node in enumerate(ordered_node_list):
            # print(node.type, map_torch_to_onnx(node.type), child_info[1], map_torch_to_onnx(node.type) == child_info[1])
            if map_torch_to_onnx(node.type) == child_info[1]:
                # and \
                #     map_torch_to_onnx(ordered_node_list[i-1].type) == child_info[1]:
                return ordered_node_list[i-1]
    return None

def generate_exit_ramps(insert_ramp_nodes, num_classes=3):
    """
    generate the exit ramp for each node in the insert_ramp_nodes list
    Args
        insert_ramp_nodes (list): ['layer_name', 'op_type (pytorch)', output_shape]
    Returns
        exit_ramps (list): the list of nn Sequential that contains the exit ramp
    """
    all_possible_ramps = []

    for node_info in insert_ramp_nodes:
        all_possible_ramps += [
            (
                node_info[0],
                nn.Sequential(
                    # nn.Conv2d(node_info[2][1], 64, kernel_size=3,
                    #           stride=1, padding=1, bias=True),
                    # nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(node_info[2][1], num_classes)
                )

            )
        ]
    return all_possible_ramps


def get_output_shape(profile, node_names):
    """
    get the output shape of the model

    Args
        profile (class Profiler): the profile of the model
        node_names (list): the list of node names
    Returns
        output_shape (list): the output shape of the model
    """
    res = {}
    nodes = [profile]
    while nodes:
        node = nodes.pop(0)

        if node.full_name in node_names:
            res[node.full_name] = node.output_shape
        
        if len(node.children) == 0:
            continue
        else:
            for child in node.children:
                nodes.append(child)
    return res


def get_exits_def(model, arch, ids, model_profile_path, num_classes=3, bert_config=None, module_name_prefix=None, dataset='video'):
    """
    get the exit ramp definition for given ramp ids

    Args
        model (nn.Module): the model to be profiled
        arch (str): the name of the model
        ids (list): the list of ramp ids
        model_profile_path (str): the path to the model profile
        bert_config (transformers.{BertConfig,RobertaConfig,DistilBertConfig}): 
            config required for initializing bert ramps
        module_name_prefix (str): the nn.module name prefix of the ramps

    Returns
        list of nn.Sequential: the list of exit ramps
    """
    if arch in utils.all_cv_models or dataset == 'video':  # CV
        dummy_input = torch.randn(1, 3, 224, 224)
        export_path = arch+".onnx"
        # if not os.path.exists(export_path):
        torch.onnx.export(model, dummy_input, export_path,
                          export_params=True, verbose=0, do_constant_folding=False)

        graph, _ = load_model_meta(export_path)
        bottleneck_nodes = get_bottleneck_nodes(graph)
        
        with open(model_profile_path, "rb") as f:
            profile = pickle.load(f)

        insert_ramp_nodes = []

        node_names = [graph.nodes[bottleneck_node]['attr']['layer_name'] for bottleneck_node in bottleneck_nodes]
        output_shape_map = get_output_shape(profile, node_names)

        for bottleneck_node in bottleneck_nodes:
            layer_name = graph.nodes[bottleneck_node]['attr']['layer_name']
            insert_ramp_nodes.append([layer_name, \
                                      graph.nodes[bottleneck_node]['attr']['op_type'], \
                                      output_shape_map[layer_name]]
                                    )
        exit_ramps = generate_exit_ramps(insert_ramp_nodes, num_classes)
        print('number of exit ramps:', len(exit_ramps))

    elif arch in utils.all_nlp_models:  # NLP
        exit_ramps = []
        if hasattr(bert_config, "num_hidden_layers"):  # bert
            num_encoders = bert_config.num_hidden_layers
        elif hasattr(bert_config, "n_layers"):  # distilbert
            num_encoders = bert_config.n_layers
        else:
            raise NotImplementedError
        for ramp_id in range(num_encoders):
            module_name = f"{module_name_prefix}.{ramp_id}"
            branch_net = BertHighway(bert_config)
            exit_ramps.append((module_name, branch_net,))
    else:
        raise NotImplementedError
    if ids[0] == -1:
        return exit_ramps
    else:
        return [exit_ramps[i] for i in ids]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--export_path', type=str, default='resnet18.onnx')
    args = parser.parse_args()

    dummy_input = torch.randn(1, 3, 224, 224)
    # model = torchvision.models.resnet50(pretrained=True)
    # model = models.waymo.resnet18_waymo(pretrained=True)
    model = models.urban.resnet18_urban(pretrained=True)

    get_exits_def(model, "resnet18_urban", [
                  8], "./profile_pickles/resnet18_urban_profile.pickle")
