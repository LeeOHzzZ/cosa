import yaml
import csv
import os
from math import ceil


def gen_conv2d_workload_yaml_from_csv(infile, out_path):
    fin = open(infile, "r")
    csv_reader = csv.DictReader(fin)
    cntr = 0
    for i, layer in enumerate(csv_reader):
        if int(layer["groups"]) > 1:
            print("Skipping conv2d layer with grouping: ", layer)
            continue
        assert int(layer["groups"]) == 1, f"{layer}, Grouped Conv2D is not supported"
        assert int(layer["dilation_h"]) == 1 and int(layer["dilation_w"]) == 1
        workload = f"""
        problem:
            C: {layer['in_chans']}
            K: {layer['out_chans']}
            N: 1
            P: {layer['output_w']}
            Q: {layer['output_h']}
            R: {layer['kernel_w']}
            S: {layer['kernel_h']}
            Padding: {layer['padding_h']}
            Hdilation: 1
            Hstride: {layer['stride_h']}
            Wdilation: 1
            Wstride: {layer['stride_w']}
            shape: cnn-layer
        """
        with open(f"{out_path}/conv{cntr}.yaml", "w") as fout:
            yaml.dump(yaml.safe_load(workload), fout)
        cntr += 1

    print(f"Generated {cntr} conv layer workloads")


def gen_conv2d_im2col_workload_yaml_from_csv_for_flexasr(infile, out_path):
    # generating 
    fin = open(infile, "r")
    csv_reader = csv.DictReader(fin)
    cntr = 0
    for i, layer in enumerate(csv_reader):
        if int(layer["groups"]) > 1:
            print("Skipping conv2d layer with grouping: ", layer)
            continue
        assert int(layer["groups"]) == 1, f"{layer}, Grouped Conv2D is not supported"
        assert int(layer["dilation_h"]) == 1 and int(layer["dilation_w"]) == 1
        # use P dimension as batch dimension
        p = int(layer["output_h"]) * int(layer["output_w"])
        p = ceil(p / 16) * 16 # pad the batch dimension to 16
        # use Q dimension as output dimension
        q = int(layer["out_chans"])
        # use C dimension as input dimension
        c = int(layer["kernel_h"]) * int(layer["kernel_w"]) * int(layer["in_chans"])
        workload = f"""
        problem:
            C: {c}
            K: 1
            N: 1
            P: {p}
            Q: {q}
            R: 1
            S: 1
            Padding: 0
            Hdilation: 1
            Hstride: 1
            Wdilation: 1
            Wstride: 1
            shape: cnn-layer
        """
        with open(f"{out_path}/gemm{cntr}.yaml", "w") as fout:
            yaml.dump(yaml.safe_load(workload), fout)
        cntr += 1

    print(f"Generated {cntr} conv layer workloads")


def gen_conv2d_im2col_workload_yaml_from_csv_for_vta(infile, out_path):
    # generating 
    fin = open(infile, "r")
    csv_reader = csv.DictReader(fin)
    cntr = 0
    for i, layer in enumerate(csv_reader):
        if int(layer["groups"]) > 1:
            print("Skipping conv2d layer with grouping: ", layer)
            continue
        assert int(layer["groups"]) == 1, f"{layer}, Grouped Conv2D is not supported"
        assert int(layer["dilation_h"]) == 1 and int(layer["dilation_w"]) == 1
        # use P dimension as batch dimension
        p = int(layer["output_h"]) * int(layer["output_w"])
        # use Q dimension as output dimension
        q = int(layer["out_chans"])
        q = ceil(q / 16) * 16 # pad the y dimension to 16
        # use C dimension as input dimension
        c = int(layer["kernel_h"]) * int(layer["kernel_w"]) * int(layer["in_chans"])
        c = ceil(c / 16) * 16 # pad the x dimension to 16
        workload = f"""
        problem:
            C: {c}
            K: 1
            N: 1
            P: {p}
            Q: {q}
            R: 1
            S: 1
            Padding: 0
            Hdilation: 1
            Hstride: 1
            Wdilation: 1
            Wstride: 1
            shape: cnn-layer
        """
        with open(f"{out_path}/gemm{cntr}.yaml", "w") as fout:
            yaml.dump(yaml.safe_load(workload), fout)
        cntr += 1

    print(f"Generated {cntr} conv layer workloads")


def gen_timeloop_conv2d_configs_from_csv(infile):
    fin = open(infile, "r")
    csv_reader = csv.DictReader(fin)
    cntr = 0
    cnn_layers = []
    for i, layer in enumerate(csv_reader):
        if int(layer["groups"]) > 1:
            print("Skipping conv2d layer with grouping: ", layer)
            continue
        assert int(layer["groups"]) == 1
        cnn_layers.append(
            (
                int(layer["input_w"]),  # W
                int(layer["input_h"]),  # H
                int(layer["in_chans"]),  # C
                1,  # N
                int(layer["out_chans"]),  # K
                int(layer["kernel_h"]),  # S
                int(layer["kernel_w"]),  # R
                int(layer["padding_w"]),  # Wpad
                int(layer["padding_h"]),  # Hpad
                int(layer["stride_w"]),  # Wstride
                int(layer["stride_h"]),  # Hstride
            )
        )
    print(cnn_layers)

def gen_model_workload_for_hlscnn(name):
    path_in = f"test_in/workload/models/{name}/{name}_non_grouped_conv_layers.csv"
    path_out = f"test_in/workload/hlscnn/{name}/"
    os.makedirs(path_out, exist_ok=True)
    print("generating workload yaml files for ", name)
    gen_conv2d_workload_yaml_from_csv(path_in, path_out)

def gen_model_workload_for_flexasr(name):
    path_in = f"test_in/workload/models/{name}/{name}_non_grouped_conv_layers.csv"
    path_out = f"test_in/workload/flexasr/{name}/"
    os.makedirs(path_out, exist_ok=True)
    print("generating workload yaml files for ", name)
    gen_conv2d_im2col_workload_yaml_from_csv_for_flexasr(path_in, path_out)

def gen_model_workload_for_vta(name):
    path_in = f"test_in/workload/models/{name}/{name}_non_grouped_conv_layers.csv"
    path_out = f"test_in/workload/vta/{name}/"
    os.makedirs(path_out, exist_ok=True)
    print("generating workload yaml files for ", name)
    gen_conv2d_im2col_workload_yaml_from_csv_for_vta(path_in, path_out)


if __name__ == "__main__":
    # gen workload for hlscnn
    gen_model_workload_for_hlscnn("alexnet")
    gen_model_workload_for_hlscnn("vgg16")
    gen_model_workload_for_hlscnn("googlenet")
    gen_model_workload_for_hlscnn("inception_v3")
    gen_model_workload_for_hlscnn("resnet18")
    gen_model_workload_for_hlscnn("densenet121")
    gen_model_workload_for_hlscnn("mobilenet_v2")
    gen_model_workload_for_hlscnn("squeezenet")
    gen_model_workload_for_hlscnn("maskrcnn_resnet50")
    gen_model_workload_for_hlscnn("ssd300_vgg16")

    # gen workload for flexasr
    gen_model_workload_for_flexasr("alexnet")
    gen_model_workload_for_flexasr("vgg16")
    gen_model_workload_for_flexasr("googlenet")
    gen_model_workload_for_flexasr("inception_v3")
    gen_model_workload_for_flexasr("resnet18")
    gen_model_workload_for_flexasr("densenet121")
    gen_model_workload_for_flexasr("mobilenet_v2")
    gen_model_workload_for_flexasr("squeezenet")
    gen_model_workload_for_flexasr("maskrcnn_resnet50")
    gen_model_workload_for_flexasr("ssd300_vgg16")

    # gen workload for vta
    gen_model_workload_for_vta("alexnet")
    gen_model_workload_for_vta("vgg16")
    gen_model_workload_for_vta("googlenet")
    gen_model_workload_for_vta("inception_v3")
    gen_model_workload_for_vta("resnet18")
    gen_model_workload_for_vta("densenet121")
    gen_model_workload_for_vta("mobilenet_v2")
    gen_model_workload_for_vta("squeezenet")
    gen_model_workload_for_vta("maskrcnn_resnet50")
    gen_model_workload_for_vta("ssd300_vgg16")