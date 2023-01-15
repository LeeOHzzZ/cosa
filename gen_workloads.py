import yaml
import csv


def gen_workload_yaml_from_csv(infile, out_path):
    fin = open(infile, "r")
    csv_reader = csv.DictReader(fin)
    cntr = 0
    for i, layer in enumerate(csv_reader):
        if int(layer["groups"]) > 1:
            print("Skipping conv2d layer with grouping: ", layer)
            continue
        assert int(layer["groups"]) == 1, f"{layer}, Grouped Conv2D is not supported"
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

if __name__ == "__main__":
    # gen_workload_yaml_from_csv(
    #     "test_in/workload/resnet18/resnet18_conv_layers.csv",
    #     "test_in/workload/resnet18/",
    # )
    gen_workload_yaml_from_csv(
        "test_in/workload/mobilenetv2/mobilenetv2_conv_layers.csv",
        "test_in/workload/mobilenetv2/",
    )