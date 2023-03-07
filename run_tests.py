import subprocess
import csv
import json
import os

def run_model_hlscnn(name):
    ap_file = "test_in/hlscnn.yaml"
    mp_file = "test_in/hlscnn_mapspace.yaml"
    result = []

    conv_model_path = f"test_in/workload/hlscnn/{name}"
    numLayer = len(list(os.listdir(conv_model_path)))
    out_path = f"test_out/hlscnn/{name}"
    os.makedirs(out_path, exist_ok=True)

    for i in range(numLayer):
        pp_file = f"{conv_model_path}/conv{i}.yaml"
        out_file = f"{out_path}/conv{i}.json"
        cmd = [
            "cosa",
            "-o",
            out_file,
            "-ap",
            ap_file,
            "-mp",
            mp_file,
            "-pp",
            pp_file,
        ]
        print("\nrunning test:", pp_file)
        subprocess.run(cmd)
        with open(out_file, "r") as fin:
            layerResult = json.load(fin)
        result.append(layerResult)
    with open(f"{out_path}/summary.json", "w") as fout:
        json.dump(result, fout, indent=2)

def run_model_flexasr(name):
    ap_file = "test_in/flexasr.yaml"
    mp_file = "test_in/flexasr_mapspace.yaml"
    result = []

    conv_model_path = f"test_in/workload/flexasr/{name}"
    numLayer = len(list(os.listdir(conv_model_path)))
    out_path = f"test_out/flexasr/{name}"
    os.makedirs(out_path, exist_ok=True)

    for i in range(numLayer):
        pp_file = f"{conv_model_path}/gemm{i}.yaml"
        out_file = f"{out_path}/gemm{i}.json"
        cmd = [
            "cosa",
            "-o",
            out_file,
            "-ap",
            ap_file,
            "-mp",
            mp_file,
            "-pp",
            pp_file,
        ]
        print("\nrunning test:", pp_file, "with command: ", " ".join(cmd))
        subprocess.run(cmd)
        with open(out_file, "r") as fin:
            layerResult = json.load(fin)
        result.append(layerResult)
    with open(f"{out_path}/summary.json", "w") as fout:
        json.dump(result, fout, indent=2)


if __name__ == "__main__":
    # run cosa tests for models on hlscnn
    run_model_hlscnn("alexnet")
    run_model_hlscnn("vgg16")
    run_model_hlscnn("googlenet")
    run_model_hlscnn("inception_v3")
    run_model_hlscnn("resnet18")
    run_model_hlscnn("densenet121")
    run_model_hlscnn("mobilenet_v2")
    run_model_hlscnn("squeezenet")
    run_model_hlscnn("maskrcnn_resnet50")
    run_model_hlscnn("ssd300_vgg16")

    # run cosa tests for models on flexasr
    run_model_flexasr("alexnet")
    run_model_flexasr("vgg16")
    run_model_flexasr("googlenet")
    run_model_flexasr("inception_v3")
    run_model_flexasr("resnet18")
    run_model_flexasr("densenet121")
    run_model_flexasr("mobilenet_v2")
    run_model_flexasr("squeezenet")
    run_model_flexasr("maskrcnn_resnet50")
    run_model_flexasr("ssd300_vgg16")