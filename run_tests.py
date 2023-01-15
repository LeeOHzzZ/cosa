import subprocess
import csv
import json


def test_vgg16_hlscnn():
    ap_file = "test_in/hlscnn.yaml"
    mp_file = "test_in/hlscnn_mapspace.yaml"
    numLayer = 13
    result = []
    for i in range(numLayer):
        pp_file = f"test_in/workload/vgg16/conv{i+1}.yaml"
        out_file = f"test_out/hlscnn/vgg16/conv{i+1}.json"
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
        subprocess.run(cmd)
        with open(out_file, "r") as fin:
            layerResult = json.load(fin)
        result.append(layerResult)
    with open("test_out/hlscnn/vgg16/summary.json", "w") as fout:
        json.dump(result, fout, indent=4)

def test_resnet18_hlscnn():
    ap_file = "test_in/hlscnn.yaml"
    mp_file = "test_in/hlscnn_mapspace.yaml"
    result = []

    numLayer = 20
    for i in range(numLayer):
        pp_file = f"test_in/workload/resnet18/conv{i}.yaml"
        out_file = f"test_out/hlscnn/resnet18/conv{i}.json"
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
    with open("test_out/hlscnn/resnet18/summary.json", "w") as fout:
        json.dump(result, fout, indent=4)

def test_mobilenetv2_hlscnn():
    ap_file = "test_in/hlscnn.yaml"
    mp_file = "test_in/hlscnn_mapspace.yaml"
    result = []

    numLayer = 35
    for i in range(numLayer):
        pp_file = f"test_in/workload/mobilenetv2/conv{i}.yaml"
        out_file = f"test_out/hlscnn/mobilenetv2/conv{i}.json"
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
    with open("test_out/hlscnn/mobilenetv2/summary.json", "w") as fout:
        json.dump(result, fout, indent=4)

if __name__ == "__main__":
    # test_vgg16_hlscnn()
    # test_resnet18_hlscnn()
    test_mobilenetv2_hlscnn()