import subprocess
import csv
import json

def test_vgg16_hlscnn():
    ap_file = "test_in/hlscnn.yaml"
    mp_file = "test_in/hlscnn_mapspace.yaml"
    numLayer = 13
    for i in range(numLayer):
        pp_file = f"test_in/workload/vgg16/conv{i+1}.yaml"
        out_file = f"test_out/hlscnn/vgg16/conv{i+1}.json"
        cmd = [
            "cosa",
            "-o", out_file,
            "-ap", ap_file,
            "-mp", mp_file,
            "-pp", pp_file,
        ]
        subprocess.run(cmd)

def summary_vgg16_hlscnn():
    # fieldnames = ["tile_sizes", "loopOrder", "timeToSolution"]
    fout = open("test_out/hlscnn/vgg16/summary.json", "w")
    # csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # csvwriter.writeheader()
    result = []
    numLayer = 13
    for i in range(numLayer):
        with open(f"test_out/hlscnn/vgg16/conv{i+1}.json", "r") as fin:
            layerResult = json.load(fin)
        result.append(layerResult)
    json.dump(result, fout, indent=4)

test_vgg16_hlscnn()
summary_vgg16_hlscnn()
