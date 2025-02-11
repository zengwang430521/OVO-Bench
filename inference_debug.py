"""
    Inference and save results to results/[model]/
"""

import argparse
import os
import json
from models import *
import os

parser = argparse.ArgumentParser(description='Run OVBench')
parser.add_argument("--anno_path", type=str, default="data/ovo_bench.json", help="Path to the annotations")
parser.add_argument("--video_dir", type=str, default="", help="Root directory of source videos")
parser.add_argument("--result_dir", type=str, default="results", help="Root directory of results")
parser.add_argument("--mode", type=str, required=True, choices=["online", "offline"], help="Online of Offline model for testing")
parser.add_argument("--task", type=str, required=False, nargs="+", \
                    choices=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"], \
                    default=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"], \
                    help="Tasks to evaluate")
parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
parser.add_argument("--save_results", type=bool, default=True, help="Save results to a file")

# For GPT init, use GPT-4o as default
parser.add_argument("--gpt_api", type=str, required=False, default=None)
# For Geimini init, use Gemini 1.5-pro as default
parser.add_argument("--gemini_project", type=str, required=False, default=None)
# For local running model init
parser.add_argument("--model_path", type=str, required=False, default=None)
parser.add_argument("--lora_path", type=str, required=False, default=None)

args = parser.parse_args()


print(f"Inference Model: {args.model}; Task: {args.task}")

# import pdb; pdb.set_trace()


with open(args.anno_path, "r") as f:
    annotations = json.load(f)

for i, item in enumerate(annotations):
    annotations[i]["video"] = os.path.join(args.video_dir, item["video"])

backward_anno = []
realtime_anno = []
forward_anno = []
backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

tasks = set(args.task)
for anno in annotations:
    if anno["task"] in tasks:
        if anno["task"] in backward_tasks:
            backward_anno.append(anno)
        if anno["task"] in realtime_tasks:
            realtime_anno.append(anno)
        if anno["task"] in forward_tasks:
            forward_anno.append(anno)

        tasks.remove(anno["task"]) # 每种task只需要1个，便于debug

anno = {
    # "backward": backward_anno[len(backward_anno)//2:],
    "backward": backward_anno,
    "realtime": realtime_anno,
    "forward": forward_anno
}


if args.model == "GPT":
    from models.GPT import EvalGPT
    assert not args.gpt_api == None
    model = EvalGPT(args)
elif args.model == "Gemini":
    from models.Gemini import EvalGemini
    assert not args.gemini_project == None
    model = EvalGemini(args)
elif args.model == "InternVL2":
    from models.InternVL2 import EvalInternVL2
    assert os.path.exists(args.model_path)
    model = EvalInternVL2(args)
elif args.model == "QWen2VL_7B":
    from models.QWen2VL import EvalQWen2VL
    assert os.path.exists(args.model_path)
    model = EvalQWen2VL(args)
elif args.model == "QWen2VL_7B_V2":
    from models.QWen2VL_v2 import EvalQWen2VL2
    assert os.path.exists(args.model_path)
    model = EvalQWen2VL2(args)
elif args.model == "QWen2VLStream_7B":
    from models.QWen2VL_Stream import EvalQWen2VLStream
    assert os.path.exists(args.model_path)
    model = EvalQWen2VLStream(args)
elif args.model == "QWen2VLStream_7B_v2":
    from models.QWen2VL_Stream_v2 import EvalQWen2VLStreamV2
    assert os.path.exists(args.model_path)
    model = EvalQWen2VLStreamV2(args)
else:
    raise ValueError(f"Unsupported model: {args.model}. Please implement the model.")

model.eval(anno, args.task, args.mode)