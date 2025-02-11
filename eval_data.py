import json
from os.path import exists
from tqdm import tqdm

data_file = 'data/ovo_bench.json'
with open(data_file, 'r') as f:
    data = json.load(f)

for item in tqdm(data):
    video_file = item['video']
    task = item['task']
    if not exists(video_file):
        print(f"{task}: {video_file}")

