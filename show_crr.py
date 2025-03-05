import json
from utils.OVOBenchScore import plot_histogram
import numpy as np


src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/stream_v2_lora_10_dense/QWen2VLStream_7B_v2/QWen2VLStream_7B_v2_CRR_online_1.json'

with open(src_file, 'r') as f:
    data = json.load(f)

offset_gt, offset_pred = [],  []
all_scores = []
for item in data["forward"]:
    # print(item)

    ask_time = item['ask_time']
    clue_time = item['clue_time']
    if len(item['all_responses']) > 0:
        response_time = int(item['all_responses'][0][0])
    else:
        response_time = 1000

    test_times = [tmp['realtime'] for tmp in item['test_info']]
    scores = []
    for t in test_times:
        if t<clue_time and t < response_time:
            scores.append(1)
        elif t>=clue_time and t >= response_time:
            scores.append(1)
        else:
            scores.append(0)
    
    # print('-'*100)
    # print(f"ID:{item['id']}")
    # print(f"{item['ask_time']}\tQ:{item['question']}")
    # print(f"{item['clue_time']}\tGT:{item['answer']}")
    # if len(item['all_responses']) > 0:
    #     print(f"{int(item['all_responses'][0][0])}\tA:{item['all_responses'][0][1]}")
    # else:
    #     print(f"None A:None")
    # print(test_times, sum(scores))
    
    print(clue_time-ask_time, response_time-ask_time)
    offset_gt.append(clue_time-ask_time)
    if len(item['all_responses']) > 0:
        offset_pred.append(response_time-ask_time)
    else:
        offset_pred.append(100)
    all_scores.append(sum(scores))

plot_histogram(offset_gt, bins=100, save_path="gt.png", range=(0,100))
plot_histogram(offset_pred, bins=100, save_path="pred.png", range=(0,100))

all_scores = np.array(all_scores)
offset_gt, offset_pred = np.array(offset_gt), np.array(offset_pred)
print(all_scores.mean()/5)
print(all_scores[offset_gt<=40].mean()/5)
print(all_scores[offset_gt>40].mean()/5)
print((offset_pred<offset_gt).mean())
print((offset_pred<10).mean())
print((offset_gt<10).mean())

t = 0