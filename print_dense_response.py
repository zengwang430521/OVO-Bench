import json

# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_epoch_4_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_epoch_4_2_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_epoch_4_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_v5_epoch_1_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_v5_epoch_2_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_v5_2_epoch_1_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_v5_2_epoch_2_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
#
#
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_2_epoch_1_32r/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_3_epoch_1/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_5_epoch_1/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'

src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/baseline/QWen2VL_7B_V3/QWen2VL_7B_V3_REC_online_1.json'

accept_delay = 0


tar_file = src_file.replace('.json', '.txt')
with open(src_file, 'r', encoding='utf-8') as f:
    results = json.load(f)

total_test = 0
total_response = 0
total_silence = 0
total_correct_response = 0
total_extra_response_in_seg = 0
total_extra_response_out_seg = 0
total_covered_seg = 0
total_missed_seg = 0

with open(tar_file, 'w', encoding='utf-8') as f_tar:
    for idx, item in enumerate(results['forward']):
        if item['task'] != 'REC':
            continue
        all_responses = item['all_responses']
        end_time = all_responses[-1][0]
        end_time = max(end_time, item['test_info'][-1]['realtime'])

        times = list(range(int(end_time+1)))
        responses = [' '] * len(times)
        for tmp in all_responses:
            responses[int(tmp[0])] = tmp[1] if tmp[1] is not None else '-'

        related_spots = [''] * len(times)
        gts = [''] * len(times)

        for t_st, t_end in zip(item['start_times'], item['end_times']):
            for i, t in enumerate(times):
                if t_st < t < t_end:
                    if related_spots[i] == '':
                        related_spots[i] = '*'
            related_spots[int(t_st)] += '['
            related_spots[int(t_end)] += ']'

        for tmp in item['test_info']:
            t = tmp['realtime']
            count = tmp['count']
            gts[int(t)] = count

        times = [str(t) for t in times]

        # 计算每个元素的固定宽度
        col_width = 3  # 使数字对齐
        label_width = 6  # "times" 和 "responses" 的宽度
        max_columns = 100000  # 每行最多显示的列数

        for i in range(0, len(times), max_columns):
            # 取出当前块的数据
            times_chunk = times[i:i + max_columns]
            responses_chunk = responses[i:i + max_columns]
            related_spots_chunk = related_spots[i:i + max_columns]
            gt_chunk = gts[i:i + max_columns]

            # 格式化每行数据
            times_str = " ".join(f"{t:^{col_width}}" for t in times_chunk)
            responses_str = " ".join(f"{r:^{col_width}}" for r in responses_chunk)
            related_spots_str = " ".join(f"{r:^{col_width}}" for r in related_spots_chunk)
            gt_str = " ".join(f"{r:^{col_width}}" for r in gt_chunk)

            if i == 0:
                # 打印表头信息（只在第一块打印）
                header = "=" * 50 + "\n"
                header += f"ID: {item['id']}\n"
                header += f"Activity: {item['activity']}\n"
                header += "=" * 50 + "\n\n"

                print(header)
                f_tar.write(header)

            # 组织数据内容
            content = (
                f"{'time'.ljust(label_width)}{times_str}\n"
                f"{'res'.ljust(label_width)}{responses_str}\n"
                f"{'act'.ljust(label_width)}{related_spots_str}\n"
                f"{'gt'.ljust(label_width)}{gt_str}\n\n"
            )
            print(content)
            f_tar.write(content)




        # 统计正确率
        target_segs = [[t_st, t_end] for t_st, t_end in zip(item['start_times'], item['end_times'])]
        response_times = [t for t, r in all_responses if r is not None]

        # 计算目标段落的回复情况
        num_test = len(all_responses)
        num_response = len(response_times)
        num_silence = num_test - num_response
        num_correct_response = 0
        num_extra_response_in_seg = 0
        num_extra_response_out_seg = 0
        num_covered_seg = 0
        num_missed_seg = 0

        target_responses = {tuple(seg): [] for seg in target_segs}
        for t in response_times:
            in_target = False
            for i, seg in enumerate(target_segs):
                if seg[0] <= t <= seg[1]+accept_delay:
                    target_responses[tuple(seg)].append(t)
                    in_target = True
                    break
            if not in_target:
                num_extra_response_out_seg += 1

        for seg, times in target_responses.items():
            if times:
                num_correct_response += 1
                num_covered_seg += 1
                if len(times) > 1:
                    num_extra_response_in_seg += len(times) - 1
            else:
                num_missed_seg += 1

        total_test += num_test
        total_response += num_response
        total_silence += num_silence
        total_correct_response += num_correct_response
        total_extra_response_in_seg += num_extra_response_in_seg
        total_extra_response_out_seg += num_extra_response_out_seg
        total_covered_seg += num_covered_seg
        total_missed_seg += num_missed_seg

        log_text = (f"总共测试次数: {num_test}\n"
                    f"回复的个数: {num_response}\n"
                    f"沉默的个数: {num_silence}\n"
                    f"正确回复的个数: {num_correct_response}\n"
                    f"目标段落里多余的回复个数: {num_extra_response_in_seg}\n"
                    f"目标段落外，错误的回复个数: {num_extra_response_out_seg}\n"
                    f"有回复的目标段落的个数: {num_covered_seg}\n"
                    f"无回复的目标段落的个数: {num_missed_seg}\n")
        print(log_text)  # 打印到控制台
        f_tar.write(log_text)  # 写入文件

        print('\n\n')
        f_tar.write('\n\n\n')

        t = 0

        # if idx >= 10:
        #     break

    log_text = (
        f"{'=' * 100}\n"
        f"汇总:\n\n"
        f"总共测试次数: {total_test}\n"
        f"回复的个数: {total_response}\n"
        f"沉默的个数: {total_silence}\n"
        f"正确回复的个数: {total_correct_response}\n"
        f"目标段落里多余的回复个数: {total_extra_response_in_seg}\n"
        f"目标段落外，错误的回复个数: {total_extra_response_out_seg}\n"
        f"有回复的目标段落的个数: {total_covered_seg}\n"
        f"无回复的目标段落的个数: {total_missed_seg}\n\n"
    
        f"回复率: {total_response} / {total_test} = {total_response / total_test}\n"
        f"回复正确率: {total_correct_response} / {total_response} = {total_correct_response / total_response}\n"
        f"段落召回率: {total_covered_seg} / {total_covered_seg + total_missed_seg} = {total_covered_seg / (total_covered_seg + total_missed_seg)}\n"
        f"正确回复\t 内部错误回复\t 外部错误回复\t\n"
        f"{total_correct_response}\t {total_extra_response_in_seg}\t {total_extra_response_out_seg}\t\n"

    )
    print(log_text)  # 打印到控制台
    f_tar.write(log_text)  # 写入文件



import argparse
from utils.OVOBenchScore import OVOBenchOfflineScore, OVOBenchOnlineScore

parser = argparse.ArgumentParser(description='Eval OVBench')
parser.add_argument("--result_dir", type=str, default="results", help="Root directory of results")
parser.add_argument("--model", type=str, default='None', help="Model to evaluate")
parser.add_argument("--force_response", action="store_true")
args = parser.parse_args()

results = {
    "backward": [],
    "realtime": [],
    "forward": []
}
with open(src_file, "r") as f:
    result = json.load(f)
    results["backward"] += result["backward"]
    results["realtime"] += result["realtime"]
    results["forward"] += result["forward"]

score_model = OVOBenchOnlineScore(args, results)
score_model.score()
