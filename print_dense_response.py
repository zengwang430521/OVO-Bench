import json
import matplotlib.pyplot as plt
import matplotlib
# 设置支持中文字体（自动使用系统中存在的字体）
matplotlib.rcParams['font.sans-serif'] = ['AR PL UKai CN']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
from collections import Counter
import math
import numpy as np


# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_epoch_4_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_epoch_4_2_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_epoch_4_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_v5_epoch_1_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_v5_epoch_2_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'

# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_v5_2_epoch_1_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_v5_2_epoch_2_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_2_epoch_1_32r/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_2_epoch_1_lr/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'

src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/rec_stream_v5_2_epoch_1_dense/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_3_epoch_1/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_4_epoch_1/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_5_epoch_1/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_6_epoch_1/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'
# src_file = '/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results/dense/rec_stream_v5_7_epoch_1/QWen2VLStream_7B_v3_align/QWen2VLStream_7B_v3_align_REC_online_1.json'


accept_delay = 0
flag_plot = False
flag_score = True
flag_re_count = False
# flag_re_count = True


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
total_seg_infos = []
total_extra_response_out_distances = []

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

                # 计算与所有段落的距离
                distances = []
                for seg in target_segs:
                    if t < seg[0]:
                        # 提前
                        dist = t - seg[0]
                    elif t > seg[1] :
                        # 落后
                        dist = t - seg[1]
                    else:
                        dist = 0  # 不应该进来这里，但加个保险
                    distances.append(dist)
                min_dist = min(abs(d) for d in distances)
                closest_dists = [d for d in distances if abs(d) == min_dist]
                total_extra_response_out_distances.extend(closest_dists)

        for seg, times in target_responses.items():
            total_seg_infos.append({
                "time": seg,
                "res_times": times
            })

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


if flag_plot:
    '''画直方图'''
    # 配置：你可以在这里调整阈值
    THRESHOLD = 20
    histogram_file = src_file.replace('.json', 'time_offset.png')

    # 分桶处理：<-THRESHOLD 为一组，>THRESHOLD 为一组，其余保留原始值
    # 分桶
    binned = []
    for val in total_extra_response_out_distances:
        if val < -THRESHOLD:
            binned.append(f"<-{THRESHOLD}")
        elif val > THRESHOLD:
            binned.append(f">{THRESHOLD}")
        else:
            binned.append(str(round(val)))

    # 统计频次
    counter = Counter(binned)

    # 构造横坐标顺序（从 <-THRESHOLD 到 >THRESHOLD）
    x_labels = [f"<-{THRESHOLD}"] + [str(i) for i in range(-THRESHOLD, THRESHOLD + 1)] + [f">{THRESHOLD}"]
    y_counts = [counter.get(label, 0) for label in x_labels]

    # 绘图
    plt.figure(figsize=(14, 6))
    plt.bar(x_labels, y_counts)
    plt.xlabel("时间差（秒）")
    plt.ylabel("频次")
    plt.title("错误回复距离最近段落的时间差分布（含提前与延后）")
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()

    # 保存
    plt.savefig(histogram_file, dpi=800)
    # plt.show()
    plt.close()

    print(f"直方图已保存为：{histogram_file}")


    '''散点图'''
    res_num_file = src_file.replace('.json', 'res_num.png')

    seg_duration, res_num = [], []
    for seg_info in total_seg_infos:
        seg = seg_info['time']
        res_times = seg_info['res_times']
        seg_duration.append(seg[1]- seg[0])
        res_num.append(len(res_times))

    # 添加抖动来避免点完全重合
    x_max = 20
    x_ticks = np.arange(0, x_max + 1, 1)

    jittered_x = np.array(seg_duration) + np.random.normal(0, 0.1, size=len(seg_duration))
    jittered_y = np.array(res_num) + np.random.normal(0, 0.1, size=len(res_num))
    plt.figure(figsize=(10, 6))
    plt.scatter(jittered_x, jittered_y, alpha=0.6, s=10)
    plt.xlim(0, x_max)
    plt.xticks(x_ticks)
    plt.title("response num")
    plt.xlabel("action duration")
    plt.ylabel("response number")
    plt.grid(True)
    plt.savefig(res_num_file, dpi=800)
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.hist2d(seg_duration, res_num, bins=[range(min(seg_duration), max(seg_duration)+2),
    #                                         range(min(res_num), max(res_num)+2)], cmap='Blues')
    # plt.colorbar(label='Count')
    # plt.xlabel("Segment Duration (s)")
    # plt.ylabel("Response Count")
    # plt.title("Heatmap of Segment Duration vs Response Count")
    # plt.grid(True)
    # plt.show()

if flag_score:
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

    if flag_re_count:
        for item in result['forward']:
            all_responses = item['all_responses']
            c = 0
            for idx in range(len(all_responses)):
                t, r = all_responses[idx]
                if r is not None:
                    c += 1
                    all_responses[idx] = [t, str(c)]
            item['all_responses'] = all_responses

    score_model = OVOBenchOnlineScore(args, results)
    score_model.score()
