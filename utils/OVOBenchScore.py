class OVOBenchOnlineScore():
    def __init__(self) -> None:
        pass

    def eval():
        pass

import matplotlib.pyplot as plt
def plot_histogram(data, bins=10, title="Histogram", xlabel="Value", ylabel="Frequency", save_path=None, **kwargs):
    """
    绘制直方图，并可选择保存图片。

    参数：
    - data: list or array-like, 数据列表
    - bins: int, 直方图的分箱数
    - title: str, 图表标题
    - xlabel: str, X 轴标签
    - ylabel: str, Y 轴标签
    - save_path: str, 如果提供路径，则保存图片，否则显示图片
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, edgecolor='black', **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')  # dpi=300 确保高质量，bbox_inches='tight' 去除多余空白
        print(f"图片已保存至: {save_path}")
    else:
        plt.show()



class OVOBenchOfflineScore():
    def __init__(self, args, results):
        self.args = args
        self.results = results

    def calculate_score_backward_realtime(self, results):
        def get_score(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        # Calculate Score for Every Result
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
        
        scores = {}
        for i in range(len(results)):
            if not results[i]["task"] in scores.keys():
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(self, results):
        def get_score_REC(response, gt):
            if response == None:
                return 0
            import re
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)
        
        def get_score_SSR_CRR(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        
        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        errors = {}
        for task in tasks:
            errors[task] = []

        for i, result in enumerate(results):
            # Calculate score for REC
            if result["task"] == "REC":
                for j, test_info_ in enumerate(result["test_info"]):
                    scores["REC"].append(get_score_REC(test_info_["response"], test_info_["count"]))
            # Calculate score for SSR
            if result["task"] == "SSR":
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["SSR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["SSR"].append(get_score_SSR_CRR(test_info_["response"], gt))
            # Calculate score for CRR
            if result["task"] == "CRR":
                gt_response_time = result['clue_time']
                pred_response_time = 0
                for tmp in result['test_info']:
                    pred_response_time = tmp['realtime']
                    if tmp['response'] == "Yes":
                        break
                errors["CRR"].append(pred_response_time - gt_response_time)


                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["CRR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["CRR"].append(get_score_SSR_CRR(test_info_["response"], gt))

        plot_histogram(errors["CRR"], bins=100, save_path="CRR.png")
        abs_err = [abs(t) for t in errors["CRR"]]
        print(f'CRR mean error: {sum(abs_err) / len(abs_err)}')
            
        return results, scores
    
    def score(self):
        print(f"Offline Model: {self.args.model}")
        backward_results = self.results["backward"]
        realtime_results = self.results["realtime"]
        forward_results = self.results["forward"]
        avg_scores = {
            "backward": [],
            "realtime": [],
            "forward": []
        }

        results_all = {}

        if len(backward_results) > 0:
            print("Evaluate Backward Tracing...")
            backward_results, backward_scores = self.calculate_score_backward_realtime(backward_results)
            # correct_backward, total_backward = 0, 0
            for k, v in backward_scores.items():
                print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
                # correct_backward += sum(v)
                # total_backward += len(v)
                avg_scores["backward"].append(sum(v)/len(v))

                results_all[k] = 100 * sum(v)/len(v)

            # print(f"Backward Avg.: {100 * correct_backward / total_backward:.2f}\n")
            print(f"Backward Avg.: {100 * sum(avg_scores['backward'])/len(avg_scores['backward']):.2f}\n")
            results_all["Backward_Avg"] = 100 * sum(avg_scores['backward'])/len(avg_scores['backward'])
        else:
            # correct_backward = 0
            # total_backward = 0
            pass
            
        if len(realtime_results) > 0:
            print("Evaluate Real-time Visual Perception...")
            realtime_results, realtime_scores = self.calculate_score_backward_realtime(realtime_results)
            # correct_realtime, total_realtime = 0, 0
            for k, v in realtime_scores.items():
                print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
                # correct_realtime += sum(v)
                # total_realtime += len(v)
                avg_scores["realtime"].append(sum(v)/len(v))

                results_all[k] = 100 * sum(v)/len(v)

            # print(f"Realtime Avg.: {100 * correct_realtime / total_realtime:.2f}\n")
            print(f"Realtime Avg.: {100 * sum(avg_scores['realtime'])/len(avg_scores['realtime']):.2f}\n")
            results_all["Realtime_Avg"] = 100 * sum(avg_scores['realtime'])/len(avg_scores['realtime'])

        else:
            # correct_realtime = 0
            # total_realtime = 0
            pass

        if len(forward_results) > 0:
            print("Evaluate Forward Active Responding...")
            forward_results, forward_scores = self.calculate_score_forward(forward_results)
            # correct_forward, total_forward = 0, 0
            for k, v in forward_scores.items():
                print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
                # correct_forward += sum(v)
                # total_forward += len(v)
                avg_scores["forward"].append(sum(v)/len(v))
                results_all[k] = 100 * sum(v)/len(v)

            # print(f"Forward Avg.: {100 * correct_forward / total_forward:.2f}\n")
            print(f"Forward Avg.: {100 * sum(avg_scores['forward'])/len(avg_scores['forward']):.2f}\n")
            results_all["Forward_Avg"] = 100 * sum(avg_scores['forward'])/len(avg_scores['forward'])
        else:
            # correct_forward = 0
            # total_forward = 0
            pass

        print(f"Total Avg.: {100 * (sum(avg_scores['backward']) + sum(avg_scores['realtime']) + sum(avg_scores['forward'])) / (len(avg_scores['backward']) + len(avg_scores['realtime']) + len(avg_scores['forward'])):.2f}")
        results_all['Total_Avg'] = 100 * (sum(avg_scores['backward']) + sum(avg_scores['realtime']) + sum(avg_scores['forward'])) / (len(avg_scores['backward']) + len(avg_scores['realtime']) + len(avg_scores['forward']))

        keys = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR", "Realtime_Avg",
                "EPM", "ASI", "HLD", "Backward_Avg",
                "REC", "SSR", "CRR", "Forward_Avg",
                "Total_Avg"]
        # import pdb; pdb.set_trace()

        for k in keys:
            print(f"{k}: {results_all.get(k, -1):.2f}")

        print(' '.join(keys))
        print(' '.join([f"{results_all.get(k, -1):.2f}" for k in keys]))


def get_realtime_response(all_responses, realtime):
    if all_responses is None:
        return None

    all_responses.sort(key=lambda t: t[0])

    response = None
    for i in range(len(all_responses)):
        t, res = all_responses[i]
        if realtime >= t and res is not None:
            response = res
    return response


class OVOBenchOnlineScore(OVOBenchOfflineScore):
    def calculate_score_backward_realtime(self, results):
        def get_score(response, gt):
            if response == None:
                return 0
            return int(gt in response)

        # Calculate Score for Every Result
        # 算两次， force response & stream response
        for i in range(len(results)):
            if self.args.force_response:
                response = results[i]["force_response"]
            else:
                all_responses = results[i]["all_responses"]
                response = None
                if all_responses is not None:
                    for time, res in all_responses:
                        if res is not None:
                            response = res
                            break
            results[i]["score"] = get_score(response, results[i]["ground_truth"])
        scores = {}
        for i in range(len(results)):
            if not results[i]["task"] in scores.keys():
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(self, results):
        def get_score_REC(response, gt):
            if response == None:
                return 0
            import re
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)

        def get_score_SSR(response, gt):
            if response == None:
                return 0
            return int(gt in response)

        def get_score_CRR(response, gt):
            if gt == 0:
                return int(response is None)
            else:
                return int(response is not None)


        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
            
        errors = {}
        for task in tasks:
            errors[task] = []
        
        for i, result in enumerate(results):
            all_responses = result["all_responses"]

            # Calculate score for REC
            if result["task"] == "REC":
                for j, test_info_ in enumerate(result["test_info"]):
                    realtime = test_info_["realtime"]
                    response = get_realtime_response(all_responses, realtime)
                    scores["REC"].append(get_score_REC(response, test_info_["count"]))
            
            # Calculate score for SSR
            if result["task"] == "SSR":
                if "start_time" in result.keys():
                    start_times = result["start_time"]
                else:
                    start_times = result["start_times"]

                gt_responses = []
                for idx_step, start_time in enumerate(start_times):
                    gt_responses.append((start_time, chr(65+idx_step)))

                for j, test_info_ in enumerate(result["test_info"]):
                    realtime = test_info_["realtime"]
                    response = get_realtime_response(all_responses, realtime)
                    gt = get_realtime_response(gt_responses, realtime)
                    scores["SSR"].append(get_score_SSR(response, gt))


            
            # Calculate score for CRR
            if result["task"] == "CRR":
                gt_response_time = result['clue_time']
                if len(result['all_responses']) == 0:
                    test_times = [t["realtime"] for t in result['test_info']]
                    test_times = sorted(test_times)
                    end_time = max(test_times) + 3
                    pred_response_time = end_time
                else:
                    pred_response_time = result['all_responses'][0][0]
                errors["CRR"].append(pred_response_time - gt_response_time)
                
                for j, test_info_ in enumerate(result["test_info"]):
                    realtime = test_info_["realtime"]
                    response = get_realtime_response(all_responses, realtime)
                    scores["CRR"].append(get_score_CRR(response, test_info_["type"]))
        
        
        plot_histogram(errors["CRR"], bins=100, save_path="CRR.png")
        abs_err = [abs(t) for t in errors["CRR"]]
        print(f'CRR mean error: {sum(abs_err)/len(abs_err)}')
        print(f'CRR PCK: {sum([1 if t<2 else 0 for t in abs_err])/len(abs_err)}')

        return results, scores

    def score(self):
        print(f"Offline Model: {self.args.model}")
        backward_results = self.results["backward"]
        realtime_results = self.results["realtime"]
        forward_results = self.results["forward"]
        avg_scores = {
            "backward": [],
            "realtime": [],
            "forward": []
        }

        results_all = {}

        if len(backward_results) > 0:
            print("Evaluate Backward Tracing...")
            backward_results, backward_scores = self.calculate_score_backward_realtime(backward_results)
            # correct_backward, total_backward = 0, 0
            for k, v in backward_scores.items():
                print(f"Task: {k}, Acc: {sum(v)} / {len(v)} = {100 * sum(v) / len(v):.2f}")
                # correct_backward += sum(v)
                # total_backward += len(v)
                avg_scores["backward"].append(sum(v) / len(v))

                results_all[k] = 100 * sum(v) / len(v)

            # print(f"Backward Avg.: {100 * correct_backward / total_backward:.2f}\n")
            print(f"Backward Avg.: {100 * sum(avg_scores['backward']) / len(avg_scores['backward']):.2f}\n")
            results_all["Backward_Avg"] = 100 * sum(avg_scores['backward']) / len(avg_scores['backward'])
        else:
            # correct_backward = 0
            # total_backward = 0
            pass

        if len(realtime_results) > 0:
            print("Evaluate Real-time Visual Perception...")
            realtime_results, realtime_scores = self.calculate_score_backward_realtime(realtime_results)
            # correct_realtime, total_realtime = 0, 0
            for k, v in realtime_scores.items():
                print(f"Task: {k}, Acc: {sum(v)} / {len(v)} = {100 * sum(v) / len(v):.2f}")
                # correct_realtime += sum(v)
                # total_realtime += len(v)
                avg_scores["realtime"].append(sum(v) / len(v))

                results_all[k] = 100 * sum(v) / len(v)

            # print(f"Realtime Avg.: {100 * correct_realtime / total_realtime:.2f}\n")
            print(f"Realtime Avg.: {100 * sum(avg_scores['realtime']) / len(avg_scores['realtime']):.2f}\n")
            results_all["Realtime_Avg"] = 100 * sum(avg_scores['realtime']) / len(avg_scores['realtime'])

        else:
            # correct_realtime = 0
            # total_realtime = 0
            pass

        if len(forward_results) > 0:
            print("Evaluate Forward Active Responding...")
            forward_results, forward_scores = self.calculate_score_forward(forward_results)
            # correct_forward, total_forward = 0, 0
            for k, v in forward_scores.items():
                print(f"Task: {k}, Acc: {sum(v)} / {len(v)} = {100 * sum(v) / len(v):.2f}")

                # correct_forward += sum(v)
                # total_forward += len(v)
                avg_scores["forward"].append(sum(v) / len(v))
                results_all[k] = 100 * sum(v) / len(v)

            # print(f"Forward Avg.: {100 * correct_forward / total_forward:.2f}\n")
            print(f"Forward Avg.: {100 * sum(avg_scores['forward']) / len(avg_scores['forward']):.2f}\n")
            results_all["Forward_Avg"] = 100 * sum(avg_scores['forward']) / len(avg_scores['forward'])
        else:
            # correct_forward = 0
            # total_forward = 0
            pass

        print(
            f"Total Avg.: {100 * (sum(avg_scores['backward']) + sum(avg_scores['realtime']) + sum(avg_scores['forward'])) / (len(avg_scores['backward']) + len(avg_scores['realtime']) + len(avg_scores['forward'])):.2f}")
        results_all['Total_Avg'] = 100 * (
                    sum(avg_scores['backward']) + sum(avg_scores['realtime']) + sum(avg_scores['forward'])) / (
                                               len(avg_scores['backward']) + len(avg_scores['realtime']) + len(
                                           avg_scores['forward']))

        keys = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR", "Realtime_Avg",
                "EPM", "ASI", "HLD", "Backward_Avg",
                "REC", "SSR", "CRR", "Forward_Avg",
                "Total_Avg"]
        # import pdb; pdb.set_trace()

        for k in keys:
            print(f"{k}: {results_all.get(k, -1):.2f}")

        print(' '.join(keys))
        print(' '.join([f"{results_all.get(k, -1):.2f}" for k in keys]))
