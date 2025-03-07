import numpy as np
import torch
from utils.OVOBench import OVOBenchOffline
from transformers import AutoProcessor
# from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from qwen_vl_utils.vision_process import *
import json
import decord
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, DynamicCache
from qwen_vl_utils import process_vision_info, fetch_video
import torch
from .qwen2_vl_monkey_patch import Qwen2VLStream
from tqdm import tqdm
import argparse
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
import os
import textwrap
from .QWen2VL_Stream_v2 import EvalQWen2VLStreamV2



class EvalQWen2VLStreamV3Align(EvalQWen2VLStreamV2):
    def inference(
            self,
            video_file_name,
            prompt,
            query_time,
            start_time,
            end_time,
            check_time_step=1.0,
            check_times=None,
            only_one_response=False):
        import pdb; pdb.set_trace()

        print(f"(Time: {query_time}) User:{prompt}")
        print(check_times)

        video_token_id = 151656  # <|vision_pad|>
        end_token_id = 151645   # <|im_end|>

        ele = {"type": "video", "video": video_file_name, "nframes": 64}

        # 视频对象
        vr = decord.VideoReader(video_file_name)
        total_frames, real_fps = len(vr), vr.get_avg_fps()
        frame_time_step = 1.0 / self.fps
        end_time = min((total_frames - 1) / real_fps, end_time)
        resized_height, resized_width = None, None

        messages = []
        system_prompt = "You are a helpful assistant."
        messages.append({"role": "system", "content": system_prompt, 'time': [0, 0]})
        if query_time > 0:
            messages.append({"role": "user", "content": [ele, {"type": "text", "text": ""}], "time": [0, query_time]})

        messages.append({"role": "user", "content": prompt, "time": [query_time, query_time]})
        cur_time = query_time

        def get_frames(frame_idxs):
            frames = vr.get_batch(frame_idxs).asnumpy()
            frames = torch.tensor(frames).permute(0, 3, 1, 2).cuda()  # Convert to TCHW format
            nonlocal resized_height, resized_width
            if resized_height is None or resized_width is None:
                # resize params
                nframes, _, height, width = frames.shape
                min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
                total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
                max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
                max_pixels = ele.get("max_pixels", max_pixels)
                if "resized_height" in ele and "resized_width" in ele:
                    resized_height, resized_width = smart_resize(
                        ele["resized_height"],
                        ele["resized_width"],
                        factor=IMAGE_FACTOR,
                    )
                else:
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=IMAGE_FACTOR,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    )

            frames = transforms.functional.resize(
                frames,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            # import pdb; pdb.set_trace()
            # frames = list(torch.split(frames, 1, dim=0))  # 分成list便于处理
            return frames

        def get_videos():
            video_time_segs = []
            for message in messages:
                content = message["content"]
                if isinstance(content, list):
                    time = message['time']
                    for i in range(0, len(time), 2):
                        video_time_segs.append([time[i], time[i + 1]])

            # 先处理一下time_seg
            total_duration = 0
            for i in range(len(video_time_segs)):
                video_duration = (total_frames - 1) / real_fps
                t_start, t_end = video_time_segs[i]
                t_start, t_end = max(t_start, 0), min(t_end, video_duration)
                total_duration += t_end - t_start
                video_time_segs[i] = [t_start, t_end]
            video_fps, video_maxlen = 2.0, 64

            # 给每段分配帧数
            frame_nums = []
            for time_seg in video_time_segs:
                # 先计算这一段需要采样多少帧
                t_start, t_end = time_seg
                seg_duration = t_end - t_start
                frame_num = min(seg_duration * video_fps, seg_duration * real_fps)
                frame_num = min(frame_num, video_maxlen * seg_duration / total_duration)
                frame_num = math.floor(frame_num)
                frame_num = max(frame_num, 2)  # 最少采集2帧
                if frame_num % 2 != 0:
                    # 必须是偶数
                    frame_num -= 1
                frame_nums.append(frame_num)

            # 此时各段的采样帧数可能加起来超过 video_maxlen
            current_total = sum(frame_nums)
            # 如果超过，则对各段进行迭代调整，每次从那些帧数大于2的段减少2帧，直到总数不超过总数要求
            while current_total > video_maxlen:
                reduced = False
                for i in range(len(frame_nums)):
                    if frame_nums[i] > 2:
                        frame_nums[i] -= 2
                        current_total -= 2
                        reduced = True
                        if current_total <= video_maxlen:
                            break
                if not reduced:
                    # 如果所有段都已经是2帧，无法再减少，则退出循环
                    break

            # 确定采样的frame idx
            frame_times, frame_idxs = [], []
            for time_seg, frame_num in zip(video_time_segs, frame_nums):
                t_start, t_end = time_seg
                sample_times = np.linspace(t_start, t_end, frame_num, endpoint=False)
                sample_idxs = (sample_times * real_fps).round().astype(np.int32)
                sample_idxs = sample_idxs.clip(min=0, max=total_frames - 1)
                frame_idxs.append(sample_idxs)
                frame_times.append(sample_times)

            # 采样
            videos = []
            for sample_idxs in frame_idxs:
                videos.append(get_frames(sample_idxs))
            return videos


        past_key_values, rope_deltas = None, None

        def need_response():
            nonlocal past_key_values, rope_deltas
            past_key_values, rope_deltas = None, None

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = self.processor(text=[text], images=None, videos=videos, padding=True, return_tensors="pt")
            # 这是一个重要的bug
            inputs["position_ids"], inputs["rope_deltas"] = self.model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                attention_mask=inputs["attention_mask"],
            )
            inputs = inputs.to("cuda")

            if past_key_values is None:
                past_key_values = DynamicCache()

            with torch.no_grad():
                output = self.model.forward(**inputs, past_key_values=past_key_values, use_cache=True)
            # import pdb; pdb.set_trace()
            past_key_values = output.past_key_values
            rope_deltas = output.rope_deltas

            '''找到判别点'''
            last_end_token_index = (inputs["input_ids"] == end_token_id).nonzero(as_tuple=True)[1].max().item()
            judge_token_index = last_end_token_index

            stream_logits = output.stream_logits
            last_logits = stream_logits[0, judge_token_index]
            result = last_logits[1] > last_logits[0]
            # import pdb; pdb.set_trace()
            return result.item()

        def get_response():
            nonlocal past_key_values, rope_deltas

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=None, videos=videos, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")

            # import pdb; pdb.set_trace()
            if past_key_values is not None:
                inputs['past_key_values'] = past_key_values
                inputs['rope_deltas'] = rope_deltas
                generated_ids = self.model.generate(**inputs, **self.sampling_params, use_cache=True)
            else:
                generated_ids = self.model.generate(**inputs, **self.sampling_params)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            output_text = output_text[0]

            return output_text

        # 先在 query time 强制回答一次
        all_responses = []
        force_response = None
        if query_time > 0:
            videos = get_videos()
            flag = need_response()
            force_response = get_response()
            if flag:
                all_responses.append((cur_time, force_response))
                messages.append({"role": "assistant", "content": force_response, "time": [cur_time, cur_time]})
                print(f"(Time: {cur_time}) Assistant:{force_response}")
            else:
                print(f"(Time: {cur_time})")


        if check_times is None:
            check_times = []
            t = query_time + check_time_step
            while t <= end_time:
                check_times.append(t)
                t += check_time_step
        check_times = [min(t, end_time) for t in check_times]
        check_times = sorted(check_times)

        last_time = cur_time

        for cur_time in check_times:
            # import pdb; pdb.set_trace()
            if only_one_response and len(all_responses) > 0:
                break

            if isinstance(messages[-1]["content"], list):
                messages[-1]['time'] = [last_time, cur_time]
            else:
                messages.append({"role": "user", "content": [ele, {"type": "text", "text": ""}],"time": [last_time, cur_time]})

            videos = get_videos()
            flag = need_response()
            if flag:
                response = get_response()
                all_responses.append((cur_time, response))
                messages.append({"role": "assistant", "content": response, "time": [cur_time, cur_time]})
                last_time = cur_time
                print(f"(Time: {cur_time}) Assistant:{response}")
            else:
                if not only_one_response:
                    # only_one_response 模式下，加入None会让推理提前停止
                    all_responses.append((cur_time, None))
                print(f"(Time: {cur_time})")

        return force_response, all_responses
