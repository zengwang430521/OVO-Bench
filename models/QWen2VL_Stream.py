import torch
from utils.OVOBench import OVOBenchOffline
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from qwen_vl_utils.vision_process import *
import json
import decord
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info, fetch_video
import torch
from .qwen2_vl_monkey_patch import Qwen2VLStream
from tqdm import tqdm
import argparse
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model


class EvalQWen2VLStream(OVOBenchOffline):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args
        self._model_init()
        self.fps = 2
        self.max_frame_num = 64

    def _model_init(self):
        model_path = self.args.model_path
        lora_path = self.args.lora_path

        model = Qwen2VLStream.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        if lora_path is not None:
            # import pdb; pdb.set_trace()
            # print('Debug: load lora!')
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
        model = model.eval()
        self.model = model
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.sampling_params = dict(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_new_tokens=256,
        )

    def inference(
            self,
            video_file_name,
            prompt,
            query_time,
            start_time,
            end_time,
            check_time_step=1.0,
            only_one_response=False):
        # import pdb; pdb.set_trace()
        print(f"(Time: {query_time}) User:{prompt}")

        ele = {
            "type": "video",
            "video": video_file_name,
            "fps": self.fps,
            # "max_pixels": 256*256,
        }
        # 视频对象
        vr = decord.VideoReader(video_file_name)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        frame_time_step = 1.0 / self.fps
        end_time = min((total_frames-1) / video_fps, end_time)

        # frames in history when query
        sample_times = []
        cur_time = start_time
        while cur_time < query_time:
            sample_times.append(cur_time)
            cur_time += frame_time_step

        if len(sample_times) > self.max_frame_num:
            sample_times = sample_times[-self.max_frame_num:]
        sample_idxs = [round(t * video_fps) for t in sample_times]
        frames = vr.get_batch(sample_idxs).asnumpy()
        frames = torch.tensor(frames).permute(0, 3, 1, 2) # Convert to TCHW format

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
        frames = list(torch.split(frames, 1, dim=0))    # 分成list便于处理

        video_token_id = 151656
        system_prompt = "You are a helpful assistant."
        text_historys = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        video_message = {"role": "user", "content": [ele, {"type": "text", "text": ""}]}

        def get_response():
            messages = text_historys + [video_message]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if len(frames) % 2 == 0:
                frames_input = torch.cat(frames, dim=0)
            else:
                frames_input = torch.cat(frames + [frames[-1]], dim=0)
            inputs = self.processor(text=[text], images=None, videos=[frames_input], padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")
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

        def need_response():
            messages = text_historys + [video_message]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if len(frames) % 2 == 0:
                frames_input = torch.cat(frames, dim=0)
            else:
                frames_input = torch.cat(frames + [frames[-1]], dim=0)
            inputs = self.processor(text=[text], images=None, videos=[frames_input], padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")
            with torch.no_grad():
                output = self.model.forward(**inputs)
            last_frame_token_index = (inputs["input_ids"] == video_token_id).nonzero(as_tuple=True)[1].max().item()
            stream_logits = output.stream_logits
            last_logits = stream_logits[0, last_frame_token_index]
            result = last_logits[1] > last_logits[0]
            return result.item()

        # 先在 query time 强制回答一次
        force_response = get_response()
        all_responses = []
        if need_response():
            all_responses.append((cur_time, force_response))
            text_historys.append({"role": "assistant", "content": force_response})
            print(f"(Time: {cur_time}) Assistant:{force_response}")
        else:
            print(f"(Time: {cur_time}) None")

        # stream 循环处理
        cur_time += frame_time_step
        check_time = query_time + check_time_step
        while cur_time <= end_time:
            if only_one_response and len(all_responses) > 0:
                break
            new_sample_idx = round(cur_time * video_fps)
            new_frame = vr.get_batch([new_sample_idx]).asnumpy()
            new_frame = torch.tensor(new_frame).permute(0, 3, 1, 2) # Convert to TCHW format
            new_frame = transforms.functional.resize(
                new_frame,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            frames.append(new_frame)
            if len(frames) > self.max_frame_num:
                frames = frames[-self.max_frame_num:]

            if cur_time >= check_time:
                # 如果到了检查的节点，就检查
                if need_response():
                    response = get_response()
                    all_responses.append((cur_time, response))
                    text_historys.append({"role": "assistant", "content": response})
                    print(f"(Time: {cur_time}) Assistant:{response}")
                else:
                    if not only_one_response:
                        # only_one_response 模式下，加入None会让推理提前停止
                        all_responses.append((cur_time, None))
                    print(f"(Time: {cur_time}) None")

                check_time += check_time_step
            cur_time += frame_time_step

        return force_response, all_responses

    def eval(self, anno, task_list, mode="offline"):
        # import pdb; pdb.set_trace()

        # Inference
        if len(anno["backward"]) > 0:
            backward_results = []
            for _anno_ in tqdm(anno["backward"], desc="Backward Tasks"):
                id = _anno_["id"]
                video = _anno_["video"]
                task = _anno_["task"]
                question = _anno_["question"]
                options = _anno_["options"]
                realtime = _anno_["realtime"]
                assert not question == None
                assert not options == None
                prompt = self.build_prompt(task=task, question=question, options=options, _anno_=None, index=None)
                try:
                    # chunk_video_path = self.chunk_video(video_path=video, end_time=realtime)
                    # response = self.inference(chunk_video_path, prompt)
                    force_response, all_responses = self.inference(video, prompt, start_time=0, query_time=realtime, end_time=realtime+3, only_one_response=True)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    force_response, all_responses = None, None

                result = {
                    "id": id,
                    "video": video,
                    "task": task,
                    "question": question,
                    # "response": response,
                    "force_response": force_response,
                    "all_responses": all_responses,
                    "ground_truth": chr(65 + _anno_["gt"])
                }
                backward_results.append(result)

        if len(anno["realtime"]) > 0:
            realtime_results = []
            for _anno_ in tqdm(anno["realtime"], desc="Realtime Tasks"):
                id = _anno_["id"]
                video = _anno_["video"]
                task = _anno_["task"]
                question = _anno_["question"]
                options = _anno_["options"]
                realtime = _anno_["realtime"]
                assert not question == None
                assert not options == None
                prompt = self.build_prompt(task=task, question=question, options=options, _anno_=None, index=None)
                try:
                    # chunk_video_path = self.chunk_video(video_path=video, end_time=realtime)
                    # response = self.inference(chunk_video_path, prompt)
                    # response = self.inference(video, prompt, start_time=0, end_time=realtime)
                    force_response, all_responses = self.inference(video, prompt, start_time=0, query_time=realtime, end_time=realtime+3, only_one_response=True)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    force_response, all_responses = None, None


                result = {
                    "id": id,
                    "video": video,
                    "task": task,
                    "question": question,
                    # "response": response,
                    "force_response": force_response,
                    "all_responses": all_responses,
                    "ground_truth": chr(65 + _anno_["gt"])
                }
                realtime_results.append(result)

        if len(anno["forward"]) > 0:
            forward_results = []
            for _anno_ in tqdm(anno["forward"], desc="Forward Tasks"):
                id = _anno_["id"]
                video = _anno_["video"]
                task = _anno_["task"]
                # test_info = _anno_["test_info"]

                test_time = [t["realtime"] for t in _anno_['test_info']]
                end_time = max(test_time) + 3

                if "ask_time" in _anno_.keys():
                    query_time = _anno_["ask_time"]
                elif "start_times" in _anno_.keys():
                    query_time = max(_anno_["start_times"][0] - 1, 0)
                elif "start_time" in _anno_.keys():
                    query_time = max(_anno_["start_time"][0] - 1, 0)

                prompt = self.build_prompt(task=task, question=None, options=None, _anno_=_anno_, index=None)
                try:
                    force_response, all_responses = self.inference(video, prompt, start_time=0, query_time=query_time, end_time=end_time, only_one_response=False)
                except:
                    force_response, all_responses = None, None

                _anno_["force_response"] = force_response
                _anno_["all_responses"] = all_responses
                forward_results.append(_anno_)

        # Calculate Score
        if len(anno["backward"]) == 0:
            backward_results = []
        if len(anno["realtime"]) == 0:
            realtime_results = []
        if len(anno["forward"]) == 0:
            forward_results = []

        # Save Results
        if self.args.save_results:
            os.makedirs(f"{self.args.result_dir}/{self.args.model}", exist_ok=True)
            with open(f"{self.args.result_dir}/{self.args.model}/{self.args.model}_{'_'.join(task_list)}_{mode}_1.json",
                      "w") as f:
                json.dump({
                    "backward": backward_results,
                    "realtime": realtime_results,
                    "forward": forward_results
                }, f, indent=4)

    def build_prompt(self, task, question, options, _anno_, index):
        if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
            formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
            prompt = f"""
                Question: {question}
                Options:
                {formatted_options}
                Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
                Do not include any additional text or explanation in your response.
            """
        elif task == "REC":
            activity = _anno_["activity"]
            prompt = f""" 
                In the video, the man/woman is {activity} repetitively. 
                Your task is to count how many times he/she has completed the action of {activity}.
                Remind me every time when he/she finishes one.
                Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
                Do not include any additional text or explanation in your response.
            """

        elif task == "SSR":
            tutorial = _anno_["tutorial"]
            all_steps = _anno_["all_steps"]
            formatted_steps = '; '.join(f'{chr(65 + i)}. {step}' for i, step in enumerate(all_steps)) + ';'
            prompt = f"""
                You're watching a tutorial video of {tutorial}. It contains the following steps:
                {formatted_steps}
                Your task is to tell me what step the video is at.
                Remind me every time when he/she turns to a new step.
                Respond only with the letter corresponding to current step (e.g., A, B, C). 
                Do not include any additional text or explanation in your response.
            """

        elif task == "CRR":
            question = _anno_["question"]
            answer = _anno_["answer"]
            prompt = f"""{question}"""
        return prompt