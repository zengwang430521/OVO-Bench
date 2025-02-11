import torch
from utils.OVOBench import OVOBenchOffline
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from qwen_vl_utils.vision_process import *
import json


def _read_video_decord_v2(
    ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    start_time, end_time = ele.get("start_time", None), ele.get("end_time", None)
    st = time.time()
    vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    idx_start, idx_end = 0, total_frames - 1
    if start_time is not None:
        idx_start = max(round(start_time * video_fps), idx_start)
    if end_time is not None:
        idx_end = min(round(end_time * video_fps), idx_end)

    nframes = smart_nframes(ele, total_frames=(idx_end - idx_start + 1), video_fps=video_fps)
    idx = torch.linspace(idx_start, idx_end, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video


def fetch_video_v2(ele: dict, image_factor: int = IMAGE_FACTOR):
    if isinstance(ele["video"], str):
        # video_reader_backend = get_video_reader_backend()
        # video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        # import pdb; pdb.set_trace()
        video = _read_video_decord_v2(ele)
        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


class EvalQWen2VL2(OVOBenchOffline):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args
        self._model_init()

    def _model_init(self):
        model_path = self.args.model_path
        self.llm = LLM(
            model=model_path,
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.7,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=256,
            stop_token_ids=[],
        )

        self.processor = AutoProcessor.from_pretrained(model_path)

    def inference(self, video_file_name, prompt, start_time, end_time):
        # import pdb; pdb.set_trace()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_file_name,
                        "nframes": 64,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

        # image_inputs, video_inputs = process_vision_info(messages)
        image_inputs = None
        vision_info = {
            "type": "video",
            "video": video_file_name,
            "nframes": 64,
            "start_time": start_time,
            "end_time": end_time
        }
        video_inputs = [fetch_video_v2(vision_info)]

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        response = outputs[0].outputs[0].text
        print(f"Prompt:{prompt}")
        print(f"Response:{response}")
        return response

    def eval(self, anno, task_list, mode="offline"):
        # Inference
        if len(anno["backward"]) > 0:
            backward_results = []
            for _anno_ in tqdm(anno["backward"], desc="Backward Tasks"):
                # import pdb; pdb.set_trace()
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
                    response = self.inference(video, prompt, start_time=0, end_time=realtime)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    response = None
                finally:
                    pass
                    # if chunk_video_path:
                    #     os.remove(chunk_video_path)

                result = {
                    "id": id,
                    "video": video,
                    "task": task,
                    "question": question,
                    "response": response,
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
                    response = self.inference(video, prompt, start_time=0, end_time=realtime)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    response = None
                finally:
                    # if chunk_video_path:
                    #     os.remove(chunk_video_path)
                    pass

                result = {
                    "id": id,
                    "video": video,
                    "task": task,
                    "question": question,
                    "response": response,
                    "ground_truth": chr(65 + _anno_["gt"])
                }
                realtime_results.append(result)

        if len(anno["forward"]) > 0:
            forward_results = []
            for _anno_ in tqdm(anno["forward"], desc="Forward Tasks"):
                id = _anno_["id"]
                video = _anno_["video"]
                task = _anno_["task"]
                test_info = _anno_["test_info"]
                for i in range(len(test_info)):
                    prompt = self.build_prompt(task=task, question=None, options=None, _anno_=_anno_, index=i)
                    realtime = test_info[i]["realtime"]
                    try:
                        # chunk_video_path = self.chunk_video(video_path=video, end_time=realtime)
                        # response = self.inference(chunk_video_path, prompt)
                        response = self.inference(video, prompt, start_time=0, end_time=realtime)
                    except Exception as e:
                        print(f"Error during inference: {e}")
                        response = None
                    finally:
                        # if chunk_video_path:
                        #     os.remove(chunk_video_path)
                        pass

                    _anno_["test_info"][i]["response"] = response
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
