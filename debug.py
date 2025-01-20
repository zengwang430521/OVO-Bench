from qwen_vl_utils.vision_process import fetch_video

video_file_name = '/tmp/tmp0v1nr4gx.mp4'
ele = {
    "type": "video",
    "video": video_file_name,
    "nframes": 64
}
import pdb; pdb.set_trace()
video = fetch_video(ele)
