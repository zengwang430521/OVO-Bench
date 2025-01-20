# from qwen_vl_utils.vision_process import fetch_video
#
# video_file_name = '/tmp/tmpre6h3gah.mp4'
# ele = {
#     "type": "video",
#     "video": video_file_name,
#     "nframes": 64
# }
# import pdb; pdb.set_trace()
# video = fetch_video(ele)
#

video_path = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/video_event/push-up_2.mp4'
import decord
vr = decord.VideoReader(video_path)
total_frames, video_fps = len(vr), vr.get_avg_fps()


t = 0