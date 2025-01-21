from moviepy.editor import VideoFileClip

video_path = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/video_event/bicycle.mp4'
start_time = 1
end_time = 2
video = VideoFileClip(video_path)
clip = video.subclip(start_time, end_time)
t = 0