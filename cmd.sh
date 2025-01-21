ln -s /afs/zengwang/projects/task_define_service/data/OVO-Bench /afs/zengwang/projects/task_define_service/OVO-Bench/data

cd  /afs/zengwang/projects/task_define_service/OVO-Bench


python inference.py \
    --mode offline \
    --task EPM ASI HLD STU OJR ATR ACR OCR FPD REC SSR CRR \
    --model QWen2VL_7B_V2 \
    --model_path /afs/zengwang/ckpt/Qwen2-VL-7B-Instruct


python inference.py \
    --mode offline \
    --task CRR \
    --model QWen2VL_7B_V2 \
    --model_path /afs/zengwang/ckpt/Qwen2-VL-7B-Instruct


python score.py --model QWen2VL_7B_V2 --mode offline



huggingface-cli download \
--repo-type dataset \
--resume-download JoeLeelyf/OVO-Bench \
--local-dir /home/SENSETIME/zengwang/myprojects/task_define_service/data/OVO-Bench \
--local-dir-use-symlinks False


huggingface-cli download \
--repo-type model \
--resume-download Qwen/Qwen2-VL-7B \
--local-dir /home/SENSETIME/zengwang/myprojects/task_define_service/Qwen2-VL-7B \
--local-dir-use-symlinks False


~/ads-cli sync \
/home/SENSETIME/zengwang/myprojects/task_define_service/data/OVO-Bench  \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss.cn-sh-01.sensecoreapi-oss.cn/data/OVO-Bench

/afs/zengwang/ads-cli sync \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss-internal.cn-sh-01.sensecoreapi-oss.cn/data/OVO-Bench \
/afs/zengwang/projects/task_define_service/data/OVO-Bench


/afs/zengwang/ads-cli sync \
/afs/zengwang/projects/task_define_service/OVO-Bench/results \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss-internal.cn-sh-01.sensecoreapi-oss.cn/ovo-bench-results

~/ads-cli sync \
s3://196FFD00B6184227B65B3D92C01A8724:DD1D004D80834448B276F125F8310F2A@zengwang.aoss.cn-sh-01.sensecoreapi-oss.cn/ovo-bench-results \
/home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/results

ln -s /home/SENSETIME/zengwang/myprojects/task_define_service/data/OVO-Bench /home/SENSETIME/zengwang/myprojects/task_define_service/OVO-Bench/data

unzip AutoEvalMetaData.zip
unzip COIN.zip
unzip Ego4D.zip
unzip YouTube_Games.zip
unzip cross_task-hirest-OpenEQA.zip
unzip thumos.zip
unzip youcook2.zip
ls
