from transformers import AutoProcessor

model_path = '/afs/zengwang/ckpt/Qwen2-VL-7B'
processor = AutoProcessor.from_pretrained(model_path)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": 'data/demo',
                "nframes": 64,
            },
            {
                "type": "text",
                "text": 'aaaaa'
            }
        ]
    }
]

prompt = processor.apply_chat_template(messages[0]["content"], tokenize=False, add_generation_prompt=True)
print(prompt)

# /usr/local/lib/python3.8/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py