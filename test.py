import os

import torch
from transformers import AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def bge():

    MODEL_NAME = "/data/wzh_fd/workspace/Models/BGE-VL-base" # or "BAAI/BGE-VL-large"

    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True) # You must set trust_remote_code=True
    model.set_processor(MODEL_NAME)
    model.eval()

    with torch.no_grad():
        query = model.encode(
            images = "/data/wzh_fd/workspace/Models/BGE-VL-base/assets/cir_query.png", 
            text = "Make the background dark, as if the camera has taken the photo at night"
        )

        candidates = model.encode(
            images = ["/data/wzh_fd/workspace/Models/BGE-VL-base/assets/cir_candi_1.png", "/data/wzh_fd/workspace/Models/BGE-VL-base/assets/cir_candi_2.png"]
        )
        
        scores = query @ candidates.T
    print(scores)

def qwen():
    

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/data/wzh_fd/workspace/Models/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processor
    min_pixels = 256*28*28
    max_pixels = 512*28*28
    processor = AutoProcessor.from_pretrained("/data/wzh_fd/workspace/Models/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

if __name__=="__main__":
    # bge()
    qwen()
    