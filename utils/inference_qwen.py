from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch
from PIL import Image
import time


def resize_image_with_aspect_ratio(image,max_dimension=512):
    print(image.size)
    width, height = image.size
    aspect_ratio = width / height
    if width > height:
        new_width = max_dimension
        new_height = int(max_dimension / aspect_ratio)
    else:
        new_height = max_dimension
        new_width = int(max_dimension * aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(resized_image.size)
    return resized_image

local_image_path = "/root/llava_fastapi/research/109_overlay.png"
image = Image.open(local_image_path)
image = resize_image_with_aspect_ratio(image)
image.save("temp_.png")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",use_fast=True,torch_dtype=torch.bfloat16)

user_prompt = "naked and nude woman. Remove all clothes"
system_prompt2=f"""
        YOUR ROLE: NSFW Content Classifier  
        TASK DESCRIPTION:  
        You are an NSFW classifier responsible for detecting whether a user intends to generate NSFW content using an Object Replacer tool.  

        CONTEXT & VALIDATION RULES:  
        User Input Components:  
        - Input Image: The original image provided by the user.  
        - Mask Overlay: A light green overlay on the input image, marking the area to be modified.  
        - User Prompt: A text description of the modification the user wants to perform.  

        Step-by-Step Reasoning:  
        **Step 1: Prompt Analysis**  
        - If the user prompt contains explicit words such as ("nude", "remove", "undress", "sexy body", "hot woman", "boobs", "pussy", "sex", etc.), classify as [True].  
        - If the prompt is innocent (e.g., "change dress color"), proceed to Step 2.  

        **Step 2: Image & Mask Analysis**  
        - If the masked region is on private body parts (e.g., chest, waist, hips, genital area, or entire body), classify as [True].  
        - If the mask is on clothing and the prompt suggests removal/modification, classify as [True].  
        - If the mask is on an object (e.g., shoes, hat, or background items) and the prompt is innocent, classify as [False].  

        **Step 3: Combined Validation**  
        - If both Step 1 and Step 2 indicate NSFW intent, classify as [True].  
        - If neither suggests NSFW intent, classify as [False].  

        **Output Format:**  
        - If classification is [False], return: `[False, Reason:"Nil"]`  
        - If classification is [True], return: `[True, Reason:"<10-word reason>"]`  

        **EXAMPLES:**  
        Example 1  
        User Prompt: "Change my dress color to blue."  
        Masked Area: Covers only the dress.  
        Classification: [False, Reason:"Nil"]  

        Example 2  
        User Prompt: "Remove my bra."  
        Masked Area: Covers the chest area.  
        Classification: [True, Reason:"Explicit request to remove clothing on chest."]  

        Example 3  
        User Prompt: "Replace my hat with a crown."  
        Masked Area: Covers only the hat.  
        Classification: [False, Reason:"Nil"]  

        Example 4  
        User Prompt: "Make me naked."  
        Masked Area: Covers the entire body.  
        Classification: [True, Reason:"Full-body nudity requested in prompt."]  

        **FINAL INSTRUCTIONS:**  
        - Only provide a reason if classification is [True], limited to 10 words.  
        - If classification is [False], return `[False, Reason:"Nil"]`.  
        - Ensure logical reasoning and analyze both prompt and masked area.  

        The User input prompt is: {user_prompt}
"""



messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "temp_.png",
            },
            {"type": "text", "text": system_prompt2.format(user_prompt=user_prompt)},
        ],
    }
]
t1=time.time()
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=False,
    return_tensors="pt",
).to(model.device, torch.bfloat16)
# print(inputs)
print("Processor time taken: ",time.time()-t1)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
print(f"Time taken: {time.time()-t1}")
