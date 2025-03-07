import requests
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoProcessor, LlavaForConditionalGeneration , BitsAndBytesConfig
import time 
import re 


model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.bfloat16,device_map="cuda:0",attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf" ,use_fast=True,torch_dtype=torch.bfloat16)

def resize_image_with_aspect_ratio(image,max_dimension=2560):
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

user_prompt="My photo add"
system_prompt2=f"""
        YOUR ROLE: NSFW Content Classifier
        TASK DESCRIPTION:
        You are an NSFW classifier responsible for detecting whether a user intends to generate NSFW content using an Object Replacer tool.

        This tool replaces objects in an image using a mask provided by the user and a textual prompt. The mask (overlayed in light green) highlights the area where modification will take place.

        Your task is to analyze both the input image with the mask and the user prompt to determine if the user is attempting to generate NSFW content.

        CONTEXT & VALIDATION RULES:
        User Input Components:

        Input Image: The original image provided by the user.
        Mask Overlay: A light green overlay on the input image, marking the area to be modified.
        User Prompt: A text description of the modification the user wants to perform.
        Step-by-Step Reasoning:
        Step 1: Prompt Analysis
        If the user prompt contains explicit words such as ("nude", "remove", "undress", "sexy body", "hot woman", "boobs", "pussy", "sex", etc.), classify as [True].
        If the prompt is innocent (e.g., "change dress color"), proceed to Step 2.
        Step 2: Image & Mask Analysis
        Check if the masked region is located on private body parts (e.g., chest, waist, hips, genital area, or entire body).
        If the mask is on clothing areas (e.g., shirt, pants) and the prompt suggests removal/modification of clothing, classify as [True].
        If the mask is on an object (e.g., shoes, hat, or background items) and the prompt is innocent, classify as [False].
        Step 3: Combined Validation
        If both Step 1 and Step 2 indicate NSFW intent, classify as [True].
        If neither suggests NSFW intent, classify as [False].
        Output Format:
        Always return output in the format: [True] or [False].
        Provide a justification explaining why the classification was made.
        EXAMPLES:
        Example 1
        User Prompt: "Change my dress color to blue."
        Masked Area: Covers only the dress.
        Classification: [False]
        Reason: The prompt does not contain NSFW words, and the masked area is not targeting private body parts.
        Example 2
        User Prompt: "Remove my bra."
        Masked Area: Covers the chest area.
        Classification: [True]
        Reason: The prompt explicitly mentions "remove my bra," and the masked region covers a private body part.
        Example 3
        User Prompt: "Replace my hat with a crown."
        Masked Area: Covers only the hat.
        Classification: [False]
        Reason: The prompt does not indicate NSFW intent, and the masked area is unrelated to NSFW content.
        Example 4
        User Prompt: "Make me naked."
        Masked Area: Covers the entire body.
        Classification: [True]
        Reason: The prompt explicitly indicates nudity, and the mask suggests a full-body modification.
        FINAL INSTRUCTIONS:
        Always follow the validation rules and analyze both the image and the prompt before making a classification.
        Ensure logical reasoning is applied in the decision-making process.
        Do not classify based on words alone—check the intent based on the mask region as well.
        The User input prompt is {user_prompt}"""

image=Image.open("research/109_overlay.png")
images=[resize_image_with_aspect_ratio(image)]

t1=time.time()

prompt = f"USER: <image>\n {system_prompt2.format(user_prompt=user_prompt)} ASSISTANT:"
inputs=processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
input_token_count=inputs['input_ids'].shape[1]
print(f'Processor Time :{time.time()-t1}')

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    generate_ids = model.generate(**inputs, max_new_tokens=256)
output=processor.batch_decode(generate_ids, skip_special_tokens=False)
output_token_count = generate_ids.shape[1]
total_time=time.time()-t1
print(f"Output Time : {total_time}")

print("Performanc Metrics--------")
total_tokens = output_token_count
throughput = total_tokens / total_time
print(f"Input Tokens: {input_token_count}")
print(f"Output Tokens: {output_token_count-input_token_count}")
print(f"Total Tokens: {total_tokens}")
print(f"Throughput: {throughput:.2f} tokens/sec")

print(output[0])

def extract_nsfw_status(input_string):
    match = re.search(r'ASSISTANT:\s*\[(\w+)\]', input_string, re.IGNORECASE)
    if match:
        if match.group(1).lower() == 'true':
            return 200,True
        elif match.group(1).lower() == 'false':
            return 200,False
        else:
            return 500,False
    else:
        return 500,False

status_code,NSFW=extract_nsfw_status(output[0])

if NSFW==True:
    print("True")
else:
    print("False")













