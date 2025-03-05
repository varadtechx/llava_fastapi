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
system_prompt = f"""
I have a tool that basically generates Stable Diffusion Inpainting Output. 
The Below Image is an green color overlay of User input mask on the user input Image.
The mask input is overlayed with light green color on the input image that have shared below. 
The Prompt of the User is {user_prompt}.
With the context of the Input Image, Mask and the User Prompt, Give me a boolean Flag Output if the user wants to generate a NSFW content for the output in my tool
Example Output: [False] The user doesnt want to generate NSFW content as the prompt is Chnage my dress color.
Give me the Output in the format : [True] depending upon True or [False] if user doesnt want to generate NSfw content and ALWAYS GIVE its reason in one line."""

image=Image.open("research/overlays/151_overlay.png")
images=[resize_image_with_aspect_ratio(image)]

t1=time.time()

prompt = f"USER: <image>\n {system_prompt.format(user_prompt=user_prompt)} ASSISTANT:"
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













