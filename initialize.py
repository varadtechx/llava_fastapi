import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.bfloat16,device_map="cuda:0",attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf" ,use_fast=True,torch_dtype=torch.bfloat16)
print("[INFO]: Model Loaded")
torch.cuda.empty_cache()  
torch.cuda.ipc_collect()
print("[INFO]: GPU Memory Cleared")  