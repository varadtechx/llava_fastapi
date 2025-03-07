from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time
from io import BytesIO
import logging
from os.path import dirname, abspath, join
from os import makedirs
from logging.handlers import RotatingFileHandler
import re

from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Setup logging
project_dir = dirname(abspath(__file__))
logger = logging.getLogger('qwen-nsfw')
logger.setLevel(logging.INFO)
BASE_PATH = project_dir
makedirs(join(BASE_PATH, 'logs'), exist_ok=True)
path = join(BASE_PATH, 'logs', 'qwen-nsfw.log')
handler = RotatingFileHandler(path,
                              maxBytes=2 * 1024 * 1024,
                              backupCount=5)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

class Request(BaseModel):
    order_id: int
    user_prompt: str
    image_path: str

class Response(BaseModel):
    order_id: int
    is_nsfw: bool
    order_status_code: int
    nsfw_reason: str
    order_response_message: str

class QwenNSFWClassifier():
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0"
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            use_fast=True, 
            torch_dtype=torch.bfloat16
        )
        logger.info("[INFO] Qwen2.5-VL Model and Processor Loaded.")
    
    def resize_image_with_aspect_ratio(self, image, max_dimension=512):
        width, height = image.size
        aspect_ratio = width / height
        if width > height:
            new_width = max_dimension
            new_height = int(max_dimension / aspect_ratio)
        else:
            new_height = max_dimension
            new_width = int(max_dimension * aspect_ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    def load_image(self, source):
        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        else:
            return Image.open(source)
    
    def classify_nsfw(self, user_prompt: str, image: Image) -> str:
        system_prompt = f"""
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

        # Prepare the image
        image = self.resize_image_with_aspect_ratio(image)
        print(image.size)
        
        t1 = time.time()
        
        # Create messages format for Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        ).to(self.model.device, torch.bfloat16)
        
        processor_time = time.time() - t1
        logger.info(f"Processor time: {processor_time}")
        
        # Using flash attention sdpa_kernel for optimal performance
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        total_time = time.time() - t1
        logger.info(f"Total inference time: {total_time}")
        logger.info(f"Model output: {output_text}")
        
        return output_text

    def extract_nsfw_status(self, input_string):
        try:
            # Look for [True, Reason:"..."] or [False, Reason:"Nil"] pattern
            match = re.search(r'\[(True|False)', input_string, re.IGNORECASE)
            if match:
                is_nsfw = match.group(1).lower() == 'true'
                # Extract reason if available
                reason_match = re.search(r'Reason:"([^"]*)"', input_string)
                reason = reason_match.group(1) if reason_match else "No reason provided"
                
                if is_nsfw:
                    logger.info(f"NSFW detected: {reason}")
                
                return 200, is_nsfw, reason
            else:
                logger.warning(f"Failed to extract NSFW status from: {input_string}")
                return 500, False, "Failed to parse classifier output"
        except Exception as e:
            logger.error(f"Error extracting NSFW status: {e}")
            return 500, False, f"Error parsing classifier output: {e}"

    def classify(self, request: Request):
        try:
            image = self.load_image(request.image_path)
        except Exception as e:
            logger.error(f"Error while loading image: {e}")
            return 500, False, f"Error loading image: {e}"
        
        try:
            output_text = self.classify_nsfw(request.user_prompt, image)
            status_code, is_nsfw, reason = self.extract_nsfw_status(output_text)
            
            if status_code == 200:
                message = f"NSFW Classification Completed: {'NSFW' if is_nsfw else 'Safe'}"
                return status_code, is_nsfw, reason, message
            else:
                return status_code, False, reason,"NSFW Classification Failed"
        except Exception as e:
            logger.error(f"Error while classifying NSFW: {e}")
            reason="Error in NSFW classification"
            return 500, False, reason,f"Error in NSFW classification: {e}"

# Initialize the classifier
qwen_nsfw_classifier = QwenNSFWClassifier()

@app.post("/classify_nsfw")
async def classify_nsfw(request: Request):
    logger.info(f"Received Request for order_id {request.order_id}: {request}")
    status_code, is_nsfw,reason,order_response_message = qwen_nsfw_classifier.classify(request)
    logger.info(f"Response for order_id {request.order_id}: {is_nsfw}, {status_code}, {order_response_message}")
    return Response(
        order_id=request.order_id, 
        is_nsfw=is_nsfw, 
        order_status_code=status_code, 
        nsfw_reason=reason,
        order_response_message=order_response_message
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)