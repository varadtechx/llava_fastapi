from fastapi import FastAPI , Depends, HTTPException, status, Request, Header
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
import cv2
import numpy as np
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
AUTH_KEY = "Basic ndedsi2i323rfwffqtdednondwnns"

class Request(BaseModel):
    order_id: str
    user_prompt: str
    image_url: str
    mask_url: str

class Response(BaseModel):
    order_id: str
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
    
    def resize_image_with_aspect_ratio(self, image : Image, max_dimension=512):
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
        - If the user prompt contains explicit words such as ("nude", "undress", "sexy body", "hot woman", "boobs", "pussy", "sex", "dick","penis","vagina","naked men", etc.), classify as [True].
        - If the user prompt contains words like "discard", "remove", "delete", "discard selected item", "replace with nothing", etc., check the masked area and if the masked area is on private body parts, classify as [True] else classify as [False].
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

        Example 5
        User Prompt: "discard selected item and replace with nothing"
        Masked Area: Covers the chest area.
        Classification: [True, Reason:"Explicit request to remove clothing on chest."]

        Example 6
        User Prompt: "discard selected item and replace with nothing"
        Masked Area: Does not cover any private body parts.
        Classification: [False, Reason:"Nil"]

        **FINAL INSTRUCTIONS:**  
        - Only provide a reason if classification is [True], limited to 10 words.  
        - If classification is [False], return `[False, Reason:"Nil"]`.  
        - Ensure logical reasoning and analyze both prompt and masked area.  

        The User input prompt is: {user_prompt}
        """
        # print(image.size)
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
        
        # Using flash attention sdpa_kernel for optimal performance
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
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

    def apply_mask(self, image_pil: Image, mask_pil: Image):
        if image_pil.mode == 'RGBA':
            image_pil = image_pil.convert('RGB')
        image = np.array(image_pil)
        mask = np.array(mask_pil.convert('L'))
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        binary_mask = np.where(mask > 240, 255, 0).astype(np.uint8)
        mask_indices = binary_mask > 0
        result = image.copy()
        if np.any(mask_indices):
            result[mask_indices] = (
                result[mask_indices] * (1 - 0.3) + 
                np.array([0, 255, 128])[None, None, :] * 0.3
            ).astype(np.uint8)
        return Image.fromarray(result)

    def classify(self, request: Request):
        t1 = time.time()
        try:
            image = self.load_image(request.image_url)
            mask = self.load_image(request.mask_url)
            image = self.resize_image_with_aspect_ratio(image)
            mask = self.resize_image_with_aspect_ratio(mask)
            logger.info(f"Image and Mask From URL Time :{time.time()-t1}")
        except Exception as e:
            logger.error(f"Error while loading image: {e}")
            reason="Nil"
            return 500, False, reason ,f"Error loading image: {e}"
        
        try:
            overlayed = self.apply_mask(image,mask)
            logger.info(f"Apply Mask Overlay Function Time:{time.time()-t1}")
            print(f"Apply Mask Overlay Function Time:{time.time()-t1}")
            output_text = self.classify_nsfw(request.user_prompt, overlayed)
            status_code, is_nsfw, reason = self.extract_nsfw_status(output_text)
            
            if status_code == 200:
                message = f"NSFW Classification Completed: {'NSFW' if is_nsfw else 'Safe'}"
                print(f"Total Inference Time :{time.time()-t1}")
                logger.info(f"Total Inference Time :{time.time()-t1}")
                return status_code, is_nsfw, reason, message
            else:
                return status_code, False, reason,"NSFW Classification Failed"
        except Exception as e:
            logger.error(f"Error while classifying NSFW: {e}")
            reason="Error in NSFW classification"
            return 500, False, reason,f"Error in NSFW classification: {e}"

# Initialize the classifier
qwen_nsfw_classifier = QwenNSFWClassifier()

async def verify_api_key(x_api_key: str = Header(None)):
    """Verify the API key provided in the X-API-Key header."""
    if x_api_key is None or x_api_key != AUTH_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return x_api_key

@app.get("/")
def server_status(api_key: str = Depends(verify_api_key)):
    return {"status": "running", "message": "Server is up and running"}

@app.post("/classify_nsfw")
async def classify_nsfw(request: Request, api_key: str = Depends(verify_api_key)):
    logger.info(f"Received Request for order_id {request.order_id}: {request}")
    status_code, is_nsfw, reason, order_response_message = qwen_nsfw_classifier.classify(request)
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