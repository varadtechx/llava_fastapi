from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time 
from io import BytesIO
import logging
from os.path import dirname, abspath, join
from os import makedirs
from logging.handlers import RotatingFileHandler

from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoProcessor, LlavaForConditionalGeneration , BitsAndBytesConfig
import re 

project_dir = dirname(abspath(__file__))
logger = logging.getLogger('textify')
logger.setLevel(logging.INFO)
BASE_PATH = project_dir
makedirs(join(BASE_PATH, 'logs'), exist_ok=True)
path = join(BASE_PATH, 'logs', 'llava-nsfw.log')
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
    order_response_message: str

class LlavaNSFWClassifier():
    def __init__(self):
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",torch_dtype=torch.bfloat16,device_map="cuda:0",attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf" ,use_fast=True,torch_dtype=torch.bfloat16)
        logger.info("[INFO] Model and Processor Loaded.")
    
    def resize_image_with_aspect_ratio(self,image,max_dimension=512):
        # print(image.size)
        width, height = image.size
        aspect_ratio = width / height
        if width > height:
            new_width = max_dimension
            new_height = int(max_dimension / aspect_ratio)
        else:
            new_height = max_dimension
            new_width = int(max_dimension * aspect_ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # print(resized_image.size)
        return resized_image

    def load_image(self,source):
        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        else:
            return Image.open(source)
    
    def classify_nsfw(self,user_prompt: str ,image:Image) -> str:
        system_prompt = f"""
        YOUR ROLE: NSFW Classifier

        CONTEXT:
        I have a tool that basically generates Stable Diffusion Inpainting Output. 
        The Below Image is an green color overlay of User input mask on the user input Image.
        The mask input is overlayed with light green color on the input image that have shared below. 
        The Prompt of the User is {user_prompt}.

        YOUR TASK:
        With the context of the Input Image, Mask and the User Prompt, Give me a boolean Flag Output if the user wants to generate a NSFW content for the output in my tool
        
        VALIDATION RULES:
        Always Classify nsfw True if the prompts contains words like nude,remove,undress,sexy body,sex,hot woman,boobs,pussy etc.
        Also Give me why you have classified the output as True or False.

        EXAMPLES:
        Example Output 1 : [False] The user doesnt want to generate NSFW content as the prompt is Chnage my dress color.
        Example Output 2 : [True] The user wants to generate NSFW content as the prompts (nude,sexy body) are present in the prompt.

        OUTPUT FORMAT:
        Give me the Output in the format : [True] depending upon True or [False] if user doesnt want to generate NSfw content.

        Complete the task with the above context.Validate your response with validation rules , output format and examples.
        Give the Reason for your classification as well.The user prompt is {user_prompt}.
        """

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
        
        images=[self.resize_image_with_aspect_ratio(image)]
        
        t1=time.time()
        
        prompt = f"USER: <image>\n {system_prompt2.format(user_prompt=user_prompt)} ASSISTANT:"
        inputs=self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device, torch.float16)
        input_token_count=inputs['input_ids'].shape[1]
        # print(f'Processor Time :{time.time()-t1}')
        
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            generate_ids = self.model.generate(**inputs, max_new_tokens=256)
        output=self.processor.batch_decode(generate_ids, skip_special_tokens=False)
        output_token_count = generate_ids.shape[1]
        total_tokens = output_token_count
        total_time=time.time()-t1
        print("Model Inference Time : ",total_time)
        throughput = total_tokens / total_time
        logger.info(f"Model Inference Time : {total_time}")
        logger.info(f"Total Throughput: {throughput:.2f} tokens/sec")
        output_text = output[0]
        logger.info(f"[INFO]: Model Output: {output_text}")
        return output_text

    def extract_nsfw_status(self,input_string):
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

    def classify(self,request: Request):
        try:
            image = self.load_image(request.image_path)
        except Exception as e:
            logger.error(f"Error while loading image: {e}")
            return 500,False,f"Error while loading image {e}"
        try:
            output_text = self.classify_nsfw(request.user_prompt,image) 
            status_code,nsfw_status = self.extract_nsfw_status(output_text)
            if status_code == 500:
                return status_code,nsfw_status,"NSFW Classification Failed"
            elif status_code == 200:
                return status_code,nsfw_status,"NSFW Classification Completed Successfully"
            else:
                return 500,False,"NSFW Classification Failed"
        except Exception as e:
            logger.error(f"Error while classifying NSFW: {e}")
            return 500,False,f"Error while classifying NSFW {e}"
        

llava_nsfw_classifier = LlavaNSFWClassifier()
@app.post("/classify_nsfw")
async def classify_nsfw(request: Request):
    logger.info(f"Received Request for order_id {request.order_id}: {request}")
    status_code,is_nsfw,order_response_message = llava_nsfw_classifier.classify(request)
    logger.info(f"Response for order_id {request.order_id}: {is_nsfw},{status_code},{order_response_message}")
    return Response(order_id=request.order_id, is_nsfw=is_nsfw, order_status_code=status_code, order_response_message=order_response_message)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)

       



