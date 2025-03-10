import requests
import time
def nsfw_classifier(order_id:str,image_url : str , mask_url: str, user_prompt:str):
    AUTH_KEY = "Basic ndedsi2i323rfwffqtdednondwnns"
    HEADERS = {
        "X-API-Key": AUTH_KEY,
        "Content-Type": "application/json"
    }
    nsfw_url = "http://3.17.17.238:8110/classify_nsfw"
    payload = {
        'image_url': image_url, 
        'mask_url' : mask_url,
        'user_prompt': user_prompt,
        'order_id': order_id
    }
    try:
        response = requests.post(nsfw_url, headers=HEADERS,json=payload)
        if response.status_code == 200:
            response=response.json()
            print(f"NSFW Classifier API Response{response}")
            if response['order_status_code']== 200:
                if response['is_nsfw']== True:
                    is_nsfw = True
                    nsfw_reason = response['nsfw_reason']
                else:
                    is_nsfw = False
                    nsfw_reason = None
            else:
                is_nsfw = False
                nsfw_reason = None
                response=response.json()
                print(f"NSFW Classifier Failed for order_id : {order_id}")
                print(response)
        else:
            is_nsfw = False
            nsfw_reason = None
            response=response.json()
            print(f"NSFW Classifier Failed for order_id : {order_id}")
            print(response)
    except Exception as e:
        is_nsfw = False
        nsfw_reason = None
        print(f"NSFW Classifier Failed for order_id : {order_id}")  
    return is_nsfw,nsfw_reason

image_url = 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/frontend_upload/file_drops/203cc69a-1e0d-408d-b1d4-c47e0358f783.jpg'
mask_url='https://phot-user-uploads.s3.us-east-2.amazonaws.com/base64URLs/2025-03-10/728b0da2-d029-40c8-bdf0-6ce143b1ebaf.webp'
order_id="j3oj2oififjpiip"
user_prompt="remove"
is_nsfw,nsfw_reason=nsfw_classifier(order_id,image_url,mask_url,user_prompt)
print(is_nsfw,nsfw_reason)