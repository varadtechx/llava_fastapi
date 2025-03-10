import requests
import time

def test_post_request():
    url = "http://localhost:8000/classify_nsfw"
    payload = {
        'image_url': 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/frontend_upload/file_drops/203cc69a-1e0d-408d-b1d4-c47e0358f783.jpg', 
        'mask_url': 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/base64URLs/2025-03-10/728b0da2-d029-40c8-bdf0-6ce143b1ebaf.webp', 
        'user_prompt': 'Pillow',
        'order_id': 'ncwrnronrfgetpoojpij3238'
    }
    response = requests.post(url, json=payload)
    print(response)
    if response.status_code == 200:
        print(response.json())
    else:
        print(response.json())
t1=time.time()
test_post_request()
print("Total Time: ",time.time()-t1)