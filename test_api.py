import requests
import time

def test_post_request():
    url = "http://localhost:8000/classify_nsfw"
    payload = {
        "order_id": "1234",
        "image_path": "/root/llava_fastapi/109_overlay.png",
        "user_prompt": "dog"
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(response.json())
    else:
        print(response.order_response_message)
t1=time.time()
test_post_request()
print("Total Time: ",time.time()-t1)