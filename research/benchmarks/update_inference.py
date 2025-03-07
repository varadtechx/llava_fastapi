import pandas as pd
import requests
import time

API_RESPONSE_TIMES = time.time()
def classify_nsfw(image_name: str, user_prompt: str) -> bool:
    url = "http://localhost:8000/classify_nsfw"
    API_RESPONSE_TIMES = []
    image_name = image_name.split(".")[0]
    image_name = image_name + "_overlay.png"
    payload = {
        "order_id": "1234",
        "image_path": f"research/benchmarks/overlays/{image_name}",
        "user_prompt": user_prompt
    }
    t1=time.time()
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        # API_RESPONSE_TIMES.append(time.time()-t1)
        return response.json().get("is_nsfw", False)
    print(time.time()-t1)
    return False

def process_csv(input_csv: str, output_csv: str):
    required_columns = ["image_name", "user_prompt", "is_nsfw", "overlay_path"]
    df = pd.read_csv(input_csv)

    #only take first 10 rows
    df = df.head(10)
    df["detected_nsfw"] = df.apply(lambda row: classify_nsfw(row["image_name"], row["user_prompt"]), axis=1)
    # df.to_csv(output_csv, index=False)

process_csv("nsfw-data-benchmark.csv", "test.csv")
print(f"Average API response time: {time.time()-API_RESPONSE_TIMES/10}")