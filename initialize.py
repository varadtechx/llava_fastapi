from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def download_model_and_processor():
    Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        use_fast=True
    )
    print("[INFO] Qwen2.5-VL-7B-Instruct model and processor downloaded and cached.")

if __name__ == "__main__":
    download_model_and_processor()
