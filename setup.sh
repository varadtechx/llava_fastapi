python3 -m venv llava-venv
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
source llava-venv/bin/activate
pip install git+https://github.com/huggingface/transformers peft  accelerate bitsandbytes safetensors sentencepiece optimum fsspec==2024.10.0
pip install pillow
pip install flash-attn --no-build-isolation
pip install torchvision
exit
