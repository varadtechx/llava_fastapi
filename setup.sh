python3 -m venv llava-venv
source llava-venv/bin/activate
cd llava_fastapi
pip install wheel
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/transformers 
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
python3 initialize.py 
python3 server_qwen.py

