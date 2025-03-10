FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --upgrade pip wheel && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install git+https://github.com/huggingface/transformers && \
    pip3 install flash-attn --no-build-isolation && \
    pip3 install -r requirements.txt

COPY . .

RUN python3 initialize.py

EXPOSE 8000

CMD ["python3", "server_qwen.py"]