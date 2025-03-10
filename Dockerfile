FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    VENV_DIR=/opt/llava-venv \
    APP_DIR=/opt/llava_fastapi

RUN apt-get update && apt-get install -y \
    python3-venv \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"

WORKDIR $APP_DIR

RUN pip install --upgrade pip wheel \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && pip install git+https://github.com/huggingface/transformers \
    && pip install flash-attn --no-build-isolation

COPY . $APP_DIR/

RUN pip install -r requirements.txt

RUN python3 initialize.py

EXPOSE 8000

CMD ["python3", "server_qwen.py"]
