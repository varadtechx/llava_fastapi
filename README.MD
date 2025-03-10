```bash

docker build -t nsfw_classifier . 

docker run --gpus all -d \
  -p 8000:8000 \
  -v /path/to/instance/logs:/app/logs \
  --name nsfw_classifier nsfw_classifier

```