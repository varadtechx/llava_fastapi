```bash

docker build -t nsfw_classifier . 

docker run --gpus all -d \
  -p 8110:8000 \
  -v /home/ubuntu/Varad/logs:/opt/llava_fastapi/logs \
  --gpus '"device=2"' \
  --name nsfw_classifier nsfw_classifier 

```