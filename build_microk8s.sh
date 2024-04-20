#!/bin/bash

microk8s enable gpu
microk8s enable registry
docker build -f docker/Dockerfile.infer -t localhost:32000/ml-infer:prod .
docker push localhost:32000/ml-infer:prod
microk8s kubectl create namespace prod

# cat <<EOF > /etc/docker/daemon.json
# {
#     "runtimes": {
#         "nvidia": {
#             "args": [],
#             "path": "nvidia-container-runtime"
#         }
#     },
#     "insecure-registries" : ["localhost:32000"]
# }
# EOF
# sudo systemctl restart docker