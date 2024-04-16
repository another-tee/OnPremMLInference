# Best Practice:
1. Push the docker images to Azure Container Registry (In this test -> Use offline docker image)
2. You should tag each image:
    - docker tag ml-infer:latest ml-infer:dev
    - docker tag ml-infer:latest ml-infer:uat
    - docker tag ml-infer:latest ml-infer:prd
3. Test

# Install microk8s
1. Install microk8s via **snap**
    
    ```bash 
    sudo snap install microk8s --classic --channel=1.29
    ```
2. Join the group
    
    ```bash
    sudo usermod -a -G microk8s $USER
    sudo mkdir -p ~/.kube
    sudo chown -f -R $USER ~/.kube
    su - $USER
    ```
3. Check status
    
    ```bash
    microk8s status --wait-ready
    ```
# Enable GPU on microk8s
- Use host NVIDIA drivers
    
    ```bash
    microk8s enable gpu
    ```

# Create Built-in registry
1. `docker build -f dockerfiles/Dockerfile.infer -t localhost:32000/ml-infer:latest .`
2. `docker push localhost:32000/ml-infer:latest`
3. Edit /etc/docker/daemon.json
    
    ```json
    {
        "insecure-registries" : ["localhost:32000"]
    }
    ```
4. `sudo systemctl restart docker`

# Useful commands for dev
- microk8s kubectl delete --all services,pods,deployment -n dev
- microk8s kubectl apply -f k8s/.
- microk8s kubectl get all -n dev
- microk8s kubectl exec --stdin --tty {pod} -n dev -- /bin/bash
