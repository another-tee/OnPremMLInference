#!/bin/bash

# Still under experiment

# Download microk8s for linux ("support GPU")
snap install microk8s --classic

# Join the group
sudo usermod -a -G microk8s $USER
sudo mkdir -p ~/.kube
sudo chown -f -R $USER ~/.kube
su - $USER

