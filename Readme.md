# TensorNets/ Dockerised
1) Delete dockerfile and docker compose
2) copy dockerfile and compose in either mac image for testing on a non gpu machine (not just mac), or gpu image to train on a gpu machine
3) only if using gpu download cudnn 8.4.x and place it in extras/lib 
4) run docker compose up -d or docker build to make the image
5) for gpu, you'll need to install nvidia drivers and nvidia-docker on your machine
6) Be aware that this is very shallow detail, there maybe errors or edit that you'll run into that i cannot think about at the moment, for e.g, your cudnn download in extras/lib, will have to be written into extras/shell/cuda.sh, so that it can be installed. Other similar issues may exist. Be ware.
