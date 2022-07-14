# cuda shell script
echo '---cuda shell script---'
#Cuda
nvcc --version
#Cudnn
cd /workspace/lib
tar -xvf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

