source venv/bin/activate
pip install -r vastai-requirements.txt
pip install tensorboard

pip install unsloth==2025.5.7 unsloth-zoo==2025.5.8

# flash-attention (requires some additional setup)
# Download and install CUDA 12.1+
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --toolkit --silent

# Update path
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

pip install flash-attn --no-build-isolation


cd data && tar -xzvf repos.tar.gz
cd ..
git pull



