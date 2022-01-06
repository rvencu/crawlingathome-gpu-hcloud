#!/bin/bash

echo "insert your nickname for the leaderboard or press Enter for anonymous..."
read nickname
export CAH_NICKNAME=$nickname

sudo apt-get update
sudo apt-get install -y git build-essential python3-dev python3-pip libjpeg-dev zip libwebp-dev

git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
#pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#pip3 install git+https://github.com/rvencu/asks
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r gpu-requirements.txt --no-cache-dir
#pip install tensorflow==2.5 --no-cache-dir
pip install clip-anytorch

git clone "https://github.com/hetznercloud/hcloud-python" hcloud
pip3 install -e ./hcloud

pip install parallel-ssh

yes | ssh-keygen -t rsa -b 4096 -f $HOME/.ssh/id_cah -q -P ""

yes | pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
