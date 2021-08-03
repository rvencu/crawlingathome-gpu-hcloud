#!/bin/bash

echo "insert your nickname for the leaderboard or press Enter for anonymous..."
read nickname
export CAH_NICKNAME=$nickname

sudo apt-get update
sudo apt-get install -y git build-essential python3-dev python3-pip libjpeg-dev zip

git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install git+https://github.com/rvencu/asks
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r requirements.txt --no-cache-dir
conda install tensorflow --no-cache-dir
pip install clip-anytorch

git clone "https://github.com/hetznercloud/hcloud-python" hcloud
pip3 install -e ./hcloud

pip install parallel-ssh

yes | ssh-keygen -t rsa -b 4096 -f $HOME/.ssh/id_cah -q -P ""

yes | pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
