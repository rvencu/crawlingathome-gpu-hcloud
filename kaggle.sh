#!/bin/bash

pip install kaggle --upgrade

rm kaggle.py
cp kaggle-script.py kaggle.py

echo "insert your nickname for the leaderboard or press Enter for anonymous..."
read nickname
sed -i -e "s/<<your_nickname>>/$nickname/" kaggle.py

echo "insert your Hetzner Cloud API Token..."
read token
sed -i -e "s/<<your_hcloud_api_token>>/$token/" kaggle.py

echo "insert desired number of nodes in the swarm..."
read nodes
sed -i -e "s/<<your_swarm_nodes>>/$nodes/" kaggle.py


kaggle kernels push -p ./

sleep 60*60*8.5

kaggle kernels pull -p ./ -m