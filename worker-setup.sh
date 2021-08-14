#!/bin/bash
# use this if you manually install download worker on your box

echo "insert your nickname for the leaderboard or press Enter for anonymous..."
read nickname
export CAH_NICKNAME=$nickname

git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r worker-requirements.txt --no-cache-dir
pip install fake-useragent

