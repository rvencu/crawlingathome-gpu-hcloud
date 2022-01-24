# Crawling@Home GPU controlled Hetzner Cloud swarm of scrapers

> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP. At the time of this writing we are up to 5 billion high quality pairs ready for training various models but we still expect your help to advance to the potential 6 billion quality pairs estimated to exist in the commoncrawl data. This dataset is intended for public use and towards a truly open access to AI for everyone !

## Concept
This image-text scraping task comes with specific characteristics: link lists might be old and images might not be online anymore, even entire domains might be missing. Also there are seldom multiple links pointing to the same domain, so the DNS queries are many and often. Finally after the actual scraping there is a computational intensive task to calculate similarities between images themselves and their captions.

On a normal CPU machine, scraping and filtering take almost the same time. On a GPU though filtering is much faster, in order of 60x faster than on single CPU.

Hence this concept for crawling@home where we created a data pipeline on 3 levels:
1. commoncrawl preprocessing, where we use a swarm of about 500 cpus to download, parse and send results to a database node with candidates for our dataset, meaning image urls with alt text, plus the detected language using gcld3. By the language detection we split the candidates into English, Multilanguage (non English) and Nolang (language not detected with confidence) categories.
2. image downloading and inspection, prefiltering by image type and resolution, producing further candidates for CLIP or mCLIP inference
3. CLIP style inference where we calculate similarity of image embeddings with text embeddings and retain only pairs with higher similarity than a manually set threshold

Common Crawl jobs are coordinated by a tracker with dashboard at http://cah.io.community/

## Cloud workers
We used AWS workers for first level of the above pipeline, Hetzner and Alibaba workers for the second level and home GPU plus AWS GPU nodes for the third level.

Thus the code migrated to:
1. Hetzner swarm control: use `infrastructure.py` to control the swarm at Hetzner Cloud via commands like `python3 infrastructure.py up 20 fsn1` where up means bring up swarm, 20 is the desired number of nodes, and fsn1 is the desired datacenter location.
2. Alibaba swarm control: due to cost restrictions we used Simple Application Servers with Alibaba, and developed a limited scope control script
3. CPU clients:
    a) `ccpp.py` is used to preprocess common crawl wat files. Nodes require minimum one CPU core and 1GB RAM for each CPU.
    b) `dbdl.py` is used to download images. Nodes require minimum one CPU core and 1GB RAM for each CPU.
3. GPU clients only consume max 3.5GB of GPU VRAM so any nVidia GPU card with 4GB VRAM or more is deemed compatible:
    a) run `python3 gpu_inference.py` from any Linux based PC with an Nvidia GPU and correct drivers installed

If you want to install on your own box, then
## Prerequisites
1. Ubuntu box with 4GB+ Nvidia GPU
2. Nvidia driver installed
3. Cuda toolkit 11+ (also corresponding cudnn is recommended for future)
4. check driver installation with `nvidia-smi` command
5. your user is able to run `sudo` commands
6. install `python3-pip` and `git` packages
## Distributed infrastructure setup and run
1. Make an account at Hetzner Cloud (https://www.hetzner.com/) and issue an API token
2. create the `.env` file and paste your HCLOUD API key in it. optionally, if you have more than one account, paste all API keys each on a separate line
3. bring up infrastructure at any time with `python3 infrastructure.py up N` in order to raise *N* nodes. It will scan all API keys and create maximum available servers on each until *N* limit is met
4. tear down infrastructure at any time with `python3 infrastructure.py down` in order to shutdown things (and save cash). this will shut down all cloud servers that belong to all API tokens saved in the `.env` file. Be aware, this command will delete all servers in the accounts even if they are NOT related to this project !!!

If you wish to SSH into any droplet you can use this command: `ssh -oStrictHostKeyChecking=no -oIdentitiesOnly=yes -i~/.ssh/id_cah crawl@<<droplet_ip>>`. The crawling script is ran as a service, check logs with `tail -f crawl.log`. Access service status or commands with `sudo systemctl stop|restart|start crawl`

If you are asked for any droplet root password at any time, it means you need to rerun `git pull` and `source conda-setup.sh` to refresh the files and regenerate the ssh keys pair.

## How to run GPU node from home computer
1. run `git clone https://github.com/rvencu/crawlingathome-gpu-hcloud`, to download crawlingathome GPU node script
2. run `cd crawlingathome-gpu-hcloud`, to enter the newly created directory
3. run `source conda-setup.sh` to setup the environment if you use anaconda. otherwise use `source pip-setup.sh`. the script will ask for a nickame to be used on leaderboard as well as for the sudo password
4. run `gpu_inference.py`. The script will run in a loop that can be interrupted at any time with Ctrl-C.

This work is based on code written by:
- https://github.com/TheoCoombes/crawlingathome
- https://github.com/Wikidepia/crawlingathome-worker

This is a subproject ran by the community around https://github.com/lucidrains/DALLE-pytorch
