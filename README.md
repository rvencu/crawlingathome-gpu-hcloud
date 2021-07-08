# Crawling@Home GPU controlled Hetzner Cloud swarm of scrapers

> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP. At the time of this writing we are up to 35 million high quality pairs ready for training various models but we still expect your help to advance to the potential 30 billion pairs estimated to exist in the commoncrawl data. This dataset is intended for public use and towards a truly open access to AI for everyone !

## Concept
This scraping task comes with specific characteristics: link lists might be old and images might not be online anymore, even entire domains might be missing. Also there are seldom multiple links pointing to the same domain, so the DNS queries are many and often. Finally after the actual scraping there is a computational intensive task to calculate similarities between images themselves and their captions.

On a normal CPU machine, scraping and filtering take almost the same time. On a GPU though filtering is much faster, in order of 60x faster than on single CPU.

Hence this concept for crawling@home where a cental GPU machine can drive a swarm of cloud workers then perform computing intensive task on GPU.

At this time the script is tested on a single GPU driving 20 workers. At full load we estimate getting about 6M pairs per 24 hours for the cost of using the local GPU and 6 Euro in Hetzner could computing.

Remember to watch your progress at http://crawlingathome.duckdns.org/

## Prerequisites
1. Ubuntu box with 8GB+ Nvidia GPU
2. Nvidia driver installed
3. Cuda toolkit 11.0 and corresponding cudnn installed
4. check driver installation with `nvidia-smi` command
5. your user is able to run `sudo` commands
6. install `python3-pip` and `git` packages
## Distributed infrastructure setup and run
1. Make an account at Hetzner Cloud (https://www.hetzner.com/) and issue an API token
2. run `git clone https://github.com/rvencu/crawlingathome-gpu-hcloud`, to download crawlingathome GPU node script
3. run `cd crawlingathome-gpu-hcloud`, to enter the newly created directory
4. create the `.env` file and paste your HCLOUD API key in it. optionally, if you have more than one account, paste all API keys each on a separate line
5. run `source conda-setup.sh` to setup the environment if you use anaconda. otherwise use `source pip-setup.sh`. the script will ask for a nickame to be used on leaderboard as well as for the sudo password
6. run `python3 gpu.py N`, to start Distributed Crawling with Central GPU Processing with a swarm of `N` scrapers! The script will run in a loop that can be interrupted at any time with Ctrl-C. The cloud infrastructure will be automatically shut down after logs from all nodes would have been collected on GPU computer. Change `N` with any number you like provided it is withing your cloud account limits.
7. tear down infrastructure at any time with `python3 infrastructure.py down` in order to shutdown things (and save cash). this will shut down all cloud servers that belong to all API tokens saved in the `.env` file

The GPU console will cycle status messages from all droplets. If you wish to SSH into any droplet you can use this command: `ssh -oStrictHostKeyChecking=no -oIdentitiesOnly=yes -i~/.ssh/id_cah crawl@<<droplet_ip>>`. The crawling script is ran as a service, check logs with `tail -f crawl.log`. Access service status or commands with `sudo systemctl stop|restart|start crawl`

If you are asked for any droplet root password at any time, it means you need to rerun `git pull` and `source conda-setup.sh` to refresh the files and regenerate the ssh keys pair.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rvencu/crawlingathome-gpu-hcloud/blob/main/gpucah.ipynb)
## Notebook version
1. open the notebook from Google Colab or Kaggle by looking it up on Github or using direct url https://raw.githubusercontent.com/rvencu/crawlingathome-gpu-hcloud/main/gpucah.ipynb or clicking the button above
2. run all the cells and insert proper values into the form (nickname, Hetzner API token, number of nodes in the swarm)

## scripted notebook run on Kaggle (alpha version)
1. make a Kaggle account and issue an API Token
2. from project folder run `. kaggle.sh`
3. input nickname, Hetzner API token and number of desired nodes in the swarm when asked for
4. the script will stop automatically in 9 hours. relaunch it once per day for 3 days per week
## TODO
- [x] Save image embedding 
- [x] Convert images to tfrecords
- [x] Upload to google drive
- [x] Prevent corrupt image to be processed
- [x] Shard of chunk (it needs to read all WAT file which will be bad for low ram server)
- [x] Crawling@Home integration
- [x] Verify output
- [X] Automate infrastructure from main script
- [X] Replace Pillow with Pillow-SIMD
- [x] Automate nickname as environment variable
- [x] Detect stalled nodes and restart jobs
- [x] Manage GPU process crashes
- [x] Make crash resilient workers
- [x] Spread droplets to all locations to avoid cpu/network competition on same hardware
- [x] Add option to use multiple HCLOUD API keys (to aggregate multiple accounts into the same swarm)
- [x] Add Colab compatible notebook with hcould swarm. Swarm ratio is about 5 nodes for 1 colab notebook
- [x] Add Kaggle automation (launch scripts on Kaggle with GPU)
- [x] Optimize GPU workflow (separate processes for jobs downloading, inference and uploading)
- [x] Optimize cloud workers: do not install unnecessary packages, remove swap file, use ramdisk for downloading images
- [x] Use SSH and SCP libraries from Python instead of subprocess calls (gpu and worker)
- [x] Add deduplication check for top 5M duplicates accrued in 2021 Q2


This work is based on code written by:
- https://github.com/TheoCoombes/crawlingathome
- https://github.com/Wikidepia/crawlingathome-worker

This is a subproject ran by the community around https://github.com/lucidrains/DALLE-pytorch

## Alternative single computer solutions to contribute to the Crawling@Home dataset
- this notebook that can run in Google Colab and Kaggle: [![Open In Colab] (https://colab.research.google.com/assets/colab-badge.svg)] (https://colab.research.google.com/github/rvencu/crawlingathome-gpu-hcloud/blob/main/gpucah.ipynb) (https://raw.githubusercontent.com/rvencu/crawlingathome-worker/colab-mod-asks/fastcah.ipynb)
- this notebook in Google Colab: [![Open In Colab] (https://colab.research.google.com/assets/colab-badge.svg)] (https://colab.research.google.com/github/ARKseal/crawlingathome-worker/blob/colab-gpu/colab-gpu.ipynb)
- this notebook in Google Colab: [![Open In Colab] (https://colab.research.google.com/assets/colab-badge.svg)] (https://colab.research.google.com/drive/1o8MndyY-l9vaox8pb0xfe7VQXUt8Qq0s)
- this repo for autonomous script (on home computer or cloud virtual computer): https://github.com/rvencu/crawlingathome-worker/tree/master
- this alternate repo for the same: https://github.com/christophschuhmann/crawlingathome-worker
