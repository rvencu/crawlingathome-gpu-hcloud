# use this if you have a spare computer with multiple CPUs and a very good internet link
# command line: python3 worker-multicpu.py N nickname
# where N = max number of CPU to use
# nickname is you nickname for the leaderboard
# examples:
# 10Gbps internet link and 6 million PPS routing - use N = 40 (max)
# 1Gbps internet link and 4 million PPS - suggested N = 30
# 1Gbps internet link and 2 million PPS - N = 15
# 1Gbps internet link and 500,000 PPS - N = 8
# 1Gbps internet link and 100,000 PPS - N = 2
# 1Gbps internet link and 50,000 PPS - N = 1
# most home modems do not allow more than 50,000 PPS for fiber link of 1Gbps. Check with your internet provider

import gc
import os
import re
import ssl
import sys
import time
import trio
import uuid
import math
import ftfy
import ujson
import shutil
import random
import hashlib
import tarfile
import requests
import numpy as np
import pandas as pd
import pycld2 as cld2
from glob import glob
from _thread import *
from uuid import uuid1
from io import BytesIO
from datetime import datetime
import crawlingathome_client as cah
from bloom_filter2 import BloomFilter
from urllib.parse import urljoin, urlparse
sys.path.append('./crawlingathome-worker/')
from multiprocessing import Process, cpu_count, Queue
from crawlingathome_client.temp import TempCPUWorker

def remove_bad_chars(text):
    # cleanup text so language can be detected
    return "".join(c for c in text if c.isprintable())

def parse_wat(content, start, line_count, i):
    """
    This function checks the wat file content and attempts to extract valid candidates of image urls and alt texts

    input: content = wat file content; start = start line number; line_count = how many lines to parse
            usually a wat file is split in 2 halfs or 2 shards. shard 0 starts at the first line and line_count is about 1/2 of wat file lines
            shard 1 starts at the middle of wat file and ends with the last line of wat
    
    output: a list of tuples (url, text, license)
    """

    # blocklist-domains.txt contains a list of domains to block based on previous results of CLIP filtering.
    # the domains are not likely to pass CLIP for either bad captions or the content is almost always NSFW

    # failed-domains.txt contains failed domains, i.e. domains with image links and suitable alt texts that actually
    # do not produce any image. domains that mayb dissapeared, or are good at blocking scrapers. List is also learned from
    # past crawling effort
    bloomip = "116.202.162.146"

    print (f"[{i} parser] start parsing")

    blocked = BloomFilter(max_elements=10000000, error_rate=0.01, filename=(f"/home/crawl/{i}/.bloom/failed-domains.bin",-1))

    clpd = 0
    valid_data = []
    content.seek(start)
    check_flag = set() # track urls and make them unique

    try:
        for _ in range(line_count):
            line = content.readline()
            if "IMG@" not in line:
                continue
            line_str = line.strip()
            data = ujson.loads(line_str)
            # find all links inside the line
            linklist = data["Envelope"]["Payload-Metadata"]["HTTP-Response-Metadata"][
                "HTML-Metadata"
            ]["Links"]
            # get base url
            base_url = os.path.dirname(
                data["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
            )
            license = "?"
            for e in linklist:
                if "url" in e and "creativecommons.org/licenses/" in e["url"]:
                    license = e["url"]
                # reject links if ALT tag is not present
                if "alt" not in e:
                    continue
                url = e["url"]
                # reject links of svg, gif or scripted images content
                if any( x in url for x in [".svg", ".gif", "data:image", "javascript:"] ):
                    continue
                # reject links found in blocked list
                domain = "unknown"
                try:
                    domain = urlparse(url).netloc
                    if domain in blocked:
                        continue
                except:
                    # cannot even parse the url
                    continue
                # detect ALT text language, we want to retain only English captions
                alt_text = ftfy.fix_text(e["alt"].replace("\n", " ")).strip()
                try:
                    _, _, details = cld2.detect(alt_text)
                except Exception as e:
                    alt_text = remove_bad_chars(alt_text)
                    _, _, details = cld2.detect(alt_text)
                # keep pair if we made it so far
                if details[0][1] == "en":
                    if not url.startswith("http"):
                        url = urljoin(base_url, url)
                    hash = hashlib.md5((url + alt_text).encode("utf-8")).hexdigest()
                    if url not in check_flag:
                        valid_data.append((url, alt_text, license, domain, hash))
                        check_flag.add(url)

    except Exception as e:
        print(f"[{i} parser] parser exception: {e}")
    
    print(f"[debug] lenght of pairs to filter {len(valid_data)}")
    s = time.time()

    # remove from valid_data elements rejected by clipped bloom server
    with open('hash.txt', 'w') as f:
        for item in valid_data:
            f.write(item[-1].strip()+"\n")
    post = {
        'file': ('hash.txt', open('hash.txt', 'rb')),
        'key': (None, 'clipped'),
    }
    
    failure = True
    for _ in range(5):
        response = requests.post(f'http://{bloomip}:8000/deduplicate/', files=post)
        if response.status_code != 200:
            print(f"bloom server error, retrying...")
            time.sleep(1)            
        else:
            failure = False
            break
    if failure:
        print(f"crash, cannot contact the bloom server, please fix")
        sys.exit() # maybe fallback to file based filters? too depressing...

    valid_hashes = response.content.decode("utf-8").split("\n")
    print(f"[debug] bloom server returned {len(valid_hashes)} in {round(time.time()-s,3)} sec")

    valid_data = [t for t in {tuple(i) for i in valid_data}]
    kept_data = []
    clpd = len(valid_data)

    for item in valid_data:
        if item[-1].strip() in valid_hashes:
            kept_data.append(item)
            clpd -= 1

    print (f"[{i} parser] parsed {len(kept_data)} preparing to return")
    return (kept_data, clpd)  # use a dict in order to remove duplicate tuples from list


class FileData:
    """
    Helper class to easily find wat file size, mid position, etc
    """

    def __init__(self, filename):
        self._filename = filename
        self._line_to_position = [0]
        self._length = 0

        with open(self._filename, 'r') as f:
            while f.readline():
                self._line_to_position.append(f.tell())
                self._length += 1
    
    def __getitem__(self, line):
        return self._line_to_position[line]

    def __len__(self):
        return self._length

def proc_worker(i: int, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL):
    # initialize working folders
    output_folder = f"./save/"
    img_output_folder = output_folder + "images/"
    tmp_folder = f"./{i}/.tmp/"

    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    # connect to C@H server and initialize client
    client = TempCPUWorker(url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD)

    # initialize stats variables for previous job
    last = 0
    localbloom = BloomFilter(max_elements=1000000000, error_rate=0.05, filename=("/home/crawl/crawlingathome-gpu-hcloud/localbloom.bin",-1))

    # this makes a loop to download new jobs while the script is running
    # normally it reads while client.jobCount() > 0
    while True:
        try:
            print(f"[{i} multicpu] clock is {datetime.now().strftime('%H:%M:%S')}")

            start = time.time()
            start0 = start

            # get new job and download the wat file
            client.newJob()
            client.downloadWat(tmp_folder)
            client_data = client.dump()
            out_fname = []
            parsed = []

            result = 0
            prefixes = {}

            fd = FileData(tmp_folder + 'shard.wat')
            lines = int(len(fd)*0.5)

            for shard_of_chunk in range(2):

                # clear working folders for a new job
                if os.path.exists(output_folder):
                    shutil.rmtree(output_folder, ignore_errors=True)
                os.makedirs(img_output_folder)

                # retrieve job details and determine what part of the wat file to parse
                first_sample_id = np.int64(client.shards[shard_of_chunk][1]["start_id"])
                last_sample_id = np.int64(client.shards[shard_of_chunk][1]["end_id"])
                shard = client.shards[shard_of_chunk][1]["shard"]
                
                if shard == 0:
                    start_index = fd[0]
                if shard == 1:
                    start_index = fd[ int(len(fd)*0.5) ]

                # compute output file names base
                out_fname.append(f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard}")
                
                start = time.time()
                # parse valid links from wat file
                with open(tmp_folder + "shard.wat", "r") as infile:
                    parsed_data, clpd = parse_wat(infile, start_index, lines, i)
                print (f"[{i} multicpu] parsed wat in {round(time.time()-start,2)}")
                start = time.time()

                # convert to dataframe and save to disk (for statistics and generating blocking lists)
                parsed_df = pd.DataFrame(parsed_data, columns=["URL","TEXT","LICENSE","DOMAIN","HASH"])
                parsed.append(parsed_df.drop_duplicates(subset=["URL"]))
                #parsed_df.to_csv(output_folder+out_fname + "_parsed.csv", index=False, sep="|")

            last = round(time.time() - start0)

            print(f"[{i} stats] WAT job completed in {last} seconds")
           
        except Exception as e:
            print (e)
            print (f"[{i} multicpu] worker crashed")
            time.sleep(60)

if __name__ == "__main__":

    # initialize client variables
    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')

    if len(sys.argv) > 2:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = sys.argv[2]

    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "Caricature, Inc"
    CRAWLINGATHOME_SERVER_URL = "http://cah.io.community/"

    print (f"starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    output_folder = f"./save/"
    img_output_folder = output_folder + "images/"
    bloom_folder = f"./.bloom/"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    if os.path.exists(bloom_folder):
        shutil.rmtree(bloom_folder)

    os.makedirs(img_output_folder)
    os.makedirs(bloom_folder)


    procs = cpu_count() - 16
    if len(sys.argv) > 1:
        procs = min(int(sys.argv[1]), cpu_count() - 5)

    workers = []
    for i in range ( procs ):
        #use this queue to annount that bloom is currently processing and please do not update filters. if queue is not empty please wait, if queue is empty you may update filters
        workers.append(Process(target=proc_worker, args= [i, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL], daemon=True))

    if os.path.exists("/home/crawl/crawlingathome-gpu-hcloud/blocklists/"):
        shutil.rmtree("/home/crawl/crawlingathome-gpu-hcloud/blocklists/")
    os.makedirs("/home/crawl/crawlingathome-gpu-hcloud/blocklists/")
    os.system(f'wget -m -np -c -U "Crawling@Home" --tries=15 -R "index.html*,bloom*.bin,clipped*.bin" "http://the-eye.eu/public/AI/cahblacklists/"')
    os.system("mv ./the-eye.eu/public/AI/cahblacklists/* /home/crawl/crawlingathome-gpu-hcloud/blocklists/")

    time.sleep(10)

    for worker in workers:
        worker.start()
        time.sleep(8)
    
    while True:
        #keep main process alive
        time.sleep(60)