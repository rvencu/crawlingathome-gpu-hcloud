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
import ftfy
import ujson
import shutil
import random
import hashlib
import tarfile
import requests
import numpy as np
import pandas as pd
import gcld3
from _thread import *
from uuid import uuid1
from io import BytesIO
from datetime import datetime
#import crawlingathome_client as cah
from urllib.parse import urljoin, urlparse
sys.path.append('./crawlingathome-worker/')
from multiprocessing import Process, cpu_count
from crawlingathome_client.temp import TempCPUWorker
from PIL import Image, ImageFile, UnidentifiedImageError 

from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem

import asks
asks.init("trio")

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

class Tracer(trio.abc.Instrument):

    def __init__(self, i=""):
        self.exceptions = 0
        self.requests = 0
        self.downloads = 0
        self.imgproc_duration = 0
        self.download_duration = 0
        self.error_duration = 0
        self.i = i

    def task_exited(self, task):
        if task.custom_sleep_data is not None:
            if task.custom_sleep_data[0] in [1, 3]: # this is exception
                self.exceptions += 1
                self.error_duration += task.custom_sleep_data[2]
            if task.custom_sleep_data[0] == 0: # this is image downloaded
                self.download_duration += task.custom_sleep_data[1]
                self.imgproc_duration += task.custom_sleep_data[2]
                self.downloads += 1

    def after_run(self):
        rate = round(self.exceptions / (self.exceptions + self.downloads + sys.float_info.epsilon), 2)
        avg_download = round(self.download_duration / (self.downloads + sys.float_info.epsilon), 2)
        avg_process = round(self.imgproc_duration / (self.downloads + sys.float_info.epsilon), 2)
        avg_error = round(self.error_duration / (self.exceptions + sys.float_info.epsilon), 2)
        print(f"[{self.i} instrumentation] While scraping there were {self.exceptions} errors within {self.downloads + self.exceptions} candidates (error rate = {round(rate * 100,2)} %). {self.downloads} images were downloaded.")
        print(f"[{self.i} instrumentation] Cumulative image processing duration {round(self.imgproc_duration, 2)} s.")
        print(f"[{self.i} instrumentation] Average downloading time {avg_download} s/img, image processing time {avg_process} s/img, exceptions processing time {avg_error} s/link")

def log(e):
    with open("errors.txt","a") as f:
        f.write(str(e.__class__.__name__) + " " + str(e) + "\n")

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

    bloomip = "116.202.162.146"
    bloom2ip = "94.130.167.172"

    print (f"[{i} parser] start parsing")  

    clpd = 0
    valid_data = []
    check_flag = set() # track urls and make them unique
    content.seek(start)
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
            try:
                domain = urlparse(url).netloc
            except:
                continue
            # detect ALT text language, we want to retain only English captions
            alt_text = ftfy.fix_text(e["alt"].replace("\n", " ")).strip()
            detector = gcld3.NNetLanguageIdentifier(min_num_bytes=6, max_num_bytes=1000)
            detlang = ""
            try:
                res = detector.FindLanguage(alt_text)
                detlang = res.language
            except Exception as e:
                alt_text = remove_bad_chars(alt_text)
                res = detector.FindLanguage(alt_text)
                detlang = res.language            # keep pair if we made it so far
            if detlang == "en":
                if not url.startswith("http"):
                    url = urljoin(base_url, url)
                hash = hashlib.md5((url + alt_text).encode("utf-8")).hexdigest()
                if url not in check_flag:
                    valid_data.append((url, alt_text, license, domain, hash))
                    check_flag.add(url)
            
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
        print(f"crash, cannot contact the clipped bloom server, please fix")
        return

    valid_hashes = response.content.decode("utf-8").split("\n")

    print(f"[debug] clipped bloom server returned {len(valid_hashes)} in {round(time.time()-s,3)} sec")

    valid_data = [t for t in {tuple(i) for i in valid_data}]
    kept_data = []
    clpd = len(valid_data)

    for item in valid_data:
        if item[-1].strip() in valid_hashes:
            kept_data.append(item)
            clpd -= 1
    
    s = time.time()
    # remove from valid_data elements rejected by parsed bloom server
    with open('hash.txt', 'w') as f:
        for item in kept_data:
            f.write(item[0].strip()+"\n")
    post = {
        'file': ('hash.txt', open('hash.txt', 'rb')),
        'key': (None, 'parsed'),
    }
    
    failure = True
    for _ in range(5):
        response = requests.post(f'http://{bloom2ip}:8000/deduplicate/', files=post)
        if response.status_code != 200:
            print(f"bloom server error, retrying...")
            time.sleep(1)            
        else:
            failure = False
            break
    if failure:
        print(f"crash, cannot contact the parsed bloom server, please fix")
        return

    valid_urls = response.content.decode("utf-8").split("\n")

    print(f"[debug] parsed bloom server returned {len(valid_urls)} in {round(time.time()-s,3)} sec")

    valid_data = [t for t in {tuple(i) for i in kept_data}]
    final_kept_data = []
    prsd = len(kept_data)

    for item in kept_data:
        if item[0].strip() in valid_urls:
            final_kept_data.append(item)
            prsd -= 1

    print(f"[{i} parser] lenght of deduplicated pairs to return {len(final_kept_data)}")

    return (final_kept_data, clpd, prsd)  # use a dict in order to remove duplicate tuples from list


def process_img_content(response, alt_text, license, sample_id, img_output_folder):
    """
    Function to process downloaded image. Use use PIL from pillow-simd 
        (faster than open cv that in return is faster than original pillow)
    
    input: web request response, ALT text, license and sample id

    output: list of image parameters or None if image is rejected
    """

    def _resize(im: Image):
        width, height = im.size
        ratio = min(width, height) / 224
        new_width = int(round(width/ratio,0))
        new_height = int(round(height/ratio,0))
        im = im.resize((new_width, new_height), resample=Image.BICUBIC)
        if new_width > 224 or new_height > 224:
            left = (new_width - 224)/2
            top = (new_height - 224)/2
            right = (new_width + 224)/2
            bottom = (new_height + 224)/2
            # Crop the center of the image
            im = im.crop((left, top, right, bottom))
        return im
    try:
        # reject too small images
        if len(response.content) < 5000:
            return
        img_data = BytesIO(response.content)
        with Image.open(img_data) as im:
            width, height = im.size
            # reject if too large (might be a DOS decompression bomb)
            if width * height > 89478484:
                return
            im_format = im.format
            out_fname = f"{img_output_folder}{str(sample_id)}.{im_format.lower()}"
            # reject if format is not in this list
            if im_format not in ["JPEG", "JPG", "PNG", "WEBP"]:
                return
            if min(width, height) > 224:
                im = _resize(im)
            
            # convert all images to RGB (necessary for CLIP, also CLIP is doing it again so do we need it here?)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out_fname)
    except (KeyError, UnidentifiedImageError):
        return

    return [str(sample_id), out_fname, response.url, alt_text, width, height, license]


async def request_image(datas, start_sampleid, img_output_folder, tmp_folder):
    """
    This function initiates many parallel async connections to try download the images from provided links
    
    input: dataset of validated links, the sample id to start with

    output: list of lists with succesfully downloaded images and their parameters. this list is dumped on disk as json file
    """

    tmp_data = []
    limit = trio.CapacityLimiter(512)

    software_names = [SoftwareName.CHROME.value]
    operating_systems = [OperatingSystem.LINUX.value]   

    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=2000)
    user_agent = user_agent_rotator.get_random_user_agent()

    # change the number of parallel connections based on CPU speed, network capabilities, etc.
    # the number of 192 is optimized for 1 vCPU droplet at Hetzner Cloud (code CX11)
    session = asks.Session(connections=64, ssl_context=ssl_ctx)
    # try to make the bot website friendly
    session.headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://google.com",
        "DNT": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml,image/;q=0.9,*/*;q=0.8",
    }

    async def _request(data, sample_id, img_output_folder):
        async with limit:
            start=time.time()
            url, alt_text, license, domain, hash = data
            # the following 2 lines are related to Trio Instrument to capture events from multiple threads
            # task = trio.lowlevel.current_task()
            # task.custom_sleep_data = None # custom_sleep_data can transport information from thread to main thread
            task = trio.lowlevel.current_task()
            try:
                response = await session.get(url, timeout=10, connection_timeout=20)
                dltime = round(time.time()-start, 2)
                start=time.time()
                proces = process_img_content(
                    # tune timeout and connection_timeout to grab more or less files. shorter timeouts will exclude bad performing websites
                    response, alt_text, license, sample_id, img_output_folder
                )
                proctime = round(time.time()-start, 2)
                task.custom_sleep_data = (0, dltime, proctime) # for success do not count errors
                if proces is not None:
                    tmp_data.append(proces)
                
            except Exception as e:
                log(e)
                task.custom_sleep_data = (1, 0, round(time.time()-start,2)) # when exception is hit, count it
        return

    # this section launches many parallel requests
    async with trio.open_nursery() as n:
        for data in datas:
            async with limit:
                n.start_soon(_request, data, start_sampleid, img_output_folder)
            start_sampleid += 1

    fn = uuid1()
    # trio makes sure at this point all async tasks were executed
    with open(f"{tmp_folder}/{fn}.json", "w") as f:
        ujson.dump(tmp_data, f)
    gc.collect()

    # add downloaded urls to parsed bloom server
    bloom2ip = "94.130.167.172"
    with open('hash.txt', 'w') as f:
        for item in datas:
            f.write(item[0].strip()+"\n")
    post = {
        'file': ('hash.txt', open('hash.txt', 'rb')),
        'key': (None, 'parsed'),
    }
    
    failure = True
    for _ in range(5):
        response = requests.post(f'http://{bloom2ip}:8000/add/', files=post)
        if response.status_code != 200:
            print(f"bloom server error, retrying...")
            time.sleep(1)            
        else:
            failure = False
            break
    if failure:
        print(f"crash, cannot contact the parsed bloom server, please fix")

    return ujson.load(open(f"{tmp_folder}/{fn}.json"))


def dl_wat(valid_data, first_sample_id, img_output_folder, tmp_folder, i):
    """
    This function initiates download attempt of validated parsed links
    It launches multithreaded tasks by using trio module
    
    input: dataset of validated links, the sample id to start with

    output: dataframe of downloaded images and their parameters
    """

    # Download every image available
    processed_samples = []
    #trio.run(request_image, valid_data, first_sample_id, instruments=[TrioProgress(len(valid_data), False)] )
    result = trio.run( request_image, valid_data, first_sample_id, img_output_folder, tmp_folder, instruments=[Tracer(i)])
    processed_samples.extend(result)
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH", "LICENSE"],
    )

def upload(source: str, clientType: str, target: str):
    with tarfile.open(f"{source}.tar.gz", "w:gz") as tar:
        tar.add(source, arcname=os.path.basename(source))
    result = os.system(f"rsync -av {source}.tar.gz {target}")
    if os.path.exists(f"{source}.tar.gz"):
        os.remove(f"{source}.tar.gz")
    if os.path.exists(f"{source}"):
        shutil.rmtree(f"{source}", ignore_errors=True)
    return result
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
    output_folder = f"./{i}/save/"
    img_output_folder = output_folder + "images/"
    tmp_folder = f"./{i}/.tmp/"
    bloom_folder = f"./{i}/.bloom/"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    if os.path.exists(bloom_folder):
        shutil.rmtree(bloom_folder)

    os.makedirs(img_output_folder)
    os.makedirs(tmp_folder)
    os.makedirs(bloom_folder)

    # connect to C@H server and initialize client
    client = TempCPUWorker(url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD)

    # initialize stats variables for previous job
    last = 0
    #localbloom = BloomFilter(max_elements=1000000000, error_rate=0.05, filename=("crawlingathome-gpu-hcloud/localbloom.bin",-1))

    # this makes a loop to download new jobs while the script is running
    # normally it reads while client.jobCount() > 0
    while True:
        try:
            lastext = f". Last job duration: {last}"
            print(f"[{i} multicpu] clock is {datetime.now().strftime('%H:%M:%S')}")

            start = time.time()
            start0 = start

            # get new job and download the wat file
            client.newJob()
            client.downloadWat(tmp_folder)

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
                out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard}"
                print(f"[{i} multicpu] shard {out_fname} acquired in {round(time.time()-start,2)} sec")
                
                start = time.time()
                # parse valid links from wat file
                with open(tmp_folder + "shard.wat", "r") as infile:
                    parsed_data, clpd, prsd = parse_wat(infile, start_index, lines, i)
                print (f"[{i} multicpu] parsed wat in {round(time.time()-start,2)}")
                start = time.time()

                # convert to dataframe and save to disk (for statistics and generating blocking lists)
                parsed_df = pd.DataFrame(parsed_data, columns=["URL","TEXT","LICENSE","DOMAIN","HASH"])
                parsed_df = parsed_df.drop_duplicates(subset=["URL"])
                parsed_df.to_csv(output_folder+out_fname + "_parsed.csv", index=False, sep="|")

                # attempt to spread out clusters of links pointing to the same domain name, improves crawling
                random.shuffle(parsed_data) 
            
                lastlinks = len(parsed_data)
                print (f"[{i} multicpu] this job has {lastlinks} links left after removing {clpd} already clipped and {prsd} already parsed")
            
                start = time.time()            
                # attempt to download validated links and save to disk for stats and blocking lists
                dlparse_df = dl_wat( parsed_data, first_sample_id, img_output_folder, tmp_folder, i )
                dlparse_df["PATH"] = dlparse_df.PATH.apply(lambda x: re.sub(r"^./save/\d{1,2}/(.*)$", r"save/\1", x))
                dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
                dlparse_df.to_csv(output_folder + out_fname + "_unfiltered.csv", index=False, sep="|")
                print (f"[{i} stats shard {shard_of_chunk}] pairs retained {len(dlparse_df)} in {round(time.time() - start, 2)}")
                print (f"[{i} stats shard {shard_of_chunk}] scraping efficiency {len(dlparse_df)/(time.time() - start)} img/sec")
                print (f"[{i} stats shard {shard_of_chunk}] crawling efficiency {lastlinks/(time.time() - start)} links/sec")

                # at this point we finishes the CPU node job, need to make the data available for GPU worker
                prefix = uuid.uuid4().hex
                prefixes[str(client.shards[shard_of_chunk][0])] = f"rsync {prefix}"
                os.mkdir(f"{prefix}")
                os.system(f"mv {output_folder}/* {prefix}/")
                
                result += upload(prefix, "CPU", client.upload_address)
            if result == 0:
                client.completeJob(prefixes)

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

    procs = cpu_count()
    if len(sys.argv) > 1:
        procs = int(sys.argv[1])

    workers = []
    for i in range ( procs ):
        #use this queue to annount that bloom is currently processing and please do not update filters. if queue is not empty please wait, if queue is empty you may update filters
        workers.append(Process(target=proc_worker, args= [i, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL], daemon=True))

    time.sleep(10)

    for worker in workers:
        worker.start()
        time.sleep(8)
    
    while True:
        #keep main process alive
        time.sleep(60)