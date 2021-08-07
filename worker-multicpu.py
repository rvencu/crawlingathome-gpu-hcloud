import gc 
import os
import re
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
import pycld2 as cld2
import pandas as pd
from glob import glob
from uuid import uuid1
from io import BytesIO
from requests import get
from threading import Thread
import crawlingathome_client as cah
from bloom_filter2 import BloomFilter
from urllib.parse import urljoin, urlparse
sys.path.append('./crawlingathome-worker/')
from multiprocessing import Process, cpu_count, JoinableQueue
from PIL import Image, ImageFile, UnidentifiedImageError 

import asks
asks.init("trio")

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486

def remove_bad_chars(text):
    # cleanup text so language can be detected
    return "".join(c for c in text if c.isprintable())


def parse_wat(content, start, line_count, blocked, bloom, clipped, want_update, bloom_processing):
    """
    This function checks the wat file content and attempts to extract valid candidates of image urls and alt texts

    input: content = wat file content; start = start line number; line_count = how many lines to parse
            usually a wat file is split in 2 halfs or 2 shards. shard 0 starts at the first line and line_count is about 1/2 of wat file lines
            shard 1 starts at the middle of wat file and ends with the last line of wat
    
    output: a list of tuples (url, text, license)
    """

    import ftfy
    import pycld2 as cld2

    # blocklist-domains.txt contains a list of domains to block based on previous results of CLIP filtering.
    # the domains are not likely to pass CLIP for either bad captions or the content is almost always NSFW

    # failed-domains.txt contains failed domains, i.e. domains with image links and suitable alt texts that actually
    # do not produce any image. domains that mayb dissapeared, or are good at blocking scrapers. List is also learned from
    # past crawling effort
    

    deduped = 0
    clpd = 0
    valid_data = []
    content.seek(start)

    #wait while bloom filters are updating
    while want_update.qsize() > 0:
        time.sleep(5)
    #block updates for a little while
    bloom_processing.put(1) # value does not matter

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
            try:
                if urlparse(url).netloc in blocked:
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
                # reject if pair is a duplicate
                #concat = str(hash(url + alt_text))
                concat = hashlib.md5((url + alt_text).encode("utf-8")).hexdigest()
                if concat in bloom: #duplicates:
                    deduped += 1
                    continue
                if concat in clipped: #duplicates:
                    clpd += 1
                    continue
                valid_data.append((url, alt_text, license))    
    bloom_processing.get()
    bloom_processing.task_done()
    return ([
        t for t in {tuple(i) for i in valid_data}
    ], deduped, clpd)  # use a dict in order to remove duplicate tuples from list



def process_img_content(response, alt_text, license, sample_id, img_output_folder):
    """
    Function to process downloaded image. Use use PIL from pillow-simd 
        (faster than open cv that in return is faster than original pillow)
    
    input: web request response, ALT text, license and sample id

    output: list of image parameters or None if image is rejected
    """

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
            # convert all images to RGB (necessary for CLIP, also CLIP is doing it again so do we need it here?)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out_fname)
    except (KeyError, UnidentifiedImageError):
        return

    return [str(sample_id), out_fname, response.url, alt_text, width, height, license]


async def request_image(datas, start_sampleid, img_output_folder):
    """
    This function initiates many parallel async connections to try download the images from provided links
    
    input: dataset of validated links, the sample id to start with

    output: list of lists with succesfully downloaded images and their parameters. this list is dumped on disk as json file
    """

    tmp_data = []

    # change the number of parallel connections based on CPU speed, network capabilities, etc.
    # the number of 192 is optimized for 1 vCPU droplet at Hetzner Cloud (code CX11)
    session = asks.Session(connections=256)
    # try to make the bot website friendly
    session.headers = {
        "User-Agent": "Crawling at Home Project (http://cah.io.community)",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://commoncrawl.org",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async def _request(data, sample_id):
        url, alt_text, license = data
        # the following 2 lines are related to Trio Instrument to capture events from multiple threads
        # task = trio.lowlevel.current_task()
        # task.custom_sleep_data = None # custom_sleep_data can transport information from thread to main thread
        try:
            proces = process_img_content(
                # tune timeout and connection_timeout to grab more or less files. shorter timeouts will exclude bad performing websites
                await session.get(url, timeout=3, connection_timeout=10), alt_text, license, sample_id, img_output_folder
            )
            if proces is not None:
                tmp_data.append(proces)
                # task.custom_sleep_data = 1
        except Exception:
            return

    # this section launches many parallel requests
    async with trio.open_nursery() as n:
        for data in datas:
            n.start_soon(_request, data, start_sampleid)
            start_sampleid += 1

    fn = uuid1()
    # trio makes sure at this point all async tasks were executed
    with open(f".tmp/{fn}.json", "w") as f:
        ujson.dump(tmp_data, f)
    gc.collect()
    return ujson.load(open(f".tmp/{fn}.json"))


def dl_wat(valid_data, first_sample_id, img_output_folder):
    """
    This function initiates download attempt of validated parsed links
    It launches multithreaded tasks by using trio module
    
    input: dataset of validated links, the sample id to start with

    output: dataframe of downloaded images and their parameters
    """

    # Download every image available
    processed_samples = []
    #trio.run(request_image, valid_data, first_sample_id, instruments=[TrioProgress(len(valid_data), False)] )
    result = trio.run( request_image, valid_data, first_sample_id, img_output_folder )
    processed_samples.extend(result)
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH", "LICENSE"],
    )

def upload(source: str, clientType: str, target: str):
    with tarfile.open(f"{source}.tar.gz", "w:gz") as tar:
        tar.add(source, arcname=os.path.basename(source))
    print(f"client type is {clientType}")
    result = os.system(f"rsync -av {source}.tar.gz {target}")
    if os.path.exists(f"/home/crawl/{source}.tar.gz"):
        os.remove(f"/home/crawl/{source}.tar.gz")
    if os.path.exists(f"/home/crawl/{source}"):
        shutil.rmtree(f"/home/crawl/{source}", ignore_errors=True)
    return result

def updateBloom(want_update: JoinableQueue, queues: JoinableQueue, target):
    while True:
        flag = 0
        for queue in queues:
            flag += queue.qsize()
        if flag==0:
            want_update.put(1) # the value does not matter
            start = time.time()
            if os.path.exists("/home/crawl/crawlingathome-gpu-hcloud/blocklists/"):
                shutil.rmtree("/home/crawl/crawlingathome-gpu-hcloud/blocklists/")
            os.makedirs("/home/crawl/crawlingathome-gpu-hcloud/blocklists/")
            if (os.getenv("CLOUD") in ["hetzner","alibaba"]):
                os.system(f"rsync -av --partial --inplace --progress {target}/*.bin /home/crawl/crawlingathome-gpu-hcloud/blocklists/")
            else:
                os.system(f'wget -m -np -c -U "Crawling@Home" --tries=15 -R "index.html*" "http://the-eye.eu/public/AI/cahblacklists/"')
                os.system("mv ./the-eye.eu/public/AI/cahblacklists/* /home/crawl/crawlingathome-gpu-hcloud/blocklists/")
            print(f"[multicpu bloom] Updated bloom filters in {round(time.time()-start, 2)} sec")
            want_update.get()
            want_update.task_done()
            time.sleep(250)
        else:
            time.sleep(5)

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

def proc_worker(i: int, want_update: JoinableQueue, bloom_processing: JoinableQueue, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL):
    # initialize working folders
    output_folder = f"./save/{i}/"
    img_output_folder = output_folder + "images/"

    # connect to C@H server and initialize client
    client = cah.init(
        url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD, type="CPU"
    )

    # initialize stats variables for previous job
    last = 0
    # this makes a loop to download new jobs while the script is running
    # normally it reads while client.jobCount() > 0
    while client.jobCount() > 0 and client.isAlive():
        try:
            lastext = f". Last job duration: {last}"

            start = time.time()
            start0 = start

            # clear working folders for a new job
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder, ignore_errors=True)

            os.makedirs(img_output_folder)

            # get new job and download the wat file
            client.newJob()
            client.downloadShard(output_folder)
            
            #wait while bloom filters are updating
            while want_update.qsize() > 0:
                time.sleep(5)
            #block updates for a little while
            bloom_processing.put(1) # value does not matter
            bloom = BloomFilter(max_elements=200000000, error_rate=0.05, filename=("/home/crawl/crawlingathome-gpu-hcloud/blocklists/bloom200M.bin",-1))
            clipped = BloomFilter(max_elements=200000000, error_rate=0.05, filename=("/home/crawl/crawlingathome-gpu-hcloud/blocklists/clipped.bin",-1))
            blocked = BloomFilter(max_elements=10000000, error_rate=0.01, filename=("/home/crawl/crawlingathome-gpu-hcloud/blocklists/failed-domains.bin",-1))
            bloom_processing.get()
            bloom_processing.task_done()

            # retrieve job details and determine what part of the wat file to parse
            first_sample_id = int(client.start_id)
            last_sample_id = int(client.end_id)
            shard_of_chunk = client.shard_piece # TODO

            fd = FileData(output_folder+'shard.wat')

            if shard_of_chunk == 0:
                start_index = fd[0]
            if shard_of_chunk == 1:
                start_index = fd[ int(len(fd)*0.5) ]

            lines = int(len(fd)*0.5)

            # compute output file names base
            out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard_of_chunk}"
            print(f"[multicpu {i}] shard {out_fname} acquired in {round(time.time()-start,2)} sec (including bloom updates)")
            start = time.time()

            print (f"[multicpu {i}] sync filters in {round(time.time()-start,2)} sec")         

            # parse valid links from wat file
            with open(output_folder+"shard.wat", "r") as infile:
                parsed_data, deduped, clpd = parse_wat(infile, start_index, lines, blocked, bloom, clipped, want_update, bloom_processing)
            print (f"[multicpu {i}] parsed wat in {round(time.time()-start,2)}")
            os.remove(output_folder+"shard.wat")
            start = time.time()

            # convert to dataframe and save to disk (for statistics and generating blocking lists)
            parsed_df = pd.DataFrame(parsed_data, columns=["URL","TEXT","LICENSE"])
            parsed_df.to_csv(output_folder+out_fname + "_parsed.csv", index=False, sep="|")

            # attempt to spread out clusters of links pointing to the same domain name, improves crawling
            random.shuffle(parsed_data) 
        
            lastlinks = len(parsed_data)
            print (f"[multicpu {i}] this job has {lastlinks} links left; deduped {deduped} and already clipped {clpd}")
        
            start = time.time()            
            # attempt to download validated links and save to disk for stats and blocking lists
            dlparse_df = dl_wat( parsed_data, first_sample_id, img_output_folder)
            dlparse_df["PATH"] = dlparse_df.PATH.apply(lambda x: re.sub(r"^./save/\d{1,2}/(.*)$", r"save/\1", x))
            dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
            dlparse_df.to_csv(output_folder + out_fname + "_unfiltered.csv", index=False, sep="|")
            print (f"[multicpu {i}] downloaded {len(dlparse_df)} in {round(time.time() - start, 2)}")
            print (f"[multicpu {i}] download efficiency {len(dlparse_df)/(time.time() - start)} img/sec")
            print (f"[multicpu {i}] crawl efficiency {lastlinks/(time.time() - start)} links/sec")

            # at this point we finishes the CPU node job, need to make the data available for GPU worker
            prefix = uuid.uuid4().hex
            os.mkdir(f"{prefix}")
            os.system(f"mv {output_folder}/* {prefix}/")
            
            result = upload(prefix, client.type, client.upload_address)
            if result == 0:
                client.completeJob(f"rsync {prefix}")

            last = round(time.time() - start0)

            print(f"[multicpu {i}] job completed in {last} seconds")
           
        except Exception as e:
            print (e)
            print ("[multicpu {i}] Worker crashed")
            time.sleep(60)

if __name__ == "__main__":

    # initialize client variables
    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')
    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "anonymous"
    CRAWLINGATHOME_SERVER_URL = "http://cah.io.community/"

    print (f"starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    if not os.path.exists(".tmp"):
        os.mkdir(".tmp")
    
    #use a queue where bloom updating process announces its intention. if something is in queue, then updating, if removed from queue then free to process at all workers
    want_update = JoinableQueue()

    workers = []
    queues = []
    for i in range ( cpu_count()-1 ):
        #use this queue to annount that bloom is currently processing and please do not update filters. if queue is not empty please wait, if queue is empty you may update filters
        bloom_processing = JoinableQueue()
        queues.append(bloom_processing)
        workers.append(Process(target=proc_worker, args= [i, want_update, bloom_processing, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL], daemon=True))

    Process(target=updateBloom, args= [want_update, queues, "archiveteam@88.198.2.17::bloom"], daemon=True).start()

    for worker in workers:
        worker.start()
        time.sleep(10)
    proc_worker(-1, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL)