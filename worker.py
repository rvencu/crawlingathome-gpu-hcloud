import gc 
import os
import sys
import time
import trio
import ujson
import shutil
import random
import hashlib
import zipfile
import pandas as pd
from glob import glob
from uuid import uuid1
from io import BytesIO
from requests import get
from bloom_filter2 import BloomFilter
from urllib.parse import urljoin, urlparse
sys.path.append('./crawlingathome-worker/')
from PIL import Image, ImageFile, UnidentifiedImageError 

import asks
asks.init("trio")

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486


def zipfolder(filename, target_dir):
    """
    Function to unzip files received from GPU node

    input: filename to unzip, target_dir location where to unzip the content

    output: does not return an output

    """

    zipobj = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])

def remove_bad_chars(text):
    # cleanup text so language can be detected
    return "".join(c for c in text if c.isprintable())


def parse_wat(content, start, line_count, blocked, bloom):
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
    valid_data = []
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
                valid_data.append((url, alt_text, license))
    return ([
        t for t in {tuple(i) for i in valid_data}
    ], deduped)  # use a dict in order to remove duplicate tuples from list


def process_img_content(response, alt_text, license, sample_id):
    """
    Function to process downloaded image. Use use PIL from pillow-simd 
        (faster than open cv that in return is faster than original pillow)
    
    input: web request response, ALT text, license and sample id

    output: list of image parameters or None if image is rejected
    """

    img_output_folder = "save/images/"
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


async def request_image(datas, start_sampleid):
    """
    This function initiates many parallel async connections to try download the images from provided links
    
    input: dataset of validated links, the sample id to start with

    output: list of lists with succesfully downloaded images and their parameters. this list is dumped on disk as json file
    """

    tmp_data = []

    # change the number of parallel connections based on CPU speed, network capabilities, etc.
    # the number of 192 is optimized for 1 vCPU droplet at Hetzner Cloud (code CX11)
    session = asks.Session(connections=192)
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
                await session.get(url, timeout=3, connection_timeout=10), alt_text, license, sample_id
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

    # trio makes sure at this point all async tasks were executed
    with open(f".tmp/{uuid1()}.json", "w") as f:
        ujson.dump(tmp_data, f)
    gc.collect()
    return


def dl_wat(valid_data, first_sample_id):
    """
    This function initiates download attempt of validated parsed links
    It launches multithreaded tasks by using trio module
    
    input: dataset of validated links, the sample id to start with

    output: dataframe of downloaded images and their parameters
    """

    import pandas as pd
    
    # Download every image available
    processed_samples = []
    #trio.run(request_image, valid_data, first_sample_id, instruments=[TrioProgress(len(valid_data), False)] )
    trio.run( request_image, valid_data, first_sample_id )

    for tmpf in glob(".tmp/*.json"):
        processed_samples.extend(ujson.load(open(tmpf)))
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH", "LICENSE"],
    )

def upload_gdrive(output_filename, unfiltered=False):
    """
    This function automates data upload to common repository (gdrive)
    
    input: filname to upload, unfiltered False for basic location, True for alternate location (files not supposed to be part of the dataset, for internal use)
    
    output: None
    """
    
    import requests

    client_id = (
        "648172777761-onv1nc5f93nhlhf63flsq6onrmjphpfo.apps.googleusercontent.com"
    )
    client_secret = "HZ4Zw-_jVJ-3mwicz1NM5W5x"
    refresh_token = "1//04N2Kysz1LObLCgYIARAAGAQSNwF-L9IrntHNWi2_nEVu2QX5fmlW0Ea0qA-ToBJLSdatDATYxiKcNFI8eZQ_fYN53gjF7b8MGmA"

    def refresh_gdrive_token():
        params = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }

        authorization_url = "https://www.googleapis.com/oauth2/v4/token"

        r = requests.post(authorization_url, data=params)

        if r.ok:
            return r.json()["access_token"]
        else:
            return None

    access_t = refresh_gdrive_token()
    headers = {"Authorization": "Bearer " + access_t}
    
    para = {
        "name": output_filename.split("/")[-1],
        "parents": ["1CIgcIR7nX2xNBPB577jwEqbbwxAJR_nt"],
    }
    if unfiltered:
        para = {
        "name": output_filename.split("/")[-1],
        "parents": ["1j1CVjxRNgxTwx5dVKz_rbrIKKdniL3WC"],
    }    

    files = {
        "data": ("metadata", ujson.dumps(para), "application/json; charset=UTF-8"),
        "file": ("application/zip", open(output_filename, "rb")),
    }
    requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files,
    )

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

if __name__ == "__main__":

    # helper function to find worker IP
    myip = ip = get('https://api.ipify.org').text

    # initialize working folders
    output_folder = "./save/"
    img_output_folder = output_folder + "images/"

    # initialize client variables
    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')
    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "anonymous"
    CRAWLINGATHOME_SERVER_URL = "http://cah.io.community/"

    print (f"starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    import crawlingathome_client as cah

    # connect to C@H server and initialize client
    client = None
    while True:
        try:
            client = cah.init(
                url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD
            )
            break
        except:
            time.sleep(5)

    # initialize stats variables for previous job
    last = 0
    lasteff = 0
    lastcount = 0
    lastlinks = 0

    # this makes a loop to download new jobs while the script is running
    # normally it reads while client.jobCount() > 0
    while True: # since we are so early into the project...
        try:
            lastext = f". Last job duration: {last}"

            start = time.time()
            start0 = start

            # clear working folders for a new job
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder, ignore_errors=True) # fix for ramdisk already existing at location
            if os.path.exists(".tmp"):
                shutil.rmtree(".tmp")

            os.mkdir(output_folder)
            os.mkdir(img_output_folder)
            os.mkdir(".tmp")

            # get new job and download the wat file
            while True:
                try:
                    client.newJob()
                    client.downloadShard()
                except:
                    time.sleep(30)
                    continue
                break
            
            # retrieve job details and determine what part of the wat file to parse
            first_sample_id = int(client.start_id)
            last_sample_id = int(client.end_id)
            shard_of_chunk = client.shard_piece # TODO

            fd = FileData('shard.wat')

            if shard_of_chunk == 0:
                start_index = fd[0]
            if shard_of_chunk == 1:
                start_index = fd[ int(len(fd)*0.5) ]

            lines = int(len(fd)*0.5)

            # compute output file names base
            out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard_of_chunk}"
            print(time.time()-start)
            start = time.time()
            print (f"[crawling@home] shard id {out_fname}") # in case test fails, we need to remove bad data

            blocked = set()
            with open("crawlingathome-gpu-hcloud/blocklists/blocklist-domain.txt","r") as f:
                blocked = set(f.read().splitlines())
            failed = set()
            with open("crawlingathome-gpu-hcloud/blocklists/failed-domains.txt","r") as f:
                failed = set(f.read().splitlines())
            blocked |= failed # merge the 2 sets and use this to reduce the number of attempted links, reduce crawling time.
            '''
            duplicates = set()
            with open("crawlingathome-gpu-hcloud/blocklists/5Mduplicates.txt","rt") as f:
                duplicates = set(f.read().splitlines())
            print (f"duplicates of size {len(duplicates)}")
            '''
            bloom = BloomFilter(max_elements=10000000, error_rate=0.01, filename=("crawlingathome-gpu-hcloud/blocklists/bloom.bin",-1))

            while True:
                try:
                    client.log("Processing shard" + lastext)
                except:
                    time.sleep(5)
                    continue
                break

            # parse valid links from wat file
            with open("shard.wat", "r") as infile:
                parsed_data, deduped = parse_wat(infile, start_index, lines, blocked, bloom)
            print(time.time()-start)
            start = time.time()
            print ("parsed wat")

            # convert to dataframe and save to disk (for statistics and generating blocking lists)
            parsed_df = pd.DataFrame(parsed_data, columns=["URL","TEXT","LICENSE"])
            parsed_df.to_csv(output_folder + out_fname + "_parsed.csv", index=False, sep="|")

            # attempt to spread out clusters of links pointing to the same domain name, improves crawling
            random.shuffle(parsed_data) 
            
            lastlinks = len(parsed_data)
            print(time.time()-start)
            start = time.time()
            print (f"this job has {lastlinks} links and deduped {deduped} links")

            while True:
                try:
                    client.log("Downloading images" + lastext)
                except:
                    time.sleep(5)
                    continue
                break
            
            # attempt to download validated links and save to disk for stats and blocking lists
            dlparse_df = dl_wat( parsed_data, first_sample_id)
            dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
            dlparse_df.to_csv(output_folder + out_fname + "_unfiltered.csv", index=False, sep="|")
            print (f"downloaded {len(dlparse_df)} in {round(time.time() - start)} seconds")
            print (f"download efficiency {len(dlparse_df)/(time.time() - start)} img/sec")
            print (f"crawl efficiency {lastlinks/(time.time() - start)} links/sec")

            start2 = time.time()
            
            # at this point we need to perform CLIP filtering, then save embeddings and tfrecords of filtered images
            # since inference is best done with GPU, this particular worker is zipping the csv and all downloaded images
            # and sends them to the GPU node
            shutil.make_archive("gpujob", "zip", ".", output_folder)
            # when zip is ready, create semaphore file to signal job data is ready
            with open('semaphore', 'w') as f:
                pass

            while True:
                try:
                    client.log("@GPU: dropping NSFW keywords" + lastext)
                except:
                    time.sleep(5)
                    continue
                break

            # wait for GPU results
            print (f"waiting for GPU node to complete job")
            status = False
            abort = False
            gpulocal = False
            while not (status | abort | gpulocal):
                print(".", end = "", flush=True)
                time.sleep(10)
                status = os.path.exists("gpusemaphore")
                abort = os.path.exists("gpuabort")
                gpulocal = os.path.exists("gpulocal")

            if abort:
                os.remove("gpuabort")
                continue
            print()
            print(f"receiving results from GPU")

            # GPU results received
            
            if gpulocal:
                with open("gpulocal","rt") as f:
                    filtered = f.read().strip()
                os.remove("gpulocal")
            else:
                with zipfile.ZipFile("gpujobdone.zip", 'r') as zip_ref:
                    zip_ref.extractall(".")
                os.remove("gpujobdone.zip")
                os.remove("gpusemaphore")

                while True:
                    try:
                        client.log("Uploading results" + lastext)
                    except:
                        time.sleep(5)
                        continue
                    break
            
                # reconstruct the filtered dataset from received csv, save to file and upload to dataset storage
                filtered_df = pd.read_csv(output_folder + out_fname + ".csv", sep="|")
                print (f"CLIP filtered {len(filtered_df)} in {round(time.time() - start2)} seconds")
                print (f"CLIP efficiency {len(dlparse_df)/(time.time() - start2)} img/sec")

                upload_gdrive(f"{output_folder}image_embedding_dict-{out_fname}.pkl")
                upload_gdrive(f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord")
                upload_gdrive(output_folder + out_fname + ".csv")
                upload_gdrive(output_folder + out_fname + "_unfiltered.csv", True)
                upload_gdrive(output_folder + out_fname + "_parsed.csv", True)

                # update job stats to be displayed on next run on leaderboard
                filtered = len(filtered_df)
            last = round(time.time() - start0)
            #lasteff = round( (filtered_df.shape[0] * 100) / (time.time() - start0)) / 100

            # we use job efficiency as KPI, i.e. the number of final pairs divided by total time taken by the entire job
            print(f"job completed in {last} seconds")
            #print(f"job efficiency {lasteff} pairs/sec")

            while True:
                try:
                    client._markjobasdone(filtered)
                except:
                    time.sleep(5)
                    continue
                break
            
        except Exception as e:
            print (e)
            print ("Worker crashed")
            time.sleep(30)