import argparse
import gc
import hashlib
import multiprocessing as mp
import os
import random
import shutil
import time
import traceback
import warnings
from glob import glob
from io import BytesIO
from urllib.parse import urljoin, urlparse
import uuid

import asks
import ftfy
import pandas as pd
import pycld2 as cld2
import tractor
import trio
import ujson
from bloom_filter2 import BloomFilter
from PIL import Image, ImageFile, UnidentifiedImageError

asks.init('trio')

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486


warnings.filterwarnings("ignore")


def chunk_using_generators(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def remove_bad_chars(text):
    return "".join(c for c in text if c.isprintable())


def parse_wat(content, start, line_count, blocked, bloom_filter):
    dedupes = 0
    valid_data = []
    content.seek(start)
    for _ in range(line_count):
        line = content.readline()

        if "IMG@" not in line:
            continue

        line_str = line.strip()
        data = ujson.loads(line_str)

        linklist = data["Envelope"]["Payload-Metadata"]["HTTP-Response-Metadata"][
            "HTML-Metadata"
        ]["Links"]

        base_url = os.path.dirname(
            data["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
        )  # get base url

        license = "?"
        for e in linklist:
            if "url" in e and "creativecommons.org/licenses/" in e["url"]:
                license = e["url"]
            if "alt" not in e:
                continue
            url = e["url"]

            if any(x in url for x in [".svg", ".gif", "data:image", "javascript:"]):
                continue

            try:
                if urlparse(url).netloc in blocked:
                    continue
            except:
                continue

            alt_text = ftfy.fix_text(e["alt"].replace("\n", " ")).strip()
            try:
                _, _, details = cld2.detect(alt_text)
            except Exception as e:
                alt_text = remove_bad_chars(alt_text)
                _, _, details = cld2.detect(alt_text)

            if details[0][1] == "en":
                if not url.startswith("http"):
                    url = urljoin(base_url, url)

                if hashlib.md5((url + alt_text).encode("utf-8")).hexdigest() in bloom_filter:
                    dedupes += 1
                    continue

                valid_data.append((url, alt_text, license))
    return [
        t for t in {tuple(i) for i in valid_data}
    ], dedupes  # Remove duplicate tuple from list


def process_img_content(response, alt_text, license, sample_id):
    img_output_folder = "save/images/"

    try:
        if len(response.content) < 5000:
            return
        img_data = BytesIO(response.content)
        with Image.open(img_data) as im:
            width, height = im.size
            im_format = im.format
            out_fname = f"{img_output_folder}{str(sample_id)}.{im_format.lower()}"
            if im_format not in ["JPEG", "JPG", "PNG"]:
                return
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out_fname)
    except (KeyError, UnidentifiedImageError):
        return

    return [str(sample_id), out_fname, response.url, alt_text, width, height, license]


async def request_image(datas, start_sampleid):
    tmp_data = []
    session = asks.Session(connections=165)
    session.headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async def _request(data, sample_id):
        url, alt_text, license = data
        try:
            proces = process_img_content(
                await session.get(url, timeout=5), alt_text, license, sample_id
            )
            if proces is not None:
                tmp_data.append(proces)
        except Exception:
            return

    async with trio.open_nursery() as n:
        for data in datas:
            n.start_soon(_request, data, start_sampleid)
            start_sampleid += 1

    with open(f".tmp/{uuid1()}.json", "w") as f:
        ujson.dump(tmp_data, f)
    gc.collect()
    return


def dl_wat(valid_data, first_sample_id):
    # Download every image available
    processed_samples = []
    n_processes = mp.cpu_count()

    if n_processes == 1:
        trio.run(request_image, valid_data, first_sample_id)
    else:
        async def _runtractor():
            async with tractor.open_nursery() as n:
                chunk_size = len(valid_data) // n_processes + 1
                for i, data in enumerate(chunk_using_generators(valid_data, chunk_size)):
                    await n.run_in_actor(
                        request_image, datas=data, start_sampleid=first_sample_id + i * chunk_size
                    )

        trio.run(_runtractor)

    for tmpf in glob(".tmp/*.json"):
        processed_samples.extend(ujson.load(open(tmpf)))
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL",
                 "TEXT", "HEIGHT", "WIDTH", "LICENSE"],
    )


def upload(source: str, client_type: str):
    client_type = client_type.upper()
    target = 'gpujobs' if client_type == 'CPU' else 'CAH'
    options = '-rsh' if client_type == 'CPU' else '-zh'
    return os.system(f'rsync {options} {source} archiveteam@88.198.2.17::{target}')


class FileData:
    def __init__(self, filename):
        self._filename = filename
        self._line_to_position = [0]
        self._length = 0

        with open(self._filename, "r") as f:
            while f.readline():
                self._line_to_position.append(f.tell())
                self._length += 1

    def __getitem__(self, line):
        return self._line_to_position[line]

    def __len__(self):
        return self._length


if __name__ == "__main__":

    # initialize working folders
    output_folder = "./save/"
    img_output_folder = output_folder + "images/"

    # initialize client variables
    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')
    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "anonymous"
    CRAWLINGATHOME_SERVER_URL = "http://cah.io.community/"

    print (f"starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    

    # connect to C@H server and initialize client
    client = None
    while True:
        try:
            client = cah.init(
                url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD, type="CPU"
            )
            break
        except:
            time.sleep(5)

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
            print (f"parsed wat in {round(time.time()-start,2)}")
            start = time.time()

            # convert to dataframe and save to disk (for statistics and generating blocking lists)
            parsed_df = pd.DataFrame(parsed_data, columns=["URL","TEXT","LICENSE"])
            parsed_df.to_csv(output_folder + out_fname + "_parsed.csv", index=False, sep="|")

            # attempt to spread out clusters of links pointing to the same domain name, improves crawling
            random.shuffle(parsed_data) 
            
            lastlinks = len(parsed_data)
            print (f"this job has {lastlinks} links and deduped {deduped} links in {round(time.time()-start,2)}")
            start = time.time()

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
            print (f"downloaded {len(dlparse_df)} in {round(time.time() - start, 2)}")
            print (f"download efficiency {len(dlparse_df)/(time.time() - start)} img/sec")
            print (f"crawl efficiency {lastlinks/(time.time() - start)} links/sec")

            # at this point we finishes the CPU node job, need to make the data available for GPU worker
            prefix = uuid.uuid4().hex
            os.mkdir(prefix)
            os.system(f"mv save/* {prefix}/")
            result = upload(prefix, client.type, f"archiveteam@88.198.2.17::gpujobs")
            if result == 0:
                client.completeJob(f"rsync {prefix}")

            shutil.rmtree(prefix)
            last = round(time.time() - start0)

            print(f"job completed in {last} seconds")
            
        except Exception as e:
            print (e)
            print ("Worker crashed")
            time.sleep(30)