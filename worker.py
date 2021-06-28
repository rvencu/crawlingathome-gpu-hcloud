import gc 
import os
import sys
import time
import trio
import ujson
import pipes
import shutil
import random
import zipfile
import urllib
import subprocess
import pandas as pd
from glob import glob
from uuid import uuid1
from io import BytesIO
from requests import get
from urllib.parse import urljoin
sys.path.append('./crawlingathome-worker/')
from PIL import Image, ImageFile, UnidentifiedImageError 

import asks
asks.init("trio")

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486


class TrioProgress(trio.abc.Instrument):

    def __init__(self, total, notebook_mode=False, **kwargs):
        if notebook_mode:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        self.tqdm = tqdm(total=total, desc="Downloaded: [ 0 ] / Links ", **kwargs)

    def task_exited(self, task):
        if task.custom_sleep_data == 0:
            self.tqdm.update(7)
        if task.custom_sleep_data == 1:
            self.tqdm.update(7)
            self.tqdm.desc = self.tqdm.desc.split(":")[0] + ": [ " + str( int(self.tqdm.desc.split(":")[1].split(" ")[2]) + 1 ) + " ] / Links "
            self.tqdm.refresh()

def zipfolder(filename, target_dir):            
    zipobj = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])

def remove_bad_chars(text):
    return "".join(c for c in text if c.isprintable())


def parse_wat(content, start, line_count):
    import ftfy
    import pycld2 as cld2

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
            alt_text = ftfy.fix_text(e["alt"].replace("\n", " ")).strip()
            if url.endswith(".svg") or url.endswith(".gif") or "data:image" in url:
                continue
            try:
                _, _, details = cld2.detect(alt_text)
            except Exception as e:
                alt_text = remove_bad_chars(alt_text)
                _, _, details = cld2.detect(alt_text)

            if details[0][1] == "en":
                if not url.startswith("http"):
                    url = urljoin(base_url, url)
                valid_data.append((url, alt_text, license))
    return [
        t for t in {tuple(i) for i in valid_data}
    ]  # Remove duplicate tuple from list


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
            if im_format not in ["JPEG", "JPG", "PNG", "WEBP"]:
                return
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out_fname)
    except (KeyError, UnidentifiedImageError):
        return

    return [str(sample_id), out_fname, response.url, alt_text, width, height, license]


async def request_image(datas, start_sampleid):
    tmp_data = []

    session = asks.Session(connections=192)
    session.headers = {
        "User-Agent": "Googlebot-Image",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async def _request(data, sample_id):
        url, alt_text, license = data
        task = trio.lowlevel.current_task()
        task.custom_sleep_data = None
        try:
            proces = process_img_content(
                await session.get(url, timeout=3, connection_timeout=10), alt_text, license, sample_id
            )
            if proces is not None:
                tmp_data.append(proces)
                task.custom_sleep_data = 1
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
    myip = ip = get('https://api.ipify.org').text
    output_folder = "./save/"
    csv_output_folder = output_folder
    img_output_folder = output_folder + "images/"

    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')
    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "anonymous"
    CRAWLINGATHOME_SERVER_URL = "http://api.gagepiracy.com:4483/"

    print (f"starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")


    import crawlingathome_client as cah

    client = cah.init(
        url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD
    )

    last = 0
    lasteff = 0
    lastcount = 0
    lastlinks = 0

    while client.jobCount() > 0:

        try:

            lastext = f". Last job eff: {lasteff}"

            start = time.time()

            if os.path.exists(output_folder):
                shutil.rmtree(output_folder, ignore_errors=True) # fix for ramdisk already existing at location
            if os.path.exists(".tmp"):
                shutil.rmtree(".tmp")

            #os.mkdir(output_folder)
            os.mkdir(img_output_folder)
            os.mkdir(".tmp")

            while True:
                try:
                    client.newJob()
                    client.downloadShard()
                except:
                    time.sleep(30)
                    continue
                break
            
            first_sample_id = int(client.start_id)
            last_sample_id = int(client.end_id)
            shard_of_chunk = client.shard_piece # TODO

            fd = FileData('shard.wat')

            if shard_of_chunk == 0:
                start_index = fd[0]
            if shard_of_chunk == 1:
                start_index = fd[ int(len(fd)*0.5) ]

            lines = int(len(fd)*0.5)

            out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard_of_chunk}"
            print (f"[crawling@home] shard id {out_fname}") # in case test fails, we need to remove bad data

            client.log("Processing shard" + lastext)
            with open("shard.wat", "r") as infile:
                parsed_data = parse_wat(infile, start_index, lines)

            random.shuffle(parsed_data) # attempt to spread out clusters of urls pointing to the same domain name
            
            lastlinks = len(parsed_data)
            print (f"this job has {lastlinks} links")

            client.log("Downloading images" + lastext)
            dlparse_df = dl_wat( parsed_data, first_sample_id)
            dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
            dlparse_df.to_csv(output_folder + out_fname + "_unfiltered.csv", index=False, sep="|")
            print (f"downloaded {len(dlparse_df)} in {round(time.time() - start)} seconds")
            print (f"download efficiency {len(dlparse_df)/(time.time() - start)} img/sec")
            print (f"crawl efficiency {lastlinks/(time.time() - start)} links/sec")

            start2 = time.time()

            client.log("@GPU: dropping NSFW keywords" + lastext)
            # insert GPU job
            subprocess.call(
                ["zip", "-r", "gpujob.zip", output_folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.call(
                ["touch", "semaphore"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # wait for GPU results
            print (f"waiting for GPU node to complete job")
            status = True
            while status:
                print(".", end = "", flush=True)
                time.sleep(10)
                status = subprocess.call(
                    ["test", "-f", "{}".format(pipes.quote("gpusemaphore"))],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            print()
            print(f"receiving results from GPU")

            # receive GPU results
        
            with zipfile.ZipFile("gpujobdone.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("gpujobdone.zip")
            os.remove("gpusemaphore")

            client.log("Uploading results" + lastext)
            filtered_df = pd.read_csv(output_folder + out_fname + ".csv", sep="|")
            print (f"CLIP filtered {len(filtered_df)} in {round(time.time() - start2)} seconds")
            print (f"CLIP efficiency {len(dlparse_df)/(time.time() - start2)} img/sec")
            upload_gdrive(f"{output_folder}image_embedding_dict-{out_fname}.pkl")
            upload_gdrive(f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord")
            upload_gdrive(output_folder + out_fname + ".csv")
            upload_gdrive(output_folder + out_fname + "_unfiltered.csv", True)

            # update job stats to be displayed on next run on leaderboard
            lastcount = len(filtered_df)
            last = round(time.time() - start)
            lasteff = round( (filtered_df.shape[0] * 100) / (time.time() - start)) / 100

            print(f"job completed in {last} seconds")
            print(f"job efficiency {lasteff} pairs/sec")

            client._markjobasdone(len(filtered_df))
        except Exception as e:
            print (e)
            print ("Worker crashed")
            #attempt to solve temporary faliure in name resolution error
            subprocess.call(
                ["sudo", "apt", "clean"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(30)