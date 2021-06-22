import gc 
import os
import sys
import time
import trio
import ujson
import pipes
import string
import random
import pickle
import shutil
import zipfile
import itertools
import subprocess
import pandas as pd
import infrastructure
from glob import glob
from copy import copy
from tqdm import tqdm
from uuid import uuid1
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse
sys.path.append('./crawlingathome-worker/')
from PIL import Image, ImageFile, UnidentifiedImageError

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
            self.tqdm.update(1)
        if task.custom_sleep_data == 1:
            self.tqdm.update(1)
            self.tqdm.desc = self.tqdm.desc.split(":")[0] + ": [ " + str( int(self.tqdm.desc.split(":")[1].split(" ")[2]) + 1 ) + " ] / Links "
            self.tqdm.refresh()

def zipfolder(filename, target_dir):            
    zipobj = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])

def chunk_using_generators(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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
            format = im.format
            out_fname = f"{img_output_folder}{str(sample_id)}.{format.lower()}"
            if format not in ["JPEG", "JPG", "PNG", "WEBP"]:
                return
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out_fname)
    except (KeyError, UnidentifiedImageError):
        return

    return [str(sample_id), out_fname, response.url, alt_text, width, height, license]


async def request_image(datas, start_sampleid):
    import asks
    asks.init("trio")

    tmp_data = []
    session = asks.Session(connections=64)
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
                await session.get(url, timeout=5, connection_timeout=40), alt_text, license, sample_id
            )
            task.custom_sleep_data = 0
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
    trio.run(request_image, valid_data, first_sample_id, instruments=[TrioProgress(len(valid_data), False)] )

    for tmpf in glob(".tmp/*.json"):
        processed_samples.extend(ujson.load(open(tmpf)))
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH", "LICENSE"],
    )


def df_clipfilter(df):
    sim_threshold = 0.3
    underaged_text = ["teen", "kid", "child", "baby"]
    import clip_filter

    clip = clip_filter.CLIP()
    img_embedding, similarities = clip.preprocess_images(df)
    nsfw_filters = clip.filter(img_embedding, clip.categories)
    underage_filters = clip.filter(img_embedding, clip.underaged_categories)
    animal_filters = clip.filter(img_embedding, clip.animal_categories)
    tmp_embed = copy(img_embedding)
    for i, (nsfw_prob, underage_prob, animal_prob, img_embed) in enumerate(
        zip(nsfw_filters, underage_filters, animal_filters, tmp_embed)
    ):
        df.at[i, "similarity"] = similarities[i]
        df.at[i, "NSFW"] = "UNSURE"

        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        # If image is nsfw and (text is containing underaged or image is containing underage or image is containing animal)
        is_nsfw_underaged = (
            df.at[i, "NSFW"] == "NSFW" or df.at[i, "NSFW"] == "UNSURE"
        ) and (
            underage_prob[0] < 4
            or underage_prob[1] < 4
            or any(x in df.at[i, "TEXT"] for x in underaged_text)
            or animal_prob[0] > 20
        )

        unfiltered_df = df

        # Remove image containing underage or not similar image-alttext
        if similarities[i] < sim_threshold or is_nsfw_underaged:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
    df.reset_index(drop=True, inplace=True)
    return df, img_embedding, unfiltered_df


def df_tfrecords(df, output_fname):
    import tensorflow as tf
    from tfr_image.utils import bytes_feature, int64_feature

    def image_to_tfexample(sample_id, image_data, image_format, height, width, caption):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "sampleID": bytes_feature(sample_id),
                    "image": bytes_feature(image_data),
                    "format": bytes_feature(image_format),
                    "label": bytes_feature(caption),
                    "height": int64_feature(height),
                    "width": int64_feature(width),
                }
            )
        )

    with tf.io.TFRecordWriter(output_fname) as tfrecord_writer:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image["PATH"]
            file_type = image_fname.split(".")[-1]
            with tf.io.gfile.GFile(image_fname, "rb") as f:
                image_data = f.read()
            example = image_to_tfexample(
                str(df_image["SAMPLE_ID"]).encode("utf_8"),
                image_data,
                file_type.encode("utf_8"),
                df_image["HEIGHT"],
                df_image["WIDTH"],
                df_image["TEXT"].encode("utf_8"),
            )
            tfrecord_writer.write(example.SerializeToString())


def upload_gdrive(output_filename):
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
    output_folder = "./save/"
    csv_output_folder = output_folder
    img_output_folder = output_folder + "images/"

    node_type = "GPU" # "worker" # changed by cloud-init script

    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')
    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "anonymous"
    CRAWLINGATHOME_SERVER_URL = "http://crawlingathome.duckdns.org/"

    print (f"[infrastructure] starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    if node_type == "worker":
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
                lastext = f". Last job had {lastlinks} links and got {lastcount} img in {last} s = {lasteff} eff"

                start = time.time()

                if os.path.exists(output_folder):
                    shutil.rmtree(output_folder)
                if os.path.exists(".tmp"):
                    shutil.rmtree(".tmp")

                os.mkdir(output_folder)
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
                
                lastlinks = len(parsed_data)

                client.log("Downloading images" + lastext)
                dlparse_df = dl_wat( parsed_data, first_sample_id)
                dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
                print (f"[crawling@home] Downloaded {len(dlparse_df)} in {round(time.time() - start)} seconds")
                print (f"[crawling@home] Download efficiency {len(dlparse_df)/(time.time() - start)} img/sec")

                start2 = time.time()

                client.log("[w/GPU] Dropping NSFW keywords" + lastext)
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
                print (f"[crawling@home] Waiting for GPU node to complete job")
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
                print(f"[crawling@home] Receiving GPU payload")

                # receive GPU results
            
                with zipfile.ZipFile("gpujobdone.zip", 'r') as zip_ref:
                    zip_ref.extractall(".")
                os.remove("gpujobdone.zip")
                os.remove("gpusemaphore")

                client.log("Uploading results" + lastext)
                filtered_df = pd.read_csv(output_folder + out_fname + ".csv", sep="|")
                print (f"[crawling@home] CLIP filtered {len(filtered_df)} in {round(time.time() - start2)} seconds")
                print (f"[crawling@home] CLIP efficiency {len(dlparse_df)/(time.time() - start2)} img/sec")
                upload_gdrive(f"{output_folder}image_embedding_dict-{out_fname}.pkl")
                upload_gdrive(f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord")
                upload_gdrive(output_folder + out_fname + ".csv")
                upload_gdrive(output_folder + out_fname + "_unfiltered.csv")                
                
                # update job stats to be displayed on next run on leaderboard
                lastcount = len(filtered_df)
                last = round(time.time() - start)
                lasteff = round( (filtered_df.shape[0] * 100) / (time.time() - start)) / 100
                
                print(f"[crawling@home] job completed in {last} seconds")
                print(f"[crawling@home] job efficiency {lasteff} pairs/sec")

                client._markjobasdone(len(filtered_df))
            
            except:
                time.sleep(30)


    elif node_type == "GPU":

        nodes = sys.argv[1]
        workers = []

        try:
            start = time.time()
            # generate cloud workers
            workers = trio.run(infrastructure.up, nodes)
            trio.run(infrastructure.wait_for_infrastructure, workers)
            print(f"[infrastructure] {len(workers)} nodes cloud infrastructure is up and initialized in {round(time.time() - start)}s")
        except KeyboardInterrupt:
            print(f"[infrastructure] Abort! Deleting cloud infrastructure...")
            trio.run(infrastructure.down, nodes)
            print (f"[infrastructure] Cloud infrastructure was shutdown")
        except:
            print(f"[infrastructure] Error, could not bring up infrastructure... please consider shutting down all workers via `python3 infrastructure.py down`")

        # poll for new GPU job
        for ip in itertools.cycle(workers): # make sure we cycle all workers
            try:
                print (f"[GPU] Checking {ip} node")
                print (f"[{ip}] " + infrastructure.last_status("crawl@"+ip, '/home/crawl/crawl.log').split("Downloaded:")[-1].rstrip())
                newjob = infrastructure.exists_remote("crawl@"+ip, "/home/crawl/semaphore", True)
                if not newjob:
                    time.sleep(10) # wait until cloud-init finishes then until jobs are ready for GPU
                else:
                    start = time.time()
                    print (f"[{ip}] sending job to GPU")
                    if os.path.exists(output_folder):
                        shutil.rmtree(output_folder)
                    if os.path.exists(".tmp"):
                        shutil.rmtree(".tmp")

                    os.mkdir(output_folder)
                    os.mkdir(img_output_folder)
                    os.mkdir(".tmp")

                    start2 = time.time()
                    time.sleep(30) # wait for the zip file to complete at the worker
                    # receive gpu job data (~500MB)
                    subprocess.call(
                        ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" + ip + ":" + "gpujob.zip", output_folder],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    # delete file on remote so there is no secondary download
                    subprocess.call(
                        ["ssh", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" + ip, "rm -rf gpujob.zip"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    subprocess.call(
                        ["ssh", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" + ip, "rm -rf semaphore"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    with zipfile.ZipFile(output_folder+"gpujob.zip", 'r') as zip_ref:
                        zip_ref.extractall("./")
                    os.remove(output_folder+"gpujob.zip")

                    all_csv_files = []
                    for path, subdir, files in os.walk(output_folder):
                        for file in glob(os.path.join(path, "*.csv")):
                            all_csv_files.append(file)

                    # get name of csv file
                    out_path = all_csv_files[0]
                    out_fname = Path(out_path).stem.strip(".")
                    #print (out_fname)

                    # recreate parsed dataset and run CLIP filtering
                    dlparse_df = pd.read_csv(output_folder + out_fname + ".csv", sep="|")
                    filtered_df, img_embeddings, unfiltered_df = df_clipfilter(dlparse_df)
                    filtered_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
                    unfiltered_df.to_csv(output_folder + out_fname + "_unfiltered.csv", index=False, sep="|")                    
                    img_embeds_sampleid = {}
                    for i, img_embed_it in enumerate(img_embeddings):
                        dfid_index = filtered_df.at[i, "SAMPLE_ID"]
                        img_embeds_sampleid[str(dfid_index)] = img_embed_it
                    with open(f"{output_folder}image_embedding_dict-{out_fname}.pkl", "wb") as f:
                        pickle.dump(img_embeds_sampleid, f)
                    
                    df_tfrecords(
                        filtered_df,
                        f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord",
                    )

                    # clean img_output_folder now since we have all results do not want to transfer back all images...
                    try:
                        shutil.rmtree(img_output_folder)
                        os.mkdir(img_output_folder)
                    except OSError as e:
                        print("[GPU] Error deleting images: %s - %s." % (e.filename, e.strerror))

                    # send GPU results
                    subprocess.call(
                        ["zip", "-r", "gpujobdone.zip", output_folder],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    subprocess.call(
                        ["touch", "gpusemaphore"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                    subprocess.call(
                        ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "gpujobdone.zip", "crawl@"+ip + ":~/gpujobdone.zip"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    subprocess.call(
                        ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "gpusemaphore", "crawl@"+ip + ":~/gpusemaphore"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    os.remove("gpujobdone.zip")
                    os.remove("gpusemaphore")

                    print(f"[GPU] GPU job completed in {round(time.time() - start2)} seconds")
                    print (f"[{ip}] resuming job with GPU results")
                    
            except KeyboardInterrupt:
                print(f"[GPU] Abort! Deleting cloud infrastructure...")
                letters = string.ascii_lowercase
                suffix = ''.join(random.choice(letters) for i in range(3))
                for ip in workers:
                    subprocess.call(
                            ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" + ip + ":" + "crawl.log", ip + "_" + suffix + ".log"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                trio.run(infrastructure.down, nodes)
                print (f"[infrastructure] Cloud infrastructure was shutdown")
            
            except:
                # todo shutdown and restart the offending ip
                
                print (f"[GPU] fault detected in job at cah-worker-{index}. Respawning offending worker...")
                workers = trio.run(infrastructure.respawn, workers, ip)
                continue

    else:
        print (f"[infrastructure] You must set node type between worker node and GPU node")