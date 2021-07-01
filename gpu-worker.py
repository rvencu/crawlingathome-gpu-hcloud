import os
import sys
import time
import trio
import string
import random
import ujson
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
from pathlib import Path
sys.path.append('./crawlingathome-worker/')
from PIL import Image, ImageFile, UnidentifiedImageError
from multiprocessing import Pool, JoinableQueue, Process, Manager

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486

def df_clipfilter(df):
    sim_threshold = 0.3
    underaged_text = ["teen", "kid", "child", "baby"]
    import clip_filter

    clip = clip_filter.CLIP()
    img_embedding, similarities = clip.preprocess_images(df)
    tmp_embed = copy(img_embedding)
    for i, img_embed in enumerate(tmp_embed):
        if similarities[i] < sim_threshold:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
            continue

        # get most similar categories
        nsfw_prob = clip.prob(img_embed, clip.categories)
        df.at[i, "NSFW"] = "UNSURE"
        df.at[i, "similarity"] = similarities[i]
        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
            continue
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        underage_prob = clip.prob(img_embed, clip.underaged_categories)
        if (
            underage_prob[0] < 4
            or underage_prob[1] < 4
            or any(x in df.at[i, "TEXT"] for x in underaged_text)
        ):
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
            continue

        animal_prob = clip.prob(img_embed, clip.animal_categories)
        if animal_prob[0] > 20:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)

    df.reset_index(drop=True, inplace=True)
    return df, img_embedding


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

def incoming_worker(clients, queue):
    print (f"inbound worker started")

    async def _get_job(clients, i, queue):
            clients[i].newJob()

            output_folder = "./" + str(i) + "/save/"
            img_output_folder = output_folder + "images/"

            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            if os.path.exists(".tmp"):
                shutil.rmtree(".tmp")

            os.makedirs(output_folder)
            os.makedirs(img_output_folder)
            os.makedirs(".tmp")

            try:
                clients[i].downloadShard() # <- extracts all images from TAR into ./images/...
                queue.put(i)
                print(f"{i} inserted to the inbound queue")
            except Exception:
                print("download failed")
                clients[i].invalidURL()

    async def find_jobs(clients, queue):
        async with trio.open_nursery() as n:
            for i in range(len(clients)):
                n.start_soon(_get_job, clients, i, queue)
    
    while True:
        if queue.qsize()==0:
            trio.run(find_jobs, clients, queue)
        else:
            time.sleep(1)

def outgoing_worker(clients, queue):
    print (f"outbound worker started")
    while True:
        if queue.qsize() > 0:
            i = queue.get()
            base = "./" + str(i)
            output_folder = base + "/save/"
            img_output_folder = output_folder + "images/"
            # clean img_output_folder now since we have all results do not want to transfer back all images...
            try:
                shutil.rmtree(img_output_folder)
            except OSError as e:
                print("[GPU] Error deleting images: %s - %s." %
                        (e.filename, e.strerror))

            # send GPU results


            queue.task_done()
        else:
            time.sleep(1)

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

if __name__ == "__main__":

    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')
    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "anonymous"
    CRAWLINGATHOME_SERVER_URL = "https://api.gagepiracy.com:4483/"

    print(
        f"[GPU] starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    import crawlingathome as cah

    clients = []
    for i in range(5):
        clients[i] = cah.init(
            url=CRAWLINGATHOME_SERVER_URL,
            nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD,
            type=cah.core.GPUClient # OR type="GPU"
        )

    inbound = JoinableQueue()
    outbound = JoinableQueue()

    inb = Process(target=incoming_worker, args=[clients, inbound], daemon=True).start()
    time.sleep(1)
    otb = Process(target=outgoing_worker, args=[clients, outbound], daemon=True).start()
    time.sleep(1)

    probar = tqdm(total=20, desc="Executed GPU jobs", position=2, bar_format='{desc}: {n_fmt} ({rate_fmt})                    ')
    incbar = tqdm(total=20, desc="Inbound pipeline", position=1, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:0.0f}%)                    ')
    outbar = tqdm(total=20, desc="Outbound pipeline", position=0, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:0.0f}%)                    ')

    while True:
        incbar.n = inbound.qsize()
        outbar.n = outbound.qsize()
        incbar.refresh()
        outbar.refresh()
        while inbound.qsize() > 0:
            ip = inbound.get()
            #print(f"gpu processing job for {ip}")
            output_folder = "./" + ip.replace(".", "-") + "/save/"
            img_output_folder = output_folder + "images/"

            all_csv_files = []
            for path, subdir, files in os.walk(output_folder):
                for file in glob(os.path.join(path, "*.csv")):
                    all_csv_files.append(file)

            # get name of csv file
            out_path = all_csv_files[0]
            out_fname = Path(out_path).stem.strip("_unfiltered").strip(".")

            # recreate parsed dataset and run CLIP filtering
            dlparse_df = pd.read_csv(
                output_folder + out_fname + ".csv", sep="|")

            dlparse_df["PATH"] = "./" + \
                ip.replace(".", "-") + "/" + dlparse_df["PATH"]

            filtered_df, img_embeddings = df_clipfilter(dlparse_df)
            filtered_df.to_csv(output_folder + out_fname +
                            ".csv", index=False, sep="|")

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

            outbound.put(ip)
            inbound.task_done()
            incbar.n = inbound.qsize()
            outbar.n = outbound.qsize()
            incbar.refresh()
            outbar.refresh()
            probar.update(1)
