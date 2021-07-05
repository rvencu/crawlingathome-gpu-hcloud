import os
import sys
import time
import trio
import string
import random
import pickle
import shutil
import zipfile
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


if __name__ == "__main__":

    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')
    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "anonymous"
    CRAWLINGATHOME_SERVER_URL = "http://cah.io.community/"

    print(
        f"[GPU] starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    nodes = sys.argv[1]
    location = None
    skip = None
    if len(sys.argv) > 2:
        location = sys.argv[2]
    if len(sys.argv) > 3:
        skip = sys.argv[3]
    workers = []

    if skip is None:
        try:
            start = time.time()
            # generate cloud workers
            workers = trio.run(infrastructure.up, nodes, location)
            with open("workers.txt", "w") as f:
                for ip in workers:
                    f.write(ip + "\n")

            trio.run(infrastructure.wait_for_infrastructure, workers)
            print(
                f"[infrastructure] {len(workers)} nodes cloud infrastructure is up and initialized in {round(time.time() - start)}s")
        except KeyboardInterrupt:
            print(f"[infrastructure] Abort! Deleting cloud infrastructure...")
            trio.run(infrastructure.down, nodes)
            print(f"[infrastructure] Cloud infrastructure was shutdown")
        except Exception as e:
            print(f"[infrastructure] Error, could not bring up infrastructure... please consider shutting down all workers via `python3 infrastructure.py down`")
            print(e)
    else:
        with open("workers.txt", "r") as f:
            for line in f.readlines():
                workers.append(line.strip("\n"))

    def incoming_worker(workers, queue):
        print (f"new inbound worker started")
        from pssh.clients import ParallelSSHClient
        from gevent import joinall

        pclient = ParallelSSHClient(workers, user='crawl', pkey="~/.ssh/id_cah")
        
        while True:
            ready = []
            #_start = time.time()
            output = pclient.run_command('test -f /home/crawl/semaphore')
            pclient.join(output)
            for host_output in output:
                hostname = host_output.host
                exit_code = host_output.exit_code
                if exit_code == 0:
                    ready.append(hostname)
            #print(f"Took {time.time()-_start} seconds to check {len(ready)} workers that have ready jobs. starting downloading...")

            if len(ready) > 0:
                #_start = time.time()
                dclient = ParallelSSHClient(ready, user='crawl', pkey="~/.ssh/id_cah")
                try:
                    cmds = dclient.scp_recv('/home/crawl/gpujob.zip', 'gpujob.zip')
                    joinall (cmds, raise_error=True)
                except Exception as e:
                    print(e)
                #print (f"all jobs downloaded in {time.time()-_start} seconds")

                #_start = time.time()
                dclient.run_command('rm -rf /home/crawl/gpujob.zip')
                dclient.run_command('rm -rf /home/crawl/semaphore')

                for file in glob('gpujob.zip_*'):
                    name, ip = file.split("_")
                    output_folder = "./" + ip.replace(".", "-") + "/save/"
                    img_output_folder = output_folder + "images/"
                    if os.path.exists(output_folder):
                        shutil.rmtree(output_folder)
                    if os.path.exists(".tmp"):
                        shutil.rmtree(".tmp")

                    os.makedirs(output_folder)
                    os.makedirs(img_output_folder)

                    try:
                        with zipfile.ZipFile(file, 'r') as zip_ref:
                            zip_ref.extractall(ip.replace(".", "-")+"/")
                        queue.put(ip)
                    except:
                        subprocess.call(
                            ["touch", ip.replace(".", "-") + "/gpuabort"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        subprocess.call(
                            ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", ip.replace(".", "-") + "/gpuabort", "crawl@"+ip + ":~/gpuabort"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        os.remove(ip.replace(".", "-") + "/gpuabort")
                    os.remove(file)

                #print (f"unzipped in {time.time()-_start} seconds")
                    
            else:
                time.sleep(20)

    def outgoing_worker(queue):
        print (f"outbound worker started")
        while True:
            if queue.qsize() > 0:
                ip = queue.get()
                base = "./" + str(ip.replace(".", "-"))
                output_folder = base + "/save/"
                img_output_folder = output_folder + "images/"
                # clean img_output_folder now since we have all results do not want to transfer back all images...
                try:
                    shutil.rmtree(img_output_folder)
                except OSError as e:
                    print("[GPU] Error deleting images: %s - %s." %
                          (e.filename, e.strerror))

                # send GPU results
                shutil.make_archive(base + "/gpujobdone", "zip", base, "save")
                
                subprocess.call(
                    ["touch", base + "/gpusemaphore"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                subprocess.call(
                    ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", base + "/gpujobdone.zip", "crawl@"+ip + ":~/gpujobdone.zip"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.call(
                    ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", base + "/gpusemaphore", "crawl@"+ip + ":~/gpusemaphore"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                os.remove(base + "/gpujobdone.zip")
                os.remove(base + "/gpusemaphore")

                #print(f"[{ip}] resuming job with GPU results")
                queue.task_done()
            else:
                time.sleep(1)

try:

    inbound = JoinableQueue()
    outbound = JoinableQueue()

    inb = Process(target=incoming_worker, args=[workers, inbound], daemon=True).start()
    time.sleep(10)
    otb = Process(target=outgoing_worker, args=[outbound], daemon=True).start()
    time.sleep(10)

    probar = tqdm(total=int(nodes), desc="Executed GPU jobs", position=2, bar_format='{desc}: {n_fmt} ({rate_fmt})                    ')
    incbar = tqdm(total=int(nodes), desc="Inbound pipeline", position=1, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:0.0f}%)                    ')
    outbar = tqdm(total=int(nodes), desc="Outbound pipeline", position=0, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:0.0f}%)                    ')


    print (f"gpu worker started")
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
            out_fname = Path(out_path).stem.strip("_unfiltered").strip("_parsed").strip(".")

            # recreate parsed dataset and run CLIP filtering
            dlparse_df = pd.read_csv(output_folder + out_fname + ".csv", sep="|")

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

except KeyboardInterrupt:
    print(f"[GPU] Abort! Deleting cloud infrastructure...")
    inb.join()

    letters = string.ascii_lowercase
    suffix = ''.join(random.choice(letters) for i in range(3))
    for ip in workers:
        subprocess.call(
            ["scp", "-oIdentitiesOnly=yes", "-i~/.ssh/id_cah", "crawl@" +
                ip + ":" + "crawl.log", ip + "_" + suffix + ".log"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    trio.run(infrastructure.down)

    print(f"[infrastructure] Cloud infrastructure was shutdown")
