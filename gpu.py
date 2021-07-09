import os
import re
import sys
import time
import trio
import string
import random
import pickle
import shutil
import curses
import zipfile
import pandas as pd
import infrastructure
from glob import glob
from copy import copy
from tqdm import tqdm
from pathlib import Path
from colorama import Fore
sys.path.append('./crawlingathome-worker/')
from PIL import ImageFile
from multiprocessing import JoinableQueue, Process, Pool
from pssh.clients import ParallelSSHClient, SSHClient
from gevent import joinall

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


def incoming_worker(workers, queue: JoinableQueue, inpsize: JoinableQueue, errors: JoinableQueue):
    print (f"inbound worker started")
    
    pclient = ParallelSSHClient(workers, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False )
    while True:
        try:
            ready = []
            output = pclient.run_command('test -f /home/crawl/semaphore')
            pclient.join(output)
            for host_output in output:
                hostname = host_output.host
                exit_code = host_output.exit_code
                if exit_code == 0:
                    ready.append(hostname)
            errors.put(f"Ready workers for download: {len(ready)}")

            if len(ready) > 0:
                inpsize.put(len(ready))
                _start = time.time()
                dclient = ParallelSSHClient(ready, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False)
                try:
                    cmds = dclient.copy_remote_file('/home/crawl/gpujob.zip', 'gpujob.zip')
                    joinall (cmds, raise_error=False)
                except Exception as e:
                    print(e)
                errors.put(f"all jobs downloaded in {round(time.time()-_start, 2)} seconds")

                dclient.run_command('rm -rf /home/crawl/gpujob.zip')
                dclient.run_command('rm -rf /home/crawl/semaphore')

                for file in glob('gpujob.zip_*'):
                    name, ip = file.split("_")
                    output_folder = "./" + ip.replace(".", "-") + "/save/"
                    img_output_folder = output_folder + "images/"
                    if os.path.exists(output_folder):
                        shutil.rmtree(output_folder)

                    os.makedirs(output_folder)
                    os.makedirs(img_output_folder)

                    try:
                        with zipfile.ZipFile(file, 'r') as zip_ref:
                            zip_ref.extractall(ip.replace(".", "-")+"/")
                        queue.put(ip)
                    except:
                        aclient = SSHClient(ip, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False)
                        aclient.execute('touch /home/crawl/gpuabort')
                        aclient.disconnect()

                    os.remove(file)

                inpsize.get()
                inpsize.task_done() # empty impsize queue to signal no work to be done
                
            else:
                time.sleep(15)
        except Exception as e:
            print(f"some inbound problem occured: {e}")

def outgoing_worker(queue: JoinableQueue, errors: JoinableQueue, local=False):
    print (f"outbound worker started")
    while True:
        try:
            while queue.qsize() > 0:
                ip = queue.get()
                
                aclient = SSHClient(ip, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False)
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

                cmd = aclient.scp_send(base + "/gpujobdone.zip", "gpujobdone.zip")
                if local:
                    os.system(f"mv {base}/gpujobdone.zip results/{time.time()}.zip")
                    aclient.execute("touch gpulocal")
                else:    
                    os.remove(base + "/gpujobdone.zip")
                    aclient.execute("touch gpusemaphore")

                aclient.disconnect()
                queue.task_done()

                    #print(f"[{ip}] resuming job with GPU results")
            else:
                time.sleep(5)
        except Exception as e:
            print(f"some outbound problem occured: {e}")

def gpu_worker(inbound: JoinableQueue, outbound: JoinableQueue, counter: JoinableQueue, errors: JoinableQueue, gpuflag: JoinableQueue):
    print (f"gpu worker started")
    while True:
        if inbound.qsize() > 0:
            ip = inbound.get()
            gpuflag.put(1)
            errors.put(f"gpu processing job for {ip}")
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
            counter.put(1)
            gpuflag.get()
            gpuflag.task_done()

def monitor(nodes, inbound, outbound, counter):
    # crude term monitor with 3 custom bars.. todo: move to curses module
    probar = tqdm(total=int(nodes), desc="Executed GPU jobs", position=2, bar_format='{desc}: {n_fmt} ({rate_fmt})                    ')
    incbar = tqdm(total=int(nodes), desc="Inbound pipeline", position=1, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:0.0f}%)                    ')
    outbar = tqdm(total=int(nodes), desc="Outbound pipeline", position=0, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:0.0f}%)                    ')
    # keep main process for monitoring
    while True:
        incbar.n = inbound.qsize()
        outbar.n = outbound.qsize()
        if inpsize.qsize() > 0:
            incbar.desc = Fore.GREEN + incbar.desc.strip(Fore.RED).strip(Fore.GREEN).strip(Fore.RESET) + Fore.RESET
        else:
            incbar.desc = Fore.RED + incbar.desc.strip(Fore.RED).strip(Fore.GREEN).strip(Fore.RESET) + Fore.RESET
        if inbound.qsize()>0:
            probar.desc = Fore.GREEN + probar.desc.strip(Fore.RED).strip(Fore.GREEN).strip(Fore.RESET) + Fore.RESET
        else:
            probar.desc = Fore.RED + probar.desc.strip(Fore.RED).strip(Fore.GREEN).strip(Fore.RESET) + Fore.RESET
        if outbound.qsize()>0:
            outbar.desc = Fore.GREEN + outbar.desc.strip(Fore.RED).strip(Fore.GREEN).strip(Fore.RESET) + Fore.RESET
        else:
            outbar.desc = Fore.RED + outbar.desc.strip(Fore.RED).strip(Fore.GREEN).strip(Fore.RESET) + Fore.RESET
        incbar.refresh()
        outbar.refresh()
        probar.refresh()
        while counter.qsize() > 0:
            counter.get()
            probar.update(1)
            counter.task_done()
        time.sleep(1)

def monitor2(nodes, inbound, outbound, counter, inpsize, stdscr, errors, gpuflag):
    gpujobsdone = 0
    start = time.time()
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    # do stuff
    while True:
        stdscr.clear()
        while counter.qsize() > 0:
            counter.get()
            gpujobsdone += 1
            counter.task_done()

        if inpsize.qsize() > 0:
            stdscr.addstr(0,0,"Downloading..", curses.A_BLINK + curses.color_pair(1))
        else:
            stdscr.addstr(0,0,"-----------  ")
        stdscr.addstr(0,13,f"Incoming pipeline ({inbound.qsize()}/{nodes})")

        if outbound.qsize()>0:
            stdscr.addstr(1,0,"Uploading..",curses.A_BLINK + curses.color_pair(1))
        else:
            stdscr.addstr(1,0,"-----------  ")
        stdscr.addstr(1,13,f"Outgoing pipeline ({outbound.qsize()}/{nodes})")

        if gpuflag.qsize()>0:
            stdscr.addstr(2,0,"Processing..", curses.A_BLINK + curses.color_pair(1))
        else:
            stdscr.addstr(2,0,"-----------  ")
        stdscr.addstr(2,13,f"GPU jobs done:    {gpujobsdone}")

        stdscr.addstr(3,0,f"GPU velocity: {round((gpujobsdone/(time.time()-start)), 2)} jobs/s")
        if errors.qsize() > 0:
            msg = errors.get()
            errors.task_done()
        stdscr.addstr(5,0,f"messages: {msg}                                                                      ")

        stdscr.refresh()
        time.sleep(1)
        #stdscr.getkey()

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
    local = True
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
                f"[swarm] {len(workers)} nodes cloud swarm is up and was initialized in {round(time.time() - start)}s")
        except KeyboardInterrupt:
            print(f"[swarm] Abort! Deleting cloud swarm...")
            trio.run(infrastructure.down)
            print(f"[swarm] Cloud swarm was shutdown")
            sys.exit()
        except Exception as e:
            print(f"[swarm] Error, could not bring up swarm... please consider shutting down all workers via `python3 infrastructure.py down`")
            print(e)
            sys.exit()
    else:
        with open("workers.txt", "r") as f:
            for line in f.readlines():
                workers.append(line.strip("\n"))

    try:
        # initial cleanup - delete all working files in case of crash recovery
        reg_compile = re.compile(r"^\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}$")
        for root, dirnames, filenames in os.walk("."):
            for filename in filenames:
                if filename.startswith("gpujob.zip_"):
                    os.remove(filename)
            for dir in dirnames:
                if reg_compile.match(dir):
                    shutil.rmtree(dir)

        

        #initialize 3 joinable queues to transfer messages between multiprocess processes
        inbound = JoinableQueue()
        outbound = JoinableQueue()
        counter = JoinableQueue()
        inpsize = JoinableQueue() # use this to communicate number of jobs downloading now
        gpuflag = JoinableQueue() # use this to flag that gpu is processing
        errors = JoinableQueue() # use this to capture errors and warnings and route them to curses display

        # launch separate processes with specialized workers
        inb = Process(target=incoming_worker, args=[workers, inbound, inpsize, errors], daemon=True).start()
        time.sleep(5)
        otb = Process(target=outgoing_worker, args=[outbound, errors, local], daemon=True).start()
        time.sleep(5)
        #P = Pool(processes=3)
        #gpu = P.map(gpu_worker, [inbound, outbound, counter])
        gpu1 = Process(target=gpu_worker, args=[inbound, outbound, counter, errors, gpuflag], daemon=True).start()
        gpu2 = Process(target=gpu_worker, args=[inbound, outbound, counter, errors, gpuflag], daemon=True).start()
        gpu3 = Process(target=gpu_worker, args=[inbound, outbound, counter, errors, gpuflag], daemon=True).start()

        stdscr = curses.initscr()
        #sys.stdout.close()
        #monitor(nodes, inbound, outbound, counter)
        curses.wrapper(monitor2(nodes, inbound, outbound, counter, inpsize, stdscr, errors, gpuflag))

    except KeyboardInterrupt:
        curses.nocbreak()
        curses.echo()
        curses.endwin()

        print(f"[GPU] Abort! Deleting cloud infrastructure...")

        letters = string.ascii_lowercase
        suffix = ''.join(random.choice(letters) for i in range(3))

        pclient = ParallelSSHClient(workers, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False )
        pclient.scp_recv('/home/crawl/crawl.log', suffix + '_crawl.log')

        trio.run(infrastructure.down)
        print(f"[infrastructure] Cloud infrastructure was shutdown")
        sys.exit()
    except Exception as e:
        print (f"general exception: {e}")
        sys.exit()