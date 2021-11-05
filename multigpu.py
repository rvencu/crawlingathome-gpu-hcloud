
# run as: python3 multigpu.py 0 and python3 multigpu.py 1 where 0 and 1 are GPU ids

import os
import re
import sys
import time
import uuid
import clip
import torch
import queue
import shutil
import curses
import tarfile
import hashlib
import requests
import threading
import subprocess
import pandas as pd
from glob import glob
from PIL import Image
from pathlib import Path
from statistics import mode
from datetime import datetime
import crawlingathome_client as cah
from sqlalchemy import create_engine
from configparser import ConfigParser
sys.path.append('./crawlingathome-worker/')
from multiprocessing import JoinableQueue, Process, Queue

# basic watcher that sends email when the script crashes as it is long ran
import sentry_sdk
sentry_sdk.init(
    "https://78667479988545ec9fa78fba79638986@o576504.ingest.sentry.io/5909507",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)

use_jit = torch.cuda.is_available() and '1.7.1' in torch.__version__
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, preprocess):
        self.dataframe = dataframe
        self.image_transform = preprocess
        self.tokenizer = clip.tokenize

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            self.image_transform(Image.open(row["PATH"])),
            self.tokenizer(str(row["TEXT"]), truncate=True)[0],
        )

class CLIP:
    def __init__(self, gpuid):
        self.device = f"cuda:{gpuid}" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=use_jit)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        with torch.no_grad():
            self.categories = self.model.encode_text(clip.tokenize(["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breasts, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]).to(self.device))
            self.underaged_categories = self.model.encode_text(clip.tokenize(["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]).to(self.device))
            self.animal_categories = self.model.encode_text(clip.tokenize(["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]).to(self.device))

    def similarity_imgalt(self, image_tensor, text_tokens):
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor.to(self.device)).float()
            text_features = self.model.encode_text(text_tokens.to(self.device)).float()
            similarity = self.cosine_similarity(image_features, text_features).tolist()

        image_features = image_features.detach().cpu().numpy()
        return image_features, similarity

    def preprocess_images(self, df):
        ret_image_features = []
        ret_similarity = []
        batch_size = 256 if "cuda" in self.device else 8
        dataset = CLIPDataset(df, self.preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=4)
        for tensors, tokens in dataloader:
            image_features, similarities = self.similarity_imgalt(tensors, tokens)
            ret_image_features.extend(image_features)
            ret_similarity.extend(similarities)
        return ret_image_features, ret_similarity

    def prob(self, image_features, text_features):
        text_features = text_features.float()
        image_features = torch.as_tensor(image_features).to(self.device, dtype=torch.float32)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity.topk(2)
        return indices

def df_clipfilter(df, clip_filter):
    sim_threshold = 0.3
    underaged_text = ["teen", "kid", "child", "baby"]

    img_embedding, similarities = clip_filter.preprocess_images(df)
    tmp_embed = []

    df["dropped"] = False

    for i, img_embed in enumerate(img_embedding):
        if similarities[i] < sim_threshold:
            df.at[i, 'dropped'] = True
            continue

        # get most similar categories
        nsfw_prob = clip_filter.prob(img_embed, clip_filter.categories)
        df.at[i, "NSFW"] = "UNSURE"
        df.at[i, "similarity"] = similarities[i]
        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
            tmp_embed.append(img_embed)
            continue
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        underage_prob = clip_filter.prob(img_embed, clip_filter.underaged_categories)
        if underage_prob[0] < 4 or underage_prob[1] < 4 or any(x in df.at[i, "TEXT"] for x in underaged_text):
            df.at[i, 'dropped'] = True
            continue

        animal_prob = clip_filter.prob(img_embed, clip_filter.animal_categories)
        if animal_prob[0] > 20:
            df.at[i, 'dropped'] = True
            continue
        tmp_embed.append(img_embed)
        
    df = df[df["dropped"] != True]
    df.reset_index(drop=True, inplace=True)
    return tmp_embed, df


def filter(df, out_fname, output_folder, clip_filter):
    with open(f"{output_folder}hashes-{out_fname}.clp", "wt") as f:
        for item in df["hash"]:
            f.write(item + "\n")
    results = []
    img_embeddings, dff = df_clipfilter(df, clip_filter)
    dff.to_csv(f"{output_folder}{out_fname}.csv", index=False, sep="|")

    dff.loc[:,"shard"] = dff.PATH.apply(lambda x: x.split("/")[1])
    results = dff["shard"].value_counts()

    with open(f"{output_folder}hashes-{out_fname}.hsh", "wt") as f:
        for item in dff["hash"]:
            f.write(item + "\n")

    return len(dff), results


def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db

def log(logqueue:Queue, msg):
    logqueue.put(msg)

def monitor_curses(logqueue: Queue, screen):
    tick = 0
    classic_count = 0
    database_count = 0
    current_gpu_job = ""
    group_size = 0
    log = ""
    duration = 0
    pairs = 0
    qsize = 0
    while True:
        try:
            raw = logqueue.get()
            type, msg = raw.split(":")
            if type == "tick":
                tick = msg
            elif type == "classic_count":
                classic_count = msg
            elif type == "database_count":
                database_count = msg
            elif type == "current_gpu_job":
                current_gpu_job = msg
            elif type == "qsize":
                qsize = msg
            elif type == "group_size":
                group_size = msg
            elif type == "pairs":
                pairs = msg
            elif type == "duration":
                duration = msg
            else:
                log = msg
            print_curses(screen, tick=tick, classic_count=classic_count, database_count=database_count, current_gpu_job=current_gpu_job, log=log, group_size=group_size, pairs=pairs, duration=duration, qsize=qsize)
        except:
            time.sleep(10)

def print_curses(screen, tick, classic_count, database_count, current_gpu_job, log, group_size, pairs, duration, qsize):
    screen.erase()
    screen.addstr(0, 0, "GPU off gap:           %s sec.\n" % (tick))
    screen.addstr(1, 0, "classic jobs count:    %s jobs\n" % (classic_count))
    screen.addstr(2, 0, "database jobs count:   %s jobs\n" % (database_count))
    screen.addstr(3, 0, "current GPU job id:    %s\n" % (current_gpu_job))
    screen.addstr(4, 0, "GPU queue size:        %s jobs\n" % (qsize))
    screen.addstr(5, 0, "group target size:     %s jobs\n" % (group_size))
    screen.addstr(6, 0, "last inference result: %s pairs\n" % (pairs))
    screen.addstr(7, 0, "last inference speed:  %s sec/job\n" % (duration))
    screen.addstr(8, 0, "                                                    \n")
    screen.addstr(9, 0, "%s\n" % (log))
    screen.addstr(10, 0, "                                                    \n")
    screen.addstr(11, 0, "                                                    \n")
    screen.addstr(12, 0, "                                                    \n")
    screen.refresh()

'''
GPU workflow:
    GPU workflow is divided in 3 processes to provide enough parallelism and ensure maximal GPU utilization

1. IO worker
    Incoming worker polls CAH server for available GPU jobs. We want to bring in a number of `group` shards, 
    combine them and process them at once for efficiency
    a) CAH client initialization and get name for the job
        also make a folder with the job name
    d) rsync from staging server all into the jobname folder
    e) move the stats files out of the way to ./stats folder
    f) transfer jobname to GPU worker then start waiting for response
    a) when response is received mark job done if number of final pairs is > 0
    c) clean up
2. GPU worker
    GPU worker keeps GPU cuda cores as busy as possible. the workflow consists in
    a) wait for the incoming queue to accumulate groupsize jobs then make a groupname and a folder with same name to hold result files
    b) make a list of shards in the groupjob
    c) create and group pandas dataframes for each shard
    d) run CLIP filtering on the resulted data
    e) save the result in ./save folder and cleanup the job folder
    f) transfer completed jobname back to IO worker
3. Monitor
    The monitor displays the status of the workers as well as performance metrics about the jobs performed
'''

def invalidURL (client, job, jobtype, engine):
    if jobtype == 0:
        client.invalidURL()
    if jobtype == 1:
        update_stmt1 = "UPDATE jobs set status=9 WHERE jobid='{}'".format(job)
        #update_stmt2 = "UPDATE dataset SET status = 0 WHERE prefix='{}'".format(job)
        conn = engine.raw_connection()
        cur = conn.cursor()
        cur.execute(update_stmt1)
        #cur.execute(update_stmt2)
        conn.commit()
        cur.close()
        conn.close()
    return

def completeJob (client, job, jobtype, engine, pairs):
    if jobtype == 0:
        client.completeJob(int(pairs))
    if jobtype == 1:
        update_stmt1 = "UPDATE jobs SET status = 2 WHERE jobid='{}'".format(job)
        conn = engine.raw_connection()
        cur = conn.cursor()
        cur.execute(update_stmt1)
        conn.commit()
        cur.close()
        conn.close()
    return

def get_dbjobscount(engine):
    select_stmt1 = "select count(*) from jobs where status = 0"
    conn = engine.raw_connection()
    cur = conn.cursor()
    cur.execute(select_stmt1)
    jobcount = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return jobcount[0]

# spawn this interface to double or more than shard groups so they can download jobs and communicate with the tracker in parallel with GPU processing. this will keep GPU busy almost continuously
def gpu_cah_interface(i:int, incomingqueue: JoinableQueue, outgoingqueue: JoinableQueue, logqueue: queue.Queue, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL, engine):
    # initiate and reinitiate a GPU type client if needed
    # print(f"   |___ inbound worker started")
    while True:
        client = cah.init(
            url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD, type="GPU"
        )
        try:
            while client.isAlive():
                jobcount = client.jobCount()
                while jobcount > 0:
                    try:
                        log(logqueue,f"classic_count:{jobcount}")
                    except:
                        print("[io {i}] error adding to logqueue")
                    # each thread gets a new job, passes it to GPU then waits for completion
                    jobtype = 0
                    job = ""
                    if jobcount > 10000: 
                        log(logqueue,f"log:[io {i}] started CLASSIC job")
                        try:
                            client.newJob()
                        except:
                            time.sleep(10)
                            continue
                        if client.shard.startswith('rsync'):
                            try:
                                job = client.shard.split(" ")[1]
                            except:
                                invalidURL (client, job, jobtype, engine)
                                log(logqueue,f"error:[io {i}] invalid job detected {job}")
                                continue
                            # found repeating shards, need to clear old files before continuing
                            if os.path.exists("./"+ job):
                                shutil.rmtree("./"+ job, ignore_errors=True)
                            #os.mkdir("./"+ job)
                            client.downloadShard()
                        elif client.shard.startswith('postgres'):
                            log(logqueue,f"error:[io {i}] this is a database job not classic, marking it complete in tracker since progress continues to be tracked in database")
                            try:
                                completeJob (client, job, jobtype, engine, 0)
                            except:
                                pass
                            continue
                    else:
                        log(logqueue,f"log:[io {i}] started DATABASE job")
                        jobtype = 1
                        select_stmt1 = "UPDATE jobs SET status = 1 WHERE jobid in (SELECT jobid from jobs where status = 0 LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING jobid"
                        conn = engine.raw_connection()
                        cur = conn.cursor()
                        cur.execute(select_stmt1)
                        job = cur.fetchone()
                        conn.commit()
                        cur.close()
                        conn.close()

                        # if there are no database jobs available
                        if job is None:
                            time.sleep(60)
                            continue

                        job = job[0]

                        # found repeating shards, need to clear old files before continuing
                        if os.path.exists("./"+ job):
                            shutil.rmtree("./"+ job, ignore_errors=True)

                        resp = 1
                        for _ in range(5):
                            p = subprocess.Popen(["rsync", "-av", f"archiveteam@176.9.4.150::gpujobs/{job}.tar.gz", f"{job}.tar.gz"], stdout=subprocess.PIPE)
                            output = p.communicate()[0]
                            resp = p.returncode
                            #resp = os.system(f'rsync -av archiveteam@176.9.4.150::gpujobs/{job}.tar.gz {job}.tar.gz')
                            if resp == 5888:
                                log(logqueue,'error:[io {i}] rsync job not found')
                                invalidURL (client, job, jobtype, engine)
                            if resp == 0:
                                with tarfile.open(f"{job}.tar.gz", "r:gz") as tar:
                                    tar.extractall()
                                break

                    # test for csv and for images folder
                    if len(glob(f"{job}/*.csv")) == 0 or not os.path.exists(f"./{job}/images"):
                        invalidURL (client, job, jobtype, engine)
                        log(logqueue,f"error:[io {i}] invalid job detected {job}")
                        continue
                    for file in glob(f"{job}/*_parsed.csv"):
                        os.system(f"mv {file} stats/")
                    for file in glob(f"{job}/*_unfiltered.csv"):
                        os.system(f"rm {file}")
                    for file in glob(f"{job}/*.csv"):
                        # Read in the file
                        with open(file, 'rt') as f :
                            filedata = f.read()
                        # Replace the target string
                        filedata = filedata.replace('\n|', '|')
                        # Write the file out again
                        with open(file, 'wt') as f:
                            f.write(filedata)
                    # search for corrupt images
                    for file in glob(f"{job}/*.csv"):
                        df = pd.read_csv(file, sep="|")
                        df["PATH"] = df.PATH.apply(lambda x: re.sub(r"^(.*)./save/[-]?[0-9][0-9]?[0-9]?/(.*)$", r"save/\2", x)) # when path is like /save/12/images/name.jpg
                        df["PATH"] = df.PATH.apply(lambda x: re.sub(r"^(.*)./[-]?[0-9][0-9]?[0-9]?/save/(.*)$", r"save/\2", x)) # when path is like /12/save/images/name.jpg
                        df["PATH"] = df.PATH.apply(lambda x: "./" + job + "/" + x.strip("save/"))
                        for index, row in df.iterrows():
                            try:
                                im = Image.open(row["PATH"])
                                im.close()
                            except Exception as e:
                                if index < 10:
                                    log(logqueue,f"error:[io {i}] invalid image {row['PATH']} because {e}")
                                df = df.drop(index)
                        df.to_csv(file, sep="|", index=False)
                        del df
                    
                    log(logqueue,f"log:[io {i}] job sent to GPU {job}")
                    incomingqueue.put((i, job, client.upload_address))
                    
                    # wait until job gets processes
                    while True:
                        if outgoingqueue.qsize() > 0:
                            outjob, pairs = outgoingqueue.get() # I am poping out from queue only if my current job is finished
                            if pairs >= 0:
                                #print(f"[io {i}] mark job as complete: {job}")
                                # cleanup temp storage now
                                if pairs == 0:
                                    pairs = 1
                                try:
                                    completeJob (client, job, jobtype, engine, pairs)
                                except:
                                    log(logqueue,f"log:[io {i}] invalid trying to complete with {pairs} pairs")
                                    invalidURL (client, job, jobtype, engine)
                            else:
                                log(logqueue,f"log:[io {i}] invalid with negative {pairs} pairs?")
                                invalidURL (client, job, jobtype, engine)
                            if os.path.exists("./"+ job):
                                shutil.rmtree("./"+ job)
                            if os.path.exists(f"{job}.tar.gz"):
                                os.remove(f"{job}.tar.gz")
                            outgoingqueue.task_done()
                            break # we can let the worker request a new job
                        else:
                            time.sleep(1)
                else:
                    log(logqueue,f"error:[io {i}] no classic jobs, switching to database jobs")
                    time.sleep(120)
            else:
                log(logqueue,f"error:[io {i}] client forgotten")
                time.sleep(30)
        except Exception as e:
            log(logqueue,f"error:[io {i}] client crashed, respawning...")
            log(logqueue,f"error:{e}") #see why clients crashes
            time.sleep(30)

# process to spawn many interfaces with the tracker
def io_worker(incomingqueue: JoinableQueue, outgoingqueue: list, groupsize: int, logqueue: Queue, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL, engine):
    # separate process to initialize threaded workers
    log(logqueue,f"log:[io] inbound workers")
    thqueue = queue.Queue()
    try:
        # just launch how many threads we need to group jobs into single output
        for i in range(int(2.7 * groupsize)):
            threading.Thread(target=gpu_cah_interface, args=(i, incomingqueue, outgoingqueue[i], thqueue, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL, engine)).start()
    except Exception as e:
        log(logqueue,f"error:[io] some inbound problem occured {e}")
    while True:
        if not thqueue.empty():
            logqueue.put(thqueue.get())
            thqueue.task_done()
        else:
            dbcount = get_dbjobscount(engine)
            logqueue.put(f"database_count:{dbcount}")
            qsize = incomingqueue.qsize()
            log(logqueue,f"qsize:{qsize}")
            time.sleep(0.1)


# process to upload the results
def upload_worker(uploadqueue: JoinableQueue, counter: JoinableQueue, outgoingqueue: list, logqueue: Queue):
    log(logqueue,f"log:upload worker started")

    while True:
        if uploadqueue.qsize() > 0:
            group_id, upload_address, shards, results = uploadqueue.get()
            p = subprocess.Popen(f"rsync -av save/*{group_id}* {upload_address}", shell=True, stdout=subprocess.PIPE)
            output = p.communicate()[0]
            resp = p.returncode
            #resp = os.system(f"rsync -av save/*{group_id}* {upload_address}") # to do get target from client
            if resp == 0:
                #print(f"[io2] sending all jobs to be marked as completed")
                for i, job, item in shards:
                    cnt = results.get(job)
                    if cnt is None:
                        cnt = 0
                    outgoingqueue[i].put((job, cnt))
                    for file in glob((f"save/*{group_id}*")):
                        os.remove(file)
                    counter.put(1)
            else:
                for i, job, item in shards:
                    outgoingqueue[i].put((job, 0)) # if upload crashes, then do NOT mark completeJob()
            uploadqueue.task_done()
        else:
            time.sleep(5)

# main gpu workers. perhaps this worker needs to be run in as many processes as GPUs are present in the system. (todo)
def gpu_worker(incomingqueue: JoinableQueue, uploadqueue: JoinableQueue, gpuflag: JoinableQueue, groupsize: int, logqueue: Queue, gpuid: int):
    log(logqueue,f"log:[gpu] worker started")
    log(logqueue,f"current_gpu_job:preparing...")
    first_groupsize = groupsize
    bloomip = "116.202.162.146"
    tick = time.time()
    # watch for the incoming queue, when it is big enough we can trigger processing    
    while True:
        qsize = incomingqueue.qsize()
        #log(logqueue,f"qsize:{qsize}")
        log(logqueue,f"group_size:{groupsize}")
        log(logqueue,f"tick:{round(time.time() - tick, 2)}")
        if qsize >= groupsize:
            start = time.time()
            shards = []
            addresses = []
            group_id = uuid.uuid4().hex            
            log(logqueue,f"current_gpu_job:{group_id}")
            group_parse = None
            for _ in range(groupsize):
                i, job, address = incomingqueue.get()

                all_csv_files = []
                for path, subdir, files in os.walk(job):
                    for file in files:
                        if file.endswith(".csv"):
                            all_csv_files.append(file)
                # get name of csv file
                out_path = all_csv_files[0]
                out_path = Path(out_path).stem
                shards.append((i, job, out_path))
                addresses.append(address)

                incomingqueue.task_done()


            for i, job, item in shards:
                dlparse_df = pd.read_csv(job + "/" + item + ".csv", sep="|")
                
                if group_parse is None:
                    group_parse = dlparse_df
                else:
                    group_parse = group_parse.append(dlparse_df, ignore_index=True)
                
            with open("./save/" + group_id + ".txt", "wt") as f:
                for i, job, item in shards:
                    f.write(item + "\n")
            
            group_parse.drop_duplicates(subset=["URL","TEXT"], keep='last', inplace=True)
            group_parse.reset_index(inplace=True, drop=True)
            group_parse.loc[:,"hash"] = group_parse.apply(lambda row: hashlib.md5((str(row.URL)+str(row.TEXT)).encode("utf-8")).hexdigest(), axis=1)
            
            with open('hash.txt', 'w') as f:
                f.write(group_parse['hash'].str.cat(sep='\n'))
            post = {
                'file': ('hash.txt', open('hash.txt', 'rb')),
                'key': (None, 'main'),
            }
            os.remove('hash.txt')
            
            failure = True
            for _ in range(10):
                response = requests.post(f'http://{bloomip}:8000/deduplicate/', files=post)
                if response.status_code != 200:
                    log(logqueue,f"error:[gpu] bloom server error, retrying...")
                    time.sleep(15)            
                else:
                    failure = False
                    break
            if failure:
                log(logqueue,f"error:[gpu] crash, cannot contact the bloom server, please fix")
                continue

            valid_hashes = response.content.decode("utf-8").split("\n")
            log(logqueue,f"log:bloom server has validated {len(valid_hashes)} pairs")

            group_parse = group_parse[group_parse.hash.isin(valid_hashes)]
            group_parse.reset_index(inplace=True, drop=True)
            #temporary fix
            group_parse["LANGUAGE"] = "en"

            print (f"before en selection {len(group_parse.index)}")
            #force en language to continue with English dataset
            en_parse = group_parse[group_parse["LANGUAGE"] == "en"]
            #int_parse = group_parse[~group_parse.LANGUAGE.isin(['en', 'bn', 'co', 'eo', 'fil', 'fy', 'gd', 'ha', 'haw', 'hmn', 'ig', 'km', 'ku', 'ky', 'lo', 'mi', 'mn', 'mt', 'ny', 'sd', 'si', 'sm', 'sn', 'so', 'st', 'su', 'sw', 'xh', 'yi', 'zu', "", None])]
            print (f"after en selection {len(en_parse.index)}")

            log(logqueue,f"log:[gpu] preparation done in {round(time.time()-start, 2)} sec.")

            start = time.time()
            clip_filter_obj = CLIP(gpuid)
            final_images, results = filter(en_parse, group_id, "./save/", clip_filter_obj)
            #TODO: add here processing command for int_parse, perhaps secondary location and different group_id
            
            log(logqueue,f"pairs:{final_images}")
            log(logqueue,f"duration:{round((time.time()-start)/groupsize,2)}")

            # find most required upload address among the grouped shards
            upload_address = mode(addresses)
            uploadqueue.put((group_id, upload_address, shards, results))
            tick = time.time()
            log(logqueue,f"current_gpu_job:not ready")
            
            # dynamic adjustment of groupsize so we can get close to 8000 pairs per group as fast as possible
            gradient = int((final_images-20000)/7000)
            oldgroupsize = groupsize
            groupsize = min( int(3 * first_groupsize) - 5 , groupsize - gradient )
            groupsize = max( groupsize - gradient, 3 )
            if groupsize != oldgroupsize:
                log(logqueue,f"log:[gpu] groupsize changed to {groupsize}")
            
        else:
            time.sleep(1)

if __name__ == "__main__":
    # script initialization
    YOUR_NICKNAME_FOR_THE_LEADERBOARD = os.getenv('CAH_NICKNAME')
    if YOUR_NICKNAME_FOR_THE_LEADERBOARD is None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "anonymous"
    CRAWLINGATHOME_SERVER_URL = "http://cah.io.community/"
    
    gpuid = 0
    if len(sys.argv) > 1:
        gpuid = sys.argv[1]
    
    print(
        f"[{datetime.now().strftime('%H:%M:%S')} GPU{gpuid}] starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    time.sleep(10)

    groupsize = 17 # how many shards to group for CLIP

    gpuid = 0
    if len(sys.argv) > 1:
        gpuid = sys.argv[1]

    params = config()
    engine = create_engine(f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:5432/{params["database"]}',pool_size=50, max_overflow=100)

    # folders cleanup (remove previous runs artifacts)

    if not os.path.exists("./stats/"):
        os.makedirs("./stats/")
    if not os.path.exists("./save/"):
        os.makedirs("./save/")

    # initial cleanup - delete all working files in case of crash recovery
    reg_compile = re.compile(r"^\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}$")
    for root, dirnames, filenames in os.walk("."):
        for filename in filenames:
            if filename.startswith("gpujob.zip_"):
                os.remove(filename)
        for dir in dirnames:
            if reg_compile.match(dir):
                shutil.rmtree(dir)
    re_uuid = re.compile(r'[0-9a-f]{32}', re.I)
    for root, dirnames, filenames in os.walk("."):
        for dir in dirnames:
            if re_uuid.match(dir):
                shutil.rmtree(dir)
    re_gz = re.compile(r'.*.tar.gz.*', re.I)
    for root, dirnames, filenames in os.walk("."):
        for file in filenames:
            if re_gz.match(file):
                os.remove(file)
    
    #initialize joinable queues to transfer messages between multiprocess processes
    # Outbound queues, we need one for each io worker
    outbound = []
    for _ in range(int(3 * groupsize)): # we need 3x IO workers to keep GPU permanently busy
            outbound.append(JoinableQueue())
    inbound = JoinableQueue()
    uploadqueue = JoinableQueue()
    counter = JoinableQueue() # count number of jobs done    
    gpuflag = JoinableQueue() # use this to flag that gpu is processing
    logqueue = Queue() # use this to send log lines to monitor
    
    sys.stderr = open('gpuerr.txt', 'a')
    sys.stdout = open('gpuout.txt', 'a')

    screen = curses.initscr()
    #mon = Process(target=monitor, args=[logqueue], daemon=True).start()
    mon = Process(target=monitor_curses, args=[logqueue, screen], daemon=True).start()

    # launch separate processes with specialized workers
    io = Process(target=io_worker, args=[inbound, outbound, groupsize, logqueue, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL, engine], daemon=True).start()
    upd = Process(target=upload_worker, args=[uploadqueue, counter, outbound, logqueue], daemon=True).start()
    
    gpu_worker(inbound, uploadqueue, gpuflag, groupsize, logqueue, gpuid)
