import os
import re
import sys
import time
import uuid
import shutil
import curses
import hashlib
import requests
import threading
import clip_filter
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from colorama import Fore
from statistics import mode
import crawlingathome_client as cah
from bloom_filter2 import BloomFilter
sys.path.append('./crawlingathome-worker/')
from multiprocessing import JoinableQueue, Process

import sentry_sdk
sentry_sdk.init(
    "https://78667479988545ec9fa78fba79638986@o576504.ingest.sentry.io/5909507",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)

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

def gpu_cah_interface(i:int, incomingqueue: JoinableQueue, outgoingqueue: JoinableQueue, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL):
    # initiate and reinitiate a GPU type client if needed
    print (f"   |___ inbound worker started")
    while True:
        client = cah.init(
            url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD, type="GPU"
        )
        try:
            while client.isAlive():
                while client.jobCount() > 0: 
                    # each thread gets a new job, passes it to GPU then waits for completion
                    try:
                        client.newJob()
                    except:
                        time.sleep(10)
                        continue
                    job = ""
                    try:
                        job = client.shard.split(" ")[1]
                    except:
                        client.invalidURL()
                        print (f"[io {i}] invalid job detected: {job}")
                        continue
                    # found repeating shards, need to clear old files before continuing
                    if os.path.exists("./"+ job):
                        shutil.rmtree("./"+ job, ignore_errors=True)
                    #os.mkdir("./"+ job)
                    client.downloadShard()

                    # test for csv and for images folder
                    if len(glob(f"{job}/*.csv")) == 0 or not os.path.exists(f"./{job}/images"):
                        client.invalidURL()
                        print (f"[io {i}] invalid job detected: {job}")
                        continue
                    for file in glob(f"{job}/*_parsed.csv"):
                        os.system(f"mv {file} stats/")
                    for file in glob(f"{job}/*_unfiltered.csv"):
                        os.system(f"mv {file} stats/")
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
                        df["PATH"] = df.apply(lambda x: "./" + job + "/" + x["PATH"].strip("save/"), axis=1)
                        for index, row in df.iterrows():
                            try:
                                im = Image.open(row["PATH"])
                                im.close()
                            except Exception as e:
                                if index < 10:
                                    print (f"[io {i}] invalid image {row['PATH']} because {e}")
                                df = df.drop(index)
                        df.to_csv(file, sep="|", index=False)
                        del df
                    
                    #print (f"[io] job sent to GPU: {job}")
                    incomingqueue.put((i, job, client.upload_address))
                    
                    # wait until job gets processes
                    while True:
                        if outgoingqueue.qsize() > 0:
                            outjob, pairs = outgoingqueue.get() # I am poping out from queue only if my current job is finished
                            if pairs >= 0:
                                #print (f"[io {i}] mark job as complete: {job}")
                                # cleanup temp storage now
                                if pairs == 0:
                                    pairs = 1
                                try:
                                    client.completeJob(int(pairs))
                                except:
                                    print(f"[io {i}] invalid trying to complete with {pairs} pairs")
                                    client.invalidURL()
                            else:
                                print(f"[io {i}] invalid with negative {pairs} pairs?")
                                client.invalidURL()
                            if os.path.exists("./"+ job):
                                shutil.rmtree("./"+ job)
                            if os.path.exists(f"{job}.tar.gz"):
                                os.remove(f"{job}.tar.gz")
                            outgoingqueue.task_done()
                            break # we can let the worker request a new job
                        else:
                            time.sleep(1)
                else:
                    print (f"[io {i}] no jobs")
                    time.sleep(120)
            else:
                print (f"[io {i}] client forgotten")
                time.sleep(30)
        except Exception as e:
            print (f"[io {i}] client crashed, respawning...")
            print (e) #see why clients crashes
            time.sleep(30)

def io_worker(incomingqueue: JoinableQueue, outgoingqueue: list, groupsize: int, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL):
    # separate process to initialize threaded workers
    print (f"[io] inbound workers:")
    try:
        # just launch how many threads we need to group jobs into single output
        for i in range(int(2.7 * groupsize)):
            threading.Thread(target=gpu_cah_interface, args=(i, incomingqueue, outgoingqueue[i], YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL)).start()
    except Exception as e:
        print(f"[io] some inbound problem occured: {e}")

def upload_worker(uploadqueue: JoinableQueue, counter: JoinableQueue, outgoingqueue: list):
    print(f"upload worker started")
    #mega = Mega()
    #m = mega.login(email, password)

    while True:
        if uploadqueue.qsize() > 0:
            group_id, upload_address, shards, results = uploadqueue.get()
            response = os.system(f"rsync -av save/*{group_id}* {upload_address}") # to do get target from client
            if response == 0:
                #print (f"[io2] sending all jobs to be marked as completed")
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
            #folder = m.find('crawliongathome/tfrecords-2021-q3')
            #m.upload('myfile.doc', folder[0])
            uploadqueue.task_done()
        else:
            time.sleep(10)

def gpu_worker(incomingqueue: JoinableQueue, uploadqueue: JoinableQueue, gpuflag: JoinableQueue, groupsize: int):
    print (f"[gpu] worker started")
    first_groupsize = groupsize
    bloomip = "116.202.162.146"
    # watch for the incoming queue, when it is big enough we can trigger processing    
    while True:
        #print (f"[gpu] testing incoming queue size")
        if incomingqueue.qsize() >= groupsize:
            start = time.time()

            gpuflag.put(1)
            shards = []
            addresses = []
            group_id = uuid.uuid4().hex
            print (f"[gpu] got new {groupsize} jobs to group in id {group_id}")
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
                #print(out_path)
                out_path = Path(out_path).stem
                #print(out_path)
                shards.append((i, job, out_path))
                addresses.append(address)

                incomingqueue.task_done()
            #print (f"[gpu] adjusted image paths")

            for i, job, item in shards:
                dlparse_df = pd.read_csv(job + "/" + item + ".csv", sep="|")
                
                if group_parse is None:
                    group_parse = dlparse_df
                else:
                    group_parse = group_parse.append(dlparse_df, ignore_index=True)
                
            with open("./save/" + group_id + ".txt", "wt") as f:
                for i, job, item in shards:
                    f.write(item + "\n")
            
            #print (f"[gpu] saving stats")

            duped = len(group_parse.index)
            group_parse.drop_duplicates(subset=["URL","TEXT"], keep='last', inplace=True)
            group_parse.reset_index(inplace=True, drop=True)
            total = len(group_parse.index)
            
            group_parse.loc[:,"hash"] = group_parse.apply(lambda row: hashlib.md5((str(row.URL)+str(row.TEXT)).encode("utf-8")).hexdigest(), axis=1)
            
            with open('hash.txt', 'w') as f:
                f.write(group_parse['hash'].str.cat(sep='\n'))
            post = {
                'file': ('hash.txt', open('hash.txt', 'rb')),
                'key': (None, 'main'),
            }
            response = requests.post(f'http://{bloomip}:8000/deduplicate/', files=post)
            os.remove('hash.txt')
            if response.status_code != 200:
                print(f"crash, cannot contact the bloom server, please fix")
                sys.exit()

            valid_hashes = response.content.decode("utf-8").split("\n")
            print(f"bloom server has validated {len(valid_hashes)} pairs")
            group_parse.loc[:,"bloom"] = group_parse.apply(lambda row: str(row.hash) in valid_hashes, axis=1)

            group_parse = group_parse[group_parse["bloom"] == True] # if True than row is valid
            group_parse.reset_index(inplace=True, drop=True)
            bloomed = len(group_parse.index)

            group_parse.to_csv("./stats/" + group_id + "_beforeclip.csv", index=False, sep="|") # I am using these to find out domains to filter from scraping

            print (f"{Fore.YELLOW}[gpu] preparation done in {round(time.time()-start, 2)} sec.{Fore.RESET}")
            start = time.time()
            final_images, results = clip_filter.filter(group_parse, group_id, "./save/")
            
            dedupe_ratio = round((duped - total) / duped, 4)
            print(f"{Fore.GREEN}[gpu] got {final_images} images from {bloomed} bloomed from {total} deduped from {duped} (ratio {dedupe_ratio}) in {round(time.time()-start, 2)} sec.")
            print(f"({groupsize} shards were grouped together and average duration per shard was {round((time.time()-start)/groupsize,2)} sec){Fore.RESET}")
            #print (f"[gpu] upload group results to rsync target")
            # find most required upload address among the grouped shards
            upload_address = mode(addresses)
            #print (f"most requested upload address is {upload_address}")
            uploadqueue.put((group_id, upload_address, shards, results))
            
            # dynamic adjustment of groupsize so we can get close to 8000 pairs per group as fast as possible
            gradient = int((final_images-20000)/7000)
            oldgroupsize = groupsize
            groupsize = min( int(3 * first_groupsize) - 5 , groupsize - gradient )
            groupsize = max( groupsize - gradient, 3 )
            if groupsize != oldgroupsize:
                print (f"{Fore.YELLOW}[gpu] groupsize changed to {groupsize}{Fore.RESET}")
            
            gpuflag.get()
            gpuflag.task_done()
        else:
            time.sleep(10)


def monitor(nodes, inbound, outbound, counter, inpsize):
    # crude term monitor with 3 custom bars.. todo: move to curses module
    probar = tqdm(total=int(nodes), desc="Executed GPU jobs", position=2, bar_format='{desc}: {n_fmt} ({rate_fmt})                    ')
    incbar = tqdm(total=int(nodes), desc="Inbound pipeline", position=1, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:0.0f}%)                    ')
    outbar = tqdm(total=int(nodes), desc="Outbound pipeline", position=0, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:0.0f}%)                    ')
    # keep main process for monitoring
    while True:
        incbar.n = inbound.qsize()
        outbar.n = sum (i.qsize() for i in outbound)
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

def monitor2(nodes, inbound, outbound, counter, inpsize, stdscr, gpuflag):
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

        stdscr.addstr(3,0,f"GPU velocity: {round(60*(gpujobsdone/(time.time()-start)), 2)} jobs/m")
        """if errors.qsize() > 0:
            msg = errors.get()
            errors.task_done() """
        #stdscr.addstr(5,0,f"messages: {msg}                                                                      ")

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

    time.sleep(10)

    groupsize = 20 # how many shards to group for CLIP

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
    for _ in range(int(3 * groupsize)): # we need 2x IO workers to keep GPU permanently busy
            outbound.append(JoinableQueue())
    inbound = JoinableQueue()
    uploadqueue = JoinableQueue()
    counter = JoinableQueue()
    inpsize = JoinableQueue() # use this to communicate number of jobs downloading now
    gpuflag = JoinableQueue() # use this to flag that gpu is processing
    #errors = JoinableQueue() # use this to capture errors and warnings and route them to curses display

    # launch separate processes with specialized workers
    io = Process(target=io_worker, args=[inbound, outbound, groupsize, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL], daemon=True).start()
    upd = Process(target=upload_worker, args=[uploadqueue, counter, outbound], daemon=True).start()

    #monitor = Process(target=monitor, args=[groupsize, inbound, outbound, counter, inpsize]).start()
    #curses.wrapper(monitor2(nodes, inbound, outbound, counter, inpsize, stdscr, errors, gpuflag))        
    
    gpu_worker(inbound, uploadqueue, gpuflag, groupsize)

    '''
    except KeyboardInterrupt:
        print ("Keyboard interrupt")
        #curses.nocbreak()
        #curses.echo()
        #curses.endwin()

        sys.exit()
    except Exception as e:
        print (f"general exception: {e}")
        sys.exit()
    '''