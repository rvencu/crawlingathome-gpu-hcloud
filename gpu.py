import os
import re
import sys
import time
import trio
import uuid
import string
import random
import shutil
import curses
import threading
import clip_filter
import pandas as pd
import infrastructure
from glob import glob
from tqdm import tqdm
from pathlib import Path
from colorama import Fore
from gevent import joinall
sys.path.append('./crawlingathome-worker/')
from multiprocessing import JoinableQueue, Process
from pssh.clients import ParallelSSHClient, SSHClient

'''
Initialization
    The GPU can initialize a cloud swarm on Hetzner Cloud (CX11 type). The number of nodes must be specified at launch

    `python3 gpu.py 32 fsn1` where 32 is number of nodes (use 0 for no swarm), while fsn1 is the name of preffered datacenter

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

def gpu_cah_interface(incomingqueue: JoinableQueue, outgoingqueue: JoinableQueue, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL):
    # initiate and reinitiate a GPU type client if needed
    print (f"              |___ inbound worker started")
    while True:
        client = cah.init(
            url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD, type="GPU"
        )
        while client.jobCount() > 0 and client.isAlive():
            # each thread gets a new job, passes it to GPU then waits for completion
            job = client.newJob()
            os.mkdir("./"+ job)
            response = os.system(f"rsync --remove-source-files -rzh archiveteam@88.198.2.17::gpujobs/{job}/* {job}")
            if response != 0:
                client.invalidURL(job)
                print (f"invalid job detected: {job}")
                continue
            else:
                os.system(f"mv {job}/*_parsed.csv stats/")
                os.system(f"mv {job}/*_unfiltered.csv stats/")
                print (f"job sent to GPU: {job}")
                incomingqueue.put(job)
            
            # wait until job gets processes
            while True:
                print (f"waiting to complete job: {job}")
                outjob, pairs = outgoingqueue[-1] # I hope I can read the queue without popping out the value here
                if outjob == job:
                    print (f"received results for: {job}")
                    outjob, pairs = outgoingqueue.get() # I am poping out from queue only if my current job is finished
                    outgoingqueue.task_done()
                    if pairs > 0:
                        print (f"mark job as complete: {job}")
                        client.completeJob(pairs)
                    shutil.rmtree("./"+ job)
                    break # we can let the worker request a new job
        else:
            print (f"no jobs or client forgotten")
            time.sleep(60)
            continue

def io_worker(incomingqueue: JoinableQueue, outgoingqueue: JoinableQueue, groupsize: int, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL):
    # separate process to initialize threaded workers
    print (f"inbound workers:")
    try:
        # just launch how many threads we need to group jobs into single output
        for _ in range(groupsize):
            threading.Thread(target=gpu_cah_interface, args=(incomingqueue, outgoingqueue, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL)).start()
    except Exception as e:
        print(f"some inbound problem occured: {e}")


def gpu_worker(inbound: JoinableQueue, outbound: JoinableQueue, counter: JoinableQueue, gpuflag: JoinableQueue, groupsize: int):
    print (f"gpu worker started")
    # watch for the incoming queue, when it is big enough we can trigger processing    
    while True:
        print (f"testing incoming queue size")
        if inbound.qsize() >= groupsize:
            gpuflag.put(1)
            shards = []
            group_id = uuid.uuid4().hex
            os.mkdir("./"+ group_id) # place group processing results here
            print (f"got new {groupsize} jobs to group in id {group_id}")
            group_parse = None
            for _ in range(groupsize):
                job = inbound.get()

                all_csv_files = []
                for path, subdir, files in os.walk(job):
                    for file in glob(os.path.join(path, "*.csv")):
                        all_csv_files.append(file)
                # get name of csv file
                out_path = all_csv_files[0]
                shards.append((job, Path(out_path).stem.strip("_unfiltered").strip("_parsed").strip(".")))

                inbound.task_done()
            print (f"adjusting images paths")

            for job, item in shards:
                dlparse_df = pd.read_csv(item + ".csv", sep="|")
                dlparse_df["PATH"] = "./" + job + "/" + dlparse_df["PATH"]
                if group_parse is None:
                    group_parse = dlparse_df
                else:
                    group_parse = group_parse.append(dlparse_df, ignore_index=True)
                
            with open("./save/" + group_id + ".txt", "wt") as f:
                for job, item in shards:
                    f.write(item + "\n")
            
            print (f"saving stats")

            group_parse.to_csv("./stats/" + group_id + "_groupduped.csv", index=False, sep="|") # I am using these to find out domains to filter from scraping
            group_parse.drop_duplicates(subset=["URL","TEXT"], keep='last', inplace=True)
            group_parse.reset_index(inplace=True, drop=True)

            group_parse.to_csv("./stats/" + group_id + "_groupdeduped.csv", index=False, sep="|") # I am using these to find out domains to filter from scraping

            print (f"sending group to CLIP filter")
            start = time.time()
            final_images, results = clip_filter.filter(group_parse, group_id, "./save/")
            print(f"last filtered {final_images} images in {round(time.time()-start,2)} sec")

            print (f"upload group results to rsync target")
            response = os.system(f"rsync -zh --remove-source-files save/{group_id}/* archiveteam@88.198.2.17::CAH") # to do get target from client
            if response == 0:
                print (f"sending all jobs to be marked as completed")
                for job, item in shards:
                    outbound.put((job, results.get(job)))
                    counter.put(1)
            else:
                for job, item in shards:
                    outbound.put((job, 0)) # if upload crashes, then do NOT mark completeJob()
            print (f"cleaning up group folders")
            #shutil.rmtree("./save/"+ group_id)
            
            gpuflag.get()
            gpuflag.task_done()


def monitor(nodes, inbound, outbound, counter, inpsize):
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

    import crawlingathome_client as cah

    print(
        f"[GPU] starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    nodes = sys.argv[1]
    location = None
    skip = None
    local = True
    groupsize = 16 # how many shards to group for CLIP
    if len(sys.argv) > 2:
        location = sys.argv[2]
    if len(sys.argv) > 3:
        skip = sys.argv[3]

    if not os.path.exists("./stats/"):
        os.makedirs("./stats/")
    if not os.path.exists("./save/"):
        os.makedirs("./save/")

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
        #errors = JoinableQueue() # use this to capture errors and warnings and route them to curses display

        # launch separate processes with specialized workers
        io = Process(target=io_worker, args=[inbound, outbound, groupsize, YOUR_NICKNAME_FOR_THE_LEADERBOARD, CRAWLINGATHOME_SERVER_URL], daemon=True).start()

        monitor = Process(target=monitor, args=[nodes, inbound, outbound, counter, inpsize]).start()
        #curses.wrapper(monitor2(nodes, inbound, outbound, counter, inpsize, stdscr, errors, gpuflag))        
        
        gpu_worker(inbound, outbound, counter, gpuflag, groupsize)

    except KeyboardInterrupt:
        #curses.nocbreak()
        #curses.echo()
        #curses.endwin()

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