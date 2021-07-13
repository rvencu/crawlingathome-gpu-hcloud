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
import zipfile
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
    GPU workflow is divided in 4 processes to provide enough parallelism and ensure maximal GPU utilization

1. Incoming worker
    Incoming worker polls CAH server for available GPU jobs. We want to bring in a number of `concat` shards, 
    combine them and process them at once for efficiency
    a) CAH client initialization and make up a name for the job (we can utilize client display name if we want to)
        also make a folder with the job name
    b) test if server have more than `concat` jobs available
    c) bring the number of requested jobs, their payload is a string in uuid hex format
    d) rsync from staging server all into the jobname folder. job file names are unique so we merge them all into the same location
    e) move the stats files out of the way to ./stats folder
    f) transfer client dump and jobname to GPU worker
2. GPU worker
    GPU worker keeps GPU cuda cores as busy as possible. the workflow consists in
    a) open job
    b) make a list of shards in the job
    c) create and concatenate pandas dataframes for each shard
    d) run CLIP filtering on the resulted data
    e) save the result in ./save folder and cleanup the job folder
    f) transfer client dump and jobname to outgoing worker
3. Outgoing worker
    this worker simply moved the result data to the staging server via rsync
    a) open job
    b) make transfer and mark job done
    c) clean up
4. Monitor
    The monitor displays the status of the workers as well as performance metrics about the jobs performed
'''

def incoming_worker(queue: JoinableQueue, inpsize: JoinableQueue, errors: JoinableQueue, concat: int):
    print (f"inbound worker started")
    while True:
        client = cah.init(
                url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD, type="GPU"
            )
        jobname = uuid.uuid4().hex # use this name also for the temp folder to collect job files
        os.makedirs("./"+ jobname)
        while client.jobCount() > concat and client.isAlive():
            try:
                ready = client.newJob(16) # list of all job payload locations in format uuid
                inpsize.put(1)
                
                for job in ready:
                    # download all jobs content into jobname folder
                    response = os.system(f"rsync --remove-source-files -rzh archiveteam@88.198.2.17::gpujobs/{job}/* {jobname}")
                    if response != 0:
                        client.invalidURL(job)
                os.system(f"mv {jobname}/*_parsed.csv stats/")
                os.system(f"mv {jobname}/*_unfiltered.csv stats/")

                queue.put((client.dump(), jobname))
                inpsize.get()
                inpsize.task_done() # empty impsize queue to signal no work to be done
                    
            except Exception as e:
                print(f"some inbound problem occured: {e}")
        else:
            time.sleep(30)


def gpu_worker(inbound: JoinableQueue, outbound: JoinableQueue, counter: JoinableQueue, errors: JoinableQueue, gpuflag: JoinableQueue):
    print (f"gpu worker started")
    while True:
        
        if inbound.qsize() > 0:
            dump, jobname = inbound.get()            
            gpuflag.put(1)

            shards = []
            concat_parse = None

            for path, subdir, files in os.walk(jobname):
                for file in glob(os.path.join(path, "*.csv")):
                    shards.append(Path(jobname).stem.strip("_unfiltered").strip("_parsed").strip("."))
            for item in shards:
                dlparse_df = pd.read_csv(item + ".csv", sep="|")
                dlparse_df["PATH"] = "./" + jobname + "/" + dlparse_df["PATH"]
                if concat_parse is None:
                    concat_parse = dlparse_df
                else:
                    concat_parse = concat_parse.append(dlparse_df, ignore_index=True)
            
            inbound.task_done()
            with open("./save/" + jobname + ".txt", "wt") as f:
                for item in shards:
                    f.write(item + "\n")
            
            #print (f"before deduplication {concat_parse.shape[0]}")
            concat_parse.to_csv("./stats/" + jobname + "_duplicated.csv", index=False, sep="|")
            concat_parse.drop_duplicates(subset=["URL","TEXT"], keep='last', inplace=True)
            concat_parse.reset_index(inplace=True, drop=True)
            concat_parse.to_csv("./stats/" + jobname + "_unfiltered.csv", index=False, sep="|")
            #print (f"after deduplication {concat_parse.shape[0]}")
            start = time.time()
            final_images, results = clip_filter.filter(concat_parse, jobname, "./save/", errors)
            print(f"last filtered {final_images} images in {round(time.time()-start,2)} sec")

            outbound.put((dump, jobname))
            counter.put(1)

            shutil.rmtree(jobname)
            
            gpuflag.get()
            gpuflag.task_done()


def outgoing_worker(queue: JoinableQueue, errors: JoinableQueue):
    print (f"outbound worker started")
    while True:
        try:
            while queue.qsize() > 0:
                dump, jobname = queue.get()
                client = cah.load(**dump)

                if client.isAlive():
                    response = os.system(f"rsync -zh --remove-source-files save/{jobname}/* archiveteam@88.198.2.17::CAH")
                    if response == 0:
                        client.completeJob()
                
                queue.task_done()

            else:
                time.sleep(5)
        except Exception as e:
            print(f"some outbound problem occured: {e}")

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

        stdscr.addstr(3,0,f"GPU velocity: {round(60*(gpujobsdone/(time.time()-start)), 2)} jobs/m")
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

    import crawlingathome_client as cah

    print(
        f"[GPU] starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    nodes = sys.argv[1]
    location = None
    skip = None
    local = True
    concat = 16 # how many shards to group for CLIP
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
        errors = JoinableQueue() # use this to capture errors and warnings and route them to curses display

        # launch separate processes with specialized workers
        inb = Process(target=incoming_worker, args=[inbound, inpsize, errors, concat], daemon=True).start()
        time.sleep(5)
        otb = Process(target=outgoing_worker, args=[outbound, errors], daemon=True).start()
        time.sleep(5)

        monitor = Process(target=monitor, args=[nodes, inbound, outbound, counter, inpsize]).start()
        #curses.wrapper(monitor2(nodes, inbound, outbound, counter, inpsize, stdscr, errors, gpuflag))        
        
        gpu_worker(inbound, outbound, counter, errors, gpuflag)

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