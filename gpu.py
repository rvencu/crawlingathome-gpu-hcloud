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


def incoming_worker(queue: JoinableQueue, inpsize: JoinableQueue, errors: JoinableQueue):
    print (f"inbound worker started")

    while True:
        client = cah.init(
                url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD, type="GPU"
            )
        while client.jobCount() > 16 and client.isAlive():
            try:
                ready = client.newJob(16) # list of all job payload locations in format ip:filepath
                inpsize.put(1)
                workers = [x.split(":")[0] for x in ready]
                copy_args = [{'local_file': 'gpujob.zip_' + workers[i],
                    'remote_file': ready[i].split(":")[1],
                    } for i in enumerate(ready)]
                
                dclient = ParallelSSHClient(workers, user='crawl', pkey="~/.ssh/id_cah", identity_auth=False)
                try:
                    cmds = dclient.copy_remote_file('%(remote_file)s', '%(local_file)s', copy_args=copy_args)
                    joinall (cmds, raise_error=True)
                except Exception as e:
                    print(e)

                dclient.run_command('%s', host_args=(",".join(["rm -rf " + x.split(":")[1] for x in ready])))

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
                    except:
                        pass
                    os.remove(file)

                queue.put((client.dump(), workers))
                inpsize.get()
                inpsize.task_done() # empty impsize queue to signal no work to be done
                    
            except Exception as e:
                print(f"some inbound problem occured: {e}")
        else:
            time.sleep(30)

def outgoing_worker(queue: JoinableQueue, errors: JoinableQueue):
    print (f"outbound worker started")
    while True:
        try:
            while queue.qsize() > 0:
                dump, jobname = queue.get()

                client = cah.load(**dump)
                if client.isAlive():
                    # here insert logic to upload result
                    #
                    #
                    client.completeJob()
                
                # remove the payload
                # os.remove("")

                queue.task_done()

            else:
                time.sleep(5)
        except Exception as e:
            print(f"some outbound problem occured: {e}")

def gpu_worker(inbound: JoinableQueue, outbound: JoinableQueue, counter: JoinableQueue, errors: JoinableQueue, gpuflag: JoinableQueue, concat):
    print (f"gpu worker started")
    while True:
        if not os.path.exists("./save/"):
            os.makedirs("./save/")
        if not os.path.exists("./stats/"):
            os.makedirs("./stats/")

        if inbound.qsize() > 0:
            jobname = uuid.uuid4().hex # use this name also for the temp folder to collect job files
            dump, workers = inbound.get()
            os.makedirs("./"+ jobname)
            gpuflag.put(1)
            ips = []
            dframes = []
            shards = []
            concat_parse = None
            for ip in workers:
                #move the files asap so the worker can continue to send files in its incoming folder
                output_folder = "./" + jobname + "/" + ip.replace(".", "-") + "/save/"
                os.makedirs("./" + jobname + "/" + ip.replace(".", "-") + "/save/")
                os.system(f"mv -f ./{ip.replace('.', '-')}/* ./{jobname}/{ip.replace('.', '-')}/")
                ips.append(ip)

                all_csv_files = []
                for path, subdir, files in os.walk(output_folder):
                    for file in glob(os.path.join(path, "*.csv")):
                        all_csv_files.append(file)
                # get name of csv file
                out_path = all_csv_files[0]
                out_fname = Path(out_path).stem.strip("_unfiltered").strip("_parsed").strip(".")
                shards.append(out_fname)
                os.system(f"mv {output_folder + out_fname}_parsed.csv ./stats/")

                # recreate parsed dataset and run CLIP filtering
                dlparse_df = pd.read_csv(output_folder + out_fname + ".csv", sep="|")
                dlparse_df["PATH"] = "./" + jobname + "/" + ip.replace(".", "-") + "/" + dlparse_df["PATH"]

                if concat_parse is None:
                    concat_parse = dlparse_df
                else:
                    concat_parse = concat_parse.append(dlparse_df, ignore_index=True)

                dframes.append(dlparse_df)
            
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
        otb = Process(target=outgoing_worker, args=[outbound, errors], daemon=True).start()
        time.sleep(5)

        monitor = Process(target=monitor, args=[nodes, inbound, outbound, counter, inpsize]).start()
        #curses.wrapper(monitor2(nodes, inbound, outbound, counter, inpsize, stdscr, errors, gpuflag))        
        
        gpu_worker(inbound, outbound, counter, errors, gpuflag, concat)

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