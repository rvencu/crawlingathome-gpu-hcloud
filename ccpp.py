import os
import sys
import time
import ftfy
import ujson
import gcld3
import uuid
import shutil
import argparse
import hashlib
import tarfile
import psycopg2
import requests
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from random import randint
from datetime import datetime
from sqlalchemy import create_engine
from configparser import ConfigParser
from urllib.parse import urlparse, urljoin
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError
from multiprocessing import Process, cpu_count
from crawlingathome_client.temp import TempCPUWorker


def config(filename='database.ini', mode="test"):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    section='postgresql'
    if mode == "production":
        section='cah_production'

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db

def is_valid_url(url_string: str) -> bool:
    validate_url = URLValidator()
    try:
        validate_url(url_string)
    except ValidationError as e:
        return False
    return True

def log(e):
    with open("errors.txt","a") as f:
        f.write(str(e.__class__.__name__) + " " + str(e) + "\n")

def remove_bad_chars(text):
    # cleanup text so language can be detected
    return "".join(c for c in text if c.isprintable())

def timeit(debug, tick, msg):
    if not debug:
        return
    else:
        print (f"{msg} time chunk {round(time.time()-tick,2)} sec.")
        return time.time()


def parse_wat(content, i, debug):
    tick = time.time()
    """
    This function checks the wat file content and attempts to extract valid candidates of image urls and alt texts

    input: content = wat file content; start = start line number; line_count = how many lines to parse
            usually a wat file is split in 2 halfs or 2 shards. shard 0 starts at the first line and line_count is about 1/2 of wat file lines
            shard 1 starts at the middle of wat file and ends with the last line of wat
    
    output: a list of tuples (url, text, license, domain, hash)
    """

    bloomip = "116.202.162.146"
    bloom2ip = "94.130.167.172"

    print (f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] start parsing")
    tick = timeit(debug, tick, "start parsing")

    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=5, max_num_bytes=2000)

    clpd = 0
    valid_data = []
    check_flag = set() # track urls and make them unique
    content.seek(0)

    for line in tqdm(content, position=i, desc=f"{i} parser"):
        if "IMG@" not in line:
            continue
        line_str = line.strip()
        data = ujson.loads(line_str)
        # find all links inside the line
        linklist = data["Envelope"]["Payload-Metadata"]["HTTP-Response-Metadata"]["HTML-Metadata"]["Links"]
        # get base url
        base_url = os.path.dirname(
            data["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
        )
        license = "?"
        for e in linklist:
            if "url" in e and "creativecommons.org/licenses/" in e["url"]:
                license = e["url"][0:80].replace("\n","").replace('\\','\\\\')
            if not "url" in e:
                continue
            url = e["url"][0:2000].replace("\n","").replace('\\','\\\\')
            try:
                if not is_valid_url(url):
                    continue
            except:
                continue
            # reject links of svg, gif or scripted images content
            if any( x in url for x in {".svg", ".gif", "data:image", "javascript:"} ):
                continue
            try:
                domain = urlparse(url).hostname
            except:
                continue
            if domain is None or domain == "":
                continue
            if len(str(domain)) > 60:
                continue
            detlang = ""
            alt_text = ""
            try:
                if "alt" in e:
                    # detect ALT text language
                    alt_text = ftfy.fix_text(e["alt"].replace("\n", " ")).strip()
                    alt_text = remove_bad_chars(alt_text)
                    res = detector.FindLanguage(alt_text)
                    detlang = res.language
                    rel = res.is_reliable
                    if not rel:
                        detlang = ""
            except:
                pass
            # keep pair or just url if we made it so far
            """ 
            if detlang in ['bn', 'co', 'eo', 'fil', 'fy', 'gd', 'ha', 'haw', 'hmn', 'ig', 'km', 'ku', 'ky', 'lo', 'mi', 'mn', 'mt', 'ny', 'sd', 'si', 'sm', 'sn', 'so', 'st', 'su', 'sw', 'xh', 'yi', 'zu']:
            """
            # get rid of Latn suffix when detected
            if detlang != "":
                detlang = detlang.split("-")[0]
            if alt_text == "" or alt_text is None:
                continue
            if len(alt_text) < 5:
                continue
            alt_text = alt_text[0:2000].replace("\t"," ").replace("\n"," ").replace('\\','\\\\') # will use tab as field separator for copy source
            if not url.startswith("http"):
                url = urljoin(base_url, url)
            hash = hashlib.md5((url + alt_text).encode("utf-8")).hexdigest()
            if url not in check_flag:
                valid_data.append((url, alt_text, license, domain, detlang, hash))
                check_flag.add(url)


    tick = timeit(debug, tick, "loop finished")        
    print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] lenght of pairs to filter {len(valid_data)}")
    s = time.time()

    # remove from valid_data elements rejected by clipped bloom server
    with open(f'{i}/hash.txt', 'w') as f:
        for item in valid_data:
            f.write(item[-1].strip()+"\n")
    post = {
        'file': ('hash.txt', open(f'{i}/hash.txt', 'rb')),
        'key': (None, 'clipped'),
    }
    
    tick = timeit(debug, tick, "clip bloom prepared")
    failure = True
    for _ in range(10):
        try:
            response = requests.post(f'http://{bloomip}:8000/deduplicate/', files=post)
            if response.status_code != 200:
                print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] bloom server error, retrying... got {response.status_code}")
                time.sleep(randint(5,30))
            else:
                failure = False
                break
        except:
            time.sleep(30)
    if failure:
        print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] crash, cannot contact the clipped bloom server, please fix")
        return  (None, 0, 0)

    valid_hashes = set(response.content.decode("utf-8").split("\n"))

    print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] clipped bloom server returned {len(valid_hashes)} in {round(time.time()-s,3)} sec")
    tick = timeit(debug, tick, "clip bloom done")

    valid_data = [t for t in {tuple(i) for i in valid_data}]
    kept_data = []
    clpd = len(valid_data)

    for item in valid_data:
        if item[-1].strip() in valid_hashes:
            kept_data.append(item)
            clpd -= 1
    
    s = time.time()
    # remove from valid_data elements rejected by parsed bloom server
    with open(f'{i}/hash.txt', 'w') as f:
        for item in kept_data:
            f.write(item[0].strip()+"\n")
    post = {
        'file': ('hash.txt', open(f'{i}/hash.txt', 'rb')),
        'key': (None, 'parsed'),
    }
    
    tick = timeit(debug, tick, "parsed bloom prepared")
    failure = True
    for _ in range(10):
        try:
            response = requests.post(f'http://{bloom2ip}:8000/deduplicate/', files=post)
            if response.status_code != 200:
                print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] bloom server error, retrying... got {response.status_code}")
                time.sleep(randint(5,30))
            else:
                failure = False
                break
        except:
            time.sleep(30)
    if failure:
        print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] crash, cannot contact the parsed bloom server, please fix")
        return (None, 0, 0)

    valid_urls = set(response.content.decode("utf-8").split("\n"))

    print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] parsed bloom server returned {len(valid_urls)} in {round(time.time()-s,3)} sec")
    tick = timeit(debug, tick, "parsed bloom done")

    valid_data = [t for t in {tuple(i) for i in kept_data}]
    final_kept_data = []
    prsd = len(kept_data)

    for item in kept_data:
        if item[0].strip() in valid_urls:
            final_kept_data.append(item)
            prsd -= 1

    print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] lenght of deduplicated pairs to return {len(final_kept_data)}")

    # add parsed urls to parsed bloom server
    with open('hash.txt', 'w') as f:
        for url in final_kept_data:
            f.write(url[0].strip()+"\n")
    post = {
        'file': ('hash.txt', open('hash.txt', 'rb')),
        'key': (None, 'parsed'),
    }
    
    tick = timeit(debug, tick, "add to parsed bloom prepared")
    failure = True
    for _ in range(10):
        try:
            response = requests.post(f'http://{bloom2ip}:8000/add/', files=post)
            if response.status_code != 200:
                print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] bloom server error, retrying... got {response.status_code}")
                time.sleep(randint(5,30))
            else:
                failure = False
                print(f"bloom add response: {response.text}")
                break
        except:
            time.sleep(15)
    if failure:
        print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] crash, cannot contact the parsed bloom server, please fix")
        return (None, 0, 0)

    tick = timeit(debug, tick, "add to parsed bloom done")

    return (final_kept_data, clpd, prsd)  # use a dict in order to remove duplicate tuples from list

def proc_worker(i: int, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL, engine, host, debug, current_set):
    # initialize working folders
    tmp_folder = f"./{i}/.tmp/"

    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    # connect to C@H server and initialize client
    client = TempCPUWorker(url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD)

    # initialize stats variables for previous job
    last = 0

    # this makes a loop to download new jobs while the script is running
    # normally it reads while client.jobCount() > 0
    conn = engine.raw_connection()
    while True:
        try:
            # clean the folder
            if os.path.exists(f"{i}"):
                shutil.rmtree(f"{i}", ignore_errors=True)
            os.makedirs(tmp_folder)

            tick = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] clock is {datetime.now().strftime('%H:%M:%S')}")

            start = time.time()
            start0 = start

            # get new job and download the wat file
            client.newJob()
            tick = timeit(debug, tick, "got new job")
            client.downloadWat(tmp_folder)
            tick = timeit(debug, tick, "downloaded wat")

            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] downloaded wat in {round(time.time()-start,2)}")
            start = time.time()

            first_sample_id = np.int64(client.shards[0][1]["start_id"])
                        
            # parse valid links from wat file
            with open(tmp_folder + "shard.wat", "r") as infile:
                parsed_data, clpd, prsd = parse_wat(infile, i, debug)

            if parsed_data is None:
                continue
            tick = timeit(debug, tick, "parsing finalized")

            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] parsed wat in {round(time.time()-start,2)}")
            start = time.time()

            # convert to dataframe and save to disk (for statistics and generating blocking lists)
            parsed_df = pd.DataFrame(parsed_data, columns=["url","text","license","domain","language","hash"])
            parsed_df = parsed_df.drop_duplicates(subset=["url"])
            parsed_df.insert(0, 'sampleid', range(first_sample_id, first_sample_id + len(parsed_df)))
            parsed_df["wat"] = int(client.shards[-1][0])
            parsed_df = parsed_df[["sampleid","url","text","license","domain","wat","hash","language"]]

            # postgres should only ingest current working data not all
            en_df = parsed_df[parsed_df["language"]=="en"]
            nolang_df = parsed_df[parsed_df["language"]==""]
            multilang_df = parsed_df[(parsed_df["language"]!="en") & (parsed_df["language"]!="")]

            tick = timeit(debug, tick, "dataframe preparation done")
            current = en_df
            if current_set == "":
                current = nolang_df
                print(f"currently working on nolang dataset")
            if current_set == "multilang":
                current = multilang_df
                print(f"currently working on multilang dataset")

            if len(parsed_df.index) > 0:
                tick = timeit(debug, tick, "before sql copy")
                parsed_df.to_csv(f"{i}/export_sql.txt", sep='\t', index=False, header=False)
                
                cur = conn.cursor()
                with open(f"{i}/export_sql.txt", "rt") as f:
                    cur.copy_from(f, 'dataset_buffer', columns=("sampleid","url","text","license","domain","wat","hash","language"))
                conn.commit()
                cur.close()
                
                tick = timeit(debug, tick, "finished sql copy")

            uid = uuid.uuid4().hex
            
            '''
            if not current.equals(en_df):
                en_df.to_csv(f"{i}/en-{uid}.txt", sep='\t', index=False, header=False)
            if not current.equals(nolang_df):
                nolang_df.to_csv(f"{i}/nolang-{uid}.txt", sep='\t', index=False, header=False)
            if not current.equals(multilang_df):
                multilang_df.to_csv(f"{i}/intl-{uid}.txt", sep='\t', index=False, header=False)
            
            os.system(f"rsync -amv --include='*{uid}.txt' --include='*/' --exclude='*' ./{i}/ postgres@185.154.158.196::aidb")
            '''

            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] saved links in {round(time.time()-start,2)}")

            lastlinks = len(parsed_data)
            en_pairs = len(en_df.index)
            nolang_pairs = len(nolang_df.index)
            intl_pairs = len(multilang_df.index)
            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] this job has {lastlinks} links left after removing {clpd} already clipped and {prsd} already parsed")
            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] links are split into {en_pairs} english, {intl_pairs} multilanguage and {nolang_pairs} without language")
            with open("datapoints.txt", "a") as f:
                f.write(f"{time.time()}\t{en_pairs}\t{intl_pairs}\t{nolang_pairs}\n")

            prefixes = {}
            prefixes[str(client.shards[0][0])] = f"postgres {host}"
            prefixes[str(client.shards[1][0])] = f"postgres {host}"
            client.completeJob(prefixes)
            tick = timeit(debug, tick, "executed complete job")

            last = round(time.time() - start0)
            print(f"[{datetime.now().strftime('%H:%M:%S')} {i} stats] WAT job completed in {last} seconds")
           
        except Exception as e:
            print (f"[{datetime.now().strftime('%H:%M:%S')} exception {i} parser] {e}")
            print (f"[{datetime.now().strftime('%H:%M:%S')} {i} parser] worker crashed")
            time.sleep(60)
            client = TempCPUWorker(url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD)
    conn.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=sys.argv[0], usage='%(prog)s -m/--mode -c/--cpus -n/--name -d/--debug')
    parser.add_argument("-n","--name",action='append',help="Your leaderboard nickname",required=False)
    parser.add_argument("-c","--cpus",action='append',help="How many cpus to use",required=False)
    parser.add_argument("-d","--debug",action='append',help="Print debug lines?",required=False)
    parser.add_argument("-m","--mode",action='append',help="Mode to run",required=True)
    parser.add_argument("-s","--set",action='append',help="Choose current set (en, nolang, multilang)",required=True)
    args = parser.parse_args()

    # initialize client variables
    YOUR_NICKNAME_FOR_THE_LEADERBOARD = None
    if args.name is not None:
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = " ".join(args.name)

    if YOUR_NICKNAME_FOR_THE_LEADERBOARD in (None,""):
        YOUR_NICKNAME_FOR_THE_LEADERBOARD = "ccpp-dev"
    CRAWLINGATHOME_SERVER_URL = "http://cah.io.community/"

    print (f"starting session under `{YOUR_NICKNAME_FOR_THE_LEADERBOARD}` nickname")

    procs = cpu_count()
    if args.cpus is not None and int(args.cpus[0]) > 0:
        procs = int(args.cpus[0])

    debug = False
    if args.debug is not None and args.debug[0] == "true":
        debug = True

    params = config(mode=args.mode[0])

    engine = create_engine(f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:5432/{params["database"]}', pool_size=procs, max_overflow=int(procs*1.5), pool_pre_ping=True)
    workers = []
    for i in range ( procs ):
        #use this queue to annount that bloom is currently processing and please do not update filters. if queue is not empty please wait, if queue is empty you may update filters
        workers.append(Process(target=proc_worker, args= [i, YOUR_NICKNAME_FOR_THE_LEADERBOARD,  CRAWLINGATHOME_SERVER_URL, engine, params["host"], debug, args.set[0]], daemon=True))

    time.sleep(10)

    for worker in workers:
        worker.start()
        time.sleep(8)
    
    while True:
        #keep main process alive
        time.sleep(60)
