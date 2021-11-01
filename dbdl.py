'''
Encoding image analyzing errors: Add the numbers below to 8 to encode all types of errors (so status=9...23 is reserved to describe the errors)
- general exception: 1
- bad format: 2
- image too big: 4
- image too small: 8
- any combination of above 

'''


import gc 
import os
import ssl
import sys
import time
import trio
import uuid
import ujson
import shutil
import tarfile
import pandas as pd
from glob import glob
from uuid import uuid1
from io import BytesIO
from sqlalchemy import create_engine
from configparser import ConfigParser
from PIL import Image, ImageFile, UnidentifiedImageError 
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem

sys.path.append('./crawlingathome-worker/')

import asks
asks.init("trio")

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

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

class Tracer(trio.abc.Instrument):

    def __init__(self):
        self.exceptions = 0
        self.requests = 0
        self.downloads = 0
        self.imgproc_duration = 0
        self.download_duration = 0
        self.error_duration = 0

    def task_exited(self, task):
        if task.custom_sleep_data is not None:
            if task.custom_sleep_data[0] in [1, 3]: # this is exception
                self.exceptions += 1
                self.error_duration += task.custom_sleep_data[2]
            if task.custom_sleep_data[0] == 0: # this is image downloaded
                self.download_duration += task.custom_sleep_data[1]
                self.imgproc_duration += task.custom_sleep_data[2]
                self.downloads += 1

    def after_run(self):
        rate = round(self.exceptions / (self.exceptions + self.downloads + sys.float_info.epsilon), 2)
        avg_download = round(self.download_duration / (self.downloads + sys.float_info.epsilon), 2)
        avg_process = round(self.imgproc_duration / (self.downloads + sys.float_info.epsilon), 2)
        avg_error = round(self.error_duration / (self.exceptions + sys.float_info.epsilon), 2)
        print(f"[instrumentation] While scraping there were {self.exceptions} errors within {self.downloads + self.exceptions} candidates (error rate = {round(rate * 100,2)} %). {self.downloads} images were downloaded.")
        print(f"[instrumentation] Cumulative image processing duration {round(self.imgproc_duration, 2)} s.")
        print(f"[instrumentation] Average downloading time {avg_download} s/img, image processing time {avg_process} s/img, exceptions processing time {avg_error} s/link")

def log(e):
    with open("errors.txt","a") as f:
        f.write(str(e.__class__.__name__) + " " + str(e) + "\n")


def process_img_content(response, alt_text, license, sample_id, language):
    """
    Function to process downloaded image. Use use PIL from pillow-simd 
        (faster than open cv that in return is faster than original pillow)
    
    input: web request response, ALT text, license and sample id

    output: list of image parameters or None if image is rejected
    """
    img_output_folder = "save/images/"
    error_code = 8

    #temp 2 lines
    if language == "": 
        language = "en" 

    def _resize(im: Image):
        width, height = im.size
        ratio = min(width, height) / 224
        new_width = int(round(width/ratio,0))
        new_height = int(round(height/ratio,0))
        im = im.resize((new_width, new_height), resample=Image.BICUBIC)
        if new_width > 224 or new_height > 224:
            left = (new_width - 224)/2
            top = (new_height - 224)/2
            right = (new_width + 224)/2
            bottom = (new_height + 224)/2
            # Crop the center of the image
            im = im.crop((left, top, right, bottom))
        return im
    try:
        # reject too small images
        if len(response.content) < 5000:
            error_code += 8
        img_data = BytesIO(response.content)
        with Image.open(img_data) as im:
            width, height = im.size
            # reject if too large (might be a DOS decompression bomb)
            if width * height > 89478484:
                error_code += 4
            im_format = im.format
            out_fname = f"{img_output_folder}{str(sample_id)}.{im_format.lower()}"
            # reject if format is not in this list
            if im_format not in ["JPEG", "JPG", "PNG", "WEBP"]:
                error_code += 2
            if min(width, height) > 224:
                im = _resize(im)
            
            # convert all images to RGB (necessary for CLIP, also CLIP is doing it again so do we need it here?)
            if im.mode != "RGB":
                im = im.convert("RGB")
            if error_code == 8:
                im.save(out_fname) # do not retain images we do not need
    except (KeyError, UnidentifiedImageError):
        error_code += 1
    
    if error_code == 8:
        error_code = 2 # mark succesful lines with status = 2

    return [str(sample_id), out_fname, response.url, alt_text, width, height, license, language, error_code]


async def request_image(parsed_df):
    """
    This function initiates many parallel async connections to try download the images from provided links
    
    input: dataset of validated links, the sample id to start with

    output: list of lists with succesfully downloaded images and their parameters. this list is dumped on disk as json file
    """
    tmp_data = []
    limit = trio.CapacityLimiter(1000)

    # change the number of parallel connections based on CPU speed, network capabilities, etc.
    # the number of 192 is optimized for 1 vCPU droplet at Hetzner Cloud (code CX11)
    session = asks.Session(connections=64, ssl_context=ssl_ctx)

    software_names = [SoftwareName.CHROME.value]
    operating_systems = [OperatingSystem.LINUX.value]   

    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=2000)
    user_agent = user_agent_rotator.get_random_user_agent()

    # try to make the bot website friendly
    session.headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://google.com",
        "DNT": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async def _request(row):
        while True:
            start=time.time()
            sample_id = row[0]
            url = row[1]
            alt_text = row[2]
            license = row[3]
            language = row[4]
            # the following 2 lines are related to Trio Instrument to capture events from multiple threads
            task = trio.lowlevel.current_task()
            try:
                response = await session.get(url, timeout=10, connection_timeout=20)
                dltime = round(time.time()-start, 2)
                start=time.time()
                proces = process_img_content(
                    # tune timeout and connection_timeout to grab more or less files. shorter timeouts will exclude bad performing websites
                    response, alt_text, license, sample_id, language
                )
                proctime = round(time.time()-start, 2)
                task.custom_sleep_data = (0, dltime, proctime) # for success do not count errors
                if proces is not None:
                    tmp_data.append(proces)
            except Exception as e:
                log(e)
                task.custom_sleep_data = (1, 0, round(time.time()-start,2)) # when exception is hit, count it
            return

    async with trio.open_nursery() as n:
        for index, row in parsed_df.iterrows():
            async with limit:
                n.start_soon(_request, row)
            
    # trio makes sure at this point all async tasks were executed
    with open(f".tmp/{uuid1()}.json", "w") as f:
        ujson.dump(tmp_data, f)
    gc.collect()

    return


def dl_wat(parsed_df): # replace valid data and start sampleid with parsed_df
    """
    This function initiates download attempt of validated parsed links
    It launches multithreaded tasks by using trio module
    
    input: dataset of validated links, the sample id to start with

    output: dataframe of downloaded images and their parameters
    """
    
    # Download every image available
    processed_samples = []
    #trio.run(request_image, valid_data, first_sample_id, instruments=[TrioProgress(len(valid_data), False)] )
    trio.run( request_image, parsed_df, instruments=[Tracer()] )

    for tmpf in glob(".tmp/*.json"):
        processed_samples.extend(ujson.load(open(tmpf)))
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH", "LICENSE", "LANGUAGE", "STATUS"],
    )

def upload(source: str, clientType: str, target: str):
    with tarfile.open(f"{source}.tar.gz", "w:gz") as tar:
        tar.add(source, arcname=os.path.basename(source))
    print(f"client type is {clientType}")
    result = os.system(f"rsync -av {source}.tar.gz {target}")
    if os.path.exists(f"{source}.tar.gz"):
        os.remove(f"{source}.tar.gz")
    if os.path.exists(f"{source}"):
        shutil.rmtree(f"{source}", ignore_errors=True)
    return result

def newJob(engine):
    # strict selection of distinct domains
    # select_stmt1 = "UPDATE dataset SET status = 1 WHERE sampleid IN (SELECT DISTINCT ON (domain) sampleid FROM (SELECT domain, sampleid FROM dataset TABLESAMPLE SYSTEM (0.05) WHERE status = 0 LIMIT 1000000 FOR UPDATE SKIP LOCKED) as \"U\" LIMIT 10000) AND status = 0 RETURNING sampleid"
    # selection on domains based on distribution of URLs per domain
    select_stmt1 = "UPDATE dataset SET status = 1 WHERE sampleid IN (SELECT sampleid FROM dataset TABLESAMPLE SYSTEM (0.05) WHERE status = 0 LIMIT 10000 FOR UPDATE SKIP LOCKED) AND status = 0 RETURNING sampleid"
    conn = engine.raw_connection()
    cur = conn.cursor()
    cur.execute(select_stmt1)
    result = cur.fetchall()
    conn.commit()
    cur.close()

    values = ",".join([str(tuple[0]) for tuple in result])
    select_stmt2 = "SELECT sampleid, url, text, license, language FROM dataset WHERE sampleid in ({})".format(values)
    df = pd.read_sql_query(select_stmt2, conn)
    conn.close()
    return df

""" def completeJob(engine, prefix, parsed_df, dlparse_df):
    values1 = ",".join(dlparse_df["SAMPLE_ID"].astype(str))
    values2 = ",".join(parsed_df["sampleid"].astype(str))
    update_stmt1 = "UPDATE dataset SET status=2 where sampleid in ({})".format(values1)
    update_stmt2 = "UPDATE dataset SET status=9 where status=1 AND sampleid in ({})".format(values2)
    insert_stmt = "INSERT INTO jobs (jobid) VALUES ('{}')".format(prefix)

    if len(dlparse_df.index > 0):
        conn = engine.raw_connection()
        cur = conn.cursor()
        cur.execute(update_stmt1)
        cur.execute(insert_stmt)
        conn.commit()
        cur.close()
        conn.close()

    conn = engine.raw_connection()
    cur = conn.cursor()
    cur.execute(update_stmt2)
    conn.commit()
    cur.close()
    conn.close()
    return """

def completeJob2(engine, prefix, parsed_df, dlparse_df):
    # prepare data for EN
    values2 = ",".join(parsed_df["sampleid"].astype(str))
    update_stmt1 = ""
    for i, row in dlparse_df.iterrows():
        update_stmt1 += "UPDATE dataset SET status={}, width={}, height={} where sampleid = {};".format(row["STATUS"],row["HEIGHT"],row["WIDTH"],row["SAMPLE_ID"])
        # this is intentional mix between width and heigth to account for the but in previous laion release
        # the csv will go scrambled but in database we want good values
    insert_stmt = "INSERT INTO jobs (jobid) VALUES ('{}')".format(prefix)

    if len(dlparse_df.index > 0):
        conn = engine.raw_connection()
        cur = conn.cursor()
        cur.execute(update_stmt1)
        cur.execute(insert_stmt)
        conn.commit()
        cur.close()
        conn.close()

    # in case there are samples unaccounted for, we try to mark them with general error status
    update_stmt2 = "UPDATE dataset SET status=9 where status=1 AND sampleid in ({})".format(values2)

    conn = engine.raw_connection()
    cur = conn.cursor()
    cur.execute(update_stmt2)
    conn.commit()
    cur.close()
    conn.close()
    return

if __name__ == "__main__":

    # initialize working folders
    output_folder = "./save/"
    img_output_folder = output_folder + "images/"

    print (f"starting session")
    
    params = config()
    engine = create_engine(f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:5432/{params["database"]}')

    while True:
        try:
            start = time.time()
            start0 = start

            parsed_df = newJob(engine)
            prefix = uuid.uuid4().hex
            result = 0

            # clear working folders for a new job
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder, ignore_errors=True)
            if os.path.exists(".tmp"):
                shutil.rmtree(".tmp")

            os.mkdir(output_folder)
            os.mkdir(img_output_folder)
            os.mkdir(".tmp")

            # compute output file names base
            out_fname = f"3_staged_workflow_job_{prefix}_full_wat"
            print(f"[stats] Job acquired in {round(time.time()-start,2)} sec")
            start = time.time()

            print (f"[stats] This job has {len(parsed_df)} candidates")
        
            # attempt to download validated links and save to disk for stats and blocking lists
            dlparse_df = dl_wat(parsed_df)

            # at this point we finishes the CPU node job, need to make the data available for GPU worker
            os.mkdir(prefix)
            os.system(f"mv save/* {prefix}/")
            result += upload(prefix, "CPU", "archiveteam@176.9.4.150::gpujobs") #todo find the IP and endpoint
            if result == 0:
                completeJob2(engine, prefix, parsed_df, dlparse_df)

            dlparse_df = dlparse_df[dlparse_df["status"]==2] # remove rejected items from gpu jobs
            dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")

            print (f"[stats] pairs retained {len(dlparse_df)} in {round(time.time() - start, 2)}")
            print (f"[stats] scraping efficiency {len(dlparse_df)/(time.time() - start)} img/sec")
            print (f"[stats] crawling efficiency {len(parsed_df)/(time.time() - start)} links/sec")


            last = round(time.time() - start0)

            print(f"[stats] Job completed in {last} seconds")
        
        except Exception as e:
            print (e)
            print ("Worker crashed")
            time.sleep(60)