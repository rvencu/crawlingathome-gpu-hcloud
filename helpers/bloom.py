# use this file inside every minute cron in order to recalculate bloom filters. location: staging server
# folder structure
# /home/archiveteam/CAH/
#                   |_bloom         archiveteam@IP::bloom   contains bloom filters
#                   |_clipped                               contains clipped lists
#                   |_ds                                    contains files ready to be sent to the eye
#                   |_hashes                                contains list of hashes of files inserted into the dataset
#                   |_results       archiveteam@IP::CAH     incoming folder for the final results from workers

# Stacked bloom filters. Naming convention:
#   frozen filters: filter.bin, filter1.bin, filter2.bin
#   active filters: filter_active.bin
#
#
import sys
import time
import requests
import pandas as pd
from glob import glob
from pathlib import Path
from datetime import datetime
from bloom_filter2 import BloomFilter

# update the bloom server filters too
bloomip = "116.202.162.146"

serverbloom = BloomFilter(max_elements=10000000, error_rate=0.01, filename=(f"/home/archiveteam/bloom-{bloomip}.bin",-1))
serverclip = BloomFilter(max_elements=10000000, error_rate=0.01, filename=(f"/home/archiveteam/clip-{bloomip}.bin",-1))

start = time.time()
now = datetime.now().strftime("%Y/%m/%d_%H:%M")

bloom = [BloomFilter(max_elements=200000000, error_rate=0.05, filename=(x,-1)) for x in glob("/home/archiveteam/CAH/bloom/bloom[!_]*")]
bloom_active = BloomFilter(max_elements=200000000, error_rate=0.05, filename=("/home/archiveteam/CAH/bloom/bloom_active.bin",-1))
bloom.append(bloom_active)
filesbloom = BloomFilter(max_elements=10000000, error_rate=0.01, filename=("/home/archiveteam/filesbloom.bin",-1))

failed = BloomFilter(max_elements=10000000, error_rate=0.01, filename=("/home/archiveteam/CAH/bloom/failed-domains.bin",-1))
filesfailed = BloomFilter(max_elements=100000, error_rate=0.01, filename=("/home/archiveteam/filesfailed.bin",-1))

clipped = [BloomFilter(max_elements=200000000, error_rate=0.05, filename=(x,-1)) for x in glob("/home/archiveteam/CAH/bloom/clipped[!_]*")]
clipped_active = BloomFilter(max_elements=200000000, error_rate=0.05, filename=("/home/archiveteam/CAH/bloom/clipped_active.bin",-1))
clipped.append(clipped_active)
filesclipped = BloomFilter(max_elements=10000000, error_rate=0.01, filename=("/home/archiveteam/filesclipped.bin",-1))

time.sleep(5)
counter = 0
uniques = 0
for file in glob("/home/archiveteam/CAH/hashes/*.hsh"):
    stem = Path(file).stem.strip(".")
    if stem not in filesbloom:
        with open(file,"rt") as f:
            for line in f.readlines():
                line = line.strip()
                counter += 1
                infilters = False
                for filter in bloom:
                    if line in filter:
                        infilters = True
                        break
                if not infilters:
                    bloom_active.add(line)
                    uniques += 1
        filesbloom.add(stem)
    if stem not in serverbloom:
        post = {
            'file': (stem, open(file, 'rb')),
            'key': (None, 'main'),
        }
        response = requests.post(f'http://{bloomip}:8000/add/', files=post)
        if response.status_code == 200:
            serverbloom.add(stem)

failed_counter = 0
for file in glob("/home/archiveteam/CAH/bloom/*.txt"):
    stem = Path(file).stem.strip(".")
    if stem not in filesfailed:
        with open(file,"rt") as f:
            for line in f.readlines():
                line = line.strip()
                if line not in failed:
                    failed.add(line)
                    failed_counter += 1
        filesfailed.add(stem)

clipped_counter = 0
for file in glob("/home/archiveteam/CAH/clipped/*.clp"):
    stem = Path(file).stem.strip(".")
    if stem not in filesclipped:
        with open(file,"rt") as f:
            for line in f.readlines():
                line = line.strip()
                infilters = False
                for filter in clipped:
                    if line in filter:
                        infilters = True
                        break
                if not infilters:
                    clipped_active.add(line)
                    clipped_counter += 1
        filesclipped.add(stem)
    if stem not in serverclip:
        post = {
            'file': (stem, open(file, 'rb')),
            'key': (None, 'clipped'),
        }
        response = requests.post(f'http://{bloomip}:8000/add/', files=post)
        if response.status_code == 200:
            serverclip.add(stem)

pd.set_option('precision', 2)
df = pd.read_csv("bloom.log", sep=" ",header=None, names=["Date", "a", "unique pairs (5%)", "b", "total including duplicates","c","clipped filter (5%)","d","failed filter","e"])
df["Date"]=df.Date.apply(lambda x: datetime.strptime(x, "[%Y/%m/%d_%H:%M]"))
df["unique pairs (5%)"]=df["unique pairs (5%)"]/1000000
df["total including duplicates"]=df["total including duplicates"]/1000000
df["clipped filter (5%)"]=df["clipped filter (5%)"]/1000000

if uniques > 0:
    print(f"[{now}] added {uniques} \"from total of\" {counter} \"(i.e. {round((counter-uniques)*100/(counter+sys.float_info.epsilon),2)}% duplication in {round(time.time()-start,2)} sec) Also added \" {clipped_counter} \"clipped and\" {failed_counter} failed")
    with open('dashboard.txt', 'w') as file:
        file.write("<h5><a href='http://cah.io.community'>Crawling at Home project</a></h5>\n")
        file.write("<h1>Bloom filters status</h1>\n")
        file.write("<h2>All time stats</h2>\n")
        file.write("<h5>initialized from first parquet files</h5>\n")
        file.write(str(df.sum(axis=0, numeric_only=True)).replace("\n","<br/>"))
        file.write("<br/><br/>")
        file.write("<h2>Last day stats</h2>\n")
        file.write(str(df[df.Date > datetime.now() - pd.to_timedelta("1day")].sum(axis=0, numeric_only=True)).replace("\n","<br/>"))
        file.write("<h2>Last week stats</h2>\n")
        file.write("<h5>Last reset date: 02 August 2021</h5>\n")
        file.write(str(df[df.Date > datetime.now() - pd.to_timedelta("7day")].sum(axis=0, numeric_only=True)).replace("\n","<br/>"))


    
