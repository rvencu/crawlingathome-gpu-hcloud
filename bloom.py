# use this file inside every minute cron in order to recalculate bloom filters. location: staging server
# folder structure
# /home/archiveteam/CAH/
#                   |_bloom         archiveteam@IP::bloom   contains bloom filters
#                   |_clipped                               contains clipped lists
#                   |_ds                                    contains files ready to be sent to the eye
#                   |_hashes                                contains list of hashes of files inserted into the dataset
#                   |_results       archiveteam@IP::CAH     incoming folder for the final results from workers

from bloom_filter2 import BloomFilter
from pathlib import Path
import pandas as pd
from glob import glob
from datetime import datetime
import sys
import time

start = time.time()
now = datetime.now().strftime("%Y/%m/%d_%H:%M")

bloom = BloomFilter(max_elements=80000000, error_rate=0.01, filename=("/home/archiveteam/CAH/bloom/bloom.bin",-1))
filesbloom = BloomFilter(max_elements=10000000, error_rate=0.01, filename=("/home/archiveteam/filesbloom.bin",-1))
failed = BloomFilter(max_elements=10000000, error_rate=0.01, filename=("/home/archiveteam/CAH/bloom/failed-domains.bin",-1))
filesfailed = BloomFilter(max_elements=100000, error_rate=0.01, filename=("/home/archiveteam/filesfailed.bin",-1))
clipped = BloomFilter(max_elements=200000000, error_rate=0.05, filename=("/home/archiveteam/CAH/bloom/clipped.bin",-1))
filesclipped = BloomFilter(max_elements=100000, error_rate=0.01, filename=("/home/archiveteam/filesclipped.bin",-1))

counter = 0
uniques = 0
for file in glob("/home/archiveteam/CAH/hashes/*"):
    stem = Path(file).stem.strip(".")
    if stem not in filesbloom:
        with open(file,"rt") as f:
            for line in f.readlines():
                line = line.strip()
                counter += 1
                if line not in bloom:
                    bloom.add(line)
                    uniques += 1
        filesbloom.add(stem)

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
for file in glob("/home/archiveteam/CAH/clipped/*"):
    stem = Path(file).stem.strip(".")
    if stem not in filesclipped:
        with open(file,"rt") as f:
            for line in f.readlines():
                line = line.strip()
                if line not in clipped:
                    clipped.add(line)
                    clipped_counter += 1
        filesclipped.add(stem)

pd.set_option('precision', 2)
df = pd.read_csv("bloom.log", sep=" ",header=None, names=["Date", "a", "main filter (max 80M, 1%)", "b", "total including duplicates","c","clipped filter (max 200M, 5%)","d","failed filter","e"])
df["main filter (max 80M, 1%)"]=df["main filter (max 80M, 1%)"]/1000000
df["total including duplicates"]=df["total including duplicates"]/1000000
df["clipped filter (max 200M, 5%)"]=df["clipped filter (max 200M, 5%)"]/1000000
with open('dashboard.txt', 'w') as file:
    file.write("<h1>Bloom filters status</h1>\n")
    file.write(str(df.sum(axis=0, numeric_only=True)).replace("\n","<br/>"))

if uniques > 0:
    print(f"[{now}] added {uniques} \"from total of\" {counter} \"(i.e. {round((counter-uniques)*100/(counter+sys.float_info.epsilon),2)}% duplication in {round(time.time()-start,2)} sec) Also added \" {clipped_counter} \"clipped and\" {failed_counter} failed")