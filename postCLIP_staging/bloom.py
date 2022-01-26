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

with open("bloomlog.txt","a") as log:

    # update the bloom server filters too
    bloomip = "116.202.162.146"

    serverbloom = BloomFilter(max_elements=10000000, error_rate=0.01, filename=(f"/home/archiveteam/bloom-{bloomip}.bin",-1))
    intlbloom = BloomFilter(max_elements=10000000, error_rate=0.01, filename=(f"/home/archiveteam/intl-{bloomip}.bin",-1))
    serverclip = BloomFilter(max_elements=10000000, error_rate=0.01, filename=(f"/home/archiveteam/clip-{bloomip}.bin",-1))

    start = time.time()
    now = datetime.now().strftime("%Y/%m/%d_%H:%M")

    time.sleep(5)
    counter = 0
    counterintl = 0
    uniques = 0
    uniquesintl = 0
    main = [(0,0)]
    intl = [(0,0)]
    for file in glob("/home/archiveteam/CAH/hashes/*.hsh"):
        stem = Path(file).stem.strip(".")
        if stem not in serverbloom:
            with open(file,"rt") as f:
                for line in f.readlines():
                    counter += 1
            post = {
                'file': (stem, open(file, 'rb')),
                'key': (None, 'main'),
            }
            response = requests.post(f'http://{bloomip}:8000/add/', files=post)
            if response.status_code == 200:
                serverbloom.add(stem)
                uniques += int(response.text)
            main.append(tuple(map(lambda i, j: i - j, (counter,uniques), main[-1])))
    del(main[0])
    #log.write(str(main) + "\n")
    for file in glob("/home/archiveteam/CAH/hashesintl/*.hsh"):
        stem = Path(file).stem.strip(".")
        if stem not in intlbloom:
            with open(file,"rt") as f:
                for line in f.readlines():
                    counterintl += 1
            post = {
                'file': (stem, open(file, 'rb')),
                'key': (None, 'multilanguage'),
            }
            response = requests.post(f'http://{bloomip}:8000/add/', files=post)
            if response.status_code == 200:
                intlbloom.add(stem)
                uniquesintl += int(response.text)
            intl.append(tuple(map(lambda i, j: i - j, (counterintl,uniquesintl), intl[-1])))
    del(intl[0])

    clippedlist=[0]
    clipped_counter = 0
    for file in glob("/home/archiveteam/CAH/clipped/*.clp"):
        stem = Path(file).stem.strip(".")
        if stem not in serverclip:
            post = {
                'file': (stem, open(file, 'rb')),
                'key': (None, 'clipped'),
            }
            response = requests.post(f'http://{bloomip}:8000/add/', files=post)
            if response.status_code == 200:
                serverclip.add(stem)
                clipped_counter += int(response.text)
            clippedlist.append(clipped_counter-clippedlist[-1])
    del clippedlist[0]
    #log.write(str(clippedlist) + "\n")

    pd.set_option('precision', 2)
    df = pd.read_csv("bloom.log", sep=" ",header=None, names=["Date", "a", "unique pairs (5%)", "b", "total including duplicates","c","clipped filter (5%)","d","failed filter","e"])
    df["Date"]=df.Date.apply(lambda x: datetime.strptime(x, "[%Y/%m/%d_%H:%M]"))
    df["unique pairs (5%)"]=df["unique pairs (5%)"]/1000000
    df["total including duplicates"]=df["total including duplicates"]/1000000
    df["clipped filter (5%)"]=df["clipped filter (5%)"]/1000000

    #log.write("Done df calc \n")
    if uniques + uniquesintl + clipped_counter > 0:
        print(f"[{now}] added {uniques + uniquesintl} \"from total of\" {counter + counterintl} \"( {str(main)} i.e. {round((counter +  counterintl - uniques - uniquesintl)*100/(counter + counterintl + sys.float_info.epsilon), 2)}% duplication in {round(time.time()-start,2)} sec) Also added \" {clipped_counter} \" {str(clippedlist)} clipped\" and 0 failed")

        #log.write("Printed stats \n")

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
            file.write("<h5>Last reset date: 01 December 2021</h5>\n")
            file.write(str(df[df.Date > datetime.now() - pd.to_timedelta("7day")].sum(axis=0, numeric_only=True)).replace("\n","<br/>"))
        #log.write("Printed dashboard \n")