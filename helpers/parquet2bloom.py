import os
import sys
import pandas as pd
import requests
from glob import glob
from random import randint
from tqdm.auto import tqdm
import os.path as path
import time
bloomip = "116.202.162.146"

files = glob("**/*.parquet", recursive=True)
with tqdm(total=len(files), file=sys.stdout) as pbar:

    for file in files:
        
        df = pd.read_parquet(file)

        n = 100000  #chunk row size
        list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]

        for ldf in list_df:
            with open('hash.txt', 'w') as f:
                f.write(ldf['URL'].str.cat(sep='\n'))
            post = {
                'file': ('hash.txt', open('hash.txt', 'rb')),
                'key': (None, "dedup"),
            }
            os.remove('hash.txt')
            
            failure = True
            for _ in range(10):
                response = requests.post(f'http://{bloomip}:8000/add/', files=post)
                if response.status_code != 200:
                    time.sleep(randint(5,30))
                else:
                    failure = False
                    break
            if failure:
                print("could not add chunk")
                continue

            with open('hash.txt', 'w') as f:
                f.write(ldf['URL'].str.cat(sep='\n'))
            post = {
                'file': ('hash.txt', open('hash.txt', 'rb')),
                'key': (None, "nolang"),
            }
            os.remove('hash.txt')
            
            failure = True
            for _ in range(10):
                response = requests.post(f'http://{bloomip}:8000/add/', files=post)
                if response.status_code != 200:
                    time.sleep(randint(5,30))
                else:
                    failure = False
                    break
            if failure:
                print("could not add chunk")
                continue

        pbar.update(1)

