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

files = glob("*.txt")
with tqdm(total=len(files), file=sys.stdout) as pbar:
    pbar.desc = "1"
    for file in files:
        age = time.time() - path.getmtime(file)
        if not os.path.isfile(f"{file}.deduped") and age > 24*60*60:
            try:
                df = pd.read_csv(file, sep="\t", names=["s","url","a","b","c","d","e","f"])
                df.drop_duplicates(subset="url", keep='first').reset_index(drop=True)

                with open('hash.txt', 'w') as f:
                    f.write(df['url'].str.cat(sep='\n'))
                post = {
                    'file': ('hash.txt', open('hash.txt', 'rb')),
                    'key': (None, "dedup"),
                }
                os.remove('hash.txt')
                
                failure = True
                for _ in range(10):
                    response = requests.post(f'http://{bloomip}:8000/deduplicate/', files=post)
                    if response.status_code != 200:
                        time.sleep(randint(5,30))
                    else:
                        failure = False
                        break
                if failure:
                    continue

                valid_urls = response.content.decode("utf-8").split("\n")

                ratio = round(len(valid_urls) / len(df.index), 2)

                df = df[df.url.isin(valid_urls)]
                df.reset_index(inplace=True, drop=True)

                df.to_csv(file+".deduped", sep="\t", index=False, header=False)

                            # add parsed urls to parsed bloom server
                with open('hash.txt', 'w') as f:
                    for url in valid_urls:
                        f.write(url.strip()+"\n")
                post = {
                    'file': ('hash.txt', open('hash.txt', 'rb')),
                    'key': (None, 'dedup'),
                }
                os.remove('hash.txt')

                failure = True
                for _ in range(10):
                    try:
                        response = requests.post(f'http://{bloomip}:8000/add/', files=post)
                        if response.status_code != 200:
                            time.sleep(randint(5,30))
                        else:
                            failure = False
                            break
                    except:
                        time.sleep(15)
                if failure:
                    continue
                os.system(f"rm {file}")
                os.system(f"mv {file}.deduped {file}")
                pbar.desc = str(ratio)
                pbar.update(1)
            except Exception as e:
                print (e)
