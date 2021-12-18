import os
import pandas as pd
from glob import glob
from bloom_filter2 import BloomFilter
from tqdm import tqdm
unique = BloomFilter(max_elements=10000000000, error_rate=0.01, filename=("./urls.bin",-1))

def isunique(url):
    if url in unique:
        return False
    else:
        unique.add(url)
        return True

files = glob("*.csv")
for file in tqdm(files):
    if not os.path.isfile(f"{file}.deduped"):
        df = pd.read_csv(file, sep="|").drop_duplicates(subset='url', keep='first').reset_index(drop=True)
        df.loc[:,"isunique"] = df.url.apply(lambda x: isunique(x))
        df = df[df["isunique"]==True].reset_index(drop=True)
        df = df.drop(labels="isunique", axis=1)
        df.to_csv(file+".deduped", sep="|", index=False, header=True)
        os.system(f"rm {file}")
        os.system(f"mv {file}.deduped ../../staging/{file}")
