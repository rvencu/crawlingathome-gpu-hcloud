import pandas as pd
from glob import glob
from tqdm.auto import tqdm
import numpy as np
import os
import re

def wonky_parser(fn):
    txt = open(fn).read()
    #                          This is where I specified 8 tabs
    #                                        V
    preparse = re.findall('(([^\t]*\t[^\t]*){7}(\n|\Z))', txt)
    parsed = [t[0].split('\t') for t in preparse]
    return pd.DataFrame(parsed)

def is_num(x):
    try:
        x = int(float(x))
        return True
    except:
        return False

files = glob("*.bad")

for file in tqdm(files):
    try:
        df = pd.read_csv(file, sep="\t", names=["s","url","a","b","c","d","e","f"], dtype={'s': 'int', 'd': 'int', 'url': 'str', 'a': 'str'})
        df.drop_duplicates(subset="url", keep='first').reset_index(drop=True)
        df.s = df.s.astype(int)
        df.d = df.d.astype(int)
        df.to_csv(file+".fix", sep="\t", index=False, header=False)
        os.system(f"rm {file}")
        os.system(f"mv {file}.fix {file}")
    except:
        df = wonky_parser(file)
        df.columns=["s","url","a","b","c","d","e","f"]
        df = df[df.s.apply(lambda x: is_num(x))]
        df = df[df.d.apply(lambda x: is_num(x))]
        df.drop_duplicates(subset="url", keep='first').reset_index(drop=True)
        df.s = df.s.apply(lambda x: int(float(x)))
        df.d = df.d.apply(lambda x: int(float(x)))
        df["s"] = df["s"].astype(int)
        df["d"] = df["d"].astype(int)
        df.to_csv(file+".fix", sep="\t", index=False, header=False)
        os.system(f"rm {file}")
        os.system(f"mv {file}.fix {file}")
        

