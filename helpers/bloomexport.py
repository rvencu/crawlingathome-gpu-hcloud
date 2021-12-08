'''
preable:
script is adapted for large redisbloom filters and will not require double memory size since it iteratively dumps or load chunks of 512MB to disk

arguments:
-m/--mode (dump|restore)
-k/--key key to be dumped or source key for backup to restore
-d/--dest key to be restored to
-p/--path where to store/retrieve the backup files

usage:
1. backup of key "main"

python3 bloomexport.py -m dump -k main

2. restore from backup of key "main" into key "test" (destination key should not exist, it will be created)

python3 bloomexport.py -m restore -k main -d test

'''

import sys
import glob
import pickle
import argparse
from redisbloom.client import Client
r = Client()

def make_dump(r, key, path):
    iter = 0
    while True:
        iter, data = r.bfScandump(key, iter)
        if iter == 0:
            return
        else:
            print(iter)
            with open(f"{path}/{iter}.{key}.bloom","wb") as f:
                pickle.dump(data, f)    

def restore_dump(r, source, dest, path):
    iters = []
    files = glob.glob(f"{path}/*.bloom")
    for file in files:
        try:
            iter, key, ext = file.split("/")[-1].split(".")
            if key == source:
                iters.append(iter)
        except:
            pass
    # reorder chunks ascending
    iters.sort(key=lambda x: int(x))
    for iter in iters:
        with open(f"{path}/{iter}.{source}.bloom","rb") as f:
            data = pickle.load(f)
            r.bfLoadChunk(dest, iter, data)
            print(iter)
    return

if __name__ == "__main__":
    # script initialization
    parser = argparse.ArgumentParser(prog=sys.argv[0], usage='%(prog)s -m/--mode -k/--key -p/--path')
    parser.add_argument("-m","--mode",action='append',help="Choose mode dump or restore", required=True)
    parser.add_argument("-k","--key",action='append',help="Choose bloom key", required=True)
    parser.add_argument("-d","--destination",action='append',help="Choose destination bloom key at restore", required=False)
    parser.add_argument("-p","--path",action='append',help="Choose folder", required=False)
    args = parser.parse_args()
    path = "."
    if args.path is not None:
        path = args.path[0]
    key = args.key[0]
    dest = key
    if args.destination is not None:
        dest = args.destination[0]
    if args.mode[0] == "dump":
        make_dump(r, key, path)
        print(f"dump for {key} saved in {path}")
    elif args.mode[0] == "restore":
        restore_dump(r, key, dest, path)
        print(f"dump for {key} restored as {dest} from {path}")
    else:
        print("bad mode entered")