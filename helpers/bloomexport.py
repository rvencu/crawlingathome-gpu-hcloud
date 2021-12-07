import sys
import glob
import argparse
from redis.client import Redis

r = Redis()

def make_dump(r, key, path):
    chunks = []
    iter = 0

    while True:
        iter, data = r.execute_command(f"BF.SCANDUMP {key} {iter}")
        if iter == 0:
            return
        else:
            print(iter)
            with open(f"{path}/{iter}.{key}.bloom","wb") as f:
                f.write(data)

def restore_dump(r, key, path):
    # Load it back
    chunks = []
    iter = 0
    files = glob.glob(f"{path}/*.bloom")
    for file in files:
        i, k, b = file.split("/")[-1].split(".")
        if k == key:
            chunks.append(i)
    # reorder chunks by iter ascending
    chunks.sort(key=lambda x: int(x))
    for chunk in chunks:
        with open(f"{path}/{chunk}.{key}.bloom","rb") as f:
            data = f.read()
            r.execute_command(f"BF.LOADCHUNK({key}, {chunk}, {data})")
    return

if __name__ == "__main__":
    # script initialization
    parser = argparse.ArgumentParser(prog=sys.argv[0], usage='%(prog)s -m/--mode -k/--key -p/--path')
    parser.add_argument("-m","--mode",action='append',help="Choose mode dump or restore", required=True)
    parser.add_argument("-k","--key",action='append',help="Choose bloom key", required=True)
    parser.add_argument("-p","--path",action='append',help="Choose folder", required=False)
    args = parser.parse_args()
    path = "."
    if args.path is not None:
        path = args.path[0]
    key = args.key[0]
    if args.mode[0] == "dump":
        make_dump(r, key, path)
        print(f"dump for {key} saved in {path}")
    elif args.mode[0] == "restore":
        restore_dump(r, key, path)
        print(f"dump for {key} restored from {path}")
    else:
        print("bad mode entered")