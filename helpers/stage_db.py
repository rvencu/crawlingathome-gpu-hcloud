import os
import re
import sys
import psycopg2
import argparse
import fileinput
from glob import glob
import pandas as pd
from sqlalchemy import create_engine
from configparser import ConfigParser
from tqdm.auto import tqdm


def config(filename='database.ini', mode="test"):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    section='postgresql'
    if mode == "production":
        section='cah_production'

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db

def get_count(engine, ds="intl"):
    table="dataset_intl"
    if ds == "en":
        table = "dataset_en"
    elif ds == "nolang":
        table = "dataset_nolang"
    select_stmt1 = f"select count(*) from {table} where status = 0"
    conn = engine.raw_connection()
    cur = conn.cursor()
    cur.execute(select_stmt1)
    count = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return str(count[0])

parser = argparse.ArgumentParser(prog=sys.argv[0], usage='%(prog)s -m/--mode -s/--set -p/--path')
parser.add_argument("-m","--mode",action='append',help="Mode to run", required=True)
parser.add_argument("-s","--set",action='append',help="Dataset to run", required=False)
parser.add_argument("-p","--path",action='append',help="Choose source path", required=False)
args = parser.parse_args()

dir = "/mnt/md1/export/staging"
if args.path is not None:
    dir = args.path[0]

mode = "txt"
if args.mode is not None:
    mode = args.mode[0]

ds = "intl"
if args.set is not None:
    ds = args.set[0]

i = 0

params = config(mode="production")
engine = create_engine(f'postgresql://{params["user"]}:{params["password"]}@{params["host"]}:5432/{params["database"]}', pool_size=5, max_overflow=10, pool_pre_ping=True)

files = glob(f'{dir}/*.{mode}')

conn = engine.raw_connection()

j = 10
if mode == "txt":
    j = 1000

with tqdm(total=len(files), file=sys.stdout) as pbar:
    pbar.desc = get_count(engine, ds)
    for file in files:
        try:
            cur = conn.cursor()
            with open(file, "rt") as f:
                if mode == "txt":
                    cur.copy_from(f, 'dataset_buffer', columns=("sampleid","url","text","license","domain","wat","hash","language"))
                elif mode == "csv":
                    cur.copy_expert("COPY dataset_buffer from STDIN DELIMITER '|' CSV HEADER", f)
                else:
                    print("bad mode, choose txt or csv only")
                    break
            conn.commit()
            cur.close()
            os.system(f"mv {file} {file}.done")
            i+=1
            if i % j == 0:
                count = get_count(engine, ds)
                if int(count) > 500000000:
                    break
                else:
                    pbar.desc = count
            pbar.update(1)
                
        except Exception as e:
            print(f"error {file} because {e}")
            for line in fileinput.input(file, inplace = True):
                if not re.search(r'\x00', line):
                    print(line, end="")
            try:
                df = pd.read_csv(file, sep="\t", on_bad_lines='skip', header=None)
                df[2] = df[2].apply(lambda x: x.replace("\n",""))
                df[5] = df[5].apply(lambda x: int(x))
                df.to_csv(file, sep="\t", index=False, header=False)
            except:
                #os.system(f"mv {file} {file}.error")
                pass
            conn.close()
            conn = engine.raw_connection()
conn.close()

print("if you had files with error of \x00 present in file, files were automatically corrected, please rerun the script")